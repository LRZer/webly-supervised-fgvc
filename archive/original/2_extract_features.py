#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, os, sys, time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import timm
import torchvision.transforms as T

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}

def list_images_by_class(root: Path) -> Tuple[List[str], List[int], List[str]]:
    """遍历一级子目录作为类别，返回 文件路径列表、class_id 列表、class_name 列表"""
    items_path, items_cid, items_cname = [], [], []
    class_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    # 如果子目录名就是数字（如 000/001），按数字做 class_id；否则按顺序分配
    def parse_cid(name: str, idx: int) -> int:
        try:
            return int(name)
        except Exception:
            return idx
    for i, d in enumerate(class_dirs):
        cname = d.name
        cid = parse_cid(cname, i)
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                items_path.append(str(p.resolve()))
                items_cid.append(cid)
                items_cname.append(cname)
    return items_path, items_cid, items_cname

class ImageFolderFlat(Dataset):
    def __init__(self, paths: List[str], cids: List[int], transform: T.Compose):
        self.paths = paths
        self.cids = cids
        self.t = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
            return self.t(im), self.cids[i], path
        except (UnidentifiedImageError, OSError, ValueError) as e:
            # 返回 None，后面 collate_fn 会过滤
            return None

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs = torch.stack([b[0] for b in batch], dim=0)
    cids = torch.tensor([b[1] for b in batch], dtype=torch.long)
    paths = [b[2] for b in batch]
    return xs, cids, paths

def l2_normalize(x: np.ndarray, eps: float=1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def main():
    ap = argparse.ArgumentParser(description="Phase-2: 提取图像特征 (DINOv2-S/14 默认)")
    ap.add_argument("--input", required=True, type=Path, help="数据根目录（子文件夹为类别）")
    ap.add_argument("--outdir", required=True, type=Path, help="输出目录")
    ap.add_argument("--model", type=str, default="deit3_base_patch16_224_fb_in1k",
                    help="timm 模型名，默认 DINOv2-S/14")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--img-size", type=int, default=224, help="中心裁剪尺寸")
    ap.add_argument("--short-side", type=int, default=256, help="缩放的短边")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fp16", action="store_true", help="开启半精度推理")
    ap.add_argument("--save-fp16", action="store_true", help="把特征以 float16 存盘（默认）")
    ap.add_argument("--no-save-fp16", dest="save_fp16", action="store_false")
    ap.set_defaults(save_fp16=True)
    
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    log_bad = args.outdir / "phase2_bad_images.csv"
    idx_csv = args.outdir / "index.csv"
    feat_npy = args.outdir / "features.npy"

    # 列出样本
    paths, cids, cnames = list_images_by_class(args.input)
    N = len(paths)
    if N == 0:
        print("[ERR] 数据目录下没有找到图片。")
        sys.exit(1)
    print(f"[INFO] Found {N} images under {args.input}")

    # 预处理：短边 256 → 中心裁 224 → ToTensor → ImageNet 标准化
    transform = T.Compose([
        T.Resize(args.short_side, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    ds = ImageFolderFlat(paths, cids, transform)
    dl = DataLoader(
                    ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True,
                    prefetch_factor=4, persistent_workers=True,
                    collate_fn=collate_skip_none, drop_last=False
                )

    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        args.model, pretrained=False, num_classes=0,
        checkpoint_path=str(args.checkpoint) if args.checkpoint else ''
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # ✨ 多卡一键启用（1.8.2 可用）
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using DataParallel on {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)
    
    model.eval().to(device)
    print(f"[INFO] Model: {args.model}  | device={device}  | fp16={args.fp16}")

    feats_list: List[np.ndarray] = []
    idx_rows = []  # 写 index.csv
    bad_rows = []  # 失败图片日志

    processed = 0
    t0 = time.time()
    autocast_dtype = torch.float16 if args.fp16 else torch.float32

    with torch.no_grad():
        for step, batch in enumerate(dl, 1):
            if batch is None:
                continue
            x, c, p = batch
            x = x.to(device, non_blocking=True)
            # 半精度前向
            if args.fp16:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    y = model(x)
            else:
                y = model(x)
            y = y.detach().float().cpu().numpy()  # 转 float32 CPU
            y = l2_normalize(y)                   # 先做 L2 归一化，后续更稳

            feats_list.append(y)
            # 记录索引（绝对路径、class_id、class_name）
            for path_i, cid_i in zip(p, c.tolist()):
                # 注意：这里取类名=其父目录名
                cname = Path(path_i).parent.name
                idx_rows.append((path_i, cid_i, cname))

            processed += len(p)
            if step % 20 == 0 or processed == N:
                elapsed = time.time() - t0
                print(f"[PROGRESS] {processed}/{N}  elapsed={elapsed:.1f}s")

    # 合并保存
    feats = np.concatenate(feats_list, axis=0)  # (M, D)；M<=N（失败样本已被过滤）
    if args.save_fp16:
        feats = feats.astype(np.float16)
    np.save(feat_npy, feats)

    with open(idx_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path","class_id","class_name"])
        w.writerows(idx_rows)

    print(f"[DONE] features: {feat_npy}  shape={feats.shape}  dtype={feats.dtype}")
    print(f"[DONE] index:    {idx_csv}  rows={len(idx_rows)}")
    print(f"[TIME] {time.time()-t0:.1f}s total")
    # 提醒：有失败图片（如有）会在 DataLoader 里被丢弃；若想额外记录，可在 __getitem__ 里把异常写入 log_bad

if __name__ == "__main__":
    main()
