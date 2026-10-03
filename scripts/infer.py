#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, csv, time, argparse
from pathlib import Path
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import timm
from torchvision import transforms
from tqdm import tqdm

# ------------------
# Dataset: flat folder of images (no subdirs)
# ------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

class FlatImageFolder(Dataset):
    def __init__(self, root: str, tfm):
        self.root = Path(root)
        self.paths: List[Path] = []
        for p in sorted(self.root.iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                self.paths.append(p)
        self.tfm = tfm

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        try:
            im = Image.open(p)
            # 统一把带透明度/调色板的图像转到 RGB，避免 PIL 警告 & 非3通道问题
            mode = getattr(im, "mode", "RGB")
            if mode in ("P", "PA", "LA") or mode == "RGBA":
                im = im.convert("RGBA").convert("RGB")
            else:
                im = im.convert("RGB")
            if self.tfm is not None:
                im = self.tfm(im)
            return im, p.name
        except Exception as e:
            # 返回 None，让 collate 丢弃
            return None

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    xs, fnames = zip(*batch)
    return torch.stack(xs, 0), list(fnames)

# ------------------
# TTA helpers
# ------------------
def build_transforms(img_size=320):
    mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
    tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return tf

def horizontal_flip(x):
    return torch.flip(x, dims=[-1])

# ------------------
# Load class mapping from classes.txt (i \t name)
# ------------------
def load_classes(classes_txt: str):
    idx2name = []
    with open(classes_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            # format: "i\t000"  or "i\tclass_name"
            parts = line.split("\t")
            if len(parts) == 2:
                if int(parts[0]) != len(idx2name): raise ValueError("Class indices must be consecutive and ordered")
                idx2name.append(parts[1])
            else:
                # fallback: whole line is the name
                idx2name.append(parts[-1])
    if not idx2name or len(set(idx2name)) != len(idx2name) or not all(n.isdigit() and len(n) <= 4 for n in idx2name):
        raise ValueError("Expected unique numeric competition class names")
    return idx2name

# ------------------
# Main
# ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-root", required=True, help="测试集图片所在目录（平铺）")
    ap.add_argument("--weights", required=True, help="训练产出的 best.pth (EMA)")
    ap.add_argument("--classes", required=True, help="训练时生成的 classes.txt")
    ap.add_argument("--out", default="pred_results_web400.csv", help="输出 CSV 文件名")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--tta", action="store_true", help="水平翻转 TTA")
    ap.add_argument("--allow-skipped", action="store_true", help="Allow an incomplete CSV when some images cannot be decoded")
    args = ap.parse_args()
    if args.workers < 0 or args.batch_size < 1: raise ValueError("Invalid loader parameters")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")

    # 1) transforms / dataset / loader
    tf = build_transforms(args.img_size)
    ds = FlatImageFolder(args.test_root, tf)
    if len(ds) == 0:
        print(f"[ERR] 未在 {args.test_root} 发现图片")
        sys.exit(1)
    print(f"[INFO] 共发现测试图片：{len(ds)}")

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=bool(args.workers),
        prefetch_factor=2 if args.workers else None, collate_fn=safe_collate
    )

    # 2) class mapping
    idx2name = load_classes(args.classes)
    num_classes = len(idx2name)
    print(f"[INFO] 类别数 = {num_classes}")

    # 3) build model & load weights
    model = timm.create_model(
        "convnext_base.fb_in1k",
        pretrained=False,
        num_classes=num_classes,   # 与训练一致
        drop_path_rate=0.3
    ).to(device)

    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    # Require the saved model architecture and classification head to match.
    sd = torch.load(args.weights, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    print(f"[INFO] loaded weights: {args.weights}")
    if missing:   print("[WARN] missing keys:", len(missing))
    if unexpected:print("[WARN] unexpected keys:", len(unexpected))

    model.eval()

    # 4) inference
    from torch.amp import autocast
    preds = []
    t0 = time.time()

    with torch.no_grad():
        pbar = tqdm(loader, total=len(loader), desc="Infer")
        for batch in pbar:
            if batch is None: 
                continue
            x, fnames = batch
            x = x.to(device, non_blocking=True)
            if args.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)

            # TTA: center crop + hflip
            with autocast(device_type=device, enabled=(device=="cuda")):
                logits = model(x)
                if args.tta:
                    x_flip = horizontal_flip(x)
                    logits_flip = model(x_flip)
                    logits = (logits + logits_flip) * 0.5

                probs = torch.softmax(logits, dim=1)
                pred_idx = probs.argmax(dim=1).detach().cpu().tolist()

            for fname, k in zip(fnames, pred_idx):
                # name 可能已是 "000" 这种四位；若不是四位数字，按 4 位补零
                name = idx2name[k]
                if name.isdigit() and len(name) <= 4:
                    cls_str = name.zfill(4)
                else:
                    # 兜底：用索引转为四位
                    cls_str = str(k).zfill(4)
                preds.append((fname, cls_str))

    dt = time.time() - t0
    print(f"[INFO] 推理完成，共 {len(preds)} 条，用时 {dt:.1f}s")

    # 5) save CSV
    if len(preds) != len(ds) and not args.allow_skipped:
        raise RuntimeError(f"Only {len(preds)} of {len(ds)} test images decoded; CSV was not written")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for fname, cls_str in preds:
            w.writerow([fname, cls_str])
    print(f"[OK] 写出提交文件: {out_path.resolve()}")
    print("     示例行：", preds[0] if preds else "（空）")

if __name__ == "__main__":
    main()
