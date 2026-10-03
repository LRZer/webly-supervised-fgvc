#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Phase-1 极简清洗（带实时日志）
#规则：
#1. 无法解码
#2. 极小文件小于10b / 最短边太小
#3. 几乎全透明 PNG/GIF

#用法：python 1_clean_minimal.py \
#  --input /home/ma-user/work/data/webfg400_train \
#  --trash /home/ma-user/work/data/webfg400_trash_phase1 \
#  --min-bytes 8192 \
#  --alpha-check 1 \
#  --alpha-opaque-min 0.08 \
#  --workers 8 \
#  --dry_run

# 400清了10b的，5000清了8b的  10240

# 输入：/home/ma-user/work/data/webfg400_train \
# 输出：删除原文件内不符合要求的图片，并且复制一份不符要求图片进入webfg400_trash_phase1


import argparse, csv, os, shutil, sys, time
from pathlib import Path
from typing import Tuple, Optional, Dict
from PIL import Image, UnidentifiedImageError
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def is_image_ext(path: Path) -> bool:
    return path.suffix.lower() in {".jpg",".jpeg",".png",".webp",".gif",".bmp",".tif",".tiff"}

def inspect_one(p: str,
                root: str,
                min_bytes: int,
                min_side: int,
                alpha_check: int,
                alpha_opaque_min: float,
                alpha_sample_pixels: int) -> Optional[Dict]:
    """
    返回需要删除的样本信息 dict，若不删除则返回 None。
    只在必要时做一次解码；alpha 检测可关闭或降采样。
    """
    path = Path(p)
    # 规则2：过小文件，连解码都不做
    try:
        nbytes = path.stat().st_size
    except FileNotFoundError:
        return None
    if nbytes < min_bytes:
        return dict(reason=f"small_bytes<{min_bytes}", bytes=nbytes, rel=str(path.relative_to(root)))

    # 尝试解码 + 拿尺寸（一次打开干两件事）
    try:
        with Image.open(path) as im:
            im.load()
            w, h = im.width, im.height
            if min(w, h) < min_side:
                return dict(reason=f"small_side<{min_side}", bytes=nbytes, rel=str(path.relative_to(root)))

            # 规则3：几乎全透明（可关闭）
            if alpha_check and im.format in ("PNG", "GIF"):
                if im.format == "GIF":
                    try: im.seek(0)
                    except EOFError: pass
                img = im.convert("RGBA")
                if alpha_sample_pixels and (w * h > alpha_sample_pixels):
                    step = int((w * h / alpha_sample_pixels) ** 0.5)
                    img = img.resize((max(1, w // step), max(1, h // step)), Image.NEAREST)
                a = np.array(img.getchannel("A"), dtype=np.uint8)
                opaque_ratio = float((a > 0).sum()) / float(a.size) if a.size else 0.0
                if opaque_ratio < alpha_opaque_min:
                    return dict(reason=f"nearly_transparent({opaque_ratio:.3f})",
                                bytes=nbytes, rel=str(path.relative_to(root)))
        # 以上都没触发 → 保留
        return None
    except (UnidentifiedImageError, OSError, ValueError):
        # 规则1：解码失败
        return dict(reason="decode_fail", bytes=nbytes, rel=str(path.relative_to(root)))
    except Exception:
        return None  # 其他异常：保守起见先保留

def main():
    ap = argparse.ArgumentParser(description="Phase-1 极简清洗（并行加速）")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--trash", required=True, type=Path)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--min-bytes", type=int, default=10_240)
    ap.add_argument("--min-side", type=int, default=64)
    ap.add_argument("--alpha-check", type=int, default=1, help="是否检查全透明PNG/GIF(1/0)")
    ap.add_argument("--alpha-opaque-min", type=float, default=0.05)
    ap.add_argument("--alpha-sample-pixels", type=int, default=512*512,
                    help="alpha 检测时的最大采样像素数(降采样提速); 0=不降采样")
    ap.add_argument("--dry_run", "--dry-run", action="store_true", default=True)
    ap.add_argument("--apply", dest="dry_run", action="store_false", help="Apply cleaning after dry-run review")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--print-every", type=int, default=400)
    args = ap.parse_args()

    root, trash_root = args.input.resolve(), args.trash.resolve()
    if not root.is_dir(): raise ValueError("Input directory does not exist")
    if root == trash_root or root in trash_root.parents or trash_root in root.parents:
        raise ValueError("Input and trash directories must be separate trees")
    if args.min_bytes < 0 or args.min_side < 1 or not 0 <= args.alpha_opaque_min <= 1:
        raise ValueError("Invalid image filter thresholds")
    trash_root.mkdir(parents=True, exist_ok=True)
    log_path = args.log or (trash_root / "phase1_removed.csv")

    all_files = [str(p) for p in root.rglob("*") if p.is_file() and is_image_ext(p)]
    total = len(all_files)
    print(f"[INFO] Start scanning {total} image files...")
    t0 = time.time()

    removed_rows = []
    removed_cnt = 0

    # 并行处理（多进程更适合 CPU 解码）
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for p in all_files:
            futures.append(ex.submit(
                inspect_one, p, str(root),
                args.min_bytes, args.min_side,
                args.alpha_check, args.alpha_opaque_min, args.alpha_sample_pixels
            ))

        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res is not None:
                removed_cnt += 1
                rel = Path(res["rel"])
                src = root / rel
                dst = trash_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not args.dry_run:
                    try:
                        shutil.copy2(src, dst)
                        src.unlink()
                    except Exception as e:
                        print(f"[WARN] 复制/删除失败: {src} ({e})")
                removed_rows.append({
                    "orig_path": str(src), "trash_path": str(dst),
                    "reason": res["reason"], "bytes": res["bytes"]
                })
                print(f"[DEL] {rel} -> {res['reason']}")

            if i % args.print_every == 0:
                elapsed = time.time() - t0
                print(f"[PROGRESS] {i}/{total} processed | removed={removed_cnt} | {elapsed:.1f}s")

    # 写日志
    if removed_rows and not args.dry_run:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["orig_path","trash_path","reason","bytes"])
            writer.writeheader()
            writer.writerows(removed_rows)

    print(f"\n[DONE] total={total}, removed={removed_cnt}, dry_run={args.dry_run}")
    print(f"[TIME] {time.time()-t0:.1f}s | log={log_path}")

if __name__ == "__main__":
    main()
