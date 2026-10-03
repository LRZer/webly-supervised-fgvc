#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, shutil, math, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def list_images(dirp: Path):
    files = []
    for p in dirp.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_one(src: Path, dst: Path, mode: str):
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "hardlink":
        # Windows/某些挂载下可能不支持；失败就回退成 copy
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        try:
            dst.symlink_to(src)
        except Exception:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown copy mode: {mode}")

def main():
    ap = argparse.ArgumentParser(description="Split single-level class folder into train/val directories.")
    ap.add_argument("--src", required=True, help="源数据根目录（单层类别目录） e.g. /data/webfg400_train")
    ap.add_argument("--dst", required=True, help="输出根目录（会在里面创建 train/ 和 val/） e.g. /data/webfg400")
    ap.add_argument("--val-ratio", type=float, default=0.15, help="验证集比例，默认 0.15")
    ap.add_argument("--seed", type=int, default=2025, help="随机种子，保证可复现")
    ap.add_argument("--min-val", type=int, default=1, help="每类验证集最少保留多少张（极小类也能评估），默认 1")
    ap.add_argument("--mode", choices=["copy", "move", "hardlink", "symlink"], default="copy",
                    help="文件放置方式：copy/move/hardlink/symlink（默认 copy 最稳妥）")
    ap.add_argument("--workers", type=int, default=8, help="并行线程数")
    ap.add_argument("--dry-run", action="store_true", help="演练模式，不真正拷贝/移动，只打印统计")
    args = ap.parse_args()
    if not 0 < args.val_ratio < 1 or args.min_val < 0 or args.workers < 1:
        raise ValueError("Invalid split parameters")

    src = Path(args.src)
    dst = Path(args.dst).resolve()
    src = src.resolve()
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError("Source and destination must be separate trees")
    if dst.exists() and any(dst.iterdir()):
        raise ValueError("Destination must be empty to avoid mixing previous splits")
    train_root = dst / "train"
    val_root   = dst / "val"

    if not src.exists():
        print(f"[ERR] 源目录不存在: {src}")
        sys.exit(1)

    random.seed(args.seed)

    # 收集类目录
    class_dirs = [d for d in sorted(src.iterdir()) if d.is_dir()]
    if len(class_dirs) == 0:
        print(f"[ERR] 在 {src} 下没有找到任何子目录（类别）。")
        sys.exit(1)

    # 创建输出目录
    if not args.dry_run:
        safe_mkdir(train_root)
        safe_mkdir(val_root)

    stats = []  # [(cls_name, n_total, n_train, n_val)]
    tasks = []

    # 先统计与切分
    for cdir in class_dirs:
        cls_name = cdir.name
        imgs = list_images(cdir)
        n = len(imgs)
        if n == 0:
            print(f"[WARN] 类别 {cls_name} 没有图片，跳过。")
            stats.append((cls_name, 0, 0, 0))
            continue

        # 随机打乱
        random.shuffle(imgs)

        # 计算每类的 val 数量（至少 min-val，但不能等于总数）
        n_val = max(args.min_val, int(round(n * args.val_ratio)))
        if n == 1:
            # 只有 1 张图：全部给 train，val=0
            n_val = 0
        elif n_val >= n:
            n_val = n - 1  # 至少留 1 张给训练

        val_imgs = imgs[:n_val]
        trn_imgs = imgs[n_val:]

        stats.append((cls_name, n, len(trn_imgs), len(val_imgs)))

        if not args.dry_run:
            # 创建类目录
            safe_mkdir(train_root / cls_name)
            safe_mkdir(val_root / cls_name)
            # 准备拷贝任务
            for s in trn_imgs:
                d = train_root / cls_name / s.name
                tasks.append((s, d, args.mode))
            for s in val_imgs:
                d = val_root / cls_name / s.name
                tasks.append((s, d, args.mode))

    # 打印统计
    tot_n = sum(t[1] for t in stats)
    tot_train = sum(t[2] for t in stats)
    tot_val = sum(t[3] for t in stats)
    print(f"[INFO] 类别数: {len([t for t in stats if t[1]>0])} / 目录数: {len(class_dirs)}")
    print(f"[INFO] 总图片: {tot_n} | 训练: {tot_train} | 验证: {tot_val} (ratio ~ {tot_val / max(tot_n,1):.3f})")

    # 逐类打印前几项，便于抽查
    show = min(10, len(stats))
    print("[INFO] 示例（前 10 类，n_total -> train/val）：")
    for cls_name, n_total, n_tr, n_va in stats[:show]:
        print(f"  {cls_name}: {n_total} -> {n_tr}/{n_va}")

    if args.dry_run:
        print("[DRY-RUN] 仅演练，不进行文件操作。")
        return

    # Record exact split membership as well as the seed.
    import csv
    with (dst / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["source", "split", "class_name", "filename"])
        for s, d, mode in tasks:
            w.writerow([str(s), d.relative_to(dst).parts[0], d.parent.name, d.name])

    # 执行文件放置
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for s, d, mode in tasks:
            futs.append(ex.submit(copy_one, s, d, mode))
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Placing files ({args.mode})"):
            fut.result()

    print("[OK] 数据划分完成。输出目录：", dst)
    print("     训练集：", train_root)
    print("     验证集：", val_root)

if __name__ == "__main__":
    main()
