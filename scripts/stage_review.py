#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage removed samples for review, then (optionally) delete originals.

A 阶段（默认）: 复制 removed_list.csv 中的图到 review 目录（保留类层级），并输出 per-class 统计
B 阶段（可选）: 加 --delete（可配合 --force-delete），按名单从原数据集中删除

Inputs
  --root         原始数据集根目录（子文件夹为类别）
  --removed      removed_list.csv（至少包含 path,class_id）
  --score-csv    sample_score.csv（用于统计 total/removed/keep_rate）
  --review-dir   审核目录（复制的目标目录）

Safety
  - 删除前默认会检查 review 目录中是否已存在每个源文件的副本；未找到则跳过（可用 --force-delete 跳过检查）
"""

import argparse, os, shutil, time
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def delete_file(p: Path):
    try:
        p.unlink()
        return True
    except FileNotFoundError:
        return False

def relpath_under_root(src: Path, root: Path) -> Path:
    # 保留原有相对层级
    return src.relative_to(root)

def main():
    ap = argparse.ArgumentParser(description="Stage removed samples for review, then optionally delete them.")
    ap.add_argument("--root", required=True, type=Path, help="原始数据根目录（按类分文件夹）")
    ap.add_argument("--removed", required=True, type=Path, help="removed_list.csv（来自打分脚本）")
    ap.add_argument("--score-csv", required=True, type=Path, help="sample_score.csv（用于统计）")
    ap.add_argument("--review-dir", required=True, type=Path, help="复制到此目录以供目检")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--print-every", type=int, default=500)
    ap.add_argument("--delete", action="store_true", help="执行删除（阶段B）。不加该项则只执行复制（阶段A）")
    ap.add_argument("--force-delete", action="store_true", help="删除时不检查 review 副本是否存在（危险）")
    args = ap.parse_args()

    t0 = time.time()
    root = args.root.resolve()
    review = args.review_dir.resolve()
    review.mkdir(parents=True, exist_ok=True)

    # 读取名单（显式 dtype，避免 DtypeWarning）
    df_rm = pd.read_csv(args.removed, dtype={"path": str, "class_id": int}, low_memory=False)
    df_sc = pd.read_csv(args.score_csv, dtype={"path": str, "class_id": int, "class_name": str}, low_memory=False)

    # 统计 per-class
    total = df_sc.groupby("class_id").size().rename("total")
    removed = df_rm.groupby("class_id").size().rename("removed")
    stat = pd.concat([total, removed], axis=1).fillna(0).astype(int)
    # class_name 映射
    if "class_name" not in df_sc.columns:
        df_sc["class_name"] = df_sc["path"].apply(lambda p: Path(p).parent.name)
    name_map = (df_sc.groupby(["class_id","class_name"]).size()
                .reset_index(name="n")
                .sort_values(["class_id","n"], ascending=[True,False])
                .drop_duplicates("class_id")
                .set_index("class_id")["class_name"].to_dict())
    stat["class_name"] = stat.index.map(name_map)
    stat["remove_rate"] = (stat["removed"] / stat["total"]).round(4)
    stat = stat.reset_index()[["class_id","class_name","total","removed","remove_rate"]]
    stat_path = review / "class_remove_stats.csv"
    stat.to_csv(stat_path, index=False)

    # 绝对路径列表
    src_list = [Path(p).resolve() for p in df_rm["path"].tolist()]
    if root == review or root in review.parents or review in root.parents:
        raise ValueError("Review and input directories must be separate trees")
    for source in src_list:
        source.relative_to(root)  # Reject paths outside the named dataset before file operations.
    print(f"[INFO] removed candidates: {len(src_list)}")

    # 阶段A：复制到 review 目录
    if not args.delete:
        print(f"[A] Copying removed samples -> {review}")
        ok, fail = 0, 0
        def dst_from_src(src: Path) -> Path:
            rel = relpath_under_root(src, root)
            # 按 <review>/<class_name>/<filename> 存放
            return review / rel

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(copy_file, s, dst_from_src(s)) for s in src_list]
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    fut.result(); ok += 1
                except Exception as e:
                    fail += 1
                    print(f"[WARN] copy failed: {e}")
                if i % args.print_every == 0 or i == len(futs):
                    print(f"[PROGRESS] {i}/{len(futs)} copied")

        print(f"[DONE A] copied={ok}, failed={fail}, stats={stat_path}")
        print("[TIP ] 目检完成后，可加 --delete 执行阶段B删除。")
        return

    # 阶段B：删除原始数据中的对应文件
    print(f"[B] Deleting originals from {root}  (force_delete={args.force_delete})")
    # 除非 --force-delete，否则先检查 review 副本是否存在
    def review_has_copy(src: Path) -> bool:
        rel = relpath_under_root(src, root)
        cand = review / rel
        return cand.exists()

    ok_del, skip_del, miss = 0, 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for s in src_list:
            if args.force_delete or review_has_copy(s):
                futs.append(ex.submit(delete_file, s))
            else:
                skip_del += 1
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                deleted = fut.result()
                ok_del += int(deleted)
                miss   += int(not deleted)
            except Exception as e:
                miss += 1
                print(f"[WARN] delete failed: {e}")
            if i % args.print_every == 0 or i == len(futs):
                print(f"[PROGRESS] {i}/{len(futs)} deleted (ok={ok_del}, miss={miss}, skipped={skip_del})")

    print(f"[DONE B] deleted_ok={ok_del}, missing={miss}, skipped={skip_del}, stats={stat_path}")
    print(f"[TIME] {time.time()-t0:.1f}s")
    
if __name__ == "__main__":
    main()
