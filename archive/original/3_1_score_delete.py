#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-2: 打分 + 生成删/留名单（phase2_scores中的文件）（不改动文件）
输入:
  - features.npy : (N, D) 特征向量，建议已 L2 归一化；若未归一化，本脚本会兜底
  - index.csv    : 列包含 [path, class_id] （可选 class_name）
输出:
  - sample_score.csv         : 每张图的 s1/s2/S/tag
  - class_thresholds.json    : 每类分位点、均值、样本数
  - kept_list.csv            : 最终保留名单（path,class_id）
  - removed_list.csv         : 最终删除名单（path,class_id,S,tag）
说明:
  S = alpha*s1 + (1-alpha)*s2
    s1 = 与类中心的余弦相似度
    s2 = 类内 Top-K 平均相似度（不含自身）
  删除策略=按 tag(gray/low) 作为候选，并施加 per-class 上限与下限:
    - 每类最大删除占比 <= max_remove_rate
    - 每类最少保留 >= min_keep_per_class
    
# WebFG-400
python 3_score_delete.py \
  --feat   /home/ma-user/work/data/webfg400_phase2/features.npy \
  --index  /home/ma-user/work/data/webfg400_phase2/index.csv \
  --alpha 0.7 --topk 10 \
  --remove-tags gray \
  --max-remove-rate 0.15 \
  --min-keep-per-class 30 \
  --outdir /home/ma-user/work/data/webfg400_phase2_scores

# WebiNat-5000
python 3_score_delete.py \
  --feat   /home/ma-user/work/data/webinat5000_phase2/features.npy \
  --index  /home/ma-user/work/data/webinat5000_phase2/index.csv \
  --alpha 0.7 --topk 10 \
  --remove-tags gray \
  --max-remove-rate 0.15 \
  --min-keep-per-class 30 \
  --outdir /home/ma-user/work/data/webinat5000_phase2_scores    
    
"""

import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

def l2_normalize(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def topk_mean_cosine(F: np.ndarray, topk: int) -> np.ndarray:
    """
    F: (m, D) 同一类的特征（建议已L2归一化）
    返回: (m,) 每个样本与其余样本的Top-K余弦相似度均值（不含自身）
    """
    m = F.shape[0]
    if m <= 1:
        return np.zeros(m, dtype=np.float32)
    S = F @ F.T                    # 余弦相似度（因已归一化）
    np.fill_diagonal(S, -1.0)      # 去自身
    K = min(topk, m-1)
    # 取行内 Top-K
    idx_part = np.argpartition(S, -K, axis=1)[:, -K:]  # 无序 K 列
    row = np.arange(m)[:, None]
    vals = S[row, idx_part]
    return vals.mean(axis=1).astype(np.float32)

def main():
    ap = argparse.ArgumentParser(description="Phase-2 scoring → kept/removed lists (no file ops)")
    ap.add_argument("--feat",   required=True, type=Path, help="features.npy")
    ap.add_argument("--index",  required=True, type=Path, help="index.csv 需含 path,class_id")
    # 打分参数
    ap.add_argument("--alpha",  type=float, default=0.7, help="综合分 S 中 s1 的权重 ∈[0,1]")
    ap.add_argument("--topk",   type=int,   default=10,  help="类内 Top-K 平均用于 s2")
    # 删除策略
    ap.add_argument("--remove-tags", type=str, default="gray",
                    help="候选标签，逗号分隔，可选 gray,low")
    ap.add_argument("--max-remove-rate", type=float, default=0.15,
                    help="每类最大删除占比（上限）")
    ap.add_argument("--min-keep-per-class", type=int, default=30,
                    help="每类最少保留样本数（下限）")
    # 输出
    ap.add_argument("--outdir", type=Path, default=Path("./phase2_scores"))
    args = ap.parse_args()

    outdir = args.outdir; outdir.mkdir(parents=True, exist_ok=True)

    feats = np.load(args.feat)        # (N, D)
    df = pd.read_csv(args.index)      # 需有 path,class_id
    if "class_id" not in df.columns or "path" not in df.columns:
        raise ValueError("index.csv 必须包含列: path,class_id")
    assert len(df) == len(feats), "features.npy 与 index.csv 行数不一致"

    # 兜底 L2 归一化
    feats = l2_normalize(feats.astype(np.float32))

    # 逐类计算
    s1_all = np.zeros(len(df), dtype=np.float32)
    s2_all = np.zeros(len(df), dtype=np.float32)
    S_all  = np.zeros(len(df), dtype=np.float32)
    tag_all= np.empty(len(df), dtype=object)
    class_thresholds = {}

    # 为了输出一致性，先按 class_id 排序
    df = df.reset_index(drop=True)
    groups = df.groupby("class_id", sort=True)

    for cid, idxs in groups.indices.items():
        idxs = np.asarray(list(idxs))
        F = feats[idxs]                                 # (m, D)
        center = F.mean(axis=0)
        center = center / (np.linalg.norm(center) + 1e-12)

        s1 = F @ center.astype(np.float32)              # (m,)
        s2 = topk_mean_cosine(F, args.topk)             # (m,)
        S  = (args.alpha * s1 + (1.0 - args.alpha) * s2).astype(np.float32)

        # 类内分位阈值（样本极少时用极值兜底）
        m = len(S)
        p10 = float(np.percentile(S, 10)) if m >= 10 else float(S.min())
        p25 = float(np.percentile(S, 25)) if m >= 4  else float(np.partition(S, max(0, m//4-1))[max(0, m//4-1)])

        tag = np.full(m, "clean", dtype=object)
        tag[S < p10] = "gray"
        mask_low = (S >= p10) & (S < p25)
        tag[mask_low] = "low"

        s1_all[idxs] = s1; s2_all[idxs] = s2; S_all[idxs] = S; tag_all[idxs] = tag

        class_thresholds[int(cid)] = {
            "n": int(m), "p10": p10, "p25": p25,
            "s1_mean": float(s1.mean()), "s2_mean": float(s2.mean()),
            "S_mean":  float(S.mean())
        }

    # 汇总表
    out = df.copy()
    out["s1"]  = s1_all
    out["s2"]  = s2_all
    out["S"]   = S_all
    out["tag"] = tag_all
    out = out.sort_values(["class_id","S"], ascending=[True, True]).reset_index(drop=True)

    # 生成删/留名单
    remove_tags = set([t.strip() for t in args.remove_tags.split(",") if t.strip()])
    remove_flags = np.zeros(len(out), dtype=bool)

    for cid, idxs in out.groupby("class_id").indices.items():
        idxs = np.asarray(list(idxs))
        tags = out.loc[idxs, "tag"].to_numpy()
        cand = np.isin(tags, list(remove_tags))
        cand_idxs = idxs[cand]

        n_cls = len(idxs)
        max_remove = int(math.floor(args.max_remove_rate * n_cls))
        min_keep   = args.min_keep_per_class
        quota = max(0, min(len(cand_idxs), max_remove, n_cls - min_keep))
        if quota > 0:
            # 从候选里按 S（低→高）取前 quota 个
            S_cand = out.loc[cand_idxs, "S"].to_numpy()
            order = np.argsort(S_cand)
            to_remove = cand_idxs[order[:quota]]
            remove_flags[to_remove] = True

    kept    = out[~remove_flags].copy()
    removed = out[remove_flags].copy()

    # 写出
    out.to_csv(outdir / "sample_score.csv", index=False)
    kept[["path","class_id"]].to_csv(outdir / "kept_list.csv", index=False)
    removed[["path","class_id","S","tag"]].to_csv(outdir / "removed_list.csv", index=False)
    with open(outdir / "class_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(class_thresholds, f, ensure_ascii=False, indent=2)

    # 控制台摘要
    byc = removed.groupby("class_id").size()
    removed_total = int(byc.sum()) if len(byc)>0 else 0
    print(f"[DONE] total={len(out)}, kept={len(kept)}, removed={removed_total}")
    print(f"[OUT ] {outdir/'sample_score.csv'}")
    print(f"[OUT ] {outdir/'kept_list.csv'}")
    print(f"[OUT ] {outdir/'removed_list.csv'}")
    print(f"[OUT ] {outdir/'class_thresholds.json'}")

if __name__ == "__main__":
    main()
