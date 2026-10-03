# Data access and layout / 数据获取与组织

[English README](../README.md) · [中文 README](../README.zh-CN.md) · [Availability manifest](manifest.json)

## Available data / 已恢复数据

The source ZIP contained eight Python scripts and two CSVs. Both CSVs are committed in full under [results/predictions/](../results/predictions). The repository also contains report metrics, prediction counts for every possible class, and a format audit. There are no raw training/test images or ground-truth test labels in the provided project folder.

原始压缩包包含 8 个 Python 脚本和 2 份 CSV。两份 CSV 完整保存在 `results/predictions/`，并附有报告指标、所有类别的预测频数和格式校验。提供的项目目录没有原始训练/测试图像及测试真实标签。

## Obtain the competition images / 获取比赛图像

The [competition task specification](../reports/competition-task.zh-CN.pdf) states that registered participants obtain initial-stage download links through the [organizer's competition platform](https://www.aicomp.cn/); subsequent-stage links are released separately. Restore the exact stage archives from an existing backup or the organizer's participant download interface if still available. An unauthenticated, direct archive URL was not recovered.

赛题说明规定参赛者注册后获取初赛数据链接，复赛另行开放下载链接。应从已有备份或仍可用的赛事账户下载入口恢复对应阶段的数据。本次没有找回无需登录的直接数据下载地址。

The source datasets are WebFG-496 and WebiNat-5089, described in the [original benchmark paper](https://arxiv.org/abs/2108.02399). The competition uses **sampled, anonymized 400/5,000-class derivatives**. Downloading the full source benchmark does not reconstruct the competition label mapping or A/B partitions.

原始基准为 WebFG-496、WebiNat-5089，比赛使用采样并脱敏后的 400/5,000 类版本。下载原始基准不能还原比赛类别映射及 A/B 划分。

## Expected layout / 预期结构

```text
data/
├── raw/
│   ├── webfg400_train/0000/image.jpg
│   ├── webfg400_train/0001/image.jpg
│   ├── webfg400_test/<original_filename>.jpg
│   ├── webinat5000_train/0000/image.jpg
│   └── webinat5000_test/<original_filename>.jpg
├── weights/
│   ├── deit3_base_patch16_224.fb_in1k.safetensors
│   └── convnext_base.fb_in1k.safetensors
└── working/
    ├── webfg400_train/<class>/image.jpg
    ├── webfg400_features/features.npy
    ├── webfg400_features/index.csv
    ├── webfg400_scores/kept_list.csv
    ├── webfg400_scores/removed_list.csv
    ├── webfg400_review/<class>/image.jpg
    └── webfg400_split/
        ├── train/<class>/image.jpg
        ├── val/<class>/image.jpg
        └── split_manifest.csv
```

Training uses immediate child directories as classes; tests are flat folders. Keep numeric class names at a consistent width. The historical scripts used paths under `/home/ma-user/work/data/`; portable scripts accept explicit paths. Raw images, working caches, weights and experiment outputs are excluded from Git by `.gitignore`.

训练集一级子目录代表类别，测试集为平铺目录。数字类名位数应一致。原脚本使用 `/home/ma-user/work/data/` 下的目录，整理版通过参数指定路径。原图、缓存、权重和实验输出默认不纳入 Git。

## Restore an experiment / 恢复实验所需文件

| Artifact / 文件 | Purpose / 用途 | Recovery status / 状态 |
|---|---|---|
| Original stage image archives / 对应阶段原图 | Training and filename coverage / 训练及覆盖校验 | Missing / 未恢复 |
| `phase1_removed.csv` and backups | Physical filtering trace / 文件筛除追踪 | Missing / 未恢复 |
| `features.npy`, `index.csv` | Embedding/sample alignment / 特征与样本对齐 | Missing / 未恢复 |
| Scores, class thresholds, kept/removed lists | Exact selection / 恢复实际筛选 | Missing / 未恢复 |
| `split_manifest.csv` or old train/val folders | Exact split / 实际划分 | Missing / 未恢复 |
| `best.pth` and `classes.txt` | Inference and label mapping / 推理与映射 | Missing / 未恢复 |
| Training logs and score receipts | Link configuration to scores / 对应配置与成绩 | Missing / 未恢复 |
| Two prediction CSVs | Preserved submission-format outputs / 已保留预测输出 | Included / 已纳入 |

The historical PFS discussion mentions cloud copies of `webfg400_1`, but no accessible backup or mount was supplied in this recovery. The discussion is a search lead, not evidence that those files still exist.

历史 PFS 聊天提到 `webfg400_1` 的云端副本，但本次没有获得可访问的备份或挂载。该记录可用于寻找数据，不能证明文件目前仍在。

When adding recovered images, verify their redistribution terms before publishing an archive. Preserve original filenames, stage names and SHA-256 hashes. Do not infer ground truth from the archived model predictions.

后续补回图像时，记录原始文件名、阶段和 SHA-256，并核对数据的再分发条件。模型预测不能作为测试真实标签。
