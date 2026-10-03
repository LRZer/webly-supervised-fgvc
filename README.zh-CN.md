# 网络监督细粒度图像分类

[English](README.md) · [算法细节](docs/METHODS.md) · [运行指南](docs/REPRODUCIBILITY.md) · [证据与历史](docs/EVIDENCE.md)

面向 **WebFG-400** 和 **WebiNat-5000** 的图像分类项目：检查网络图像的可用性，按类别计算特征一致性，筛选疑似异常样本，微调基于 ImageNet-1K 预训练的 ConvNeXt-Base，并生成比赛预测文件。

任务是从外观相近的飞机、汽车、鸟类或自然物种中识别具体子类别。网络训练图像的标签可能有误，因此项目在监督训练前加入文件检查和基于视觉特征的样本筛选。

仓库包含找回的原始代码和预测数据、可配置的运行版本、课程报告及技术说明。**下列准确率来自历史报告，本次整理未在比赛数据集上重新训练。**

## 实验结果

| 期末报告中的方法 | WebFG-400 准确率 | WebiNat-5000 准确率 |
|---|---:|---:|
| ResNet-50 | 65.8% | 58.2% |
| EfficientNet-B4 | 67.1% | 59.6% |
| ConvNeXt-Base 项目流程 | **69.3%** | **60.9%** |
| 相对 ResNet-50 的差值 | **+3.5 个百分点** | **+2.7 个百分点** |

来源：[期末报告第 6–7 页](reports/final-report.zh-CN.pdf)、[指标 CSV](results/metrics/report_metrics.csv)。报告另记录获得 **2025 年第七届全球校园人工智能算法精英大赛省级选拔赛二等奖**；本次恢复材料未包含获奖证书和官方评分回执。

![历史报告准确率对比](docs/assets/reported_results.png)

报告未明确这些结果对应 A 榜还是 B 榜，也没有附上基线训练代码、模型权重和逐次实验日志。因此，表中的差值不能单独归因于数据净化。

## 数据与任务

| 数据集 | 类别数 | 训练图像数 | 完整测试图像数 | 内容 |
|---|---:|---:|---:|---|
| WebFG-400 | 400 | 43,087 | 13,882 | 飞机、汽车、鸟类 |
| WebiNat-5000 | 5,000 | 580,865 | 100,000 | 自然物种 |

规模来自[赛题说明第 2 页](reports/competition-task.zh-CN.pdf)。训练标签源于网络，测试标签由赛事方人工核验。比赛数据从 WebFG-496、WebiNat-5089 采样并脱敏获得。获取方式和目录结构见 [data/README.md](data/README.md)。

![比赛数据规模](docs/assets/dataset_scale.png)

找回的源码包包含 **两份预测 CSV**，没有原始图像：

| 预测文件 | 记录数 | 出现在预测中的类别数 | 重复文件名 |
|---|---:|---:|---:|
| [pred_results_web400.csv](results/predictions/pred_results_web400.csv) | 5,687 | 397 / 400 | 0 |
| [pred_results_web5000.csv](results/predictions/pred_results_web5000.csv) | 60,000 | 4,925 / 5,000 | 0 |

文件无表头，每行是 `图像文件名,四位类别编号`。标签均在任务范围内。当前无法确认 CSV 所属测试阶段和完整图像覆盖率；预测频数不代表训练集类别分布或准确率。

![已恢复预测文件的类别频数分布](docs/assets/prediction_distribution.png)

## 实现流程

![数据处理、训练与推理流程](docs/assets/pipeline.png)

| 阶段 | 工作 | 整理版脚本 | 输出 |
|---|---|---|---|
| 基础清洗 | 解码、文件大小、最短边、PNG/GIF 非透明像素占比检查 | [clean_images.py](scripts/clean_images.py) | 备份和筛除日志 |
| 特征提取 | 本地 DeiT III ImageNet-1K 权重；RGB、缩放、裁剪、L2 归一化 | [extract_features.py](scripts/extract_features.py) | 特征矩阵、索引、坏图清单 |
| 一致性评分 | 类中心和类内 Top-K 余弦相似度 | [score_samples.py](scripts/score_samples.py) | 分数、阈值、保留/移除清单 |
| 审核暂存 | 按类别复制候选图；可在备份后移除 | [stage_review.py](scripts/stage_review.py) | 审核副本、各类统计 |
| 数据划分 | 各类别按 85/15 划分，种子 2025 | [split_dataset.py](scripts/split_dataset.py) | train/val 目录、划分清单 |
| 模型训练 | ConvNeXt-Base、增强、Mixup/CutMix、AdamW、EMA | [train.py](scripts/train.py) | 类别映射、best/last 权重、配置、日志 |
| 测试推理 | 中心裁剪，可选水平翻转 TTA | [infer.py](scripts/infer.py) | 比赛 CSV |

样本分数为 **S = 0.7 × 类中心相似度 + 0.3 × 类内 Top-10 相似度均值**。每类单独计算 10% 和 25% 分位点；默认筛除候选为低于 10% 分位点的样本，每类最多移除 15%，至少保留 30 张已有样本。原本不足 30 张的类别不移除。以上是代码默认值，历史运行的实际筛除数量未恢复；公式和边界情况见[算法说明](docs/METHODS.md)。

![ConvNeXt-Base 分类器结构](docs/assets/architecture.png)

| 找回代码的训练参数 | WebFG-400 脚本 | WebiNat-5000 脚本 |
|---|---:|---:|
| 模型 | `convnext_base.fb_in1k` | `convnext_base.fb_in1k` |
| 输入 / 批次大小 | 320 × 320 / 64 | 320 × 320 / 64 |
| 训练轮数 | 100 | 50 |
| Mixup alpha / CutMix alpha | 0.2 / 0.2 | 0.4 / 1.0 |
| AdamW 学习率 / 权重衰减 | 0.0005 / 0.05 | 0.0005 / 0.05 |
| 预热 / 最低学习率 | 3 轮 / 0.000001 | 3 轮 / 0.000001 |
| Drop path / EMA 衰减 | 0.4 / 0.9999 | 0.4 / 0.9999 |

训练增强包括随机缩放裁剪、RandAugment、水平翻转、随机擦除。验证使用 EMA 模型，按验证准确率选择 `best.pth`；TTA 对同一个分类器的原图及翻转图结果取平均。两套任务配置见 [configs/](configs)。

## 使用方法

用 Python 3.11 即可检查仓库内的预测文件，无需安装机器学习依赖：

```bash
python scripts/validate_predictions.py results/predictions/pred_results_web400.csv --num-classes 400 --expected-rows 5687
python scripts/validate_predictions.py results/predictions/pred_results_web5000.csv --num-classes 5000 --expected-rows 60000
```

运行数据处理、训练或推理前，安装整理版环境：

```bash
python -m venv .venv
# 按当前终端的方式激活 .venv 后：
python -m pip install -r requirements.txt
```

完成图像目录准备并取得对应的本地预训练权重后，可训练 400 类任务：

```bash
python scripts/train.py --config configs/webfg400.json --data-root data/working/webfg400_split --out-dir experiments/webfg400 --pretrained-checkpoint data/weights/convnext_base.fb_in1k.safetensors
```

[运行指南](docs/REPRODUCIBILITY.md)提供所有处理阶段、两个任务、推理、仅加载权重继续训练及 CPU/GPU 环境的完整命令。[模型卡](https://huggingface.co/timm/convnext_base.fb_in1k)说明 ImageNet-1K 预训练来源；恢复实验时仍需核对本地权重文件。

## 归档与验证

```text
archive/original/       8 个原始脚本，内容保持原样
scripts/               整理版流程、预测校验、图表生成
configs/               从源码提取的任务配置
data/                  数据获取说明、目录结构、可用性清单
results/predictions/   两份完整的原始预测 CSV
results/metrics/       报告指标、预测校验、逐类预测数量
reports/               期末报告与赛题说明
docs/                  算法、运行、证据文档及 5 组 PNG/SVG 图表
tests/                 流程及归档校验
```

整理版修复了写死的数据路径、特征提取缺少权重仍可运行、划分顺序依赖目录枚举、推理静默遗漏坏图等问题；另外增加最后一轮权重、实际划分清单和结构化训练日志。新增内容与原项目的区别、历史模型探索见[证据与改动记录](docs/EVIDENCE.md)。

已执行 6 项流程测试、两份原始 CSV 校验、归档文件 SHA-256 校验，以及 CPU 模型试运行：覆盖一次优化更新、EMA 保存、严格权重加载与 TTA 推理、拒绝不完整输出、归一化特征和索引对齐。合成数据试运行仅验证执行流程。[验证记录](results/verification.json)。

原始图像、任务模型权重、类别映射、特征缓存及历史训练日志尚未恢复。原代码在净化之后划分验证集，因此该验证集不能独立评估净化步骤的效果。重新复现实验分数需要补回相应数据与实验产物。

## 作者与参考资料

报告列出的团队成员为 **刘润章、徐瀚章、南怡波**。报告将整体算法设计与调优归于刘润章，数据预处理、评估和报告撰写归于徐瀚章，噪声筛选、训练与结果整理归于南怡波。发布的报告副本已去除学号。

- [比赛规则](https://www.aicomp.cn/wp-content/uploads/2025/06/%E8%B5%9B%E9%A2%98%E8%A7%84%E5%88%99%EF%BC%9A%E7%BD%91%E7%BB%9C%E7%9B%91%E7%9D%A3%E7%BB%86%E7%B2%92%E5%BA%A6%E5%9B%BE%E5%83%8F%E8%AF%86%E5%88%AB-1.pdf)
- [Webly Supervised Fine-Grained Recognition: Benchmark Datasets and An Approach](https://arxiv.org/abs/2108.02399)
- [ConvNeXt 官方实现](https://github.com/facebookresearch/ConvNeXt) · [DeiT III 官方实现](https://github.com/facebookresearch/deit)
- [timm](https://github.com/huggingface/pytorch-image-models) · [DeiT III ImageNet-1K 模型卡](https://huggingface.co/timm/deit3_base_patch16_224.fb_in1k)

代码作者权利归项目贡献者所有。本次整理未为原始代码、报告、数据集或第三方权重新增再分发许可，详见 [NOTICE.md](NOTICE.md)。
