# Running the project / 运行与复现

[English README](../README.md) · [中文 README](../README.zh-CN.md) · [Data](../data/README.md)

## 1. Scope / 运行范围

The commands below operate the portable maintenance version. They retain the recovered method and task settings, while fixing path/configuration and input/output handling. The original eight scripts remain in `archive/original/`. This is an executable reconstruction of the workflow, not a claim that the historical scores have been reproduced.

以下命令使用整理版，保留原方法和任务参数并修复路径、配置及输入输出处理。8 个原始脚本保存在 `archive/original/`。当前验证了流程执行，没有重新复现历史分数。

Required external artifacts: competition training/test images; `deit3_base_patch16_224.fb_in1k` pretrained weights for selection; `convnext_base.fb_in1k` pretrained weights for training. For existing-model inference, also restore the task `best.pth` and its **original** `classes.txt`. Reconstructing the class file from numeric predictions alone is unsafe.

需要另行恢复比赛训练/测试图像、特征提取用 DeiT III 权重、训练用 ConvNeXt-Base 权重。已有模型推理还需要对应 `best.pth` 和训练时的原始 `classes.txt`，不能仅根据预测编号猜测类别映射。

## 2. Environment / 环境

Use Python **3.11**, PyTorch **2.5.1**, torchvision **0.20.1**, timm **1.0.12**. `requirements.txt` describes the maintenance environment; it is not the original server's lockfile. CUDA training requires an NVIDIA GPU and a matching driver. CPU can validate files and run small execution checks; full ConvNeXt training on 580,865 images requires separate compute resources. No recovered timing or peak-memory measurements are available.

推荐上述整理版依赖；历史云端环境的完整锁定文件未找回。CUDA 训练需要 NVIDIA GPU 及适配驱动。CPU 可进行文件校验与小规模试运行。原实验没有恢复完整耗时、显存测量记录。

```bash
python -m venv .venv
```

Activation / 激活：

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

CPU installation / CPU 安装：

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

CUDA 12.1 installation / CUDA 12.1 安装：

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Weights / 权重来源：[DeiT III model card](https://huggingface.co/timm/deit3_base_patch16_224.fb_in1k), [ConvNeXt-Base model card](https://huggingface.co/timm/convnext_base.fb_in1k). Download `model.safetensors` from each specified repository and rename locally as shown in `data/README.md`. Record the model revision, download source and file SHA-256. Do not substitute an ImageNet-22K or other-data variant for an ImageNet-1K-only experiment.

从指定模型仓库下载 `model.safetensors`，按数据文档中的文件名保存。记录版本、来源与 SHA-256。不同预训练数据来源的变体不能直接作为同一实验配置。

## 3. Prepare a working copy / 准备工作副本

Put the original images under `data/raw/` and process a separate copy. The next steps assume `data/working/webfg400_train` is such a copy. For WebiNat replace each `webfg400` component with `webinat5000`. Paths should have numeric class folders with consistent zero padding.

将原始图像放在 `data/raw/`，复制到独立工作目录后处理。以下假设 `data/working/webfg400_train` 已是工作副本；5000 类任务将相应路径换成 `webinat5000`。类别文件夹使用位数一致的数字名称。

### 3.1 File checks / 文件检查

```bash
python scripts/clean_images.py --input data/working/webfg400_train --trash data/working/webfg400_phase1_trash --workers 8
# Review dry-run output, then apply to the working copy:
python scripts/clean_images.py --input data/working/webfg400_train --trash data/working/webfg400_phase1_trash --workers 8 --apply
```

Default thresholds are 10,240 bytes, 64 pixels and alpha coverage 0.05. They are configurable. Rejected files are copied before removal; keep the backup and log. / 默认阈值可调整。先检查预演结果，再对工作副本执行，保留备份及日志。

### 3.2 Features / 特征提取

```bash
python scripts/extract_features.py --input data/working/webfg400_train --outdir data/working/webfg400_features --model deit3_base_patch16_224.fb_in1k --checkpoint data/weights/deit3_base_patch16_224.fb_in1k.safetensors --batch 64 --workers 8 --fp16
```

For CPU debugging use `--workers 0`, a small batch, and omit `--fp16`. Feature weights are required; the script will not proceed with a missing local checkpoint. / CPU 调试使用 `--workers 0`、较小批次并省略 `--fp16`；必须提供本地权重。

### 3.3 Scores and review / 评分与审核

```bash
python scripts/score_samples.py --feat data/working/webfg400_features/features.npy --index data/working/webfg400_features/index.csv --alpha 0.7 --topk 10 --remove-tags gray --max-remove-rate 0.15 --min-keep-per-class 30 --outdir data/working/webfg400_scores
python scripts/stage_review.py --root data/working/webfg400_train --removed data/working/webfg400_scores/removed_list.csv --score-csv data/working/webfg400_scores/sample_score.csv --review-dir data/working/webfg400_review
```

After inspecting candidates and retaining review backups, removal from the working copy is a separate command. / 检查候选样本并保留审核副本后，再单独对工作副本执行移除：

```bash
python scripts/stage_review.py --root data/working/webfg400_train --removed data/working/webfg400_scores/removed_list.csv --score-csv data/working/webfg400_scores/sample_score.csv --review-dir data/working/webfg400_review --delete
```

### 3.4 Split / 划分

```bash
python scripts/split_dataset.py --src data/working/webfg400_train --dst data/working/webfg400_split --val-ratio 0.15 --seed 2025 --mode copy --workers 8
```

Use a new empty destination. The membership manifest is the definitive split record, beyond the seed alone. / 输出目录应为空；保存实际划分清单，随机种子本身不足以恢复所有历史样本。

## 4. Train / 训练

```bash
python scripts/train.py --config configs/webfg400.json --data-root data/working/webfg400_split --out-dir experiments/webfg400 --pretrained-checkpoint data/weights/convnext_base.fb_in1k.safetensors
python scripts/train.py --config configs/webinat5000.json --data-root data/working/webinat5000_split --out-dir experiments/webinat5000 --pretrained-checkpoint data/weights/convnext_base.fb_in1k.safetensors
```

`best.pth` is the EMA model selected by validation accuracy; `last.pth` is the final EMA state; `last_raw.pth` is the final non-EMA model. Each run also saves `classes.txt`, `run_config.json` and `history.csv`. Use a new output directory per run. Reduce batch size explicitly if memory is insufficient; changing it changes the experiment configuration.

`best.pth` 为验证准确率最优的 EMA，`last.pth` 为最后一轮 EMA，`last_raw.pth` 为最后一轮普通模型。每次另存类别映射、实际配置和逐轮日志；不同运行使用新输出目录。降低批次大小会改变实验配置。

Weight-only continuation from an existing task checkpoint / 加载任务权重继续训练：

```bash
python scripts/train.py --config configs/webfg400.json --data-root data/working/webfg400_split --out-dir experiments/webfg400_continue --init-weights experiments/webfg400/best.pth --epochs 20 --warmup-epochs 0 --lr 0.0001
```

This starts a new optimizer, scheduler and EMA from the supplied weights. It does not restore the previous optimizer/RNG state. / 此方式重新创建优化器、学习率调度和 EMA，不恢复原优化器或随机状态。

## 5. Infer and audit / 推理与校验

```bash
python scripts/infer.py --test-root data/raw/webfg400_test --weights experiments/webfg400/best.pth --classes experiments/webfg400/classes.txt --out experiments/pred_results_web400.csv --img-size 320 --batch-size 128 --workers 8 --tta --channels-last
python scripts/infer.py --test-root data/raw/webinat5000_test --weights experiments/webinat5000/best.pth --classes experiments/webinat5000/classes.txt --out experiments/pred_results_web5000.csv --img-size 320 --batch-size 128 --workers 8 --tta --channels-last
python scripts/validate_predictions.py experiments/pred_results_web400.csv --num-classes 400 --test-root data/raw/webfg400_test
python scripts/validate_predictions.py experiments/pred_results_web5000.csv --num-classes 5000 --test-root data/raw/webinat5000_test
```

TTA is optional; the recovered files do not reveal whether it was enabled. Validation with `--test-root` checks filename coverage against the actual flat directory. A malformed/undecodable test image causes the portable inference script to reject incomplete output by default.

TTA 为可选开关，已有 CSV 无法证明是否启用。提供 `--test-root` 会比较实际文件名覆盖。默认不会为含无法解码图像的测试集静默生成不完整 CSV。

## 6. Local verification and figures / 本地验证与图表

```bash
python -m unittest discover -s tests -v
python -m compileall -q scripts archive/original
python -m pip install -r requirements-figures.txt
python scripts/make_figures.py
```

Figure generation writes five PNG/SVG pairs, the prediction audit, and per-class prediction counts. For Chinese rendering, install Microsoft YaHei or Noto Sans CJK SC. The committed figures are already rendered and do not depend on fonts on a reader's computer.

图表脚本生成 5 组 PNG/SVG、预测审计 JSON 和各类预测数量。重新生成中文图表需微软雅黑或 Noto Sans CJK SC；已提交图表不依赖读者本机字体。

The added CPU Docker recipe validates the recovered WebFG CSV by default:

```bash
docker build -t webly-fgvc .
docker run --rm webly-fgvc
```

This is a maintenance recipe, not a recovered competition submission. Docker build availability is recorded in `results/verification.json`. / 这是本次整理新增的 CPU 配方，不是找回的比赛镜像，构建验证状态见验证记录。
