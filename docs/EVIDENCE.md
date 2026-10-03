# Evidence, evolution and maintenance / 证据、演进与整理记录

[English README](../README.md) · [中文 README](../README.zh-CN.md)

## Evidence hierarchy / 资料使用方式

The eight recovered scripts establish implemented behavior and defaults. The two original prediction CSVs establish their row counts and output labels. The course report establishes what the team reported, including its comparison table and prize statement. The competition specification establishes task definitions and submission rules. Historical chats explain model alternatives and engineering decisions; assistant suggestions alone are not treated as completed experiments.

8 个源码文件用于确认实际实现和默认参数，2 份原始 CSV 用于核验输出数量及标签。课程报告用于记录团队当时提交的指标和获奖表述，赛题说明用于确认任务与提交规则。历史聊天用于补充模型尝试和工程决策，不将助手建议直接写成已完成的实验。

The student-ID fields in the report copy are redacted; the original desktop report is preserved outside this repository. Raw chat transcripts and personal cloud/session details are not committed. [Artifact SHA-256 manifest](../results/artifact_manifest.json) covers all eight originals, both CSVs and the two report PDFs.

发布报告中的学号已去除，桌面原文件保持原样。仓库整理的是聊天中可核验的项目事实，没有提交原始私人聊天全文或个人云端会话信息。SHA-256 清单覆盖 8 个原始脚本、2 份 CSV 和 2 份 PDF。

## Read historical sources / 已读取的历史记录

Dates below are conversation creation dates in Asia/Shanghai. Ten conversations were consulted: eight older links supplied by the owner and two recent project chats. This is the retrieved source set, not an assertion that every project chat was accessible.

下表日期为聊天创建日期，时区 Asia/Shanghai。本次共参考 10 段聊天：用户给出的 8 个旧链接，以及项目内 2 段近期聊天；不代表读取到了项目中所有聊天。

| Date / 日期 | Exact conversation title / 原标题 | Project evidence / 可确认内容 |
|---|---|---|
| 2025-10-28 | 训练1，优化3（新模型） | User logs confirm MambaVision-S-1K download/load and one pretrained-image inference; no fine-grained task result / 用户日志确认下载、加载和一次预训练图像推理，没有细粒度任务指标 |
| 2025-10-29 | 训练1，优化4（旧模型） | Only-best saving, desire to save last weights, and terminal-log discussion / 仅保存 best、希望保留 last、终端日志记录 |
| 2025-10-30 | 上传文件到PFS | Cloud data-copy and mount-path troubleshooting / 云端数据复制与挂载路径讨论 |
| 2025-10-31 | 训练1，优化5（旧模型） | Owner chose weight-only continuation and asked to disable warmup / 用户选择仅加载权重继续训练并关闭预热 |
| 2025-10-31 | 检查模型预训练数据 | Local ConvNeXt V1 safetensors, V2 alternative, old timm model registration failure / 本地 V1 权重、V2 备选和旧 timm 注册失败 |
| 2025-11-04 | Docker作用介绍 | Packaging discussion; no built competition image recovered / 打包讨论，未恢复比赛镜像 |
| 2025-11-29 | 技术说明文档撰写 | Eight-script workflow summary / 8 个脚本的技术流程说明 |
| 2025-12-30 | 技術文稿寫作指南 | Stage-to-code mapping, training settings and inference behavior / 阶段与代码映射、训练设置、推理流程 |
| 2026-09 | 整理图像分类项目信息 | Earlier inventory lacked local source code / 当时本地盘点没有源码 |
| 2026-09 to 2026-10 | 项目总结梳理 | Earlier summary identified source gaps and treated specifics as unverified / 近期总结明确了此前缺少源码的边界 |

## Reconciled differences / 对照后的差异

| Topic / 内容 | Recovered evidence / 证据 | Repository treatment / 处理 |
|---|---|---|
| Feature extractor / 特征提取器 | Actual default string identifies DeiT III; help text says DINOv2 and report discusses CNNs / 默认值是 DeiT III，帮助文字和报告描述不同 | Document the exact source default; portable extraction requires local weights / 按源码记录，整理版要求权重 |
| 400-class paths / 400 类路径 | Its training file points to `webinat5000_split` and the 5000 output directory / 400 文件内仍指向 5000 目录 | Keep original unchanged; explicit config/path in portable training / 保留原文件，整理版显式配置 |
| Model V1 vs V2 / 模型版本 | Recovered classifier scripts use V1 `convnext_base.fb_in1k`; chats discuss V2 / 源码是 V1，聊天尝试 V2 | V1 is the recovered main implementation; V2 has no recovered result association / 主实现按 V1，V2 未与成绩建立关联 |
| MambaVision / 新模型 | User logs show successful load and sample inference / 用户日志可确认加载与样例推理 | Record as exploration; no task training/score claim / 记录探索，不写成已完成任务成绩 |
| Optimizer / 优化器 | Report says Adam; code instantiates AdamW / 报告写 Adam，源码用 AdamW | Document AdamW for the code / 代码说明按 AdamW |
| Self-supervision / 自监督 | Report mentions it; no corresponding objective/training code recovered / 报告提及，未找到对应训练目标实现 | Keep report as source, do not claim an implemented self-supervised phase / 保留报告，不扩写为已实现阶段 |
| Distributed training / 分布式训练 | Extractor contains DataParallel; classifier scripts are single-device / 提取器用 DataParallel，分类训练为单设备 | No DDP claim for the recovered classifiers / 不将分类器描述为 DDP 训练 |
| Best/last / 权重保存 | Source training saves only best EMA weights; chats request last / 源码仅保存 best EMA，聊天希望加 last | Portable version adds last EMA/raw and structured logs / 整理版新增 last 和日志 |
| 256→320 schedule / 分辨率两阶段 | Appears in historical assistant descriptions, without recovered phase logs/config / 聊天提及，没有找回阶段日志 | Not treated as a verified completed experiment / 不作为已核验实验 |
| Results / 成绩 | Report table and prize statement; no official receipt/certificate / 报告表格与奖项表述，无官方回执 | Explicit historical-report labels / 明确标注报告记录 |

## Portable maintenance changes / 整理版改动

The original files are byte-preserved in `archive/original/`; prediction files are copied without changing their contents. All maintenance behavior lives under `scripts/` and `configs/`.

原始脚本与 CSV 内容保持不变，维护改动位于 `scripts/`、`configs/`：

1. **Cleaning / 清洗**: dry-run default, explicit apply, and separate input/backup trees.
2. **Extraction / 提取**: canonical DeiT III model identifier, required local checkpoint, strict 1,000-class loading before removing the head, sorted paths, zero-worker loaders, actual bad-image log, multi-GPU wrapping only when needed.
3. **Scoring / 评分**: parameter/shape/finite-value checks, duplicate-index rejection, negative-infinity self masking. The source score, tags and quota rule are retained.
4. **Review / 审核**: preserve full relative paths, reject paths outside the named dataset root, separate review/input trees.
5. **Split / 划分**: sort before shuffling, record exact membership, require an empty destination, propagate copy failures.
6. **Training / 训练**: task configs, explicit data/output/weight paths, class-map agreement, odd-batch handling for Mixup, averaged training loss, strict weight loading, seed/config/log saving, best/last weights, optional weight-only continuation.
7. **Inference / 推理**: strict mapping and checkpoint checks, zero-worker support, current AMP API, output-directory creation, reject incomplete predictions by default.
8. **Audit / 校验**: standard-library CSV validator, SHA-256 manifest, workflow tests, CPU Docker recipe, bilingual documentation and regenerable figures.

These changes are operational maintenance and were not used to establish the report's historical accuracies. New default environment versions, logs and reproducibility checks are dated **2026-10-04**.

以上为本次工程整理，不能作为历史报告指标的实验来源。新增环境、日志及验证记录的整理日期为 **2026-10-04**。

## Remaining gaps / 待恢复材料

Raw stage images, exact cleaning manifests, original train/validation directories, pretrained-file hashes, task-trained checkpoints and class maps, complete training logs, official score receipts and award certificates. Without them the recovered implementation cannot be bound to an exact historical run, and CSV label coverage cannot be converted into accuracy.

尚缺对应阶段原图、实际清洗清单、原训练/验证目录、预训练文件哈希、任务权重与类别映射、完整日志、官方评分回执和获奖证书。缺少这些文件，无法将实现绑定到具体历史运行，也不能把预测类别覆盖率换算为准确率。
