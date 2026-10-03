# Method details / 算法细节

[English README](../README.md) · [中文 README](../README.zh-CN.md) · [Run guide](REPRODUCIBILITY.md)

## English

### 1. Problem and model roles

The training set contains image-label pairs $(x_i,\tilde y_i)$ with potentially incorrect web labels. The feature extractor builds embeddings for sample selection; a separate ConvNeXt-Base classifier is fine-tuned on the selected images. The recovered extractor defaults to DeiT III; the two networks have different roles. There is one classifier per task, with 400 or 5,000 outputs.

### 2. Physical image checks

Source defaults: file size at least **10,240 bytes**, shortest side at least **64 pixels**, PNG/GIF fraction of pixels with alpha greater than zero at least **0.05**. A file can also be rejected if PIL cannot decode it. Alpha checks sample at most 512 × 512 pixels. The original cleaner copies rejected files into a backup tree before removing originals; the portable version defaults to dry-run and requires `--apply` for removal.

These checks detect file/input problems. They do not establish whether a label is correct. Byte thresholds can exclude valid compressed images. The source comments mention different thresholds for the two datasets but do not provide a complete execution record.

### 3. Feature construction

Processing: convert to RGB, resize the short side to 256 with bicubic interpolation, center-crop to 224 × 224, convert to a tensor, normalize using ImageNet channel mean/std. The default model is `deit3_base_patch16_224.fb_in1k`, with local ImageNet-1K weights and its classification head removed. The source help text refers to DINOv2, although the actual default string points to DeiT III. The portable version corrects the identifier/help and requires a checkpoint.

Each embedding is L2-normalized, $z_i=f(x_i)/\max(\lVert f(x_i)\rVert_2,\epsilon)$. Features default to float16 storage. `index.csv` preserves the corresponding path, numeric class ID and class name. Scoring converts features to float32 and normalizes them again. Decode failures are excluded; the portable version records their paths.

### 4. Classwise consistency scores

For the $m_c$ images assigned to class $c$, let $Z_c$ be the matrix of normalized embeddings. The normalized class center is

$$\mu_c = \frac{m_c^{-1}\sum_{i\in c}z_i}{\left\lVert m_c^{-1}\sum_{i\in c}z_i\right\rVert_2+\epsilon}.$$

The center score includes the sample itself in the center calculation:

$$s_{1,i}=z_i^\top\mu_c.$$

The neighbor score excludes the diagonal of $Z_cZ_c^\top$ and averages the $K'=\min(10,m_c-1)$ largest similarities:

$$s_{2,i}=\frac{1}{K'}\sum_{j\in\operatorname{TopK'}(c\setminus\{i\})}z_i^\top z_j,\qquad S_i=0.7s_{1,i}+0.3s_{2,i}.$$

For singleton classes, the neighbor score is defined as zero. The original diagonal masking value is −1; the portable version uses negative infinity to exclude self even in an exact tie.

Each class computes its own $P_{10}$ and $P_{25}$ of $S$. Tags are `gray` for $S<P_{10}$, `low` for $P_{10}\leq S<P_{25}$, and `clean` otherwise. With fewer than 10 samples, $P_{10}$ uses the minimum. With fewer than 4, $P_{25}$ uses the source order-statistic fallback. Tags are relative score categories, not verified noise labels.

The default candidate set $C_c$ contains only `gray` samples. Candidates are ordered by increasing score, with removal quota

$$q_c=\max\{0,\min(|C_c|,\lfloor0.15m_c\rfloor,m_c-30)\}.$$

This limits removal rather than forcing a fixed fraction. If $m_c<30$, no sample is removed. Equal scores can yield no `gray` candidates. The script writes scores, class thresholds, and kept/removed lists without changing images. `stage_review.py` copies candidates for inspection; creating that directory does not prove that human review took place.

For a class with $m$ images and $d$ feature dimensions, the full similarity matrix takes $O(m^2)$ memory and $O(m^2d)$ computation. This cost is per class; no approximate nearest-neighbor algorithm is implemented.

### 5. Split and validation boundary

The source splits **after** purification. For a class of size $n>1$, $n_\mathrm{val}=\min(n-1,\max(1,\operatorname{round}(0.15n)))$ under default settings. Singleton classes go entirely to training. The portable version sorts paths before seeded shuffling and saves every membership record. Duplicate images are not detected or grouped by the recovered scripts.

Purification already uses the feature distribution of all images subsequently split into train/validation. Validation consequently cannot independently measure purification performance. Duplicate content can also cross the split. Independent evaluation would require a separately defined holdout and duplicate handling; those changes are not represented as historical experiments here.

### 6. Training and inference

The classifier is ConvNeXt V1 Base, with stage depths [3, 3, 27, 3] and channels [128, 256, 512, 1024]. Local `convnext_base.fb_in1k` weights initialize the 1,000-class model, then its classifier is replaced for the target task. All model parameters are optimized with AdamW. Training uses Mixup/CutMix and soft-target cross entropy; validation uses ordinary cross entropy on EMA weights.

EMA updates each floating-point parameter as $\theta_\mathrm{EMA}\leftarrow0.9999\theta_\mathrm{EMA}+0.0001\theta$. The source uses three epochs of linear warmup from 20% of the base LR and cosine annealing to 1e-6. Exact task settings are in `configs/`. CUDA execution uses AMP and TF32; channel-last memory layout is used by the portable training version for both model and input.

Inference resizes the short side to `int(1.14 × image_size)` and center-crops; at size 320 the resize target is 364 pixels. Optional TTA averages original/flipped logits. The class index is mapped through the training class file and padded to four digits. The portable version validates mappings, loads weights strictly, and refuses to write incomplete predictions unless explicitly allowed.

### 7. Metrics and interpretation

Accuracy is the fraction of correctly predicted test images. The competition specification averages the two task accuracies within a given A/B stage. The arithmetic mean of the two final-report accuracies is **65.1%**, computed from 69.3% and 60.9%; it is not a recovered official leaderboard score.

No ground-truth labels accompany the archived CSVs. Precision, recall, F1, confusion matrices, error examples, uncertainty, and cleaning ablations cannot be reconstructed from them. The figures show the supplied report values, code structure, specified dataset sizes, and observed prediction frequencies only.

## 中文

### 1. 任务与模型角色

训练数据为可能带有错误网络标签的图像 $(x_i,\tilde y_i)$。特征提取器用于样本筛选，另一套 ConvNeXt-Base 分类器在筛选后的图像上微调。找回的提取脚本默认使用 DeiT III；两个模型承担不同职责。每个任务分别使用一个分类器，输出为 400 或 5,000 类。

### 2. 文件检查

源码默认阈值：文件至少 **10,240 字节**，最短边至少 **64 像素**，PNG/GIF 中 alpha 大于零的像素占比至少 **0.05**。PIL 无法解码的图像也会被筛除。透明度检测最多采样 512 × 512 像素。原脚本先备份再移除；整理版默认仅预演，添加 `--apply` 后才执行。

这些规则检查输入可用性，不能判定标签是否正确；字节阈值可能筛掉正常的压缩图片。源码注释提到两个数据集使用过不同阈值，但没有完整运行记录。

### 3. 特征构建

预处理为 RGB 转换、双三次插值缩放短边至 256、中心裁剪 224 × 224、张量转换和 ImageNet 均值/标准差归一化。实际默认模型是 `deit3_base_patch16_224.fb_in1k`，使用本地 ImageNet-1K 权重并去除分类头。原帮助文字仍写 DINOv2，整理版修正了标识和帮助文字，并要求提供权重。

提取向量进行 L2 归一化，默认以 float16 保存。`index.csv` 按行记录路径、类别 ID 和类别名。评分时转换回 float32 并再次归一化。解码失败的图像不参与特征矩阵；整理版额外记录失败路径。

### 4. 类内一致性评分

先对每类特征取均值并归一化，得到类中心 $\mu_c$。中心分数为 $s_{1,i}=z_i^\top\mu_c$，计算中心时包含样本自身。邻居分数 $s_{2,i}$ 是与同类其他样本的 Top-10 余弦相似度均值，排除自身，不足 10 个则使用所有其他样本；单样本类别设为零。综合分数为 $S_i=0.7s_{1,i}+0.3s_{2,i}$。英文部分给出完整公式。

每类独立计算 $P_{10}$、$P_{25}$：低于 $P_{10}$ 标记为 `gray`，处于 $[P_{10},P_{25})$ 为 `low`，其余为 `clean`。小类别使用源码中的最小值/顺序统计回退。标签表示相对一致性，不能等同于人工确认的错标。

默认仅从 `gray` 中按分数从低到高选择候选，数量受 `floor(0.15 × 类别数量)` 和“至少保留 30 张”约束：$q_c=\max(0,\min(|C_c|,\lfloor0.15m_c\rfloor,m_c-30))$。原本不足 30 张的类不会移除；同分样本可能没有候选。评分脚本仅生成清单，审核脚本才执行复制或移除。存在审核目录不能证明已完成人工复核。

算法对每类构建完整相似度矩阵，其存储为 $O(m^2)$，计算为 $O(m^2d)$。没有实现近似近邻检索。

### 5. 数据划分边界

原流程先净化再划分。默认每类验证数量为 `min(n-1, max(1, round(0.15 × n)))`；单图类别全部进入训练。整理版先排序路径再按种子 2025 随机打乱，并保存实际划分清单。原脚本没有图像去重或相同内容分组。

净化已使用后来进入验证集的图像特征，因此验证集不能独立评估净化的收益，重复图像也可能跨集合。独立留出集和去重策略需要另行设计，仓库未将这些改进写成历史实验。

### 6. 训练与推理

分类器为 ConvNeXt V1 Base，四阶段深度为 [3,3,27,3]，通道为 [128,256,512,1024]。先加载 1,000 类本地预训练权重，再替换任务分类头。全参数使用 AdamW 优化；Mixup/CutMix 生成软标签，训练损失为软标签交叉熵，验证使用普通交叉熵。

EMA 浮点参数更新为旧值乘 0.9999 加当前值乘 0.0001。前 3 轮从基准学习率的 20% 线性预热，再余弦衰减至 1e-6。CUDA 下启用 AMP、TF32；整理版同时对模型和输入使用 channels-last。详细参数在 `configs/`。

推理先缩放短边到 `int(1.14 × 输入大小)` 再中心裁剪，320 输入对应 364 像素短边。可选 TTA 对原图、水平翻转图的 logits 取平均。输出索引依训练类别文件映射为四位数字。整理版严格加载权重、校验类别映射，默认拒绝不完整预测文件。

### 7. 指标解释

准确率是正确预测图像占测试图像的比例。赛题按同一 A/B 阶段两个任务准确率的平均值计分。报告中 69.3%、60.9% 的算术平均为 **65.1%**，属于本次计算值，不能当作已恢复的官方榜单成绩。

预测 CSV 没有真实标签，无法据此还原精确率、召回率、F1、混淆矩阵、误分类样例、置信区间或清洗消融。图表仅呈现报告值、代码结构、赛题规模与预测频数。
