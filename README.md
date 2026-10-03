# Webly Supervised Fine-Grained Image Classification

[中文](README.zh-CN.md) · [Method](docs/METHODS.md) · [Run guide](docs/REPRODUCIBILITY.md) · [Evidence and history](docs/EVIDENCE.md)

An image classification project for **WebFG-400** and **WebiNat-5000**: clean web images, score their consistency within each assigned class, fine-tune an ImageNet-1K ConvNeXt-Base classifier, and generate competition prediction files.

For a non-specialist, the task is to distinguish similar aircraft, cars, birds, or species from images whose web-derived training labels may be incorrect. The project combines image quality checks with feature-based sample selection before supervised training.

This repository contains the recovered implementation and results, a portable maintenance version, and the course report. **The accuracy values below are historical report values; training was not rerun on the competition datasets.**

## Results

| Method in the final report | WebFG-400 accuracy | WebiNat-5000 accuracy |
|---|---:|---:|
| ResNet-50 | 65.8% | 58.2% |
| EfficientNet-B4 | 67.1% | 59.6% |
| ConvNeXt-Base pipeline | **69.3%** | **60.9%** |
| Difference from ResNet-50 | **+3.5 percentage points** | **+2.7 percentage points** |

Source: [final report, pp. 6–7](reports/final-report.zh-CN.pdf); [machine-readable metrics](results/metrics/report_metrics.csv). The report also records a **second prize in the 2025 provincial selection of the 7th Global Campus Artificial Intelligence Algorithm Elite Competition**. An award certificate and official score receipt were not included in the recovered materials.

![Historical report accuracy comparison](docs/assets/reported_results.png)

The report does not identify the A/B test subset used for these results. Baseline training code, checkpoints, and per-run logs are absent. These comparisons therefore do not establish the isolated contribution of data purification.

## Tasks and data

| Dataset | Classes | Training images | Full test images | Subject matter |
|---|---:|---:|---:|---|
| WebFG-400 | 400 | 43,087 | 13,882 | Aircraft, cars, birds |
| WebiNat-5000 | 5,000 | 580,865 | 100,000 | Natural species |

Sizes come from the [competition specification, p. 2](reports/competition-task.zh-CN.pdf). Training labels are web-derived; test labels are manually curated by organizers. Competition subsets are sampled and anonymized from WebFG-496 and WebiNat-5089. Dataset access and the expected directory layout are documented in [data/README.md](data/README.md).

![Competition dataset scale](docs/assets/dataset_scale.png)

The recovered archive includes **two prediction CSVs**, not the original image datasets:

| Preserved CSV | Rows | Classes receiving predictions | Duplicate filenames |
|---|---:|---:|---:|
| [pred_results_web400.csv](results/predictions/pred_results_web400.csv) | 5,687 | 397 / 400 | 0 |
| [pred_results_web5000.csv](results/predictions/pred_results_web5000.csv) | 60,000 | 4,925 / 5,000 | 0 |

Each row contains `image_filename,four_digit_class_id`, without a header. All recovered labels are in range. CSV subset stage and full image coverage remain unknown. Predicted class frequencies describe model outputs, not dataset balance or accuracy.

![Recovered prediction distribution](docs/assets/prediction_distribution.png)

## Implementation

![Data preparation, training and inference pipeline](docs/assets/pipeline.png)

| Stage | Operation | Portable script | Output |
|---|---|---|---|
| File checks | Decode, file size, shortest side, PNG/GIF alpha coverage | [clean_images.py](scripts/clean_images.py) | Backups and removal log |
| Features | Local DeiT III ImageNet-1K weights; RGB, resize, crop, L2 normalization | [extract_features.py](scripts/extract_features.py) | `features.npy`, `index.csv`, bad-image log |
| Consistency scoring | Class-center and within-class Top-K cosine similarities | [score_samples.py](scripts/score_samples.py) | Scores, thresholds, kept/removed lists |
| Review staging | Copy candidates with class hierarchy; optional removal after backup | [stage_review.py](scripts/stage_review.py) | Review images and per-class statistics |
| Split | Per-class 85/15 split, seed 2025 | [split_dataset.py](scripts/split_dataset.py) | Train/val folders and split manifest |
| Training | ConvNeXt-Base, augmentation, Mixup/CutMix, AdamW, EMA | [train.py](scripts/train.py) | Class mapping, best/last weights, config, log |
| Inference | Center crop, optional horizontal-flip TTA | [infer.py](scripts/infer.py) | Competition CSV |

Sample score: **S = 0.7 × class-center similarity + 0.3 × Top-10 neighbor similarity**. Each class uses its own 10th/25th percentiles. Default removal candidates are below the 10th percentile; at most 15% of a class can be removed, while retaining at least 30 existing samples. Classes already below 30 are preserved. These are source defaults, not recovered per-run cleaning statistics. See [the equations and boundary cases](docs/METHODS.md).

![ConvNeXt-Base classifier architecture](docs/assets/architecture.png)

| Recovered training setting | WebFG-400 script | WebiNat-5000 script |
|---|---:|---:|
| Model | `convnext_base.fb_in1k` | `convnext_base.fb_in1k` |
| Input / batch size | 320 × 320 / 64 | 320 × 320 / 64 |
| Epochs | 100 | 50 |
| Mixup alpha / CutMix alpha | 0.2 / 0.2 | 0.4 / 1.0 |
| AdamW learning rate / weight decay | 0.0005 / 0.05 | 0.0005 / 0.05 |
| Warmup / minimum learning rate | 3 epochs / 0.000001 | 3 epochs / 0.000001 |
| Drop path / EMA decay | 0.4 / 0.9999 | 0.4 / 0.9999 |

Training uses random resized crops, RandAugment, horizontal flips, and random erasing. EMA validation accuracy selects `best.pth`; TTA uses the same classifier on the original and horizontally flipped crop. Configurations are in [configs/](configs).

## Use the repository

Validate the included prediction files immediately with Python 3.11; these commands need no ML dependencies:

```bash
python scripts/validate_predictions.py results/predictions/pred_results_web400.csv --num-classes 400 --expected-rows 5687
python scripts/validate_predictions.py results/predictions/pred_results_web5000.csv --num-classes 5000 --expected-rows 60000
```

For data preparation, training, or inference, install the portable environment:

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.txt
```

Example training after preparing the image directories and obtaining the specified local pretrained weights:

```bash
python scripts/train.py --config configs/webfg400.json --data-root data/working/webfg400_split --out-dir experiments/webfg400 --pretrained-checkpoint data/weights/convnext_base.fb_in1k.safetensors
```

The [run guide](docs/REPRODUCIBILITY.md) gives commands for all stages, both tasks, inference, weight-only continuation, and CPU/GPU installation. [Model cards](https://huggingface.co/timm/convnext_base.fb_in1k) identify ImageNet-1K provenance; local checkpoint bytes must still be verified when restoring an experiment.

## Archive and verification

```text
archive/original/       Eight recovered scripts, unchanged
scripts/               Portable workflow, validation, figure generation
configs/               Task-specific source settings
data/                  Dataset access, layout, and availability manifest
results/predictions/   Two complete recovered CSVs
results/metrics/       Report metrics, prediction audit, class counts
reports/               Course report and competition specification
docs/                  Methods, run guide, evidence, five PNG/SVG figures
tests/                 Workflow and artifact checks
```

The maintenance version fixes hardcoded task paths, requires feature-extraction weights, records exact split membership, and rejects incomplete inference outputs by default. It also saves last-epoch weights and machine-readable training logs. [Evidence and change notes](docs/EVIDENCE.md) distinguish these additions from the original project and from historical model explorations.

Verification includes six workflow tests, both original CSV audits, preserved-file SHA-256 checks, and a CPU model smoke test covering an optimization step, EMA saving, strict inference with TTA, incomplete-output rejection, and normalized feature/index alignment. Synthetic smoke tests verify execution only. [Verification record](results/verification.json).

Raw images, task-trained weights, original class maps, feature caches, and historical logs are not recovered. The original workflow performs purification before the validation split; its validation set is therefore not an independent holdout for evaluating the purification step. Full score reproduction requires these missing experiment artifacts.

## Credits and references

The report credits **Liu Runzhang (刘润章), Xu Hanzhang (徐瀚章), and Nan Yibo (南怡波)**. It attributes overall algorithm design and tuning to Liu, preprocessing/evaluation/report writing to Xu, and sample filtering/training/result organization to Nan. Student identifiers are redacted from the published report copy.

- [Competition rules](https://www.aicomp.cn/wp-content/uploads/2025/06/%E8%B5%9B%E9%A2%98%E8%A7%84%E5%88%99%EF%BC%9A%E7%BD%91%E7%BB%9C%E7%9B%91%E7%9D%A3%E7%BB%86%E7%B2%92%E5%BA%A6%E5%9B%BE%E5%83%8F%E8%AF%86%E5%88%AB-1.pdf)
- [Webly Supervised Fine-Grained Recognition: Benchmark Datasets and An Approach](https://arxiv.org/abs/2108.02399)
- [ConvNeXt official implementation](https://github.com/facebookresearch/ConvNeXt) · [DeiT III official implementation](https://github.com/facebookresearch/deit)
- [timm](https://github.com/huggingface/pytorch-image-models) · [DeiT III ImageNet-1K model card](https://huggingface.co/timm/deit3_base_patch16_224.fb_in1k)

Code authorship remains with the project contributors. No new redistribution license is assigned to the recovered code, report, datasets, or third-party weights; see [NOTICE.md](NOTICE.md).
