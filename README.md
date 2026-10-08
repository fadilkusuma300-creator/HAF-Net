# HAF-Net

**Hierarchical Attention Learning for Imbalanced Financial Distress Prediction**

This repository contains the code, configurations, public benchmark snapshots, feature-group definitions, and figure-generation utilities used with HAF-Net.

HAF-Net organizes financial indicators with a fixed two-level accounting hierarchy. Each subgroup is encoded separately, intra-group and inter-group attention aggregate the learned representations, and an input residual preserves information that is not captured by the hierarchy. The prediction objective uses class-balanced focal loss. Two optional auxiliary objectives are available: TCAR for prior-guided attention regularization and TCL for adjacent annual observations in the American benchmark.

## Repository layout

```text
.
├── configs/                  # Dataset, training, search, and baseline settings
├── data/
│   └── raw/                  # Pinned public benchmark snapshots
├── metadata/                 # Fixed feature-group mapping
├── scripts/                  # Training, evaluation, statistics, and plotting entry points
├── src/hafnet/               # HAF-Net and comparison-model implementations
├── DATASETS.md               # Dataset sources, targets, splits, and fingerprints
├── requirements.txt
├── requirements-lock.txt
└── pyproject.toml
```

## Environment

Python 3.10 or later is recommended. Create an isolated environment and install the pinned dependencies:

```bash
python -m venv .venv

# Linux / macOS
. .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

`training.device: auto` uses CUDA when available and otherwise falls back to CPU. Tree-based learners can use multiple CPU workers; set `HAFNET_N_JOBS` to a positive integer to control that value.

## Data

The package contains the two public benchmark snapshots under `data/raw/`. Their SHA-256 fingerprints are pinned in the dataset configuration files so the code can verify that the expected files are being used.

| Dataset | Predictors | Train rows / positive | Validation rows / positive | Test rows / positive |
|---|---:|---:|---:|---:|
| Taiwan | 95 | 5,455 / 176 | 682 / 22 | 682 / 22 |
| American | 18 | 55,927 / 403 | 10,473 / 87 | 12,282 / 119 |

Taiwan uses a fixed stratified 80/10/10 split with seed 42. The released table does not provide row-level company or year identifiers, so the benchmark is treated as pooled-sample discrimination.

American uses chronological periods 1999-2011, 2012-2014, and 2015-2018. The configured `terminal_failed_proxy` labels only the last observed row of each failed firm as positive. Company identifiers and years are used for splitting, temporal pairing, and firm-clustered resampling; they are not predictor variables. The training period contains 392 eligible TCL pairs.

Full dataset provenance, licenses, target definitions, and file hashes are documented in [`DATASETS.md`](DATASETS.md).

To re-download the public datasets into the expected locations:

```bash
python scripts/download_data.py --dataset all
```

To verify the loaded data and partition sizes:

```bash
python scripts/inspect_data.py --config configs/taiwan.yaml
python scripts/inspect_data.py --config configs/american.yaml
```

## Training protocol

HAF-Net hyperparameters are selected with tuning seeds 17, 29, and 53 using mean validation average precision (AP). Neural checkpoints are also selected by validation AP. After a checkpoint is fixed, the operating threshold is selected on validation predictions by minority-class F1 and then held fixed for test evaluation.

Final evaluation uses the training seeds:

```text
42, 123, 456, 789, 1024, 2024, 3456, 5678, 7890, 9999
```

The Taiwan data partition remains fixed across all seeds. American uses the fixed chronological split defined in `configs/american.yaml`.

### HAF-Net

```bash
# Taiwan
python scripts/tune_hafnet.py \
  --config configs/taiwan.yaml \
  --output outputs/taiwan/tuning

python scripts/run_experiment.py \
  --config configs/taiwan.yaml \
  --parameters outputs/taiwan/tuning/selected_parameters.json \
  --output outputs/taiwan/hafnet \
  --save-checkpoints

# American
python scripts/tune_hafnet.py \
  --config configs/american.yaml \
  --output outputs/american/tuning

python scripts/run_experiment.py \
  --config configs/american.yaml \
  --parameters outputs/american/tuning/selected_parameters.json \
  --output outputs/american/hafnet \
  --save-checkpoints
```

### Comparison methods

```bash
python scripts/run_baselines.py \
  --config configs/taiwan.yaml \
  --baseline-config configs/baselines.yaml \
  --models all \
  --output outputs/taiwan/baselines

python scripts/run_baselines.py \
  --config configs/american.yaml \
  --baseline-config configs/baselines.yaml \
  --models all \
  --output outputs/american/baselines
```

The benchmark includes Logistic Regression, XGBoost, LightGBM, CatBoost, SMOTE+XGBoost, TabNet, FT-Transformer, Trompt, TabTransformer (numeric adaptation), CSStacking, and Two-Stage LightGBM; the American benchmark also includes the Altman Z-score. TabNet, FT-Transformer, and Trompt follow their defining published architectures. TabTransformer is explicitly identified as a numerical-input adaptation because the two benchmark tables contain continuous financial variables and no categorical columns. CSStacking follows the published two-phase feature-selection and stacking procedure, and Two-Stage LightGBM follows the published full-feature, wrapper-selection, resampling, and LightGBM procedure. Exact search spaces are defined in `configs/baselines.yaml`.

### Ablations

```bash
python scripts/run_ablation.py \
  --config configs/taiwan.yaml \
  --parameters outputs/taiwan/tuning/selected_parameters.json \
  --output outputs/taiwan/ablation

python scripts/run_ablation.py \
  --config configs/american.yaml \
  --parameters outputs/american/tuning/selected_parameters.json \
  --output outputs/american/ablation
```

The ablation entry point covers the representation, loss, residual, prior, and subgroup-aggregation controls used in the manuscript.

## Sensitivity analysis

```bash
python scripts/run_sensitivity.py \
  --config configs/taiwan.yaml \
  --parameters outputs/taiwan/tuning/selected_parameters.json \
  --parameter all \
  --output outputs/taiwan/sensitivity
```

The sensitivity run writes setting-level results and a manifest. Figure 4 can then be regenerated from those outputs with:

```bash
python scripts/plot_sensitivity.py \
  --summary outputs/taiwan/sensitivity/sensitivity_summary.csv \
  --manifest outputs/taiwan/sensitivity/sensitivity_manifest.json \
  --output outputs/taiwan/sensitivity
```

## Attention analysis

A selected-case table can be exported from a trained Taiwan checkpoint with:

```bash
python scripts/export_attention_cases.py \
  --config configs/taiwan.yaml \
  --parameters outputs/taiwan/tuning/selected_parameters.json \
  --checkpoint outputs/taiwan/hafnet/checkpoints/taiwan_full_seed42.pt \
  --seed 42 \
  --output outputs/taiwan/cases
```

The corresponding figure is generated with:

```bash
python scripts/plot_attention_cases.py \
  --attention outputs/taiwan/cases/case_attention.csv \
  --features outputs/taiwan/cases/case_features.csv \
  --output outputs/taiwan/cases
```

The evaluation workflow also writes full-test group-attention summaries and cross-seed dominant-group agreement.

## Reproduce the manuscript figures

The plotting definitions for the manuscript versions of Figures 3 and 4 are contained in `scripts/reproduce_figures.py`. Run:

```bash
python scripts/reproduce_figures.py --output figures
```

This creates `Fig3.pdf`, `Fig3.png`, `Fig4.pdf`, and `Fig4.png`. The same plotting functions are also used by `plot_attention_cases.py` and `plot_sensitivity.py` for outputs generated by the experiment scripts.

## Metrics and statistical analysis

Row-level prediction files contain the dataset, split, model, seed, row identifier, target, continuous score, selected threshold, and predicted class. American predictions additionally retain the company identifier, fiscal year, and target definition.

Metrics can be recomputed directly from prediction files:

```bash
python scripts/export_metric_tables.py \
  --predictions \
    outputs/taiwan/hafnet/hafnet_predictions.csv \
    outputs/taiwan/baselines/baseline_predictions.csv \
  --output outputs/taiwan/tables
```

Primary paired inference is run with:

```bash
python scripts/run_statistics.py \
  --taiwan-hafnet outputs/taiwan/hafnet/hafnet_predictions.csv \
  --taiwan-baselines outputs/taiwan/baselines/baseline_predictions.csv \
  --american-hafnet outputs/american/hafnet/hafnet_predictions.csv \
  --american-baselines outputs/american/baselines/baseline_predictions.csv \
  --bootstrap 10000 \
  --output outputs/statistics/primary_comparisons.csv
```

Taiwan paired intervals resample test rows. American paired intervals resample company identifiers and keep all test-period observations for each sampled company together. Matched-seed Wilcoxon comparisons are reported separately to characterize training-run variability. Holm adjustment is applied to the four prespecified dataset-by-metric primary comparisons.

## Deterministic data products

The following command rebuilds split membership, preprocessing statistics, label counts, temporal-pair counts, and the direct Altman calculation from the pinned raw CSV files:

```bash
python scripts/prepare_data_summary.py
```

Generated files are written to `data/derived/`, which is created on demand and is not included in the source distribution.

## Outputs

Experiment scripts create their outputs only under the directory supplied with `--output`. Generated checkpoints, predictions, summaries, figures, and logs are not part of the source distribution.

Typical output files include:

```text
selected_parameters.json
hafnet_runs.csv
hafnet_predictions.csv
hafnet_attention.csv
baseline_runs.csv
baseline_predictions.csv
ablation_inference.csv
sensitivity_runs.csv
sensitivity_summary.csv
primary_comparisons.csv
```

## Citation and license

Software citation metadata are provided in `CITATION.cff`.

The code is released under the MIT License. The benchmark datasets retain their original source licenses; see `DATASETS.md` for details.

---

## 中文说明

本仓库包含 HAF-Net 的模型实现、配置文件、公开数据快照、特征分组、实验入口、统计分析以及论文图 3/图 4 的绘图代码。

推荐从以下顺序开始：

1. 安装 `requirements-lock.txt` 中的依赖并执行 `pip install --no-deps -e .`；
2. 使用 `inspect_data.py` 检查两个数据集与划分；
3. 使用 `tune_hafnet.py` 完成参数选择；
4. 使用 `run_experiment.py` 训练 HAF-Net；
5. 使用 `run_baselines.py`、`run_ablation.py` 和 `run_sensitivity.py` 运行比较、消融与敏感性实验；
6. 使用 `run_statistics.py` 从逐行预测计算配对统计；
7. 使用 `reproduce_figures.py` 直接重绘论文中的图 3 和图 4。

数据来源与标签口径见 `DATASETS.md`，比较方法的搜索空间见 `configs/baselines.yaml`。
