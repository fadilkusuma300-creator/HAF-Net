# HAF-Net: Hierarchical Attention Learning for Imbalanced Financial Distress Prediction

<p align="center">
  <a href="#english"><strong>English</strong></a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="#中文"><strong>中文</strong></a>
</p>

---

<a id="english"></a>

## English

### 1. Description

HAF-Net is a neural framework for financial distress prediction under severe class imbalance. It combines a fixed two-level financial feature hierarchy with minority-sensitive optimization. The model first organizes accounting variables into financially meaningful subgroups and level-1 dimensions, learns intra-group and inter-group attention, and then optimizes the resulting representation with class-balanced focal loss.

Two optional regularizers use information available in specific benchmark settings:

- **Theory-Consistent Attention Regularization (TCAR)** applies a soft group-level financial prior to positive observations on the Taiwan benchmark.
- **Temporal Contrastive Learning (TCL)** uses ordered firm-year observations on the American benchmark to relate adjacent pre-filing states and same-year healthy firms.

The repository includes data loading, train-only preprocessing, HAF-Net, comparison methods, ablation settings, sensitivity analysis, attention-case export, evaluation metrics, validation-threshold selection, and paired statistical inference.

### 2. Repository structure

```text
HAF-Net/
├── README.md
├── DATASETS.md
├── CITATION.cff
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── taiwan.yaml
│   ├── american.yaml
│   └── baselines.yaml
├── data/
│   ├── README.md
│   └── raw/
│       ├── taiwan/
│       └── american/
├── metadata/
│   └── feature_group_mapping.csv
├── scripts/
│   ├── download_data.py
│   ├── inspect_data.py
│   ├── run_experiment.py
│   ├── run_baselines.py
│   ├── run_ablation.py
│   ├── run_sensitivity.py
│   ├── export_attention_cases.py
│   ├── run_statistics.py
│   └── summarize_results.py
└── src/
    └── hafnet/
        ├── baselines.py
        ├── config.py
        ├── data.py
        ├── experiment.py
        ├── grouping.py
        ├── losses.py
        ├── metrics.py
        ├── model.py
        ├── preprocess.py
        ├── prior.py
        ├── statistics.py
        ├── temporal.py
        ├── trainer.py
        └── utils.py
```

### 3. Dataset information

#### 3.1 Taiwan Bankruptcy Prediction

- Source: UCI Machine Learning Repository
- Dataset page: https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction
- DOI: https://doi.org/10.24432/C5004D
- License: CC BY 4.0
- Observations: 6,819
- Predictive variables: 95 financial indicators
- Positive observations: 220
- Local file: `data/raw/taiwan/data.csv`
- Split used by the code: stratified 80% / 10% / 10% for train / validation / test

The 95 variables are assigned to four level-1 financial dimensions and eight level-2 subgroups. The exact mapping is stored in `metadata/feature_group_mapping.csv`.

#### 3.2 American Bankruptcy Dataset

- Source repository: https://github.com/sowide/bankruptcy_dataset
- Dataset article: https://doi.org/10.3390/fi14080244
- License: CC BY 4.0
- Observations: 78,682 firm-years
- Companies: 8,262
- Predictive variables: `X1`-`X18`
- Period: 1999-2018
- Local file: `data/raw/american/american_bankruptcy_dataset.csv`
- Chronological split:
  - train: 1999-2011
  - validation: 2012-2014
  - test: 2015-2018

The public row-level `status_label` is used directly. A positive row is the fiscal year immediately before a Chapter 7 or Chapter 11 filing. Company identifiers and fiscal years are used for chronological splitting and TCL pair construction only; they are not predictive inputs.

The benchmark tables contain complete numeric inputs, so the HAF-Net preprocessing path does not perform missing-value imputation.

### 4. Code information

| Component | Purpose |
|---|---|
| `src/hafnet/grouping.py` | Loads the fixed 4-group / 8-subgroup financial hierarchy and builds the random-group control. |
| `src/hafnet/preprocess.py` | Fits 1st/99th percentile Winsorization and standardization statistics on the training partition only. |
| `src/hafnet/model.py` | Implements HFGE, additive intra-group/inter-group attention, residual fusion, the classifier, and flat representation controls. |
| `src/hafnet/losses.py` | Implements mean-one normalized class-balanced focal loss, TCAR, and temporal InfoNCE. |
| `src/hafnet/prior.py` | Builds the theory/statistical financial prior and the prior-permutation control. |
| `src/hafnet/temporal.py` | Builds training-only TCL anchor/positive pairs and same-year healthy negative pools. |
| `src/hafnet/trainer.py` | Trains HAF-Net, selects checkpoints by validation PR-AUC, and fixes the validation-F1 threshold before test evaluation. |
| `src/hafnet/baselines.py` | Implements the comparison methods under the common data and threshold protocol. |
| `src/hafnet/statistics.py` | Implements paired bootstrap confidence intervals, paired Wilcoxon tests, and Holm correction. |
| `scripts/*.py` | Command-line entry points for data acquisition, experiments, controls, figures, and result summaries. |

The comparison set consists of Logistic Regression, Altman Z-Score, XGBoost, LightGBM, CatBoost, SMOTE+XGBoost, TabNet, TabTransformer, FT-Transformer, Trompt, CS-Stacking, and Two-Stage LightGBM. The neural tabular methods use continuous-feature tokenization because both benchmark inputs are numeric.

### 5. Requirements

Recommended environment:

- Python 3.10 or newer
- PyTorch 2.1 or newer
- CUDA-capable GPU for neural experiments; CPU execution is supported but slower

Core Python dependencies are listed in `requirements.txt` and `pyproject.toml`.

Create an isolated environment and install the package from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 6. Dataset setup

Download both public datasets:

```bash
python scripts/download_data.py --dataset all
```

Or download one benchmark:

```bash
python scripts/download_data.py --dataset taiwan
python scripts/download_data.py --dataset american
```

Expected paths after download:

```text
data/raw/taiwan/data.csv
data/raw/american/american_bankruptcy_dataset.csv
```

Inspect the loaded data, class distribution, split sizes, and hierarchy:

```bash
python scripts/inspect_data.py --config configs/taiwan.yaml
python scripts/inspect_data.py --config configs/american.yaml
```

### 7. Usage instructions

All commands below are run from the repository root.

#### 7.1 Main HAF-Net experiments

Taiwan:

```bash
python scripts/run_experiment.py \
  --config configs/taiwan.yaml \
  --output outputs/taiwan/hafnet \
  --save-checkpoints
```

American:

```bash
python scripts/run_experiment.py \
  --config configs/american.yaml \
  --output outputs/american/hafnet \
  --save-checkpoints
```

Each run writes `hafnet_runs.csv` and `hafnet_summary.csv`. When `--save-checkpoints` is supplied, model state dictionaries are written to the corresponding `checkpoints/` directory.

To run a subset of configured seeds:

```bash
python scripts/run_experiment.py \
  --config configs/taiwan.yaml \
  --seeds 42 123 \
  --output outputs/taiwan/hafnet_subset
```

#### 7.2 Comparison methods

Taiwan:

```bash
python scripts/run_baselines.py \
  --config configs/taiwan.yaml \
  --baseline-config configs/baselines.yaml \
  --models all \
  --output outputs/taiwan/baselines
```

American:

```bash
python scripts/run_baselines.py \
  --config configs/american.yaml \
  --baseline-config configs/baselines.yaml \
  --models all \
  --output outputs/american/baselines
```

A selected subset can be requested explicitly:

```bash
python scripts/run_baselines.py \
  --config configs/taiwan.yaml \
  --models logistic xgboost lightgbm catboost two_stage_lightgbm \
  --output outputs/taiwan/baselines_selected
```

The script chooses model settings on the validation partition and stores the selected parameter dictionaries in `selected_parameters.json`.

#### 7.3 Component and structure controls

Taiwan controls include the full model, HFGE+CB-Focal core, HFGE+BCE, Flat+CB-Focal, Flat+BCE, Random-HFGE+CB-Focal, HAF-Net without the residual path, TCAR off, and permuted-prior TCAR:

```bash
python scripts/run_ablation.py \
  --config configs/taiwan.yaml \
  --output outputs/taiwan/ablation
```

American compares the full model with TCL disabled:

```bash
python scripts/run_ablation.py \
  --config configs/american.yaml \
  --output outputs/american/ablation
```

#### 7.4 Hyperparameter sensitivity

The manuscript sensitivity analysis evaluates `lambda_tcar`, `beta`, and `gamma` on the Taiwan validation partition:

```bash
python scripts/run_sensitivity.py \
  --config configs/taiwan.yaml \
  --parameter all \
  --output outputs/taiwan/sensitivity
```

This writes run-level and summary CSV files together with PDF and 600-dpi PNG figures.

#### 7.5 Attention cases

After a Taiwan main run with checkpoints enabled, export three positive cases with their group-attention distributions and prominent standardized indicators:

```bash
python scripts/export_attention_cases.py \
  --config configs/taiwan.yaml \
  --checkpoint outputs/taiwan/hafnet/checkpoints/taiwan_full_seed42.pt \
  --seed 42 \
  --output outputs/taiwan/attention_cases
```

The command writes `case_attention.csv`, `case_features.csv`, `attention_cases.pdf`, and a 600-dpi PNG.

#### 7.6 Primary paired statistical comparisons

After HAF-Net and Two-Stage LightGBM have been run for both datasets:

```bash
python scripts/run_statistics.py \
  --taiwan-hafnet outputs/taiwan/hafnet/hafnet_runs.csv \
  --taiwan-baselines outputs/taiwan/baselines/baseline_runs.csv \
  --american-hafnet outputs/american/hafnet/hafnet_runs.csv \
  --american-baselines outputs/american/baselines/baseline_runs.csv \
  --bootstrap 10000 \
  --output outputs/statistics/primary_comparisons.csv
```

The four prespecified comparisons are F1 and MCC on Taiwan and American. The script reports the paired mean difference, paired nonparametric bootstrap 95% confidence interval, paired Wilcoxon signed-rank p-value, and Holm-adjusted p-value.

#### 7.7 Compact result tables

```bash
python scripts/summarize_results.py \
  --input outputs/taiwan/hafnet/hafnet_runs.csv \
  --output outputs/taiwan/hafnet/table.md
```

The same command accepts `baseline_runs.csv`.

### 8. Methodology

#### 8.1 Preprocessing

For each run, preprocessing statistics are estimated from the training partition only:

1. Compute the 1st and 99th percentile for each numerical feature.
2. Winsorize the training, validation, and test matrices with the fixed training percentiles.
3. Compute feature means and standard deviations from the Winsorized training matrix.
4. Standardize all three partitions with the fixed training statistics.
5. Apply the fixed accounting-semantic feature hierarchy from `metadata/feature_group_mapping.csv`.

No test observation contributes to preprocessing, model selection, the TCAR statistical prior, or TCL training pairs.

#### 8.2 HFGE

For each of eight level-2 subgroups, an independent two-layer MLP maps the local features to a 32-dimensional representation. Additive attention first aggregates the two subgroups inside each level-1 financial dimension. A second additive-attention layer combines the four level-1 dimensions. The weighted representation is projected to 64 dimensions and added to a linear residual projection of the original input before the single binary classification head.

With the fixed hierarchy and dimensions in the supplied configurations, the model has:

- Taiwan: 25,505 trainable parameters
- American: 18,113 trainable parameters

#### 8.3 Class-balanced focal loss

For class count `n_y`, the effective-number weight is

```text
w_y = (1 - beta) / (1 - beta^n_y)
```

The two class weights are normalized to a class mean of one:

```text
w_tilde_y = 2 * w_y / (w_0 + w_1)
```

The sample loss is

```text
L_CBF = -w_tilde_y * (1 - p_t)^gamma * log(p_t)
```

Default settings are `beta=0.999` and `gamma=2`.

#### 8.4 TCAR

TCAR is active in the Taiwan configuration (`lambda_tcar=0.05`). Its four-dimensional theory component is:

```text
[0.35, 0.30, 0.20, 0.15]
```

for Solvency, Profitability, Operational efficiency, and Growth/capital capacity. The statistical component is derived from the standardized absolute positive-negative mean difference of each training feature, averaged within each level-1 group and normalized. With `rho=0.5`, the prior is:

```text
prior = rho * theory_prior + (1 - rho) * statistical_prior
```

For positive observations only, TCAR minimizes `KL(prior || group_attention)`. The American configuration sets `lambda_tcar=0`.

#### 8.5 TCL

TCL is active in the American configuration (`lambda_tcl=0.10`). A public positive row represents the fiscal year immediately before a filing. The paired pre-filing states are therefore:

- positive state: filing year minus 1
- anchor state: filing year minus 2

For every eligible training positive, healthy negatives are drawn from the anchor fiscal year. TCL uses cosine InfoNCE with temperature `0.10` and eight negatives per eligible positive observation. All TCL pairs are constructed inside the training interval.

#### 8.6 Model selection and decision threshold

- Neural checkpoints are selected by validation PR-AUC with a patience of 15 epochs.
- After the checkpoint is fixed, one decision threshold is selected on the validation partition by maximizing minority-class F1.
- The selected threshold is then fixed for all threshold-dependent test metrics from that run.
- ROC-AUC and PR-AUC are computed directly from continuous risk scores.
- Altman Z-Score retains its classical cutoff of `Z < 2.675`.

#### 8.7 Evaluation metrics

The code reports:

- Accuracy
- minority-class Precision
- minority-class Recall
- minority-class F1
- ROC-AUC
- PR-AUC
- Matthews Correlation Coefficient (MCC)
- G-Mean

The configured random seeds are:

```text
42, 123, 456, 789, 1024, 2024, 3456, 5678, 7890, 9999
```

### 9. Output files

Common output files include:

| File | Contents |
|---|---|
| `hafnet_runs.csv` | One HAF-Net row per seed and model configuration. |
| `hafnet_summary.csv` | Mean and standard deviation across seeds. |
| `group_attention.csv` | Mean positive-class group attention for each HAF-Net run. |
| `baseline_runs.csv` | One comparison-method row per seed. |
| `baseline_summary.csv` | Comparison-method summaries across seeds. |
| `selected_parameters.json` | Model settings selected by the validation protocol. |
| `primary_comparisons.csv` | Paired bootstrap intervals and Wilcoxon/Holm statistics. |
| `sensitivity_*.csv` | Run-level and aggregated sensitivity results. |
| `case_attention.csv` | Group-attention values for selected positive cases. |
| `case_features.csv` | Prominent standardized indicators for the selected cases. |

### 10. Citations

If this software is used in academic work, please cite the HAF-Net article and the original benchmark datasets.

**HAF-Net**

```text
Meng, Z., Wang, T., & Zhang, L.
HAF-Net: Hierarchical Attention Learning for Imbalanced Financial Distress Prediction.
```

**Taiwan Bankruptcy Prediction**

```text
UCI Machine Learning Repository.
Taiwanese Bankruptcy Prediction.
https://doi.org/10.24432/C5004D
```

**American Bankruptcy Dataset**

```text
Lombardo, G., Pellegrino, M., Adosoglou, G., Cagnoni, S., Pardalos, P. M., & Poggi, A. (2022).
Machine Learning for Bankruptcy Prediction in the American Stock Market: Dataset and Benchmarks.
Future Internet, 14(8), 244.
https://doi.org/10.3390/fi14080244
```

A machine-readable software citation is also provided in `CITATION.cff`.

### 11. License and contribution guidelines

The source code in this repository is distributed under the MIT License; see `LICENSE`. Benchmark datasets remain subject to the terms stated by their original providers and are not covered by the software license.

For contributions:

1. Open an issue describing the proposed change or problem.
2. Keep dataset access and preprocessing confined to the public benchmark protocol.
3. Add clear docstrings and comments for new public functions and command-line options.
4. Keep generated outputs, raw datasets, checkpoints, and local environments outside source control.
5. Submit focused pull requests with a concise description of the affected component and expected behavior.

<p align="right"><a href="#haf-net-hierarchical-attention-learning-for-imbalanced-financial-distress-prediction">Back to top</a></p>

---

<a id="中文"></a>

## 中文

### 1. 项目说明

HAF-Net 是一个面向严重类别不平衡财务困境预测的神经网络框架。其核心思想是将固定的两级财务语义层次与少数类敏感优化放在同一表示学习过程中：模型先按照会计含义把财务变量组织为二级子组和一级财务维度，通过组内注意力与组间注意力学习结构化表示，再使用类别平衡焦点损失强化稀有困境样本及难分类样本对训练的影响。

框架还包含两个由数据条件决定的辅助正则项：

- **理论一致性注意力正则化（TCAR）**：在 Taiwan 基准上，对正类观测的组间注意力施加软金融先验约束。
- **时间对比学习（TCL）**：在 American 基准上，利用有序企业年度观测，把相邻破产前状态与同年健康企业纳入时间对比目标。

本仓库包含数据读取、仅基于训练集的预处理、HAF-Net 主模型、对比方法、消融设置、敏感性分析、注意力案例导出、评价指标、验证集阈值选择以及配对统计推断的完整代码。

### 2. 目录结构

```text
HAF-Net/
├── README.md
├── DATASETS.md
├── CITATION.cff
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── taiwan.yaml
│   ├── american.yaml
│   └── baselines.yaml
├── data/
│   ├── README.md
│   └── raw/
│       ├── taiwan/
│       └── american/
├── metadata/
│   └── feature_group_mapping.csv
├── scripts/
│   ├── download_data.py
│   ├── inspect_data.py
│   ├── run_experiment.py
│   ├── run_baselines.py
│   ├── run_ablation.py
│   ├── run_sensitivity.py
│   ├── export_attention_cases.py
│   ├── run_statistics.py
│   └── summarize_results.py
└── src/
    └── hafnet/
        ├── baselines.py
        ├── config.py
        ├── data.py
        ├── experiment.py
        ├── grouping.py
        ├── losses.py
        ├── metrics.py
        ├── model.py
        ├── preprocess.py
        ├── prior.py
        ├── statistics.py
        ├── temporal.py
        ├── trainer.py
        └── utils.py
```

### 3. 数据集信息

#### 3.1 Taiwan Bankruptcy Prediction

- 来源：UCI Machine Learning Repository
- 数据集页面：https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction
- DOI：https://doi.org/10.24432/C5004D
- 许可：CC BY 4.0
- 观测数：6,819
- 预测变量：95 个财务指标
- 正类观测：220
- 本地路径：`data/raw/taiwan/data.csv`
- 数据划分：分层 80% / 10% / 10% 训练集、验证集、测试集

95 个变量被固定映射到 4 个一级财务维度和 8 个二级子组，完整机器可读映射位于 `metadata/feature_group_mapping.csv`。

#### 3.2 American Bankruptcy Dataset

- 数据仓库：https://github.com/sowide/bankruptcy_dataset
- 数据集论文：https://doi.org/10.3390/fi14080244
- 许可：CC BY 4.0
- 观测数：78,682 个企业年度观测
- 企业数：8,262
- 预测变量：`X1`-`X18`
- 时间范围：1999-2018
- 本地路径：`data/raw/american/american_bankruptcy_dataset.csv`
- 时间划分：
  - 训练集：1999-2011
  - 验证集：2012-2014
  - 测试集：2015-2018

代码直接使用公开数据中的逐行 `status_label`。正类行表示企业在该财年之后一年依据 Chapter 7 或 Chapter 11 进入破产程序。企业标识符和财年仅用于时间划分与 TCL 样本对构造，不进入 18 个预测变量。

两个公开基准的数值输入均完整，因此 HAF-Net 的预处理流程不执行缺失值插补。

### 4. 代码说明

| 模块 | 功能 |
|---|---|
| `src/hafnet/grouping.py` | 读取固定的 4 个一级组 / 8 个二级子组层次，并构造随机分组对照。 |
| `src/hafnet/preprocess.py` | 仅在训练集估计 1%/99% Winsorization 阈值以及标准化统计量。 |
| `src/hafnet/model.py` | 实现 HFGE、组内/组间加性注意力、残差融合、分类头和扁平表示控制模型。 |
| `src/hafnet/losses.py` | 实现类别平均权重归一到 1 的 CB-Focal、TCAR 和时间 InfoNCE。 |
| `src/hafnet/prior.py` | 构造理论/统计组合金融先验以及先验置换对照。 |
| `src/hafnet/temporal.py` | 构造仅来自训练时间区间的 TCL 锚点、正样本和同年健康负样本池。 |
| `src/hafnet/trainer.py` | 训练 HAF-Net，以验证集 PR-AUC 选择检查点，并在测试前固定验证集 F1 最优阈值。 |
| `src/hafnet/baselines.py` | 在统一数据协议和阈值协议下实现对比方法。 |
| `src/hafnet/statistics.py` | 实现配对 bootstrap 置信区间、配对 Wilcoxon 检验和 Holm 校正。 |
| `scripts/*.py` | 数据下载、实验运行、控制实验、图形导出和结果汇总的命令行入口。 |

对比方法包括 Logistic Regression、Altman Z-Score、XGBoost、LightGBM、CatBoost、SMOTE+XGBoost、TabNet、TabTransformer、FT-Transformer、Trompt、CS-Stacking 和 Two-Stage LightGBM。两个基准的输入均为数值型，因此神经表格方法使用连续特征 token 化方式，不设置类别特征嵌入分支。

### 5. 环境与依赖

推荐环境：

- Python 3.10 或更高版本
- PyTorch 2.1 或更高版本
- 神经网络实验建议使用支持 CUDA 的 GPU；CPU 也可运行，但速度较慢

完整 Python 依赖见 `requirements.txt` 与 `pyproject.toml`。

在仓库根目录创建独立环境并安装：

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 6. 数据准备

下载两个公开数据集：

```bash
python scripts/download_data.py --dataset all
```

也可以分别下载：

```bash
python scripts/download_data.py --dataset taiwan
python scripts/download_data.py --dataset american
```

下载后的预期位置：

```text
data/raw/taiwan/data.csv
data/raw/american/american_bankruptcy_dataset.csv
```

检查数据维度、类别分布、划分规模和特征层次：

```bash
python scripts/inspect_data.py --config configs/taiwan.yaml
python scripts/inspect_data.py --config configs/american.yaml
```

### 7. 使用方法

以下命令均从仓库根目录运行。

#### 7.1 HAF-Net 主实验

Taiwan：

```bash
python scripts/run_experiment.py \
  --config configs/taiwan.yaml \
  --output outputs/taiwan/hafnet \
  --save-checkpoints
```

American：

```bash
python scripts/run_experiment.py \
  --config configs/american.yaml \
  --output outputs/american/hafnet \
  --save-checkpoints
```

每组实验输出 `hafnet_runs.csv` 和 `hafnet_summary.csv`。添加 `--save-checkpoints` 后，模型参数字典写入对应的 `checkpoints/` 目录。

只运行部分随机种子时：

```bash
python scripts/run_experiment.py \
  --config configs/taiwan.yaml \
  --seeds 42 123 \
  --output outputs/taiwan/hafnet_subset
```

#### 7.2 对比方法

Taiwan：

```bash
python scripts/run_baselines.py \
  --config configs/taiwan.yaml \
  --baseline-config configs/baselines.yaml \
  --models all \
  --output outputs/taiwan/baselines
```

American：

```bash
python scripts/run_baselines.py \
  --config configs/american.yaml \
  --baseline-config configs/baselines.yaml \
  --models all \
  --output outputs/american/baselines
```

只运行指定方法：

```bash
python scripts/run_baselines.py \
  --config configs/taiwan.yaml \
  --models logistic xgboost lightgbm catboost two_stage_lightgbm \
  --output outputs/taiwan/baselines_selected
```

脚本使用验证集选择模型设置，并把选定参数写入 `selected_parameters.json`。

#### 7.3 组件与结构控制实验

Taiwan 包括完整模型、HFGE+CB-Focal 核心、HFGE+BCE、Flat+CB-Focal、Flat+BCE、Random-HFGE+CB-Focal、移除残差通路、关闭 TCAR 和先验置换 TCAR：

```bash
python scripts/run_ablation.py \
  --config configs/taiwan.yaml \
  --output outputs/taiwan/ablation
```

American 比较完整模型与关闭 TCL：

```bash
python scripts/run_ablation.py \
  --config configs/american.yaml \
  --output outputs/american/ablation
```

#### 7.4 超参数敏感性

论文中的敏感性分析在 Taiwan 验证集考察 `lambda_tcar`、`beta` 和 `gamma`：

```bash
python scripts/run_sensitivity.py \
  --config configs/taiwan.yaml \
  --parameter all \
  --output outputs/taiwan/sensitivity
```

脚本生成逐次运行 CSV、汇总 CSV、PDF 图和 600-dpi PNG 图。

#### 7.5 注意力案例

完成 Taiwan 主实验并保存模型参数后，可导出三个正类案例的组级注意力以及对应财务组中绝对标准化值较大的指标：

```bash
python scripts/export_attention_cases.py \
  --config configs/taiwan.yaml \
  --checkpoint outputs/taiwan/hafnet/checkpoints/taiwan_full_seed42.pt \
  --seed 42 \
  --output outputs/taiwan/attention_cases
```

输出包括 `case_attention.csv`、`case_features.csv`、`attention_cases.pdf` 和 600-dpi PNG。

#### 7.6 主要配对统计比较

Taiwan 与 American 的 HAF-Net 和 Two-Stage LightGBM 均完成运行后：

```bash
python scripts/run_statistics.py \
  --taiwan-hafnet outputs/taiwan/hafnet/hafnet_runs.csv \
  --taiwan-baselines outputs/taiwan/baselines/baseline_runs.csv \
  --american-hafnet outputs/american/hafnet/hafnet_runs.csv \
  --american-baselines outputs/american/baselines/baseline_runs.csv \
  --bootstrap 10000 \
  --output outputs/statistics/primary_comparisons.csv
```

四项预设比较为两个数据集上的 F1 与 MCC。输出包含配对均值差、配对非参数 bootstrap 95% 置信区间、配对 Wilcoxon 符号秩检验 p 值和 Holm 校正 p 值。

#### 7.7 简洁结果表

```bash
python scripts/summarize_results.py \
  --input outputs/taiwan/hafnet/hafnet_runs.csv \
  --output outputs/taiwan/hafnet/table.md
```

同一脚本也可读取 `baseline_runs.csv`。

### 8. 方法流程

#### 8.1 预处理

每次运行的预处理统计量仅由训练集估计：

1. 计算各数值特征的第 1 与第 99 百分位点；
2. 使用固定的训练集百分位点对训练、验证和测试矩阵进行 Winsorization；
3. 在处理后的训练矩阵上计算均值和标准差；
4. 使用固定训练统计量标准化三个数据划分；
5. 按 `metadata/feature_group_mapping.csv` 中固定的会计语义层次组织模型输入。

测试集观测不会参与预处理、模型选择、TCAR 统计先验或 TCL 训练样本对的构造。

#### 8.2 HFGE

8 个二级子组分别由参数独立的两层 MLP 映射到 32 维表示。加性注意力先在每个一级财务维度内部聚合两个子组，再在四个一级维度之间进行第二次加权。加权组表示被投影到 64 维，并与原始输入的线性残差投影相加，最后接单层二元分类头。

按照仓库中固定的分组和维度配置，可训练参数量为：

- Taiwan：25,505
- American：18,113

#### 8.3 类别平衡焦点损失

类别 `y` 的有效样本数权重为：

```text
w_y = (1 - beta) / (1 - beta^n_y)
```

两个类别权重进一步归一化，使类别平均权重为 1：

```text
w_tilde_y = 2 * w_y / (w_0 + w_1)
```

单样本损失为：

```text
L_CBF = -w_tilde_y * (1 - p_t)^gamma * log(p_t)
```

默认设置为 `beta=0.999`、`gamma=2`。

#### 8.4 TCAR

TCAR 在 Taiwan 配置启用（`lambda_tcar=0.05`），其四维理论先验为：

```text
[0.35, 0.30, 0.20, 0.15]
```

依次对应偿债能力、盈利能力、运营效率和成长/资本能力。统计成分由训练集标准化后各特征的正负类绝对均值差计算，在一级组内求平均并归一化。`rho=0.5` 时：

```text
prior = rho * theory_prior + (1 - rho) * statistical_prior
```

TCAR 仅对正类观测计算 `KL(prior || group_attention)`。American 配置设置 `lambda_tcar=0`。

#### 8.5 TCL

TCL 在 American 配置启用（`lambda_tcl=0.10`）。公开数据中的正类行对应破产申报前一个财年，因此时间对比状态为：

- 正样本状态：破产申报年份前 1 年；
- 锚点状态：破产申报年份前 2 年。

每个满足条件的训练正类观测从锚点年份抽取健康企业作为负样本。TCL 使用余弦 InfoNCE，温度为 `0.10`，每个有效正类观测使用 8 个负样本。所有时间样本对均在训练时间区间内部构造。

#### 8.6 模型选择与决策阈值

- 神经模型以验证集 PR-AUC 选择检查点，patience 为 15 个 epoch；
- 检查点确定后，仅在验证集选择使少数类 F1 最大的一个决策阈值；
- 该阈值随后固定用于对应测试运行的所有阈值相关指标；
- ROC-AUC 与 PR-AUC 直接根据连续风险分数计算；
- Altman Z-Score 保留经典的 `Z < 2.675` 判别阈值。

#### 8.7 评价指标

代码报告：

- Accuracy
- 少数类 Precision
- 少数类 Recall
- 少数类 F1
- ROC-AUC
- PR-AUC
- Matthews Correlation Coefficient（MCC）
- G-Mean

配置中的 10 个随机种子为：

```text
42, 123, 456, 789, 1024, 2024, 3456, 5678, 7890, 9999
```

### 9. 输出文件

常见输出包括：

| 文件 | 内容 |
|---|---|
| `hafnet_runs.csv` | HAF-Net 每个随机种子和模型设置的逐次指标。 |
| `hafnet_summary.csv` | 多随机种子的均值和标准差。 |
| `group_attention.csv` | 每次 HAF-Net 运行的正类平均组级注意力。 |
| `baseline_runs.csv` | 对比方法逐随机种子结果。 |
| `baseline_summary.csv` | 对比方法多随机种子汇总。 |
| `selected_parameters.json` | 通过验证集协议选择的模型设置。 |
| `primary_comparisons.csv` | 配对 bootstrap 区间及 Wilcoxon/Holm 统计结果。 |
| `sensitivity_*.csv` | 敏感性分析逐次结果与汇总结果。 |
| `case_attention.csv` | 所选正类案例的组级注意力。 |
| `case_features.csv` | 案例中代表性财务指标的标准化数值。 |

### 10. 引用

在学术工作中使用本代码时，请同时引用 HAF-Net 论文和原始基准数据集。

**HAF-Net**

```text
Meng, Z., Wang, T., & Zhang, L.
HAF-Net: Hierarchical Attention Learning for Imbalanced Financial Distress Prediction.
```

**Taiwan Bankruptcy Prediction**

```text
UCI Machine Learning Repository.
Taiwanese Bankruptcy Prediction.
https://doi.org/10.24432/C5004D
```

**American Bankruptcy Dataset**

```text
Lombardo, G., Pellegrino, M., Adosoglou, G., Cagnoni, S., Pardalos, P. M., & Poggi, A. (2022).
Machine Learning for Bankruptcy Prediction in the American Stock Market: Dataset and Benchmarks.
Future Internet, 14(8), 244.
https://doi.org/10.3390/fi14080244
```

仓库还提供机器可读的 `CITATION.cff`。

### 11. 许可与贡献指南

本仓库源代码采用 MIT License，详见 `LICENSE`。两个公开基准数据集继续遵循其原始提供方声明的许可条款，不属于本软件许可覆盖范围。

如需贡献代码：

1. 先通过 issue 描述拟解决的问题或功能；
2. 保持数据访问、划分和预处理符合仓库中的公开基准协议；
3. 新增公共函数和命令行参数时补充清晰的 docstring 与必要注释；
4. 原始数据、实验输出、模型参数文件和本地环境目录不提交到源码仓库；
5. Pull request 应聚焦单一功能，并清楚说明涉及的模块和预期行为。

<p align="right"><a href="#haf-net-hierarchical-attention-learning-for-imbalanced-financial-distress-prediction">返回顶部</a></p>
