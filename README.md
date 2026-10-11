# HAF-Net

**Hierarchical Attention Fusion Network for Imbalanced Financial Distress Prediction**

This package contains the source code and configurations used for the HAF-Net experiments reported in the manuscript. It implements the HAF-Net predictive core, dataset-specific auxiliary objectives, comparison methods, ablations, validation-only parameter selection, sensitivity analysis, attention analysis, and paired statistical inference.

## 1. Environment

Python 3.10 or later is recommended.

```bash
python -m venv .venv

# Linux / macOS
. .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

`training.device: auto` uses CUDA when available and otherwise uses CPU. For CPU-based tree models, `HAFNET_N_JOBS` can be set to a positive integer to control worker count.

## 2. Data

The two benchmark datasets are public and are not duplicated in this source package. Download the pinned releases with:

```bash
python scripts/download_data.py --dataset all
```

Expected paths are:

```text
data/raw/taiwan/data.csv
data/raw/american/american_bankruptcy_dataset.csv
```

The configured SHA-256 fingerprints are verified before experiments are run. Dataset sources, target definitions, split rules, and fingerprints are documented in `DATASETS.md`.

The Taiwan benchmark uses the public `Bankrupt?` target and one fixed stratified 80/10/10 split with seed 42.

For the American benchmark, the public status field is constant within a firm and the released table does not provide a filing-year field. The code therefore uses the manuscript's `terminal_failed_proxy`: only the final observed fiscal-year row of each failed firm is labeled positive. The chronological split is 1999-2011 for training, 2012-2014 for validation, and 2015-2018 for testing. Company identifiers and fiscal years are metadata only; they are not among the 18 predictive variables.

Neither benchmark requires missing-value imputation. Winsorization limits (1st/99th percentiles), means, and standard deviations are estimated from the training partition only and then fixed for validation and test data.

## 3. Experimental protocol

HAF-Net and all multi-candidate comparison methods are selected using mean validation average precision (AP) over three dedicated tuning seeds:

```text
17, 29, 53
```

Final evaluation uses ten disjoint seeds:

```text
42, 123, 456, 789, 1024, 2024, 3456, 5678, 7890, 9999
```

Neural checkpoints are selected by validation AP. After the checkpoint is fixed, the operating threshold is selected on validation predictions by maximizing minority-class F1 and is then held fixed for test evaluation. Test predictions are not used during parameter selection.

## 4. HAF-Net

The model uses a fixed two-level accounting hierarchy. Level-2 subgroups are encoded independently, additive attention aggregates subgroups within each level-1 financial dimension, and a second attention layer aggregates the four level-1 dimensions. The fused representation is combined with a residual projection of the original input and passed to a binary classifier.

The predictive core uses class-balanced focal loss. Two dataset-specific auxiliary objectives are available:

- **TCAR** on Taiwan: positive-class group attention is regularized toward a mixed financial prior. The configured heuristic domain vector is `[0.35, 0.30, 0.20, 0.15]`, mixed equally with a training-derived statistical prior. The exact proportions are a modeling specification rather than weights established by a financial standard.
- **TCL** on American: a proxy-positive terminal observation is paired with the same firm's preceding observed fiscal-year record; negative-labeled records from the anchor year provide contrastive negatives. The annual offset is an observed-record relation and is not interpreted as a verified time to bankruptcy filing.

The default loss settings are `beta=0.999`, `gamma=2`, `rho=0.50`, contrastive temperature `0.10`, and eight TCL negatives. `lambda_tcar=0.05` on Taiwan and `lambda_tcl=0.10` on American.

## 5. Run HAF-Net

First select HAF-Net settings on validation data:

```bash
python scripts/tune_hafnet.py \
  --config configs/taiwan.yaml \
  --output outputs/taiwan/tuning

python scripts/tune_hafnet.py \
  --config configs/american.yaml \
  --output outputs/american/tuning
```

Then run final evaluation with the selected parameter files:

```bash
python scripts/run_experiment.py \
  --config configs/taiwan.yaml \
  --parameters outputs/taiwan/tuning/selected_parameters.json \
  --output outputs/taiwan/hafnet \
  --save-checkpoints

python scripts/run_experiment.py \
  --config configs/american.yaml \
  --parameters outputs/american/tuning/selected_parameters.json \
  --output outputs/american/hafnet \
  --save-checkpoints
```

The HAF-Net search grid is defined in `configs/hafnet_search.yaml`: subgroup width `{16, 32, 64}`, dropout `{0.05, 0.10, 0.20}`, and learning rate `{1e-4, 5e-4, 1e-3}`, with fusion width fixed at 64.

## 6. Comparison methods

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

The comparison set contains Logistic Regression, XGBoost, LightGBM, CatBoost, SMOTE+XGBoost, four explicitly named numeric adaptations (`tabnet_numeric_adapted`, `tabtransformer_numeric_adapted`, `ft_transformer_numeric_adapted`, and `trompt_numeric_adapted`), plus the dataset-specific `csstacking_adapted` and `two_stage_lightgbm_adapted` procedures. The Altman Z-Score is included on the American benchmark only. The adapted identifiers are deliberate: these package implementations transfer the defining mechanisms needed by the two all-numerical benchmarks but are not claimed to reproduce the original source-study software, feature spaces, or sampling designs exactly. Multi-candidate methods use the validation search spaces in `configs/baselines.yaml`.

The validation-search budgets are fixed before final evaluation: HAF-Net uses 27 candidates; XGBoost, LightGBM, CatBoost, SMOTE+XGBoost, and each numeric neural adaptation use 27 candidates; `csstacking_adapted` uses 9 candidates; and `two_stage_lightgbm_adapted` uses 16 candidates. Every multi-candidate configuration is evaluated over tuning seeds 17, 29, and 53 and selected by mean validation AP. The Two-Stage LightGBM adaptation transfers full-feature modeling, iterative importance-tail wrapper reduction, and the four configured resampling choices. The CSStacking adaptation performs XGBoost prefiltering and forward selection within training folds before fitting cost-sensitive stacked learners.

## 7. Ablation and sensitivity analyses

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

Taiwan sensitivity analysis is run on validation data only:

```bash
python scripts/run_sensitivity.py \
  --config configs/taiwan.yaml \
  --parameters outputs/taiwan/tuning/selected_parameters.json \
  --parameter all \
  --output outputs/taiwan/sensitivity
```

The sensitivity script evaluates `lambda_tcar`, `beta`, `gamma`, and `rho` and writes the corresponding summary and figure files.

## 8. Attention-case analysis

After the Taiwan final run has saved checkpoints:

```bash
python scripts/export_attention_cases.py \
  --config configs/taiwan.yaml \
  --parameters outputs/taiwan/tuning/selected_parameters.json \
  --checkpoint outputs/taiwan/hafnet/checkpoints/taiwan_full_seed42.pt \
  --seed 42 \
  --output outputs/taiwan/attention_cases
```

This is a targeted illustrative analysis rather than a representative sample. The script records the requested dominant group, selected row, seed, checkpoint fingerprint, and fallback status, and exports Case A/Case B/Case C together with their group-attention values and prominent standardized indicators. Population-level attention distributions and cross-seed dominant-group stability are written by `run_experiment.py` for the full test set.

## 9. Metrics and statistical inference

Metrics can be recalculated directly from saved prediction files:

```bash
python scripts/export_metric_tables.py \
  --predictions \
    outputs/taiwan/hafnet/hafnet_predictions.csv \
    outputs/taiwan/baselines/baseline_predictions.csv \
  --output outputs/taiwan/metric_tables
```

Primary paired comparisons against the Two-Stage LightGBM adaptation are run with:

```bash
python scripts/run_statistics.py \
  --taiwan-hafnet outputs/taiwan/hafnet/hafnet_predictions.csv \
  --taiwan-baselines outputs/taiwan/baselines/baseline_predictions.csv \
  --american-hafnet outputs/american/hafnet/hafnet_predictions.csv \
  --american-baselines outputs/american/baselines/baseline_predictions.csv \
  --bootstrap 10000 \
  --output outputs/statistics/primary_comparisons.csv
```

The primary metrics are F1 and Matthews correlation coefficient (MCC). Bootstrap intervals resample Taiwan test rows and American company identifiers, keeping each sampled company's test-period observations together. Matched-seed Wilcoxon tests are reported separately, with Holm adjustment across the four primary dataset-by-metric comparisons.

## 10. Main outputs

Typical output files include the run-level predictions required for paired inference:

```text
selected_parameters.json
tuning_runs.csv
tuning_summary.csv
hafnet_runs.csv
hafnet_summary.csv
hafnet_predictions.csv
hafnet_attention.csv
baseline_runs.csv
baseline_summary.csv
baseline_predictions.csv
ablation_inference.csv
sensitivity_runs.csv
sensitivity_summary.csv
primary_comparisons.csv
```

Generated prediction files and checkpoints are produced by the commands above and can be regenerated from the pinned data and configurations.

## 11. Reproducibility metadata

The `reproducibility/` directory provides compact CSV records that mirror the fixed evaluation protocol, search spaces, sensitivity grids, and paired-comparison tables reported in the manuscript:

```text
reproducibility/evaluation_protocol.csv
reproducibility/hyperparameter_search_space.csv
reproducibility/sensitivity_grid.csv
reproducibility/reported_primary_comparisons.csv
reproducibility/reported_primary_seed_differences.csv
reproducibility/reported_component_effects.csv
```

The reported-result CSVs are direct machine-readable mirrors of Tables 4, C1, and C2. Observation-level `hafnet_predictions.csv` and `baseline_predictions.csv` are generated by the evaluation scripts and are the inputs to the paired statistical pipeline.

## 12. Source layout

```text
HAF-Net/
├── README.md
├── DATASETS.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── requirements-lock.txt
├── pyproject.toml
├── configs/
├── data/
│   └── README.md
├── metadata/
│   └── feature_group_mapping.csv
├── reproducibility/
│   ├── evaluation_protocol.csv
│   ├── hyperparameter_search_space.csv
│   ├── sensitivity_grid.csv
│   ├── reported_primary_comparisons.csv
│   ├── reported_primary_seed_differences.csv
│   └── reported_component_effects.csv
├── scripts/
│   ├── download_data.py
│   ├── tune_hafnet.py
│   ├── run_experiment.py
│   ├── run_baselines.py
│   ├── run_ablation.py
│   ├── run_sensitivity.py
│   ├── export_attention_cases.py
│   ├── plot_attention_cases.py
│   ├── plot_sensitivity.py
│   ├── export_metric_tables.py
│   └── run_statistics.py
└── src/hafnet/
```

## 13. Citation and license

Software citation metadata are provided in `CITATION.cff`. The source code is released under the MIT License. The benchmark datasets remain subject to their original licenses; see `DATASETS.md`.
