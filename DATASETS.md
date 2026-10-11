# Dataset information

## Taiwan Bankruptcy Prediction

- Provider: UCI Machine Learning Repository
- Dataset page: https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction
- DOI: https://doi.org/10.24432/C5004D
- License: CC BY 4.0
- Local path: `data/raw/taiwan/data.csv`
- SHA-256: `67bf2e7c75490f7ad3f76bbce57d49cdc25967cdab607527b94f944863fa14d8`
- Observations: 6,819
- Predictive variables: 95 financial indicators
- Positive observations: 220
- Target: `Bankrupt?`
- Split: fixed stratified 80/10/10 with seed 42, yielding 5,455 / 682 / 682 train / validation / test observations

The released table does not contain row-level firm identifiers or observation years. The benchmark is therefore used for pooled-sample discrimination, and TCL is disabled.

## American Bankruptcy Dataset

- Source repository: https://github.com/sowide/bankruptcy_dataset
- Dataset article: https://doi.org/10.3390/fi14080244
- License: CC BY 4.0
- Local path: `data/raw/american/american_bankruptcy_dataset.csv`
- SHA-256: `d64ed85ae786d75113455ae238feb24015d19716c10ec1de3a90a52eddc7b04a`
- Observations: 78,682 firm-year rows
- Predictive variables: `X1`-`X18`
- Period: 1999-2018
- Target mode: `terminal_failed_proxy`
- Positive observations under the proxy: 609
- Chronological split: 1999-2011 / 2012-2014 / 2015-2018 for train / validation / test
- Proxy-positive observations by split: 403 / 87 / 119

The public status field is constant within each company and the released table contains no bankruptcy filing-year field. The configured target therefore labels only the final observed fiscal-year row of each failed firm as positive; preceding rows of failed firms and all rows of firms labeled alive are negative. This target is a terminal-observation proxy and is not interpreted as a verified one-year pre-filing label.

Company identifiers and fiscal years are used for chronological partitioning, TCL pair construction, grouped inner folds where applicable, and firm-clustered bootstrap resampling. They are not included among the 18 predictive variables.

## Preprocessing

Both benchmark tables are expected to contain finite numeric predictors. The code does not perform implicit missing-value imputation. For each run, the 1st/99th percentile Winsorization limits and standardization statistics are estimated from the training partition only and then fixed for validation and test data.

## Download

Run from the repository root:

```bash
python scripts/download_data.py --dataset all
```

The downloader verifies the configured SHA-256 fingerprints before placing files in the expected paths.
