# Dataset placement

The two benchmark datasets are obtained from their public sources and are not bundled with this repository.

Expected paths:

```text
data/raw/taiwan/data.csv
data/raw/american/american_bankruptcy_dataset.csv
```

The download helper can place both files in these directories:

```bash
python scripts/download_data.py --dataset all
```

Taiwan source: UCI Machine Learning Repository, dataset 572.
American source: https://github.com/sowide/bankruptcy_dataset
