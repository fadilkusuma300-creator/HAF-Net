# Data

The public benchmark datasets are downloaded on demand to:

```text
data/raw/taiwan/data.csv
data/raw/american/american_bankruptcy_dataset.csv
```

Their pinned SHA-256 fingerprints are recorded in `configs/taiwan.yaml` and `configs/american.yaml`. Source URLs, licenses, target definitions, and split rules are documented in `DATASETS.md`. Download the exact public snapshots with:

```bash
python scripts/download_data.py --dataset all
```
