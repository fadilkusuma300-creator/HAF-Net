# Data

The repository includes pinned snapshots of the two public benchmark datasets used by the code:

```text
data/raw/taiwan/data.csv
data/raw/american/american_bankruptcy_dataset.csv
```

Their SHA-256 fingerprints are recorded in `configs/taiwan.yaml` and `configs/american.yaml`. Source URLs, licenses, target definitions, and split rules are documented in the repository-level `DATASETS.md`.

The snapshots can be downloaded again from their public sources with:

```bash
python scripts/download_data.py --dataset all
```

Deterministic products such as partition membership, preprocessing statistics, and Altman calculations are generated on demand under `data/derived/` by:

```bash
python scripts/prepare_data_summary.py
```
