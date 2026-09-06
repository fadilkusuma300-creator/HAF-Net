# Dataset information

## Taiwan Bankruptcy Prediction

- Provider: UCI Machine Learning Repository
- Dataset page: https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction
- DOI: https://doi.org/10.24432/C5004D
- License: CC BY 4.0
- Local path: `data/raw/taiwan/data.csv`
- Model inputs: 95 financial indicators
- Target: `Bankrupt?`

## American Bankruptcy Dataset

- Repository: https://github.com/sowide/bankruptcy_dataset
- Dataset article: https://doi.org/10.3390/fi14080244
- License: CC BY 4.0
- Local path: `data/raw/american/american_bankruptcy_dataset.csv`
- Model inputs: `X1`-`X18`
- Metadata: company identifier and fiscal year
- Target: the public row-level status label, where the fiscal year before a Chapter 7 or Chapter 11 filing is labeled as the positive class

Company identifiers and fiscal years are used only for chronological splitting and temporal-pair construction. They are not included among the 18 predictive variables.
