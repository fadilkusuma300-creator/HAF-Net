# Dataset information

## Taiwan Bankruptcy Prediction

- Provider: UCI Machine Learning Repository
- Dataset page: https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction
- DOI: https://doi.org/10.24432/C5004D
- License: CC BY 4.0
- Local path: `data/raw/taiwan/data.csv`
- Pinned SHA-256: `67bf2e7c75490f7ad3f76bbce57d49cdc25967cdab607527b94f944863fa14d8`
- Model inputs: 95 financial indicators
- Target: `Bankrupt?`
- Split: stratified 80/10/10 with fixed seed 42
- Row metadata: the released table contains no firm or observation-year identifier

## American Bankruptcy Dataset

- Repository: https://github.com/sowide/bankruptcy_dataset
- Dataset article: https://doi.org/10.3390/fi14080244
- License: CC BY 4.0
- Local path: `data/raw/american/american_bankruptcy_dataset.csv`
- Pinned SHA-256: `d64ed85ae786d75113455ae238feb24015d19716c10ec1de3a90a52eddc7b04a`
- Model inputs: `X1`-`X18`
- Metadata: company identifier and fiscal year
- Target: `terminal_failed_proxy` labels the last observed fiscal-year row of each failed firm as positive and other rows as negative. It must be selected explicitly in the dataset configuration. The source describes next-year bankruptcy prediction, but the pinned table contains no filing-year field and its status is constant within each firm. Equality between the terminal row and the pre-filing year has not been established; the loader therefore does not generate an event year or claim a validated one-year horizon.
- Pinned-release counts: 78,682 rows, 8,971 distinct anonymized company identifiers, 609 failed-firm identifiers, and 5,220 rows with raw failed status. The proxy yields 609 positive rows. These row and identifier counts apply to the pinned CSV and should not be conflated with the article's company count.

Company identifiers and fiscal years are used only for chronological splitting, firm-clustered bootstrap resampling, and temporal-pair construction. They are not included among the 18 predictive variables.


## Derived-data calculations

The derived tables record all split members and raw-data fingerprints. The American train/validation/test partitions contain 403/87/119 proxy-positive rows and 392 eligible adjacent-year training pairs. There are 10/2/1 rows with nonpositive reported total liabilities, respectively; in this pinned table these liabilities are negative, not zero. The deterministic Altman calculation retains the supplied signs. No missing or nonzero denominator is silently repaired. These observations limit the interpretation of that score as a conventional financial-ratio baseline.

