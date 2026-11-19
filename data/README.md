# Data Directory

This directory stores the raw and processed thyroid datasets used for data exploration, preprocessing, and machine learning model training.

## Directory Structure

```
data/
├── README.md
├── raw/
│   ├── thyroid_raw.csv           # Original 31-column UCI Thyroid Disease merged dataset
│   └── uci_thyroid_original/     # Raw benchmark UCI files (allbp, allhyper, allhypo, ann, etc.)
├── processed/
│   └── thyroid_processed.csv     # Cleaned, 14-feature, 4-class balanced dataset (20,000 records)
└── predict_samples.csv           # Sample test batch (10 records) for inference validation
```

## Dataset Summary

- **Raw Data:**
  - File: `data/raw/thyroid_raw.csv`
  - Shape: 9,172+ rows, 31 columns
  - Features: Patient age, sex, symptoms (goitre, tumor, etc.), binary measurement flags, hormone lab tests (`TSH`, `T3`, `TT4`, `T4U`, `FTI`), and target `class`.
- **Processed Data:**
  - File: `data/processed/thyroid_processed.csv`
  - Shape: 20,000 rows, 15 columns (14 features + 1 target)
  - Classes:
    - `1`: Compensated Hypothyroid (5,000 rows)
    - `7`: Negative / Normal (5,000 rows)
    - `9`: Primary Hypothyroid (5,000 rows)
    - `11`: Secondary Hypothyroid (5,000 rows)
  - Missing values: None (imputed using feature medians/means).
  - Class balancing: Balanced via Random Over-sampling.

For full clinical descriptions and feature details, see [DATASET_DOCUMENTATION.md](../docs/DATASET_DOCUMENTATION.md).
