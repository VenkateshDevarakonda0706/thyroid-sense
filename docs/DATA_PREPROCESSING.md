# Data Preprocessing Documentation: Thyroid Disease Detection

**Project:** Thyroid Disease Detection Using Machine Learning  
**Author:** Venkatesh Devarakonda  
**Course Context:** B.Tech CSE (AI & ML) — Semester 2-1  

---

## 1. Introduction

Data preprocessing is the foundational phase in the machine learning lifecycle. Raw medical datasets collected from clinical settings frequently contain noise, missing laboratory test values, unstandardized scales, redundant administrative columns, and severe class imbalances.

Feeding raw, unprocessed data directly into machine learning algorithms leads to:
- Numerical calculation errors or runtime exceptions (due to missing values or non-numeric strings).
- Algorithmic bias toward majority classes.
- Distorted distance calculations in distance-based algorithms like K-Nearest Neighbors (KNN) and Support Vector Machines (SVM).
- Data leakage, causing overly optimistic evaluation metrics that fail in real-world testing.

This document details every preprocessing stage implemented in `src/data_preprocessing.py` and the `02_data_preprocessing.ipynb` notebook.

---

## 2. Preprocessing Architecture & Pipeline

```
Raw UCI Dataset (31 cols)
   │
   ├── 1. Drop Administrative & Redundant Flag Columns (patient_id, referral source, TSH measured, etc.)
   │
   ├── 2. Data Cleaning & Type Conversion ('?' -> NaN, 't'/'f' -> 1/0)
   │
   ├── 3. Missing Value Imputation (Continuous lab features imputed with median/mean)
   │
   ├── 4. Target Class Selection & Encoding (Filter to 4 primary thyroid states: [1, 7, 9, 11])
   │
   ├── 5. Class Balancing (Random Over-sampling to 5,000 instances per class = 20,000 total)
   │
   ├── 6. Stratified Train-Test Splitting (80% Train [16,000], 20% Test [4,000])
   │
   └── 7. Feature Scaling (StandardScaler fitted strictly on Train split -> Transformed to Test)
```

---

## 3. Step-by-Step Preprocessing Operations

### Step 1: Attribute Selection and Redundancy Elimination
The raw UCI thyroid dataset includes 31 columns. A significant number of columns are administrative or duplicate information:
1. **Administrative Columns:**
   - `patient_id`: A unique database key. Retaining it would allow tree algorithms to memorize patient identities rather than physiological patterns.
   - `referral source`: Denotes the clinic or department that referred the patient (e.g., SVHC, SVI, STMW). Irrelevant to underlying endocrine disease.
   - `dataset_name`: Name of the sub-database partition (`allrep`, `allhypo`, etc.).
2. **Measurement Indicator Flags:**
   - Columns such as `TSH measured`, `T3 measured`, `TT4 measured`, `T4U measured`, and `FTI measured` are binary booleans indicating whether a lab test was performed.
   - When the lab test value is present, these flags are redundant.
   - Dropping these flags reduces dimensionality from 31 down to 14 predictive features.

### Step 2: Missing Value Treatment
In clinical practice, physicians order specific blood tests depending on a patient's primary symptoms; thus, not every blood test is recorded for every patient.
- Missing values in the raw dataset appear as `?` or `NaN`.
- In our preprocessing pipeline:
  - Binary symptoms (`goitre`, `tumor`, `lithium`, etc.) default to `0` (absence of symptom) if missing.
  - Continuous endocrine measurements (`TSH`, `T3`, `TT4`, `T4U`, `FTI`) are imputed using the **median** (or mean) of the training dataset:
    $$\text{Imputed Value} = \text{Median}(X_{\text{train, feature}})$$
  - *Median imputation is chosen for hormonal values because hormone distributions (especially TSH) are heavily skewed by extreme pathology, making the median a more robust measure of central tendency than the mean.*

### Step 3: Categorical Feature Encoding
Machine learning algorithms perform mathematical operations on vectors; therefore, text and categorical values must be numerically encoded:
1. **Binary Symptom and Demographic Features:**
   - `sex`: Encoded as `0` for Female and `1` for Male.
   - Symptoms (`on thyroxine`, `pregnant`, `goitre`, etc.): Encoded as `0` (No/False) and `1` (Yes/True).
2. **Target Variable (`class`):**
   - Transformed using Scikit-Learn's `LabelEncoder`:
     - `1`: Compensated Hypothyroid
     - `7`: Negative (Normal)
     - `9`: Primary Hypothyroid
     - `11`: Secondary Hypothyroid

### Step 4: Class Imbalance Handling
In the original raw dataset, normal/negative records outnumbered disease cases by over 95:1:
- Normal (`negative`): >21,000 cases.
- Primary hypothyroid: ~95 cases.
- Secondary hypothyroid: ~2 cases in some subsets.

**Why Class Imbalance is Dangerous:**
A naive classifier predicting "Normal" for 100% of patients would achieve >95% accuracy while completely failing to diagnose patients suffering from thyroid disease.

**Remediation:**
In the original 2-1 semester experiments (`Thyroid_(1).ipynb`), the 4 primary clinical classes were balanced using `RandomOverSampler` from `imblearn`, bringing each class to 5,000 samples (total 20,000 samples). This ensures that loss functions weight all diagnostic categories equally.

### Step 5: Stratified Train-Test Splitting
We split the 20,000 records into:
- **80% Training Set:** 16,000 samples (4,000 per class).
- **20% Testing Set:** 4,000 samples (1,000 per class).

By setting `stratify=y` in `train_test_split`, we guarantee that both training and testing partitions have identical class proportions (25% each), preventing class distribution skew.

### Step 6: Feature Standardization (StandardScaler)
Different medical measurements operate on vastly different numerical scales:
- `age`: 1 to 100
- `TSH`: 0.005 to 400+
- `TT4`: 2 to 400+
- `T4U`: 0.2 to 2.5
- Binary flags: 0 to 1

**The Standardization Formula (Z-Score):**
$$z = \frac{x - \mu}{\sigma}$$
Where:
- $x$ is the raw feature value.
- $\mu$ is the mean of the training feature.
- $\sigma$ is the standard deviation of the training feature.

**Prevention of Data Leakage:**
- The scaler calculates $\mu$ and $\sigma$ **ONLY** from `X_train`.
- `X_test` is transformed using the parameters learned from `X_train`:
  ```python
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)  # Compute mu & sigma from train
  X_test_scaled = scaler.transform(X_test)        # Use train mu & sigma on test
  ```
- This guarantees zero information from the test evaluation leaks into the training pipeline.
