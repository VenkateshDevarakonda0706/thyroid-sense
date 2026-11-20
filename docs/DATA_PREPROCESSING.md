# Data Preprocessing Documentation: ThyroidSense

**Project:** ThyroidSense: Thyroid Disease Detection Using Machine Learning  
**Project Identifier:** `ThyroidSense`  
**GitHub Repository:** [https://github.com/VenkateshDevarakonda0706/thyroid-sense](https://github.com/VenkateshDevarakonda0706/thyroid-sense)  
**Author:** Venkatesh Devarakonda  
**Course Context:** B.Tech CSE (AI & ML) — Semester 2-1 Academic Project  
**Benchmark Dataset:** Authentic UCI `allhypo` Benchmark (`allhypo.data` [2,800] & `allhypo.test` [972])

---

## 1. Introduction

Data preprocessing is the foundational phase in the machine learning lifecycle. Medical datasets collected from clinical laboratory settings frequently contain missing tests, unstandardized measurement scales, redundant administrative columns, and extreme class imbalances.

Feeding raw, unprocessed data directly into machine learning estimators leads to:
- Numerical calculation errors or runtime exceptions (due to missing values or non-numeric tokens like `?`).
- Algorithmic bias toward majority classes (predicting healthy for all queries).
- Distorted distance calculations in geometric algorithms like K-Nearest Neighbors (KNN) and Support Vector Machines (SVM).
- **Data leakage**, causing artificially inflated evaluation metrics that collapse when evaluated on genuine, unseen clinical samples.

This document details the **authentic, leakage-free preprocessing pipeline** implemented in `src/data_preprocessing.py`, the mathematical principles of each transformation, and the rationale behind our data hygiene practices.

---

## 2. Authentic UCI `allhypo` Pipeline Architecture

```
Authentic UCI allhypo Benchmark
   ├── Train Partition: allhypo.data (2,800 records)
   └── Test Partition:  allhypo.test (972 records)
            │
            ├── 1. Raw Attribute Parsing & Identification
            │      ├── Parse 29 comma-delimited attributes
            │      ├── Extract Garavan Institute patient IDs (following '|')
            │      └── Exclude patient IDs from modeling features (prevent label memorization)
            │
            ├── 2. Categorical Cleaning & Target Encoding
            │      ├── Map missing indicator '?' -> np.nan
            │      ├── Map binary flags ('f'/'t') -> 0/1, 'sex' ('F'/'M') -> 0/1
            │      └── Map hypothyroid diagnostic classes:
            │            - Compensated Hypothyroid -> 1
            │            - Negative (Healthy)     -> 7
            │            - Primary Hypothyroid     -> 9
            │            - Secondary Hypothyroid   -> 11
            │
            ├── 3. Strict Partition Isolation
            │      ├── Train Set: 2,800 genuine patient records
            │      ├── Test Set:  972 genuine patient records
            │      └── Verify ZERO patient ID overlap between Train and Test sets
            │
            ├── 4. Training-Only Missing Value Imputation
            │      ├── Compute continuous medians STRICTLY on X_train:
            │      │     age: 54.0, TSH: 1.4, T3: 2.0, TT4: 104.0, T4U: 0.98, FTI: 107.0
            │      ├── Impute missing binary/categorical features with training mode (Mode(sex)=0.0, Mode(symptoms)=0.0)
            │      └── Apply X_train statistics to impute X_test (zero test leakage)
            │
            ├── 5. Feature Standardization BEFORE Oversampling
            │      ├── Fit StandardScaler STRICTLY on imputed original X_train
            │      ├── Transform X_train -> X_train_scaled
            │      ├── Transform X_test  -> X_test_scaled
            │      └── Save models/scaler.pkl
            │
            └── 6. Class Balancing on Training Data ONLY
                   ├── Apply RandomOverSampler STRICTLY to (X_train_scaled, y_train)
                   ├── Resample minority classes to match majority (2,580 each -> 10,320 rows)
                   └── Test set remains COMPLETELY UNTOUCHED (972 genuine holdout observations)
```

---

## 3. Step-by-Step Preprocessing Operations

### Step 1: Attribute Selection and Administrative ID Exclusion
The raw UCI `allhypo` dataset contains 29 comma-delimited attributes followed by a class label and a patient identifier separated by a pipe (`|`):
1. **Patient Identifier Exclusion:**
   - Ross Quinlan's 1987 thyroid database includes clinical record numbers from the Garavan Institute of Medical Research in Sydney, Australia (e.g., `840807029`).
   - Retaining patient identifiers would allow tree-based algorithms to memorize individual patients rather than learning biological relationships.
   - We extract `patient_id` solely for provenance auditing and split verification, strictly excluding it from the feature matrix $X$.
2. **Measurement Flag Pruning:**
   - Boolean indicators indicating whether a laboratory test was performed (e.g., `TSH measured`, `T3 measured`, `TT4 measured`, `T4U measured`, `FTI measured`, `TBG measured`) are pruned.
   - When the quantitative hormone level is recorded, the measurement flag is completely redundant.
   - Pruning these flags leaves **14 clinically meaningful predictive features**: `age`, `sex`, `on thyroxine`, `query on thyroxine`, `on antithyroid medication`, `sick`, `pregnant`, `thyroid surgery`, `I131 treatment`, `query hypothyroid`, `query hyperthyroid`, `lithium`, `goitre`, `tumor`, `hypopituitary`, `psych`, `TSH`, `T3`, `TT4`, `T4U`, and `FTI`.

### Step 2: Categorical Feature Encoding & Target Mapping
Machine learning estimators perform linear algebra and distance computations on numerical matrices:
1. **Binary Symptom and Demographic Features:**
   - `sex`: Encoded as `0` for Female and `1` for Male.
   - Clinical symptom flags (`on thyroxine`, `pregnant`, `goitre`, etc.): Encoded as `0` (No/False) and `1` (Yes/True).
2. **Target Variable (`class`):**
   - Mapped to standard diagnostic codes:
     - `1`: Compensated Hypothyroid (`compensated hypothyroid.`)
     - `7`: Negative (`negative.`)
     - `9`: Primary Hypothyroid (`primary hypothyroid.`)
     - `11`: Secondary Hypothyroid (`secondary hypothyroid.`)

### Step 3: Partitioning and Patient Disjointness
We respect the canonical split established by Ross Quinlan in the 1987 benchmark:
- **Training Set (`allhypo.data`):** 2,800 records (2,580 Negative, 154 Compensated, 64 Primary, 2 Secondary).
- **Test Set (`allhypo.test`):** 972 records (901 Negative, 40 Compensated, 31 Primary, 0 Secondary).
- **Disjoint Patient Verification:** All 3,772 patient IDs across the two partitions are verified to be strictly disjoint (zero patient overlap between train and test).

### Step 4: Missing Value Imputation (Fitted Strictly on Training Data)
In real-world clinical practice, physicians order blood tests selectively; not every panel is complete for every patient:
- Missing laboratory measurements appear as `?` in the raw files.
- Binary/categorical features (such as `sex` and symptom indicators) are imputed using the **statistical mode computed strictly from `X_train`** (e.g. Mode(`sex`) = 0.0 [Female], Mode(symptoms) = 0.0), rather than assuming unknown values represent absence.
- Continuous endocrine measurements (`age`, `TSH`, `T3`, `TT4`, `T4U`, `FTI`) are imputed using the **median of X_train only**:
  $$\text{Imputed Value} = \text{Median}(X_{\text{train, feature}})$$

**Exact Training Medians Learned:**
- `age`: **54.0**
- `TSH`: **1.40 mIU/L**
- `T3`: **2.00 nmol/L**
- `TT4`: **104.00 nmol/L**
- `T4U`: **0.98 ratio**
- `FTI`: **107.00 index**

**Data Leakage Prevention:** `X_test` missing values are filled with the training medians. The test set's distribution is never examined during imputation.

### Step 5: Feature Standardization BEFORE Oversampling
Standardization scales features so that they have zero mean ($\mu = 0$) and unit variance ($\sigma = 1$):
$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

**Why Standardize BEFORE Oversampling?**
- If oversampling is conducted first, duplicate rows are created for minority classes.
- Duplicating identical feature vectors alters the empirical probability density: it artificially concentrates values around the minority cluster centroids, shrinking the computed variance $\sigma$ and skewing $\mu$.
- By fitting `StandardScaler` on the original 2,800 non-replicated training records, $\mu$ and $\sigma$ reflect the true clinical sample distribution.
- `X_test` (972 samples) is transformed using these training parameters without any data leakage:
  ```python
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)  # Fitted on original 2,800 records
  X_test_scaled  = scaler.transform(X_test)       # Transformed with training mu & sigma
  ```

### Step 6: Training Class Balancing via RandomOverSampler
In the raw training split, healthy patients outnumber disease cases by over 16:1:
- Negative (`7`): 2,580 cases
- Compensated Hypothyroid (`1`): 154 cases
- Primary Hypothyroid (`9`): 64 cases
- Secondary Hypothyroid (`11`): 2 cases

**The Balancing Strategy:**
- `RandomOverSampler` is applied **strictly to the scaled training partition** (`X_train_scaled`, `y_train`).
- Each minority class is resampled to match the majority class (2,580 samples each), creating an augmented training matrix of **10,320 rows**.
- **The test partition remains 100% untouched:** exactly 972 genuine out-of-sample patient observations with zero synthetic duplicates.

---

## 4. Provenance Audit: The Six-Task Concatenation Pitfall

During our forensic audit of the historical repository, an earlier file named `thyroid_raw.csv` (22,632 rows) was investigated:
- **Origin of 22,632 Rows:** Stacking six separate UCI subtasks (`allrep`, `allbp`, `sick`, `allhyper`, `dis`, and `allhypo`), each with 3,772 rows ($6 \times 3,772 = 22,632$).
- **Patient Duplication:** The Garavan patient IDs match row-for-row across all 6 files. The dataset only ever represented 3,772 unique clinical patients.
- **Label Contradictions:** In subtasks like `allhyper` (hyperthyroid screening) and `sick` (general illness screening), hypothyroid patients were labeled `negative.` (because they were not hyperthyroid or sick). Merging all six files introduced **1,712 contradictory rows** into the dataset.
- **Resolution:** We abandoned the incorrectly pooled multi-task dataset and migrated directly to the authentic, authoritative UCI `allhypo` dataset (`allhypo.data` and `allhypo.test`), completely eliminating all label contradictions.

---

## 5. Summary of Viva Defense Points

1. **Why do you standardize features BEFORE applying RandomOverSampler?**
   *"Standardizing before oversampling ensures that the mean and standard deviation are calculated from the genuine, un-replicated patient distribution. If we oversampled first, replicating minority patient vectors would artificially inflate the density around those points, artificially reducing the variance $\sigma$ and distorting the z-score normalization."*

2. **Why does Secondary Hypothyroidism have zero support in the test partition?**
   *"In Ross Quinlan's canonical 1987 benchmark partition, both Secondary Hypothyroid patients ($n=2$) were placed into `allhypo.data` (train), leaving 0 cases in `allhypo.test`. As a result, its test support is 0, and its F1-score is mathematically 0.0. This is an authentic reflection of the historical benchmark dataset's scarcity."*

3. **How do you ensure zero data leakage across your preprocessing steps?**
   *"First, we maintain the strict partition boundary between `allhypo.data` (2,800 train) and `allhypo.test` (972 test). Second, missing value medians are computed exclusively on `X_train`. Third, the StandardScaler is fitted exclusively on `X_train`. Fourth, RandomOverSampler is applied exclusively to the training set. The test set is transformed using training parameters and evaluated without alteration."*

4. **Why was patient ID removed from features?**
   *"The Garavan Institute patient ID is an administrative identifier. If included as a predictive feature, decision trees or nearest neighbor classifiers could achieve high accuracy simply by memorizing patient IDs rather than identifying physiological patterns in TSH, TT4, or FTI levels."*

---

## 6. Clinical Disclaimer

> [!CAUTION]
> **Academic Demonstration Only:**
> ThyroidSense is an educational B.Tech CSE (AI & ML) project developed to illustrate supervised machine learning, data leakage mitigation, and imbalanced data handling on historical benchmarks. It is NOT certified for clinical diagnostics or patient care.
