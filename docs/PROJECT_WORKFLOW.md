# Project Workflow: ThyroidSense

**Project:** ThyroidSense: Thyroid Disease Detection Using Machine Learning  
**Project Identifier:** `ThyroidSense`  
**GitHub Repository:** [https://github.com/VenkateshDevarakonda0706/thyroid-sense](https://github.com/VenkateshDevarakonda0706/thyroid-sense)  
**Author:** Venkatesh Devarakonda  
**Curriculum:** B.Tech CSE (AI & ML) — Semester 2-1 Academic Project  
**Benchmark:** Authentic UCI `allhypo` Dataset (`allhypo.data` & `allhypo.test`)

---

## 1. High-Level Lifecycle Flowchart

```
1. Authentic Ingestion & Parsing
   ├── Load data/raw/uci_thyroid_original/allhypo.data (2,800 train records)
   ├── Load data/raw/uci_thyroid_original/allhypo.test (972 test records)
   └── Parse 29 raw attributes, clinical diagnosis, and patient identifiers

2. Cleaning & Feature Selection
   ├── Extract Garavan Institute patient IDs (following '|') for verification
   ├── Strictly EXCLUDE patient IDs from feature matrix (prevent label memorization)
   ├── Prune redundant boolean measurement flags (TSH measured, TBG measured, etc.)
   ├── Map binary clinical flags ('f'/'t', 'F'/'M') to numeric 0/1
   └── Map target classes to standard codes: [1, 7, 9, 11]

3. Strict Partition Isolation
   ├── Maintain Ross Quinlan's 1987 canonical partition (2,800 Train / 972 Test)
   └── Verify strictly zero patient ID overlap across partitions

4. Missing Value Imputation (Training-Fitted)
   ├── Calculate medians of continuous features from X_train ONLY:
   │     age: 54.0, TSH: 1.4, T3: 2.0, TT4: 104.0, T4U: 0.98, FTI: 107.0
   ├── Impute missing binary/categorical features with training mode (Mode(sex)=0.0, Mode(symptoms)=0.0)
   └── Apply training medians to impute X_test (zero test leakage)

5. Feature Standardization BEFORE Oversampling
   ├── Fit StandardScaler ONLY on original imputed X_train (compute mu and sigma)
   ├── Transform X_train -> X_train_scaled
   ├── Transform untouched X_test -> X_test_scaled
   └── Save models/scaler.pkl

6. Target Balancing on Training Data ONLY
   ├── Apply RandomOverSampler strictly to (X_train_scaled, y_train)
   ├── Resample minority classes to match majority (2,580 each -> 10,320 rows)
   └── X_test remains completely untouched (972 genuine out-of-sample observations)

7. Model Training & Latency Benchmarking
   ├── Train Logistic Regression (Linear baseline)
   ├── Train Decision Tree (Interpretable tree, max_depth=8)
   ├── Train Random Forest (100 trees, max_depth=12)
   ├── Train K-Nearest Neighbors (k=3, Euclidean metric)
   ├── Train Support Vector Machine (RBF kernel, probability=True)
   └── Train Gaussian Naive Bayes (Probabilistic benchmark)

8. Model Evaluation on Untouched Holdout Samples (972 patients)
   ├── Compute Accuracy, Macro Precision, Macro Recall, Macro F1
   ├── Document zero test support for Secondary Hypothyroid (support=0)
   ├── Generate normalized Confusion Matrix heatmaps with support counts
   ├── Create comparative performance bar chart
   └── Persist best model artifact to models/best_model.pkl (Decision Tree)

9. Quality Assurance & Verification
   ├── Execute automated Pytest test suite (16 tests, 100% pass rate)
   ├── Validate single-sample and batch inference on new patient vectors
   └── Confirm zero data leakage across all pipeline stages
```

---

## 2. Step-by-Step Execution Guide

### Step 1: Run Exploratory Data Analysis
```bash
python src/generate_eda_figures.py
```
- Reads `data/processed/allhypo_processed.csv`.
- Generates publication-quality figures in `results/figures/`:
  - `class_distribution.png`
  - `feature_distributions.png`
  - `correlation_heatmap.png`
  - `hormone_boxplots_by_class.png`
  - `raw_missing_values_audit.png`

### Step 2: Run Data Preprocessing
```bash
python -m src.data_preprocessing
```
- Ingests `allhypo.data` (2,800 rows) and `allhypo.test` (972 rows).
- Parses attributes, excludes patient IDs, fits imputation and scaling strictly on `X_train`.
- Balances `X_train_scaled` to 10,320 rows with `RandomOverSampler`.
- Persists `data/processed/allhypo_processed.csv` and `models/scaler.pkl`.

### Step 3: Run Model Training & Evaluation
```bash
python -m src.model_evaluation
```
- Trains all 6 algorithms on `X_train_resampled` (10,320 samples).
- Evaluates models on `X_test_scaled` (972 untouched samples).
- Saves:
  - `results/metrics/model_comparison.csv`
  - `results/metrics/classification_reports.json`
  - `results/figures/model_comparison_chart.png`
  - `results/figures/confusion_matrix_*.png`
  - `models/best_model.pkl` (Decision Tree)
  - `models/best_model_metadata.json`

### Step 4: Run Automated Verification Tests
```bash
pytest -v tests
```
- Executes all 18 unit tests in `tests/test_preprocessing.py` and `tests/test_model_training.py`.
- Verifies parsing logic, split row counts, patient ID exclusion, disjoint splits, scaling before oversampling, untouched test set, zero test support handling, and batch inference.

### Step 5: Explore Interactive Jupyter Notebooks
Navigate to `notebooks/` and explore the educational notebooks:
1. `01_data_exploration.ipynb`
2. `02_data_preprocessing.ipynb`
3. `03_model_training.ipynb`
4. `04_model_evaluation.ipynb`
