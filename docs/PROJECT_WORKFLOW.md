# Project Workflow: Thyroid Disease Detection

**Project:** Thyroid Disease Detection Using Machine Learning  
**Author:** Venkatesh Devarakonda  
**Curriculum:** B.Tech CSE (AI & ML) — Semester 2-1 Academic Project  

---

## 1. High-Level Lifecycle Flowchart

```
1. Ingestion
   ├── Load data/raw/thyroid_raw.csv (31 columns)
   └── Inspect shapes, missing values ('?'), and class labels

2. Cleaning & Feature Engineering
   ├── Drop non-predictive columns: patient_id, referral source, dataset_name
   ├── Drop duplicate boolean flags: TSH measured, TBG measured, etc.
   ├── Impute missing lab values with feature median
   └── Map binary strings ('f'/'t', 'F'/'M') to 0/1

3. Target Selection & Balancing
   ├── Filter down to 4 major diagnostic states: [1, 7, 9, 11]
   └── Balance instances to 5,000 per class using Random Over-sampling (20,000 rows)

4. Stratified Data Splitting
   ├── Split 80% Training (16,000 samples)
   └── Split 20% Testing (4,000 samples) using stratify=y

5. Feature Standardization (Leakage Free)
   ├── Fit StandardScaler ONLY on X_train (compute mu and sigma)
   ├── Transform X_train -> X_train_scaled
   ├── Transform X_test -> X_test_scaled
   └── Save models/scaler.pkl

6. Model Training & Latency Benchmarking
   ├── Train Logistic Regression (Linear baseline)
   ├── Train Decision Tree (Interpretable tree, max_depth=8)
   ├── Train Random Forest (100 trees, max_depth=12)
   ├── Train K-Nearest Neighbors (k=3, Euclidean metric)
   ├── Train Support Vector Machine (RBF kernel, probability=True)
   └── Train Gaussian Naive Bayes (Probabilistic benchmark)

7. Model Evaluation on 4,000 Holdout Samples
   ├── Compute Accuracy, Macro Precision, Macro Recall, Macro F1
   ├── Generate normalized Confusion Matrix heatmaps
   ├── Create comparative performance bar chart
   └── Persist best model artifact to models/best_model.pkl

8. Quality Assurance & Verification
   ├── Execute automated Pytest test suite
   ├── Validate single-sample inference on new patient vectors
   └── Confirm zero data leakage
```

---

## 2. Step-by-Step Execution Guide

### Step 1: Run Exploratory Data Analysis
```bash
python src/generate_eda_figures.py
```
- Reads `data/processed/thyroid_processed.csv`.
- Generates 5 high-resolution figures in `results/figures/`:
  - `class_distribution.png`
  - `feature_distributions.png`
  - `correlation_heatmap.png`
  - `hormone_boxplots_by_class.png`
  - `raw_missing_values_audit.png`

### Step 2: Run Data Preprocessing
```bash
python -m src.data_preprocessing
```
- Loads the dataset and performs stratified 80/20 train/test splitting.
- Standardizes continuous laboratory features and outputs partition verification tables.
- Persists `models/scaler.pkl`.

### Step 3: Run Model Training & Evaluation
```bash
python -m src.model_evaluation
```
- Trains all 6 algorithms on `X_train_scaled` (16,000 samples).
- Evaluates models on `X_test_scaled` (4,000 samples).
- Outputs the complete benchmark table.
- Saves:
  - `results/metrics/model_comparison.csv`
  - `results/metrics/classification_reports.json`
  - `results/figures/model_comparison_chart.png`
  - `results/figures/confusion_matrix_*.png`
  - `models/best_model.pkl`

### Step 4: Run Automated Verification Tests
```bash
pytest -v tests
```
- Executes all 9 unit tests in `tests/test_preprocessing.py` and `tests/test_model_training.py`.
- Verifies split ratios, data hygiene, leak prevention, and model inference.

### Step 5: Explore Interactive Jupyter Notebooks
Navigate to `notebooks/` and run the numbered notebooks in sequence:
1. `01_data_exploration.ipynb`
2. `02_data_preprocessing.ipynb`
3. `03_model_training.ipynb`
4. `04_model_evaluation.ipynb`
