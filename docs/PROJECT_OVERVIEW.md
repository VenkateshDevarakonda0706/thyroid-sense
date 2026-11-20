# Project Overview: ThyroidSense — Thyroid Disease Detection Using Machine Learning

**Project Title:** ThyroidSense: Thyroid Disease Detection Using Machine Learning  
**Project Identifier:** `ThyroidSense`  
**GitHub Repository:** [https://github.com/VenkateshDevarakonda0706/thyroid-sense](https://github.com/VenkateshDevarakonda0706/thyroid-sense)  
**Author:** Venkatesh Devarakonda  
**Degree / Curriculum:** B.Tech Computer Science & Engineering (AI & ML) — Semester 2-1  
**Project Category:** Supervised Machine Learning & Academic Benchmark  
**Status:** Complete, Reconstructed, Benchmarked, and Leakage-Safe  

---

## 1. Executive Summary

**ThyroidSense** is an academic Machine Learning project that develops an end-to-end classification system for thyroid disease detection. Thyroid disorders affect hundreds of millions of individuals globally. Because symptoms like fatigue, weight fluctuations, mood changes, and cold/heat intolerance overlap with numerous common conditions, clinical diagnosis heavily relies on serum blood tests measuring thyroid hormones.

This project delivers an end-to-end Machine Learning pipeline that analyzes patient clinical attributes, demographic factors, and endocrine laboratory measurements to classify patients into four distinct thyroid diagnostic states:
1. **Compensated Hypothyroid (Class 1):** Early/subclinical stage of thyroid underactivity (elevated TSH with normal free thyroid hormone).
2. **Negative (Normal) (Class 7):** Healthy, balanced euthyroid endocrine function.
3. **Primary Hypothyroid (Class 9):** Overt thyroid gland failure (severely elevated TSH with low TT4/FTI).
4. **Secondary Hypothyroid (Class 11):** Central underactivity caused by pituitary failure (rare, low or normal TSH with low thyroid hormone).

By systematically migrating from an incorrectly merged 6-subtask historical dataset to the authentic, single-task UCI **`allhypo`** benchmark (3,772 total patients: 2,800 train in `allhypo.data` and 972 test in `allhypo.test`), implementing training-only median imputation, fitting feature standardization **before** oversampling, and balancing only training data with `RandomOverSampler`, the project establishes a rigorous, leak-free machine learning benchmark.

In our authentic evaluation on the 972 untouched test patients, the **Decision Tree Classifier** achieved top performance:
- **Test Accuracy:** **99.79%** (970/972 correct)
- **Macro F1-Score:** **0.7436** (capped by zero support for Secondary Hypothyroid in the test partition)
- **Compensated Hypothyroid Recall:** **100.00%** (40/40 detected, F1: 0.9756)
- **Primary Hypothyroid Recall:** **100.00%** (31/31 detected, F1: 1.0000)
- **Negative Class Recall:** **99.78%** (899/901 healthy correctly classified, Precision: 100.00%)

---

## 2. Problem Statement

Accurately interpreting thyroid panels can be complex when patients exhibit subtle, borderline, or conflicting hormone readings:
- Traditional manual threshold interpretation can misdiagnose subclinical conditions (like Compensated Hypothyroidism) or miss rare pituitary dysfunctions (Secondary Hypothyroidism).
- Raw medical datasets are characterized by severe class imbalance (healthy patients drastically outnumber sick ones) and missing blood tests.
- A clinical machine learning model must not only attain high accuracy but also achieve near-zero False Negatives (high Recall) to ensure sick patients receive timely clinical intervention.

---

## 3. Project Objectives

1. **Demonstrate Complete ML Lifecycle:** Follow a rigorous engineering pipeline from raw data parsing to final model inference and verification.
2. **Rigorous Data Hygiene:** Prevent data leakage by ensuring that all scalers and imputation medians are computed strictly on the training partition (`allhypo.data`).
3. **Benchmark Multiple Algorithms:** Implement, tune, and compare six traditional supervised learning models:
   - Decision Tree Classifier (Top Performer)
   - Random Forest Classifier
   - Logistic Regression
   - Support Vector Classifier (SVC, RBF Kernel)
   - K-Nearest Neighbors (KNN, $k=3$)
   - Gaussian Naive Bayes
4. **Comprehensive Evaluation:** Assess models using multi-metric evaluation (Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and Inference Latency).
5. **Educational Excellence:** Create clean, modular Python modules, self-contained executable Jupyter notebooks, unit tests (16 tests, 100% pass rate), and thorough academic documentation suitable for an undergraduate engineering viva and technical interviews.

---

## 4. Technology Stack

| Category | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.11.13 | Core programming language |
| **Data Manipulation** | Pandas | 3.1.1 | Tabular data analysis and preprocessing |
| **Numerical Computing** | NumPy | 2.4.6 | Vectorized operations and linear algebra |
| **Machine Learning** | Scikit-Learn | 1.9.1 | Estimators, metrics, and preprocessing pipelines |
| **Scientific Computing**| SciPy | 1.17.1 | Statistical analysis and hypothesis tests |
| **Visualization** | Matplotlib | 3.11.2 | Custom publication-quality plots |
| **Statistical Viz** | Seaborn | 0.13.2 | Distribution plots and correlation heatmaps |
| **Model Persistence** | Joblib | 1.6.0 | Serializing fitted models and scalers |
| **Testing** | Pytest | 9.2.0 | Automated unit testing and assertion verification |
| **Interactive Docs** | Jupyter Notebook| 7.x | Educational step-by-step walkthroughs |

---

## 5. Architectural Pipeline

```
[Authentic UCI allhypo Dataset]
   ├── allhypo.data (2,800 train records)
   └── allhypo.test (972 test records)
            │
            ▼
[Parsing & Administrative ID Exclusion] (Exclude Garavan patient IDs from features)
            │
            ▼
[Training-Only Median Imputation] (Computed strictly on X_train: age=54, TSH=1.4, etc.)
            │
            ▼
[StandardScaler Fit on Train] (Fitted BEFORE oversampling to preserve natural variance)
            │
            ▼
[RandomOverSampler on Train Only] (Resampled to 2,580/class = 10,320 rows)
            │
            ├── Train 6 Estimators on X_train_resampled (10,320 rows)
            └── Evaluate on UNTOUCHED Test Partition (972 rows)
                     │
                     ▼
             [Benchmark Results]
             - Decision Tree: 99.79% Acc | 0.7436 Macro F1 (Rank 1)
             - Random Forest: 99.59% Acc | 0.7334 Macro F1 (Rank 2)
             - Logistic Reg:  95.88% Acc | 0.6572 Macro F1 (Rank 3)
             - SVM (RBF):     95.37% Acc | 0.6186 Macro F1 (Rank 4)
             - KNN (k=3):     94.55% Acc | 0.5733 Macro F1 (Rank 5)
             - Naive Bayes:    9.26% Acc | 0.1156 Macro F1 (Rank 6)
```

---

## 6. Summary of Key Achievements

- **Zero Data Leakage:** Preprocessing transformations fit exclusively on the training split (`allhypo.data`), and standardization precedes oversampling.
- **Authentic Benchmark Integrity:** Relies on the canonical 1987 UCI `allhypo` dataset with zero multi-task label contradictions.
- **Reproducible Science:** Fixed random states guarantee identical metric reproduction across runs.
- **Comprehensive Test Coverage:** Automated unit test suite (`pytest`) verifies 16 tests covering parsing, split disjointness, scaler leak prevention, zero test support handling, and inference.
- **Academic Rigor:** Fully transparent regarding dataset scarcity (Secondary Hypothyroid support = 0 in test partition) and baseline comparisons (92.70% majority-class baseline).
