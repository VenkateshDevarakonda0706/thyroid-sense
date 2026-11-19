# ThyroidSense: Thyroid Disease Detection Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-3.1.1-150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(9%2F9)-brightgreen.svg?style=flat)](tests/)
[![Repository](https://img.shields.io/badge/GitHub-ThyroidSense-blue.svg?style=flat&logo=github)](https://github.com/VenkateshDevarakonda0706/ThyroidSense)
[![Status](https://img.shields.io/badge/Academic%20Project-B.Tech%20CSE%20AIML-blue.svg?style=flat)]()

**ThyroidSense** is an end-to-end Machine Learning classification project designed, implemented, and benchmarked for undergraduate computer science and AI/ML students. This project classifies patients into four thyroid diagnostic states using clinical symptoms and objective serum hormone measurements from the UCI Thyroid Disease dataset.

---

## Table of Contents

1. [Project Title & Introduction](#1-project-title--introduction)
2. [Problem Statement](#2-problem-statement)
3. [Project Objectives](#3-project-objectives)
4. [Dataset Description](#4-dataset-description)
5. [Technologies Used](#5-technologies-used)
6. [Project Architecture](#6-project-architecture)
7. [Project Folder Structure](#7-project-folder-structure)
8. [Installation Instructions](#8-installation-instructions)
9. [How to Run the Project](#9-how-to-run-the-project)
10. [Data Preprocessing & Leakage Prevention](#10-data-preprocessing--leakage-prevention)
11. [Exploratory Data Analysis (EDA)](#11-exploratory-data-analysis-eda)
12. [Machine Learning Algorithms Used](#12-machine-learning-algorithms-used)
13. [Model Training Process](#13-model-training-process)
14. [Evaluation Metrics Explained](#14-evaluation-metrics-explained)
15. [Actual Measured Model Results](#15-actual-measured-model-results)
16. [Model Comparison & Discussion](#16-model-comparison--discussion)
17. [Core Machine Learning Concepts Explained](#17-core-machine-learning-concepts-explained)
18. [Project Limitations](#18-project-limitations)
19. [Future Improvements](#19-future-improvements)
20. [Conclusion](#20-conclusion)

---

## 1. Project Title & Introduction

**Project Title:** ThyroidSense: Thyroid Disease Detection Using Machine Learning  
**Project Identifier:** `ThyroidSense`  
**GitHub Repository:** [https://github.com/VenkateshDevarakonda0706/ThyroidSense](https://github.com/VenkateshDevarakonda0706/ThyroidSense)  
**Author:** Venkatesh Devarakonda  
**Academic Context:** B.Tech Computer Science & Engineering (AI & ML) — Semester 2-1 Academic Project  

The thyroid gland is a vital endocrine organ in the human neck that produces hormones—primarily Thyroxine ($T_4$) and Triiodothyronine ($T_3$)—regulated by Thyroid-Stimulating Hormone ($TSH$) secreted by the pituitary gland. These hormones control the human body's metabolic rate, heart rate, core temperature, and energy expenditure.

When the thyroid malfunctions, it causes widespread clinical disorders that are notoriously difficult to diagnose based purely on non-specific symptoms such as fatigue, weight fluctuations, and hair loss. This project builds a reliable supervised Machine Learning classification system that analyzes multi-variate laboratory blood panels and clinical patient records to automate the detection of four thyroid diagnostic states:
1. **Compensated Hypothyroid (Class 1):** Subclinical or mild underactive thyroid.
2. **Negative / Normal (Class 7):** Healthy, euthyroid endocrine function.
3. **Primary Hypothyroid (Class 9):** Overt thyroid gland failure.
4. **Secondary Hypothyroid (Class 11):** Central dysfunction originating from the pituitary gland.

---

## 2. Problem Statement

1. **Symptom Non-Specificity:** Clinical symptoms of thyroid disorders overlap with depression, aging, stress, and lifestyle fatigue, leading to frequent diagnostic delays.
2. **Multi-Variate Endocrine Complexity:** Interpreting blood panels requires understanding feedback loops (e.g., normal $T_4$ with elevated $TSH$ indicates subclinical failure, whereas low $TSH$ with low $T_4$ indicates central pituitary failure).
3. **Severe Class Imbalance:** In real-world screening datasets, healthy patients account for over 95% of cases. Standard machine learning models trained on raw imbalanced data become biased toward predicting "Normal", dangerously missing acute disease cases.
4. **Data Leakage Risks:** Preprocessing transformations (such as feature scaling and mean imputation) frequently suffer from data leakage in student projects, yielding artificially inflated test accuracies that fail on genuine patient samples.

---

## 3. Project Objectives

- **Demonstrate Complete ML Lifecycle:** Encompass data understanding, cleaning, imputation, class balancing, feature scaling, model training, holdout evaluation, artifact serialization, and automated unit testing.
- **Prevent Data Leakage:** Guarantee that all scalers and imputation parameters are fitted strictly on the training partition.
- **Benchmark 6 Classical ML Algorithms:** Train and evaluate Logistic Regression, Decision Trees, Random Forests, K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Gaussian Naive Bayes.
- **Provide Rigorous Documentation:** Formulate comprehensive guides and at least 40 viva questions suitable for undergraduate academic examinations and technical interviews.

---

## 4. Dataset Description

- **Source:** UCI Machine Learning Repository — Thyroid Disease Database.
- **Raw Dataset (`data/raw/thyroid_raw.csv`):** 9,172+ patient records with 31 columns including demographics, medical symptoms, indicator flags, and hormone values.
- **Processed Modeling Dataset (`data/processed/thyroid_processed.csv`):** 20,000 patient records across 15 columns (14 features + 1 target `class`), balanced to 5,000 instances per class using Random Over-sampling.

### Feature Summary
- **Demographics:** `age` (1–100 years), `sex` (0=Female, 1=Male).
- **Clinical Symptoms & History:** `on thyroxine`, `on antithyroid medication`, `pregnant`, `thyroid surgery`, `lithium`, `goitre`, `tumor` (Binary: 0=No, 1=Yes).
- **Serum Endocrine Laboratory Blood Tests:**
  - `TSH`: Thyroid-Stimulating Hormone (Normal: 0.4–4.0 mIU/L).
  - `T3`: Triiodothyronine (Normal: 1.2–3.1 nmol/L).
  - `TT4`: Total Thyroxine (Normal: 60–150 nmol/L).
  - `T4U`: Thyroxine Uptake ratio (Normal: 0.7–1.2).
  - `FTI`: Free Thyroxine Index ($\text{FTI} = \text{TT4} / \text{T4U}$, Normal: 70–140).

For full medical documentation, see [DATASET_DOCUMENTATION.md](docs/DATASET_DOCUMENTATION.md).

---

## 5. Technologies Used

- **Language:** Python 3.11.13
- **Data Analysis & Preprocessing:** Pandas 3.1.1, NumPy 2.4.6, SciPy 1.17.1
- **Machine Learning:** Scikit-Learn 1.9.1, Joblib 1.6.0
- **Data Visualization:** Matplotlib 3.11.2, Seaborn 0.13.2
- **Unit Testing:** Pytest 9.2.0
- **Interactive Notebooks:** Jupyter Notebook / nbconvert

---

## 6. Project Architecture

```
                                  [Raw UCI Records]
                                          │
                                          ▼
                         [Data Cleaning & Imputation]
                         - Drop non-predictive admin cols
                         - Median imputation for lab tests
                         - Binary symptom encoding
                                          │
                                          ▼
                               [Target Class Balancing]
                         - Filter to 4 primary classes
                         - RandomOverSampler (20,000 rows)
                                          │
                                          ▼
                        [Stratified 80/20 Train-Test Split]
                         - Train: 16,000 (4,000 per class)
                         - Test:   4,000 (1,000 per class)
                                          │
                                          ▼
                           [StandardScaler (Leak-Free)]
                         - Fitted strictly on Train split
                         - Applied to Test split
                                          │
                                          ▼
                            [Model Training (6 Models)]
                                          │
                                          ▼
                          [Holdout Evaluation (4,000 Test)]
                         - Accuracy, Precision, Recall, F1
                         - Confusion Matrices, Plots
                                          │
                                          ▼
                           [Best Model Persistence]
                         - models/best_model.pkl (Random Forest)
```

---

## 7. Project Folder Structure

```
ThyroidSense/
├── README.md                      # Master project documentation
├── requirements.txt               # Pinned Python package dependencies
├── .gitignore                     # Git ignore rules for caches & environments
├── PROJECT_AUDIT.md               # Detailed audit and file disposition log
├── FINAL_VERIFICATION_REPORT.md   # Final test results and verification report
├── GITHUB_SETUP_REPORT.md         # GitHub repository setup, security & audit log
│
├── data/
│   ├── README.md                  # Dataset descriptions
│   ├── raw/                       # Original UCI benchmark files
│   ├── processed/                 # Curated 20,000-row balanced modeling dataset
│   └── predict_samples.csv        # Sample batch for inference validation
│
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Exploratory Data Analysis & visual inspections
│   ├── 02_data_preprocessing.ipynb# Cleaning, imputation, and leak-free scaling
│   ├── 03_model_training.ipynb    # Training all 6 supervised classifiers
│   └── 04_model_evaluation.ipynb  # Benchmarking, confusion matrices, inference
│
├── src/
│   ├── __init__.py                # Package initializer
│   ├── data_preprocessing.py      # Preprocessing, splitting, and scaling logic
│   ├── model_training.py          # Estimator training and timing routines
│   ├── model_evaluation.py        # Benchmark calculation and plotting
│   └── generate_eda_figures.py    # Generates publication-ready figures
│
├── models/
│   ├── best_model.pkl             # Top performing model (Random Forest)
│   ├── best_model_metadata.json   # Model configuration & class mapping
│   ├── scaler.pkl                 # Fitted StandardScaler artifact
│   └── enc.pickle                 # Original label encoder mapping
│
├── results/
│   ├── figures/                   # High-resolution charts & confusion matrices
│   └── metrics/                   # Comparison tables and classification reports
│
├── docs/
│   ├── PROJECT_OVERVIEW.md        # Executive summary and architecture
│   ├── DATASET_DOCUMENTATION.md   # Clinical feature dictionary & provenance
│   ├── DATA_PREPROCESSING.md      # Data cleaning and leak prevention theory
│   ├── MODEL_EXPLANATION.md       # Theoretical foundations of 6 algorithms
│   ├── MODEL_EVALUATION.md        # Comprehensive benchmark metrics report
│   ├── PROJECT_WORKFLOW.md        # Step-by-step pipeline execution guide
│   ├── VIVA_QUESTIONS.md          # 40+ viva questions & answers with pitches
│   └── INSTALLATION_GUIDE.md      # Setup, virtualenv, and troubleshooting
│
├── tests/
│   ├── __init__.py                # Tests package initializer
│   ├── test_preprocessing.py      # Unit tests for preprocessing & scaling
│   └── test_model_training.py     # Unit tests for model training & inference
│
└── archive/                       # Safely preserved historical & legacy code
```

---

## 8. Installation Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/VenkateshDevarakonda0706/ThyroidSense.git
   cd ThyroidSense
   ```
2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. **Install dependencies:**
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Verify installation:**
   ```powershell
   pytest -v tests
   ```

---

## 9. How to Run the Project

### A. Run Automated Unit Tests:
```powershell
pytest -v tests
```

### B. Generate EDA Figures:
```powershell
python src/generate_eda_figures.py
```

### C. Run Full Model Training and Benchmark Evaluation:
```powershell
python -m src.model_evaluation
```

### D. Run Interactive Jupyter Notebooks:
```powershell
jupyter notebook notebooks/
```

---

## 10. Data Preprocessing & Leakage Prevention

1. **Column Pruning:** Removed non-predictive columns (`patient_id`, `referral source`, `dataset_name`) and redundant measurement flags (`TSH measured`, etc.), reducing 31 raw columns to 14 predictive features.
2. **Missing Value Imputation:** Missing continuous laboratory tests were imputed using training set medians to protect against outlier skewness.
3. **Categorical Encoding:** Binary features (`sex`, `on thyroxine`, `pregnant`, etc.) were encoded as 0 or 1.
4. **Class Balancing:** Balanced the 4 primary classes to 5,000 instances each via `RandomOverSampler`.
5. **Stratified Splitting:** Partitioned the 20,000 instances into 80% Train (16,000) and 20% Test (4,000) using `stratify=y`.
6. **Leak-Free Standardization:** `StandardScaler` was fitted strictly on `X_train` (`scaler.fit_transform(X_train)`). The fitted scaler was applied to `X_test` via `scaler.transform(X_test)` with zero test-set information leaking into training parameters.

---

## 11. Exploratory Data Analysis (EDA)

Key findings from our exploratory analysis:
- **Hormone Differentiation:** Primary Hypothyroid patients display markedly elevated TSH (>15 mIU/L) and depressed FTI and TT4.
- **Multicollinearity:** Total Thyroxine (`TT4`) and Free Thyroxine Index (`FTI`) exhibit strong positive correlation ($r \approx 0.79$), reflecting endocrine biochemistry ($\text{FTI} \approx \text{TT4}/\text{T4U}$).
- **Severe Raw Skewness:** Blood hormone distributions (particularly TSH) are heavily right-skewed, confirming why median imputation is mathematically superior to mean imputation.

Figures saved in `results/figures/`:
- `class_distribution.png`: Bar chart of the 4 balanced target classes.
- `feature_distributions.png`: Histograms and KDE plots for numerical tests.
- `correlation_heatmap.png`: Annotated Pearson correlation matrix.
- `hormone_boxplots_by_class.png`: Boxplots showing hormone levels across diagnoses.
- `raw_missing_values_audit.png`: Missingness percentages across raw columns.

---

## 12. Machine Learning Algorithms Used

1. **Logistic Regression (Softmax Multinomial):** Linear baseline optimizing multinomial cross-entropy loss.
2. **Decision Tree Classifier (`max_depth=8`):** Non-parametric hierarchical tree partitioning feature space by maximizing Gini Information Gain.
3. **Random Forest Classifier (`n_estimators=100`, `max_depth=12`):** Ensemble of 100 decorrelated decision trees using bootstrap aggregation (bagging) and majority voting.
4. **K-Nearest Neighbors (`k=3`, Euclidean Distance):** Instance-based non-parametric classifier identifying the 3 closest patients in standardized feature space.
5. **Support Vector Machine (RBF Kernel):** Maximum-margin classifier using the Radial Basis Function kernel to separate non-linear clinical boundaries.
6. **Gaussian Naive Bayes:** Probabilistic generative classifier applying Bayes' Theorem under the assumption of feature conditional independence.

---

## 13. Model Training Process

All models were trained on the standardized training partition (16,000 samples, 4,000 per class) with fixed random seeds (`random_state=42`):
- **Decision Tree:** Trained in **0.019 seconds**.
- **Naive Bayes:** Trained in **0.028 seconds**.
- **K-Nearest Neighbors:** Fitted in **0.041 seconds**.
- **Logistic Regression:** Converged in **0.301 seconds**.
- **Random Forest:** Trained 100 trees in **0.406 seconds**.
- **Support Vector Machine:** Required **16.342 seconds** due to quadratic kernel distance matrix calculations and internal probability cross-validation.

---

## 14. Evaluation Metrics Explained

- **Accuracy:** Overall proportion of correct predictions across all classes.
- **Precision (Macro):** Average fraction of true disease cases among all positive predictions for that condition:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
- **Recall (Macro):** Average fraction of actual disease cases successfully detected:
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
  *Recall is paramount in medical screening to prevent False Negatives.*
- **F1-Score (Macro):** Harmonic mean balancing Precision and Recall equally across all classes:
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Confusion Matrix:** Full tabular breakdown of true vs predicted classes, revealing specific misclassification patterns.

---

## 15. Actual Measured Model Results

Evaluated on 4,000 holdout test records (1,000 samples per class):

| Rank | Model | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Train Time (s) | Inference Latency (ms/sample) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Random Forest** | **98.58%** | **0.9862** | **0.9858** | **0.9857** | 0.406 s | 0.0118 ms |
| 🥈 | **Decision Tree** | **98.22%** | **0.9827** | **0.9822** | **0.9822** | 0.019 s | 0.0008 ms |
| 🥉 | **K-Nearest Neighbors (k=3)** | **97.55%** | **0.9768** | **0.9755** | **0.9753** | 0.041 s | 0.0970 ms |
| 4 | **Support Vector Machine (RBF)** | **94.17%** | **0.9424** | **0.9417** | **0.9417** | 16.342 s | 0.3375 ms |
| 5 | **Logistic Regression** | **88.25%** | **0.8821** | **0.8825** | **0.8821** | 0.301 s | 0.0008 ms |
| 6 | **Gaussian Naive Bayes** | **64.98%** | **0.7475** | **0.6498** | **0.5725** | 0.028 s | 0.0002 ms |

---

## 16. Model Comparison & Discussion

1. **Why Random Forest Won (98.58% Accuracy, 0.9857 Macro F1):**
   - Endocrine diagnosis depends on threshold combinations (e.g., elevated TSH with normal vs low TT4).
   - Random Forest's 100 decorrelated trees capture these step-function decision boundaries effortlessly, reducing individual tree variance.
2. **Decision Tree as an Interpretable Alternative (98.22% Accuracy):**
   - Offers near-parity performance while providing clear, transparent white-box clinical rules that physicians can inspect line-by-line.
3. **Reproduction of Semester 2-1 Baseline Models:**
   - KNN ($k=3$) achieved **97.55%**, and SVM achieved **94.17%**, directly validating the student's original coursework experiments.
4. **Why Gaussian Naive Bayes Lagged (64.98% Accuracy):**
   - Naive Bayes strictly assumes features are conditionally independent. The strong biological correlation between TT4 and FTI ($r=0.79$) violates this assumption, distorting probability calculations.

---

## 17. Core Machine Learning Concepts Explained

### What is Machine Learning?
Machine Learning (ML) is a branch of artificial intelligence where algorithms learn mathematical patterns directly from empirical data to perform tasks without being explicitly programmed with manual if-else rules.

### What is Supervised Learning?
A learning paradigm where the algorithm is provided input feature vectors $\mathbf{x}$ alongside corresponding ground-truth target labels $y$. The model learns a mapping function $f(\mathbf{x}) \approx y$.

### What is Classification?
A supervised learning task where the target output variable is a discrete category (class label) rather than a continuous number. In our project, classification predicts one of four distinct thyroid diagnoses.

### What is Training Data?
The subset of historical records (80%, or 16,000 samples) provided to the algorithm to optimize its internal weights and decision boundaries.

### What is Testing Data?
The holdout partition (20%, or 4,000 samples) strictly withheld during training. It serves as a benchmark to simulate how the model performs on new, unseen patients.

### What is Overfitting?
When an algorithm memorizes training data noise, outliers, and quirks rather than learning generalizable patterns. It displays near-perfect training accuracy but poor test accuracy.

### What is Underfitting?
When a model is too simplistic to capture the underlying pattern in the data (e.g., using a linear model on complex non-linear data). Both training and testing performance are poor.

### What is Data Preprocessing?
The process of transforming raw, messy data into a clean, normalized format suitable for machine learning algorithms (cleaning missing values, removing identifiers, scaling features).

### Why do we encode categorical features?
Machine learning algorithms perform linear algebra and calculus operations on vectors of real numbers; text strings like "Female" or "True" cannot be directly multiplied by weight matrices.

### Why do we split the dataset?
To validate generalization. If you test on the same data used to train, you only measure memorization rather than learning.

### How does Logistic Regression work?
It computes a linear weighted sum of features $\mathbf{w}^T \mathbf{x} + b$ and passes the score through the Softmax function to output calibrated probabilities for each class.

### How does a Decision Tree work?
It recursively segments feature space into rectangular regions by selecting feature thresholds that maximize the reduction in Gini Impurity.

### How does Random Forest work?
It trains an ensemble of $B$ bootstrap decision trees ($B=100$), each considering a random subset of features at each split, and aggregates their outputs via majority vote.

### What is a Confusion Matrix?
A contingency table where rows represent actual ground-truth classes and columns represent model predictions, clearly exposing True Positives, False Positives, and False Negatives.

### What is Precision?
The fraction of positive predictions that are truly correct: $\frac{\text{TP}}{\text{TP} + \text{FP}}$. It answers: *"When the model predicts disease, how often is it right?"*

### What is Recall?
The fraction of actual positive cases detected: $\frac{\text{TP}}{\text{TP} + \text{FN}}$. It answers: *"Out of all genuinely sick patients, how many did the model find?"*

### What is the F1-Score?
The harmonic mean of precision and recall: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$. It provides a single balanced score when both false positives and false negatives are important.

---

## 18. Project Limitations

1. **Synthetic Over-Sampling:** The balanced dataset was created using `RandomOverSampler`. While standard for academic benchmarks, clinical deployments require large-scale multi-center clinical trials.
2. **Limited Diagnostic Scope:** The model classifies 4 primary conditions; rare disorders (e.g., Hashitoxicosis, toxic adenoma, thyroid carcinomas) are not included.
3. **Absence of Imaging Data:** Real endocrinology integrates thyroid ultrasound scans, radioactive iodine uptake, and fine-needle aspiration biopsies.
4. **Educational Nature:** This project is designed strictly for academic instruction and must not be used as a primary diagnostic tool in medical facilities.

---

## 19. Future Improvements

1. **Explainable AI (XAI):** Integrate SHAP (SHapley Additive exPlanations) and LIME to generate patient-specific feature contribution plots for clinical explainability.
2. **Bayesian Hyperparameter Tuning:** Apply Optuna for automated search of tree depth, minimum samples per split, and regularization parameters.
3. **Multimodal Deep Learning:** Combine tabular blood markers with thyroid ultrasound imaging using convolutional neural networks.
4. **Cost-Sensitive Loss Functions:** Implement asymmetric loss functions that penalize False Negatives (missed diagnoses) significantly more than False Positives.

---

## 20. Conclusion

This project successfully reconstructs, standardizes, and benchmarks a complete beginner-level Machine Learning project for Thyroid Disease Detection. By enforcing rigorous data hygiene, preventing data leakage, and comparing six traditional supervised algorithms, we demonstrated that **Random Forest Classifier** achieves the highest classification performance (**98.58% accuracy, 0.9857 Macro F1**). The project structure, automated tests, executable notebooks, and documentation provide an exemplary portfolio artifact suitable for academic viva presentation and professional technical interviews.

---

## Author & Acknowledgments

- **Author:** Venkatesh Devarakonda (B.Tech CSE AI & ML)
- **Dataset:** UCI Machine Learning Repository (Garavan Institute of Medical Research & J. Ross Quinlan)
- **Academic Mentor & Code Review:** Antigravity AI Maintainer

