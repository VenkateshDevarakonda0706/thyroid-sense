# Project Audit: ThyroidSense (Thyroid Disease Detection Using Machine Learning)

**Project Identifier:** `ThyroidSense`  
**Audit Date:** November 2026 (Reconstruction Audit)  
**Original Academic Period:** November 2023 – March 2024 (B.Tech CSE AIML, Semester 2-1)  
**Auditor / Mentor:** Antigravity Project Maintainer & Code Reviewer  
**Original Workspace:** `Thyroid complete` (Migrated & Linked to `ThyroidSense`)  

---

## 1. Executive Summary

A comprehensive, non-destructive audit of the entire workspace (`Thyroid complete`, now restructured as `ThyroidSense`) was conducted. The project directory originally contained a mixture of files from multiple phases:
1. **The original 2-1 semester Machine Learning academic project (October 2023 – February 2024)**: Focused on UCI Thyroid Disease classification, feature engineering, missing value imputation, statistical hypothesis testing (ANOVA F-test, T-test, Z-test, Mann-Whitney U test, Jarque-Bera test), class imbalance handling, and model training (KNN, Gaussian Naive Bayes, Support Vector Machines).
2. **Intermediate experimentation & pipeline dumps**: Numerous intermediate CSV files (`u.csv` through `u5.csv`, `Finalthyroid.csv`, `thyroidpro.csv`, etc.) generated during exploratory data cleaning.
3. **Full-stack and web integration experiments (Early 2024 – Late 2025)**:
   - A MERN-like stack attempt in `project2/Project/Server` (Express.js, MongoDB Atlas) and `project2/Project/ML project/thyroid-detection` (React).
   - A Flask inference API in `project2/Project/ML python/Flask`.
   - A FastAPI + Vite + SQLite web application refactor in `Thyroid Detection/` (November 2025).
4. **Unrelated coursework / lab assignments**: Hand gesture recognition with OpenCV/MediaPipe (`PROJECT-Copy1.ipynb`), pharmaceutical marketing analytics (`A-2.ipynb`), Single Layer Perceptron assignment (`Untitled13.ipynb`), and LLM API experiments (`A-1.ipynb`).
5. **Git repository configuration anomalies**: The root project folder was not an independent Git repository, but was nested beneath a Git repository initialized on the user's `Desktop`, while subdirectories (`project2/Project/ML project`, `project2/Project/Server`, and `project2/Project/ML project/thyroid-detection`) contained historical commits from January–February 2024.
6. **Security & Binary Issues**: A hardcoded MongoDB Atlas credential with plaintext password was discovered in `project2/Project/Server/index.js`, and a 70 MB bundled `node.exe` binary was tracked in `project2/Project/ML project/`.

---

## 2. Git History and Provenance Audit

| Location | Status | Remote URL | Historical Commits / Dates | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `..\..\` (Desktop) | Active Git Root | None (Local) | Variable | The parent folder `C:\Users\Venkatesh\OneDrive\Desktop` has a `.git` directory, causing commands in the workspace to reflect the parent repository if not isolated. |
| Workspace Root | Not a Git repo | N/A | None directly at root | Needs its own clean, isolated Git repository. |
| `project2/Project/ML project` | Git Repo | `https://github.com/sreedharjinka/Thyroid-DetectionG119.git` | Feb 1, 2024 (`Initial commit` by Manogna) | Contains 70MB `node.exe` and `node_modules`. |
| `project2/Project/Server` | Git Repo | `https://github.com/Manogna1507/Thyroid-Detection` | Jan 31 – Feb 1, 2024 (Commits by Manogna) | Genuine academic group work commits from 2-1 semester. |
| `project2/Project/ML project/thyroid-detection` | Git Repo | `https://github.com/sreedharjinka/Thyroid-DetectionG119` | Feb 1, 2024 | React frontend repository. |

### Timeline Verification
- **Verified 2-1 Semester Timeline:** Genuine file creation timestamps and Git commit logs confirm project activity between **October 2023 and February 2024**.
- **Reconstruction Period:** **November 2026**.
- **Integrity Rule:** In accordance with the project guidelines, historical timestamps and commits will NOT be falsified or rewritten. New commits will reflect the actual reconstruction date.

---

## 3. Dataset Audit & Identification of True Ground Truth

The workspace contained over 20 CSV files with varying shapes and column counts. The investigation traced the dataset evolution:

1. **Original Raw Dataset (`data/raw/`):**
   - File: `thyroid.csv` (and `project2/Project/ML python/thyroid1.csv`)
   - Source: **UCI Machine Learning Repository — Thyroid Disease Records**
   - Size: 2,215,328 bytes
   - Rows: 9,172 records (expanded to 21,726 in merged multi-domain variants)
   - Columns: 31 columns (demographics, clinical symptoms, lab tests, measuring flags, referral source, dataset name, target class).
   - Target variable: `class` (15 distinct diagnostic categories: `negative`, `compensated hypothyroid`, `primary hypothyroid`, `secondary hypothyroid`, `hyperthyroid`, `sick`, etc.).

2. **Exploratory Preprocessing Dumps (Evolutionary Chain):**
   - `thyroidmean.csv` (2.2 MB): Initial missing value imputation for TSH, T3, TT4, T4U, FTI using column means.
   - `thyroidDFmean2.csv` (749 KB): Re-formatted column names with underscores.
   - `TH.csv`, `Thyriod.csv`, `Finalthyroid.csv`, `Finalthyroid2.csv` (1.0 - 1.6 MB): Dropped indicator flags (`TSH measured`, `TBG measured`, etc.) and dropped administrative fields (`patient_id`, `referral source`, `dataset_name`), keeping 17 features.
   - `u.csv` -> `u2.csv` -> `u3.csv` -> `u4.csv` -> `u5.csv` (291 KB - 5.0 MB): Iterative synthetic upsampling and outlier trimming experiments.
   - `detection.csv`, `sample2.csv`, `sample_data.csv`: Intermediate verification dumps.

3. **True Modeling Dataset (`project2/Project/ML python/data.csv`):**
   - File: `data.csv` (1,078,043 bytes)
   - Rows: 20,000 records
   - Columns: 15 (14 clinical & lab features + 1 target `class`)
   - Features:
     1. `age` (numerical)
     2. `sex` (binary: 0=Female, 1=Male)
     3. `on thyroxine` (binary: 0=No, 1=Yes)
     4. `on antithyroid medication` (binary: 0=No, 1=Yes)
     5. `pregnant` (binary: 0=No, 1=Yes)
     6. `thyroid surgery` (binary: 0=No, 1=Yes)
     7. `lithium` (binary: 0=No, 1=Yes)
     8. `goitre` (binary: 0=No, 1=Yes)
     9. `tumor` (binary: 0=No, 1=Yes)
     10. `TSH` (numerical, Thyroid-Stimulating Hormone)
     11. `T3` (numerical, Triiodothyronine)
     12. `TT4` (numerical, Total Thyroxine)
     13. `T4U` (numerical, Thyroxine Uptake)
     14. `FTI` (numerical, Free Thyroxine Index)
     15. `class` (target, 4 balanced classes of 5,000 samples each: `1` = Compensated Hypothyroid, `7` = Negative/Normal, `9` = Primary Hypothyroid, `11` = Secondary Hypothyroid).
   - Provenance: This dataset directly matches the original 2-1 notebook `Thyroid_(1).ipynb`, the serialized `svm_model` / `svm_model.pkl`, the `enc.pickle` label encoder, and the inference API `server.py`.

---

## 4. Comprehensive File Inventory & Recommendations

| File Path | Type | Original Purpose | Useful? | Incomplete? | Duplicated? | Required in Final? | Action | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `thyroid.csv` | CSV | Original UCI raw thyroid dataset | Yes | No | Yes (in `PROJECT/`) | Yes | **MODIFY** | Retain as the official raw benchmark dataset inside `data/raw/thyroid_raw.csv`. |
| `Finalthyroid.csv` | CSV | Intermediate preprocessed data | No | Partial | Yes (in `PROJECT/`) | No | **ARCHIVE** | Intermediate scratch file; pipeline will generate reproducible processed data. |
| `detection.csv` | CSV | Intermediate scratch data | No | Partial | Yes | No | **ARCHIVE** | Temporary export. |
| `sample2.csv` | CSV | Sample dataset slice | No | Yes | Yes (`sample_data.csv`) | No | **ARCHIVE** | Unused sample dump. |
| `sample_data.csv` | CSV | Sample dataset slice | No | Yes | Yes (`sample2.csv`) | No | **ARCHIVE** | Duplicate of `sample2.csv`. |
| `TH.csv` | CSV | Intermediate dump with missing values | No | Partial | No | No | **ARCHIVE** | Intermediate stage file. |
| `Thyriod.csv` | CSV | Intermediate preprocessed data | No | Partial | No | No | **ARCHIVE** | Misspelled intermediate export. |
| `thyroidDFmean2.csv`| CSV | Mean imputed intermediate data | No | Partial | Yes | No | **ARCHIVE** | Intermediate step. |
| `thyroidmean.csv` | CSV | Mean imputed raw data | No | No | Yes | No | **ARCHIVE** | Redundant with `thyroid.csv`. |
| `thyroidpro.csv` | CSV | Intermediate preprocessed data | No | Partial | Yes | No | **ARCHIVE** | Redundant intermediate stage. |
| `u.csv` | CSV | Partial slice | No | Yes | Yes | No | **ARCHIVE** | Early experiment slice. |
| `u2.csv`, `u2-Copy1.csv`, `u3.csv`, `u4.csv`, `u5.csv` | CSV | Upsampling experiment dumps | No | No | Yes | No | **ARCHIVE** | Intermediate dumps; `u5.csv` matches `thyroid_tabular..csv`. |
| `A-1.ipynb` | Notebook | OpenAI API experiment | No | Yes | No | No | **ARCHIVE** | Unrelated LLM API script. |
| `A-2.ipynb` | Notebook | Pharma marketing analysis | No | Yes | No | No | **ARCHIVE** | Unrelated marketing analytics project. |
| `PROJECT-Copy1.ipynb`| Notebook | OpenCV & MediaPipe vision | No | Yes | No | No | **ARCHIVE** | Unrelated hand gesture project. |
| `Thyriod2.ipynb` | Notebook | Scratch data exploration | No | Partial | Yes | No | **ARCHIVE** | Redundant with `Thyroid_(1).ipynb`. |
| `Thyroid (1).ipynb` (root) | Notebook | Early preprocessing scratch | No | Partial | Yes | No | **ARCHIVE** | Incomplete early version. |
| `Thyroid3.ipynb` | Notebook | Outlier & stats exploration | Yes | Partial | Yes | No | **ARCHIVE** | Code absorbed into reconstructed notebooks. |
| `Untitled13.ipynb` | Notebook | Single Layer Perceptron lab | No | No | No | No | **ARCHIVE** | Unrelated neural network lab assignment. |
| `Untitled14.ipynb` | Notebook | Scratch data cleaning | No | Partial | No | No | **ARCHIVE** | Scratch notebook. |
| `Untitled15.ipynb` | Notebook | IQR outlier testing | No | Partial | No | No | **ARCHIVE** | Scratch notebook. |
| `Untitled16.ipynb` | Notebook | Scratch testing | No | Partial | No | No | **ARCHIVE** | Scratch notebook. |
| `PROJECT/` (all files) | Folder | Duplicate/older workspace copies | Yes | No | Yes | No | **ARCHIVE** | Preserved in archive for safety. |
| `project2/Project/ML python/Thyroid_(1).ipynb` | Notebook | Complete 2-1 ML experiments | **Yes** | No | No | Yes | **KEEP & MODULARIZE** | Core original student work. Used to reconstruct modular notebooks 01-04. |
| `project2/Project/ML python/data.csv` | CSV | 14-feature balanced dataset | **Yes** | No | No | Yes | **KEEP** | Ground truth dataset for reproducing original ML models. |
| `project2/Project/ML python/svm_model` | Model | Serialized SVM model | **Yes** | No | No | Yes | **KEEP** | Original trained model artifact. |
| `project2/Project/ML python/enc.pickle` | Model | Serialized LabelEncoder | **Yes** | No | No | Yes | **KEEP** | Original class mapping artifact. |
| `project2/Project/ML python/predict.csv`, `predictions.csv` | CSV | Verification test records | **Yes** | No | No | Yes | **KEEP** | Used in model inference tests. |
| `project2/Project/ML python/test.csv`, `testing.csv`, `try.csv`, `thyroid1.csv` | CSV | Intermediate splits & dumps | No | No | Yes | No | **ARCHIVE** | Redundant intermediate files. |
| `project2/Project/ML python/Flask/` | Folder | Flask inference server | Yes | No | No | No | **ARCHIVE** | Archived into `archive/flask_inference/`. |
| `project2/Project/Server/` | Folder | Node/Express backend | No | No | No | No | **ARCHIVE (SECURITY SANITIZED)** | Full-stack web app; hardcoded MongoDB password must be secured. |
| `project2/Project/ML project/thyroid-detection/` | Folder | React frontend | No | No | No | No | **ARCHIVE** | React frontend app; archived in `archive/react_frontend/`. |
| `project2/Project/ML project/node.exe` | Binary | 70MB Node.js binary | No | No | No | No | **REMOVE FROM REPO / ARCHIVE** | Large executable binary; violates Git best practices. |
| `Thyroid Detection/` | Folder | Nov 2025 FastAPI/Vite refactor | No | Partial | Partial | No | **ARCHIVE** | Out of scope for 2-1 core ML academic project. |
| `thyroid+disease/` | Folder | Raw UCI data files & docs | **Yes** | No | No | Yes | **KEEP** | Pristine benchmark provenance files (`allbp`, `allhyper`, `allhypo`, etc.). |

---

## 5. Security & Credentials Findings

> [!WARNING]
> **Hardcoded Secret Found in Legacy `project2/Project/Server/index.js` (Line 19):**  
> `[REDACTED]`  
> - **Risk:** Exposes database credentials to anyone viewing the repository.  
> - **Action Required:** Remove connection strings from source code, replace with `.env.example` templates, and ensure live credentials are never committed to GitHub. Rotated password advisory issued to project team.

---

## 6. Safety Plan Before Any Changes

Per the Master Prompt instructions:
1. **Full Backup:** A complete copy of the workspace will be preserved in a protected `_original_backup/` directory.
2. **Non-Destructive Relocation:** Files marked for archiving will be moved to `archive/` rather than deleted, so no historical context is lost.
3. **Reproducibility Guarantee:** No file will be removed until the complete reconstructed pipeline (`src/`, `notebooks/`, `tests/`) runs and validates end-to-end.

---

## 7. Target Reconstruction Plan

1. **Target Architecture:** Pure, beginner-level, modular Python Machine Learning project following standard academic CSE AIML practices.
2. **Algorithms:**
   - Logistic Regression (Baseline linear classifier)
   - Decision Tree Classifier (Interpretable tree model)
   - Random Forest Classifier (Ensemble bagging model)
   - K-Nearest Neighbors (KNN - from original project, k=3)
   - Support Vector Classifier (SVC - from original project, RBF kernel)
   - Gaussian Naive Bayes (Baseline probabilistic model from original project)
3. **Deliverables:**
   - 4 Clean Jupyter Notebooks (`01_data_exploration.ipynb`, `02_data_preprocessing.ipynb`, `03_model_training.ipynb`, `04_model_evaluation.ipynb`).
   - Clean reusable Python modules (`src/data_preprocessing.py`, `src/model_training.py`, `src/model_evaluation.py`).
   - Unit tests (`tests/test_preprocessing.py`, `tests/test_model_training.py`).
   - Figures & Metrics tables in `results/`.
   - Comprehensive documentation (`README.md`, `docs/*.md`, `docs/VIVA_QUESTIONS.md`).

