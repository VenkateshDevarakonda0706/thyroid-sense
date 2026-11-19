# Dataset Documentation: Thyroid Disease Classification

**Project:** Thyroid Disease Detection Using Machine Learning  
**Course Context:** B.Tech CSE (AI & ML) — Semester 2-1 Academic Project  
**Author:** Venkatesh Devarakonda  
**Source Origin:** UCI Machine Learning Repository (Thyroid Disease Records)  
**Dataset Format:** CSV (Comma-Separated Values)  

---

## 1. Overview and Provenance

The dataset utilized in this project originates from the **UCI Machine Learning Repository's Thyroid Disease Database**, originally contributed by the **Garavan Institute of Medical Research and J. Ross Quinlan**.

The thyroid gland is a butterfly-shaped endocrine gland located in the lower front of the neck. It produces hormones—primarily **Thyroxine (T4)** and **Triiodothyronine (T3)**—which regulate metabolism, heart rate, body temperature, and energy production. The pituitary gland produces **Thyroid-Stimulating Hormone (TSH)** to signal the thyroid gland how much hormone to release.

When the thyroid gland is dysfunctional, it typically leads to:
1. **Hypothyroidism (Underactive thyroid):** Insufficient production of thyroid hormones, slowing body metabolism.
2. **Hyperthyroidism (Overactive thyroid):** Excessive production of thyroid hormones, accelerating body metabolism.
3. **Euthyroid / Negative (Normal):** Healthy, balanced thyroid function.

This dataset compiles real patient records combining:
- **Demographic data** (Age, Sex).
- **Clinical history and symptoms** (Goitre, Tumor, Pregnancy, Thyroid surgery history, Medication usage).
- **Objective endocrine blood test laboratory measurements** (TSH, T3, TT4, T4U, FTI).

---

## 2. Raw Dataset vs. Curated Modeling Dataset

During the project lifecycle, the data transitions through two primary representations:

| Attribute | Raw Dataset (`data/raw/thyroid_raw.csv`) | Curated Modeling Dataset (`data/processed/thyroid_processed.csv`) |
| :--- | :--- | :--- |
| **Row Count** | 9,172 records (merged variants reach 21,726) | 20,000 records (balanced via random oversampling) |
| **Column Count** | 31 columns | 15 columns (14 features + 1 target) |
| **Missing Values** | Present in continuous lab test columns (marked as `?` or `NaN`) | Imputed using training feature medians/means (0 missing) |
| **Target Column** | `class` (15 raw multiclass diagnostic labels) | `class` (4 core clinically defined classes) |
| **Redundant Flags** | Contains 6 boolean measuring flags (`TSH measured`, etc.) | Dropped (redundant with the actual measurement presence) |
| **Identifiers** | Contains `patient_id`, `referral source`, `dataset_name` | Dropped (non-predictive administrative attributes) |

---

## 3. Feature Dictionary

The 14 input features used by the Machine Learning models are defined below:

### A. Demographic Features
| Feature Name | Type | Range / Values | Description | Clinical Importance |
| :--- | :--- | :--- | :--- | :--- |
| `age` | Numerical (Integer/Float) | 1 to 100 years | Age of the patient in years. | Thyroid disorders are significantly more prevalent in older populations. |
| `sex` | Categorical (Binary) | 0 = Female, 1 = Male | Biological sex of the patient. | Thyroid conditions are 5 to 8 times more common in females. |

### B. Medical History & Symptoms
| Feature Name | Type | Range / Values | Description | Clinical Importance |
| :--- | :--- | :--- | :--- | :--- |
| `on thyroxine` | Categorical (Binary) | 0 = No, 1 = Yes | Patient is undergoing synthetic thyroid hormone replacement (Levothyroxine). | Directly impacts blood hormone levels and indicates known previous disorder. |
| `on antithyroid medication` | Categorical (Binary) | 0 = No, 1 = Yes | Patient takes drugs to suppress hormone production (e.g., Methimazole). | Indicates treatment for hyperthyroidism. |
| `pregnant` | Categorical (Binary) | 0 = No, 1 = Yes | Patient is pregnant. | Pregnancy alters thyroid-binding globulin and hormone demand. |
| `thyroid surgery` | Categorical (Binary) | 0 = No, 1 = Yes | Patient previously underwent partial/total thyroidectomy. | Direct cause of surgical hypothyroidism. |
| `lithium` | Categorical (Binary) | 0 = No, 1 = Yes | Patient takes lithium medication (mood stabilizer). | Lithium inhibits thyroid hormone synthesis and release. |
| `goitre` | Categorical (Binary) | 0 = No, 1 = Yes | Physical enlargement of the thyroid gland. | Hallmark clinical sign of autoimmune thyroid disease or iodine deficiency. |
| `tumor` | Categorical (Binary) | 0 = No, 1 = Yes | Presence of thyroid nodule or tumor. | Requires differential diagnosis between benign disease and malignancy. |

### C. Objective Serum Blood Test Measurements
| Feature Name | Type | Normal Clinical Range | Description | Diagnostic Significance |
| :--- | :--- | :--- | :--- | :--- |
| `TSH` | Numerical (Continuous) | 0.4 – 4.0 mIU/L | Thyroid-Stimulating Hormone. | The gold standard first-line test: high TSH indicates hypothyroidism; low TSH indicates hyperthyroidism. |
| `T3` | Numerical (Continuous) | 1.2 – 3.1 nmol/L | Triiodothyronine (active hormone). | Key marker for assessing active metabolic state. |
| `TT4` | Numerical (Continuous) | 60 – 150 nmol/L | Total Thyroxine. | Total circulating T4 (both protein-bound and free). |
| `T4U` | Numerical (Continuous) | 0.7 – 1.2 | Thyroxine Uptake ratio. | Measures the saturation of thyroid hormone binding proteins. |
| `FTI` | Numerical (Continuous) | 70 – 140 | Free Thyroxine Index. | Normalized estimate of active unbound T4: $\text{FTI} = \frac{\text{TT4}}{\text{T4U}}$. |

---

## 4. Target Variable and Class Definitions

The target variable is `class`. In the curated dataset, 4 major clinical classes are investigated:

| Class Code | Clinical Diagnosis | Description | Class Frequency (Curated) |
| :---: | :--- | :--- | :---: |
| **`1`** | **Compensated Hypothyroid** | Subclinical / mild underactivity. Pituitary increases TSH to force the thyroid to produce normal T4 levels. | 5,000 (25%) |
| **`7`** | **Negative** | Healthy, normal euthyroid function. Blood hormone markers are within expected physiological boundaries. | 5,000 (25%) |
| **`9`** | **Primary Hypothyroid** | Overt thyroid failure. The gland cannot produce enough T4 despite elevated TSH stimulation. | 5,000 (25%) |
| **`11`** | **Secondary Hypothyroid** | Central / pituitary dysfunction. Both TSH and T4 levels are abnormally low. | 5,000 (25%) |

### Label Encoding Mapping
The original string diagnoses were mapped using Scikit-Learn's `LabelEncoder`:
- `1` $\rightarrow$ `compensated hypothyroid`
- `7` $\rightarrow$ `negative`
- `9` $\rightarrow$ `primary hypothyroid`
- `11` $\rightarrow$ `secondary hypothyroid`

---

## 5. Data Hygiene & Quality Checks

1. **Missing Value Analysis:**
   - Categorical and demographic features had zero missing values in the processed subset.
   - Continuous laboratory columns (`TSH`, `T3`, `TT4`, `T4U`, `FTI`) in raw UCI records contained missing test values because physicians do not order every test for every patient.
   - Missing continuous values are imputed strictly using training partition statistics (median or mean) to eliminate data leakage.
2. **Duplicate Records:**
   - Duplicates in the balanced dataset arise deliberately from `RandomOverSampler`, which clones minority instances to prevent majority-class bias.
   - The test set is evaluated independently to ensure fair generalization.
3. **Outliers:**
   - Extreme lab values (e.g., TSH > 100 mIU/L) represent severe clinical pathology rather than instrument error. While exploratory IQR capping is demonstrated for educational comparison, clinical models benefit from retaining physiological extremes.

---

## 6. Privacy, Ethics, and Open Source Redistribution

- **Patient Anonymity:** The dataset is fully de-identified. It contains no names, medical record numbers, contact information, hospital admission dates, or geographic markers.
- **License & Redistribution:** UCI datasets are standard educational benchmarks provided for open research and machine learning instruction under fair academic use.
- **Academic Disclaimer:** This dataset and derived models are intended strictly for educational and instructional demonstration. They are **not** certified diagnostic medical devices.
