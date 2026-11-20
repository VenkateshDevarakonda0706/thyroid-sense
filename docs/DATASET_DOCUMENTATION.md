# Dataset Documentation: ThyroidSense Thyroid Disease Classification

**Project:** ThyroidSense: Thyroid Disease Detection Using Machine Learning  
**Project Identifier:** `thyroid-sense`  
**GitHub Repository:** [https://github.com/VenkateshDevarakonda0706/thyroid-sense](https://github.com/VenkateshDevarakonda0706/thyroid-sense)  
**Course Context:** B.Tech CSE (AI & ML) — Academic Project  
**Author:** Venkatesh Devarakonda  
**Source Origin:** UCI Machine Learning Repository — Thyroid Disease Database (Garavan Institute of Medical Research & J. Ross Quinlan, 1987)  
**Authentic Benchmark Source:** Single-task UCI `allhypo` (`allhypo.data` and `allhypo.test`)  

---

## 1. Overview and Provenance

The dataset utilized in this project originates from the **UCI Machine Learning Repository's Thyroid Disease Database**, originally collected by the **Garavan Institute of Medical Research in Sydney, Australia** and contributed by **J. Ross Quinlan** during the 1987 Machine Learning Workshop.

The thyroid gland produces hormones—primarily **Thyroxine (T4)** and **Triiodothyronine (T3)**—regulated by pituitary **Thyroid-Stimulating Hormone (TSH)**. These hormones control the body's metabolic rate, heart rate, temperature, and cellular energy expenditure.

When the thyroid gland is dysfunctional, it manifests in distinct diagnostic states:
1. **Compensated Hypothyroid (Subclinical):** Mild thyroid underactivity where the pituitary increases TSH stimulation to maintain normal peripheral T4 levels.
2. **Primary Hypothyroid:** Overt intrinsic thyroid gland failure characterized by abnormally low circulating T4 and markedly elevated compensatory TSH.
3. **Secondary Hypothyroid:** Rare central/pituitary endocrine failure where both TSH and thyroid hormones are abnormally low.
4. **Negative (Euthyroid / Healthy):** Normal, balanced endocrine feedback function.

---

## 2. Forensic Provenance Discovery: The Multi-Task Concatenation Pitfall

In early exploratory iterations of this project, a 22,632-row CSV (`thyroid_raw.csv`) was investigated. A comprehensive technical audit established that this pooled dataset was **not a genuine 22,632-patient cohort**, but rather an invalid vertical concatenation of six distinct UCI subtask files:

| Subtask File | Clinical Question Addressed | Row Count | Negative Count in Subtask | Other Diagnostic Labels in Subtask |
| :--- | :--- | :---: | :---: | :--- |
| **`allhypo`** | Is the patient hypothyroid? | 3,772 | 3,481 | compensated, primary, secondary hypothyroid (291) |
| **`allhyper`** | Is the patient hyperthyroid? | 3,772 | 3,670 | hyperthyroid, T3 toxic, toxic goitre (102) |
| **`allbp`** | Does the patient have abnormal binding protein? | 3,772 | 3,609 | increased / decreased binding protein (163) |
| **`allrep`** | Is the patient on replacement therapy? | 3,772 | 3,648 | replacement therapy, over/under replacement (124) |
| **`dis`** | Are the test results discordant? | 3,772 | 3,714 | discordant measurements (58) |
| **`sick`** | Does the patient have euthyroid sick syndrome? | 3,772 | 3,541 | sick (231) |
| **Total Stacked** | **Merged 6-Task CSV (`thyroid_raw.csv`)** | **22,632** | **21,663** | **969 non-negative records across 15 labels** |

### Why the Merged CSV Was Invalid for 4-Class Hypothyroid Detection:
1. **Patient Replication:** Comparing the original Garavan patient IDs (encoded behind the `|` delimiter in the raw UCI `.data` and `.test` files) proved that the exact same 3,772 patients were duplicated 6 times.
2. **Catastrophic Label Corruption:** In subtasks such as `allhyper` or `sick`, a hypothyroid patient was labeled `negative.` simply because they did not have hyperthyroidism or euthyroid sick syndrome. Stacking these files caused every single hypothyroid patient to appear up to 5 additional times labeled as Negative, introducing **1,712 cross-task contradictory records**.
3. **Artificial Data Leakage:** When an 80/20 train/test split was performed on the stacked rows, the `allhypo` record for a patient fell into the test set as `compensated hypothyroid`, while the identical patient's rows from `allhyper` or `sick` fell into the training set as `negative`.
4. **Resolution:** ThyroidSense completely abandons the pooled 22,632-row artifact for modeling and transitions exclusively to the **authentic single-task UCI `allhypo` dataset**.

---

## 3. The Authentic UCI `allhypo` Dataset

The authentic single-task `allhypo` dataset is partitioned into its historical training and testing splits:
- `data/raw/uci_thyroid_original/allhypo.data` (2,800 records)
- `data/raw/uci_thyroid_original/allhypo.test` (972 records)

### Distribution Across Partitions (3,772 Unique Patients)

| Target Label | Class Code | Training Partition (`allhypo.data`) | Testing Partition (`allhypo.test`) | Combined Cohort |
| :--- | :---: | :---: | :---: | :---: |
| **Negative (Healthy)** | `7` | 2,580 (92.14%) | 901 (92.70%) | **3,481** (92.29%) |
| **Compensated Hypothyroid** | `1` | 154 (5.50%) | 40 (4.12%) | **194** (5.14%) |
| **Primary Hypothyroid** | `9` | 64 (2.29%) | 31 (3.19%) | **95** (2.52%) |
| **Secondary Hypothyroid** | `11` | 2 (0.07%) | **0 (0.00%)** | **2** (0.05%) |
| **Total Cohort** | — | **2,800 records** | **972 records** | **3,772 unique patients** |

> [!IMPORTANT]
> **Zero Support on Secondary Hypothyroidism in Test Split:**  
> In Quinlan's original benchmark partitioning, both genuine Secondary Hypothyroid patients were allocated to `allhypo.data`, leaving **0 records in `allhypo.test`**. Consequently, Secondary Hypothyroid performance cannot be evaluated on the holdout test set. This zero support is transparently reported.

---

## 4. Feature Dictionary

The 14 clinical features used for modeling are defined below:

### A. Demographic Attributes
| Feature Name | Type | Description | Normal Clinical Reference |
| :--- | :--- | :--- | :--- |
| `age` | Numerical (Continuous) | Age in years (1 to 100). | N/A |
| `sex` | Binary Categorical | Biological sex (0 = Female, 1 = Male). | N/A |

### B. Medical Symptoms & History
| Feature Name | Type | Values | Diagnostic Significance |
| :--- | :--- | :--- | :--- |
| `on thyroxine` | Binary | 0 = No, 1 = Yes | Patient takes levothyroxine hormone replacement. |
| `on antithyroid medication` | Binary | 0 = No, 1 = Yes | Patient takes drugs to suppress hormone synthesis (Methimazole). |
| `pregnant` | Binary | 0 = No, 1 = Yes | Pregnancy alters hormone demand and binding globulin levels. |
| `thyroid surgery` | Binary | 0 = No, 1 = Yes | History of thyroidectomy, causing postsurgical hypothyroidism. |
| `lithium` | Binary | 0 = No, 1 = Yes | Lithium inhibits thyroid hormone synthesis and release. |
| `goitre` | Binary | 0 = No, 1 = Yes | Physical enlargement of the thyroid gland. |
| `tumor` | Binary | 0 = No, 1 = Yes | Presence of thyroid nodules or suspected neoplasm. |

### C. Serum Endocrine Blood Laboratory Measurements
| Feature Name | Type | Normal Clinical Range | Diagnostic Significance |
| :--- | :--- | :--- | :--- |
| `TSH` | Continuous | 0.4 – 4.0 mIU/L | Thyroid-Stimulating Hormone. Primary first-line diagnostic test. |
| `T3` | Continuous | 1.2 – 3.1 nmol/L | Triiodothyronine (metabolically active thyroid hormone). |
| `TT4` | Continuous | 60 – 150 nmol/L | Total Thyroxine (circulating pro-hormone). |
| `T4U` | Continuous | 0.7 – 1.2 | Thyroxine Uptake ratio (measures binding protein saturation). |
| `FTI` | Continuous | 70 – 140 | Free Thyroxine Index ($\text{FTI} = \text{TT4} / \text{T4U}$). |

---

## 5. Preprocessing and Leakage Prevention Rules

1. **Partitioning First:** Training (`allhypo.data`) and testing (`allhypo.test`) are parsed separately. Zero patient IDs overlap.
2. **Training-Only Imputation:** Continuous missing values are filled with medians computed strictly from `X_train` (`age: 54.0, TSH: 1.4, T3: 2.0, TT4: 104.0, T4U: 0.98, FTI: 107.0`). Binary symptoms are filled with `0.0` (absence of condition, matching training mode).
3. **Scaling Before Resampling:** `StandardScaler` is fitted on original training features before `RandomOverSampler`. This preserves natural population variance statistics.
4. **Untouched Test Partition:** `allhypo.test` (972 patients) is never oversampled or resampled.

---

## 6. Limitations & Disclaimer

> [!CAUTION]
> **Educational & Academic Notice:**  
> This dataset documentation supports an undergraduate computer science machine learning portfolio. It is **not** validated for prospective healthcare deployment or clinical medical diagnostics.
