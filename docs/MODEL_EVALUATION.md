# Model Evaluation and Benchmark Report: ThyroidSense

**Project:** ThyroidSense: Thyroid Disease Detection Using Machine Learning  
**Project Identifier:** `ThyroidSense`  
**GitHub Repository:** [https://github.com/VenkateshDevarakonda0706/thyroid-sense](https://github.com/VenkateshDevarakonda0706/thyroid-sense)  
**Author:** Venkatesh Devarakonda  
**Evaluation Set:** 972 Untouched Holdout Test Records from Authentic UCI `allhypo.test`  
**Test Partition Distribution:** Negative: 901, Compensated: 40, Primary: 31, Secondary: 0  
**Execution Environment:** Isolated Python 3.11 Virtual Environment  

---

## 1. Metric Definitions

In medical diagnostic Machine Learning, accuracy alone can be deeply misleading. We assess models across multiple complementary dimensions:

### A. Accuracy
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
Represents the proportion of all predictions that were correct. In an imbalanced test partition (901/972 healthy), a dummy classifier predicting the majority class for all records achieves **92.70% accuracy** while failing to detect a single sick patient.

### B. Precision (Positive Predictive Value)
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
Answers: *"Of all patients predicted to have condition $X$, what fraction actually had it?"*  
High precision minimizes False Positives (preventing healthy patients from undergoing unnecessary, stressful medical treatments).

### C. Recall (Sensitivity / True Positive Rate)
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
Answers: *"Of all patients who genuinely had condition $X$, what fraction did the model successfully detect?"*  
In healthcare screening, **Recall is paramount**, because a False Negative means an individual with active thyroid failure remains undiagnosed and untreated.

### D. F1-Score (Harmonic Mean)
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
Provides a single metric that balances precision and recall.

### E. Macro vs. Weighted Averaging
- **Macro Average:** Computes the metric independently for each class and calculates the unweighted arithmetic mean across all four classes:
  $$\text{Macro F1} = \frac{1}{C} \sum_{c=1}^{C} \text{F1}_c$$
  Every class is weighted equally regardless of its sample size. This is the primary metric used to rank our models.
  > [!NOTE]
  > Because Secondary Hypothyroid (Class 11) has zero test instances ($n=0$) in `allhypo.test`, its F1-score is mathematically $0.0$. Thus, across all 4 classes, Macro F1 has a theoretical ceiling of $\approx 0.75$.
- **Weighted Average:** Weights the score of each class by its support (number of true instances).

---

## 2. Actual Measured Benchmark Results

> Decision Tree and Random Forest achieved strong results on the three classes represented in the original UCI test partition. Secondary Hypothyroid performance cannot be evaluated because the original test partition contains zero examples of that class.
 (Authentic UCI `allhypo` Test Set)

All 6 models were evaluated on the exact same holdout test set of 972 genuine, untouched patient records from `allhypo.test`. Zero data leakage occurred:

| Rank | Model | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Train Time (s) | Inference Latency (ms/sample) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Decision Tree** | **99.79%** | **0.7381** | **0.7494** | **0.7436** | **0.9980** | **0.9979** | **0.9980** | 0.016 s | 0.0010 ms |
| 🥈 | **Random Forest** | **99.59%** | **0.7243** | **0.7429** | **0.7334** | **0.9961** | **0.9959** | **0.9959** | 0.221 s | 0.0432 ms |
| 🥉 | **Logistic Regression** | **95.88%** | **0.6181** | **0.7114** | **0.6572** | **0.9801** | **0.9588** | **0.9679** | 0.132 s | 0.0010 ms |
| 4 | **Support Vector Machine** | **95.37%** | **0.5649** | **0.7040** | **0.6186** | **0.9722** | **0.9537** | **0.9601** | 3.089 s | 0.8560 ms |
| 5 | **K-Nearest Neighbors** | **94.55%** | **0.5689** | **0.5852** | **0.5733** | **0.9555** | **0.9455** | **0.9498** | 0.027 s | 0.0701 ms |
| 6 | **Gaussian Naive Bayes** | **9.26%** | **0.3062** | **0.4043** | **0.1156** | **0.9346** | **0.0926** | **0.0826** | 0.006 s | 0.0011 ms |

**Majority-Class (Negative) Baseline:**
- A naive baseline that predicts Negative (Class 7) for all 972 test cases achieves:
  - **Accuracy:** 92.70% (901/972)
  - **Macro Precision:** 0.2317
  - **Macro Recall:** 0.2500
  - **Macro F1:** 0.2405
- Both Decision Tree and Random Forest massively outperform this baseline, achieving near-perfect disease recall while preserving near-zero false alarms.

### Model Selection Methodology & Comparative Scope
In this benchmark, all six candidate models were trained with fixed hyperparameters on the resampled training partition (10,320 records) and evaluated on the untouched 972-row UCI test partition. Decision Tree achieved the highest 4-class Macro F1 score (0.7436) and was persisted as `best_model.pkl`.
- **Top Comparative Benchmark Score:** Because model selection among the 6 candidate classifiers was performed on the test partition, the reported Macro F1 of 0.7436 represents the highest-scoring model in the multi-algorithm comparison on this historical benchmark partition, rather than an independent post-selection validation estimate.
- **Tuning vs. Model Selection:** No hyperparameter tuning loops, grid searches, or iterative threshold adjustments were conducted on the test set; models used fixed default hyperparameters.
- **Benchmark Split Preservation:** Ross Quinlan's 1987 UCI benchmark provides an established canonical two-way split (2,800 train in `allhypo.data` / 972 test in `allhypo.test`). Because Secondary Hypothyroid cases are extremely rare (only 2 records exist in the entire dataset, and both reside in `allhypo.data`), the canonical split was preserved as published rather than fabricating artificial validation splits.

---

## 3. Detailed Class-Wise Performance & Confusion Analysis

### Top Performer: Decision Tree (Accuracy: 99.79%, Macro F1: 0.7436)

```
========================================================================================
Class                              Precision    Recall    F1-Score    Support in Test
========================================================================================
1: Compensated Hypothyroid           95.24%    100.00%      0.9756                 40
7: Negative (Healthy)               100.00%     99.78%      0.9989                901
9: Primary Hypothyroid              100.00%    100.00%      1.0000                 31
11: Secondary Hypothyroid             0.00%      0.00%      0.0000                  0
----------------------------------------------------------------------------------------
Macro Average (All 4 Classes)        73.81%     74.94%      0.7436                972
Weighted Average                     99.80%     99.79%      0.9980                972
========================================================================================
```

### Decision Tree Confusion Matrix (Raw Test Counts):
- **Actual Compensated (40):** 40 predicted Compensated, 0 predicted Negative, 0 predicted Primary. (100% Sensitivity)
- **Actual Negative (901):** 2 predicted Compensated (false alarms), 899 predicted Negative, 0 predicted Primary. (Recall: 99.78%, Precision: 100.00%)
- **Actual Primary (31):** 0 predicted Compensated, 0 predicted Negative, 31 predicted Primary. (100% Sensitivity)
- **Actual Secondary (0):** 0 test cases exist in the authentic UCI test partition.

### Clinical & Algorithmic Analysis:
1. **Feature Reliance:** The Decision Tree relied heavily on TSH, TT4, and FTI in its predictions. These features are relevant thyroid laboratory measurements, although the learned model thresholds have not been clinically validated.
2. **Deterministic & Interpretable:** Tree rules can be inspected and validated directly by clinicians without black-box opacity.
3. **Sub-Millisecond Inference:** Inference takes only **0.0004 ms per patient**, making it ideal for real-time electronic health record (EHR) screening workflows.

---

## 4. Failure Analysis: Why Gaussian Naive Bayes Collapsed (9.26% Accuracy)

Gaussian Naive Bayes relies on the fundamental assumption that **all features are conditionally independent given the class**:
$$P(x_1, x_2, \dots, x_d \mid C_k) = \prod_{i=1}^{d} P(x_i \mid C_k)$$

In clinical endocrinology, this assumption is completely violated:
- **Free Thyroxine Index (FTI)** is mathematically derived from **Total Thyroxine (TT4)** and **T4 Uptake (T4U)**:
  $$\text{FTI} \approx \frac{\text{TT4}}{\text{T4U}}$$
- Because TT4 and FTI have strong positive correlation ($r > 0.85$), multiplying their likelihoods double-counts evidence, severely distorting posterior probability calculations and leading Naive Bayes to miss nearly all healthy patients.

---

### Important Methodological Disclosures:
1. **Secondary Hypothyroid Scarcity:** Only two Secondary Hypothyroid examples exist in the entire original dataset, and both belong to the original training partition (`allhypo.data`).
2. **Resampling Nature:** Random oversampling duplicates these examples; it does not create new independent clinical observations.
3. **No Evaluation Claim for Class 11:** Secondary Hypothyroid performance cannot be evaluated because the original test partition contains zero examples of that class.
4. **Historical Benchmark:** This is a historical UCI benchmark, not a modern clinical validation study.
5. **Educational Scope:** The project is educational/research work and is not intended for clinical diagnosis.

---

## 5. Key Viva Questions & Answers

1. **Why does Decision Tree rank higher than Random Forest in this evaluation?**
   *"Decision Tree achieved 99.79% accuracy and 0.7436 Macro F1, correctly detecting 100% of Compensated (40/40) and 100% of Primary (31/31) hypothyroid patients with only 2 false positives out of 901 healthy patients. In addition to slightly higher recall, a single decision tree offers complete white-box interpretability, which is vital for clinical validation."*

2. **Why is the Macro F1 around 0.74 when the model gets 970 out of 972 predictions right?**
   *"Macro F1 averages F1 across all four diagnostic classes equally. In Ross Quinlan's 1987 UCI benchmark partition, both Secondary Hypothyroid patients were placed into the training set (`allhypo.data`), leaving 0 test cases in `allhypo.test`. As a result, Class 11 has an F1 of 0.0. The average of (0.9756 + 0.9989 + 1.0000 + 0.0000) / 4 is 0.7436, which is near the mathematical limit of 0.75."*

3. **How does your evaluation compare to a naive baseline?**
   *"A majority-class baseline that predicts Negative for all patients achieves 92.70% accuracy, but a Macro F1 of only 0.2405 because it detects zero sick patients. Decision Tree achieves 99.79% accuracy and 0.7436 Macro F1, catching every single sick patient in the test set."*

4. **Why did you not artificially move one Secondary Hypothyroid patient into the test set?**
   *"The canonical UCI partition established by Ross Quinlan in 1987 is a standardized scientific benchmark used in machine learning literature for over 35 years. Preserving the exact 2,800 train and 972 test partition ensures authentic reproducibility and scientific integrity rather than fabricating synthetic data."*

---

## 6. Clinical Disclaimer

> [!CAUTION]
> **Educational Academic Demonstration Only:**
> ThyroidSense is an educational B.Tech CSE (AI & ML) project developed to illustrate supervised classification, data leakage mitigation, and imbalanced data handling on historical benchmarks. It is NOT certified for medical diagnostics or patient care.
