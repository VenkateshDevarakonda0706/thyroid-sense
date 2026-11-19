# Model Evaluation and Benchmark Report: ThyroidSense

**Project:** ThyroidSense: Thyroid Disease Detection Using Machine Learning  
**Project Identifier:** `ThyroidSense`  
**Author:** Venkatesh Devarakonda  
**Evaluation Set:** 4,000 Stratified Holdout Test Records (1,000 samples per class)  
**Execution Environment:** Isolated Python 3.11 Virtual Environment  

---

## 1. Metric Definitions

In medical diagnostic Machine Learning, accuracy alone can be deeply misleading. We assess models across multiple complementary dimensions:

### A. Accuracy
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
Represents the proportion of all predictions that were correct. Useful on balanced datasets, but insufficient on its own for critical medical applications.

### B. Precision (Positive Predictive Value)
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
Answers: *"Of all patients predicted to have condition $X$, what fraction actually had it?"*  
High precision minimizes False Positives (preventing healthy patients from undergoing invasive, stressful medical interventions).

### C. Recall (Sensitivity / True Positive Rate)
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
Answers: *"Of all patients who genuinely had condition $X$, what fraction did the model successfully detect?"*  
In healthcare screening, **Recall is the most critical metric**, because a False Negative means a patient with severe thyroid dysfunction goes untreated.

### D. F1-Score (Harmonic Mean)
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
Balances precision and recall into a single metric.

### E. Macro vs. Weighted Averaging
- **Macro Average:** Computes the metric independently for each class and calculates the unweighted mean:
  $$\text{Macro F1} = \frac{1}{C} \sum_{c=1}^{C} \text{F1}_c$$
  Every class is treated with equal importance regardless of size.
- **Weighted Average:** Weights the score of each class by its support (number of true instances).

---

## 2. Actual Measured Benchmark Results

All models were evaluated on the exact same holdout test set of 4,000 patient records (1,000 samples per class). No metrics are fabricated:

| Rank | Model | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Training Time (s) | Inference Latency (ms/sample) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Random Forest** | **98.58%** | **0.9862** | **0.9858** | **0.9857** | **0.9862** | **0.9858** | **0.9857** | 0.406 s | 0.0118 ms |
| 🥈 | **Decision Tree** | **98.22%** | **0.9827** | **0.9822** | **0.9822** | **0.9827** | **0.9822** | **0.9822** | 0.019 s | 0.0008 ms |
| 🥉 | **K-Nearest Neighbors (k=3)** | **97.55%** | **0.9768** | **0.9755** | **0.9753** | **0.9768** | **0.9755** | **0.9753** | 0.041 s | 0.0970 ms |
| 4 | **Support Vector Machine (RBF)** | **94.17%** | **0.9424** | **0.9417** | **0.9417** | **0.9424** | **0.9417** | **0.9417** | 16.342 s | 0.3375 ms |
| 5 | **Logistic Regression** | **88.25%** | **0.8821** | **0.8825** | **0.8821** | **0.8821** | **0.8825** | **0.8821** | 0.301 s | 0.0008 ms |
| 6 | **Gaussian Naive Bayes** | **64.98%** | **0.7475** | **0.6498** | **0.5725** | **0.7475** | **0.6498** | **0.5725** | 0.028 s | 0.0002 ms |

---

## 3. Detailed Class-Wise Performance Analysis

### A. Top Performer: Random Forest (Accuracy: 98.58%, F1: 0.9857)
- **Class 1 (Compensated Hypothyroid):** Precision: 0.98, Recall: 0.98, F1: 0.98 (Support: 1,000)
- **Class 7 (Negative / Healthy):** Precision: 0.98, Recall: 0.98, F1: 0.98 (Support: 1,000)
- **Class 9 (Primary Hypothyroid):** Precision: 0.99, Recall: 0.98, F1: 0.99 (Support: 1,000)
- **Class 11 (Secondary Hypothyroid):** Precision: 0.99, Recall: 1.00, F1: 1.00 (Support: 1,000)

*Interpretation:* Random Forest achieved nearly flawless separation across all 4 thyroid conditions. Its ensemble of 100 decision trees effectively modeled non-linear interactions between blood hormones (`TSH`, `FTI`, `TT4`) and binary clinical flags (`on thyroxine`, `goitre`).

### B. Original Project Models Reproduction:
1. **K-Nearest Neighbors ($k=3$):**
   - Test Accuracy: **97.55%** (Macro F1: **0.9753**).
   - This aligns with the original student notebook `Thyroid_(1).ipynb` (which reported ~97.1% accuracy).
   - Demonstrates that patients with identical thyroid disorders form tight clusters in standardized Euclidean feature space.
2. **Support Vector Machine (RBF Kernel):**
   - Test Accuracy: **94.17%** (Macro F1: **0.9417**).
   - Directly reproduces the original student notebook's SVM performance (~94.7% accuracy).
   - However, training latency was high (16.3 seconds) due to quadratic kernel matrix computations.
3. **Gaussian Naive Bayes:**
   - Test Accuracy: **64.98%** (Macro F1: **0.5725**).
   - Notice that while Precision is moderate (0.7475), Recall drops severely (0.6498) because the model assumes blood hormones are independent Gaussian distributions, which is biologically invalid for correlated endocrine feedback loops.

---

## 4. Key Takeaways for Academic Viva

1. **Why does Random Forest beat Logistic Regression by >10%?**
   - Logistic Regression assumes linear decision boundaries between classes: $\mathbf{w}^T \mathbf{x} + b = 0$.
   - Thyroid diagnosis relies on non-linear thresholds and ratios (e.g., if TSH is high and FTI is low $\rightarrow$ Primary Hypothyroid; if TSH is high and FTI is normal $\rightarrow$ Compensated Hypothyroid). Tree ensembles naturally capture these non-linear thresholds through step-wise partitioning.
2. **Why does Naive Bayes perform poorly?**
   - `FTI` is directly derived as $\text{TT4} / \text{T4U}$. Thus, `FTI`, `TT4`, and `T4U` have strong multicollinearity, violating the Naive Bayes feature independence assumption.
3. **What is the trade-off between Decision Tree and Random Forest?**
   - Decision Tree trains faster (0.019s vs 0.406s) and produces a single white-box decision path, but has slightly lower accuracy (98.22% vs 98.58%) due to higher model variance.

---

## 5. Clinical Disclaimer & Limitations

> [!CAUTION]
> **Educational Scope vs. Clinical Deployment:**
> This machine learning pipeline was developed strictly as an educational undergraduate engineering project. In clinical medicine:
> 1. Diagnostic decisions require comprehensive patient examinations, ultrasound imaging, biopsy results, and physician judgment.
> 2. Laboratory reference ranges vary between hospitals and assay technologies.
> 3. False negatives can lead to life-threatening complications (e.g., myxedema coma).
> 
> Therefore, this software must **NEVER** be used as a substitute for certified medical diagnosis.

