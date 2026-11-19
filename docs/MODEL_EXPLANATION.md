# Machine Learning Models Explanation: Thyroid Disease Detection

**Project:** Thyroid Disease Detection Using Machine Learning  
**Author:** Venkatesh Devarakonda  
**Course Context:** B.Tech CSE (AI & ML) — Semester 2-1 Academic Curriculum  

---

## 1. Overview of Supervised Learning Algorithms

In this project, we implement and benchmark six fundamental supervised classification algorithms. These algorithms were selected because they represent core traditional Machine Learning paradigms taught in undergraduate Computer Science and AI & ML programs:

1. **Linear Model:** Logistic Regression
2. **Tree-Based Model:** Decision Tree Classifier
3. **Ensemble Bagging Model:** Random Forest Classifier
4. **Instance-Based / Non-Parametric Model:** K-Nearest Neighbors (KNN)
5. **Maximum Margin Kernel Model:** Support Vector Classifier (SVC)
6. **Probabilistic Generative Model:** Gaussian Naive Bayes

---

## 2. Detailed Algorithm Explanations

### 1. Logistic Regression (Multinomial Classification)
- **Concept:** Despite its name, Logistic Regression is a linear classification model. For binary classification, it estimates the log-odds of a positive outcome using the standard sigmoid function. For multiclass problems (such as our 4 thyroid classes), the model generalizes using the **Softmax function**:
  $$P(y = c \mid \mathbf{x}) = \frac{e^{\mathbf{w}_c^T \mathbf{x} + b_c}}{\sum_{j=1}^{C} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$
- **Optimization:** Optimized using cross-entropy loss (log-loss) with the L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) quasi-Newton optimization solver.
- **Why it is suitable:** Serves as the gold standard linear baseline. If linear hyperplanes can separate thyroid conditions, Logistic Regression will succeed with minimal compute.
- **Strengths:** Fast to train, highly interpretable weights, outputs calibrated probabilities.
- **Limitations:** Cannot capture complex non-linear interactions between blood hormones (e.g., the ratio $\text{FTI} = \frac{\text{TT4}}{\text{T4U}}$) without explicit polynomial feature engineering.

---

### 2. Decision Tree Classifier
- **Concept:** A non-parametric supervised model that partitions feature space into orthogonal rectangular regions through recursive binary splits.
- **Split Criterion (Gini Impurity):**
  $$I_G(t) = 1 - \sum_{i=1}^{C} p_i^2$$
  At each node, the feature and threshold that maximize Information Gain (decrease in Gini impurity) are selected:
  $$\Delta I_G = I_G(parent) - \left( \frac{N_{left}}{N} I_G(left) + \frac{N_{right}}{N} I_G(right) \right)$$
- **Why it is suitable:** Directly reflects clinical decision guidelines used by endocrinologists (e.g., "If $\text{TSH} > 6.0$ and $\text{FTI} < 70$, then Primary Hypothyroid").
- **Hyperparameter Tuning:** `max_depth=8` was constrained to prevent uncontrolled tree growth and memorization of training instances.
- **Strengths:** White-box interpretability, handles non-linear boundaries effortlessly, invariant to monotonic feature scaling.
- **Limitations:** High variance; a slight perturbation in training data can alter the tree structure dramatically.

---

### 3. Random Forest Classifier
- **Concept:** An ensemble learning technique based on **Bootstrap Aggregation (Bagging)** and random feature subspace projection. It builds an ensemble of $B$ decorrelated decision trees ($B=100$) and aggregates their predictions by majority voting:
  $$\hat{y} = \text{mode}\left( \{ T_b(\mathbf{x}) \}_{b=1}^{B} \right)$$
- **Why it is suitable:** Thyroid disease diagnosis depends on combinations of symptoms and lab tests. Random Forest smooths out individual tree errors, dramatically lowering prediction variance while preserving high accuracy.
- **Hyperparameters:** `n_estimators=100`, `max_depth=12`, `random_state=42`.
- **Strengths:** State-of-the-art accuracy on tabular medical datasets, robust to noise, provides feature importance rankings, rarely overfits when tree depth is regularized.
- **Limitations:** Black-box ensemble compared to a single decision tree; slightly higher inference latency.

---

### 4. K-Nearest Neighbors (KNN)
- **Concept:** An instance-based ("lazy") learning algorithm. It stores the training dataset without fitting an explicit parametric equation. Given a query patient vector $\mathbf{x}$, it calculates the Euclidean distance to all $N$ training points:
  $$d(\mathbf{x}, \mathbf{x}_i) = \sqrt{\sum_{j=1}^{d} (x_j - x_{i,j})^2}$$
  The model identifies the $k$ nearest neighbors ($k=3$) and assigns the majority class.
- **Why it is suitable:** Patients exhibiting similar clinical symptoms and identical hormone blood panels cluster closely together in Euclidean space.
- **Original Student Configuration:** Inherited directly from the original semester 2-1 project (`k=3`, `metric='minkowski'`, `p=2`).
- **Strengths:** Zero training time, non-linear decision surface, highly intuitive.
- **Limitations:** Prediction time scales linearly with dataset size ($O(N \cdot d)$); extremely sensitive to unstandardized feature scales (mitigated by `StandardScaler`).

---

### 5. Support Vector Classifier (SVC)
- **Concept:** Finds the optimal separating hyperplane that maximizes the geometric margin between classes. Because thyroid classes are not linearly separable in raw 14-dimensional space, SVC uses the **Radial Basis Function (RBF) Kernel** to map features into an infinite-dimensional Hilbert space:
  $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left( -\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2 \right)$$
- **Dual Optimization Problem:**
  $$\max_{\alpha} \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{N} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j) \quad \text{s.t.} \quad 0 \le \alpha_i \le C, \sum_{i=1}^{N} \alpha_i y_i = 0$$
- **Why it is suitable:** Effective in high-dimensional continuous feature spaces. The original 2-1 project utilized SVC with `probability=True` for soft-margin classification.
- **Strengths:** Robust against overfitting in high dimensions, effective non-linear separation.
- **Limitations:** High computational complexity ($O(N^2)$ to $O(N^3)$), leading to longer training times (16.3 seconds) compared to trees (<0.5 seconds).

---

### 6. Gaussian Naive Bayes
- **Concept:** A probabilistic generative classifier grounded in **Bayes' Theorem**:
  $$P(y = c \mid \mathbf{x}) \propto P(y = c) \prod_{j=1}^{d} P(x_j \mid y = c)$$
  Assumes that continuous features conditioned on class $c$ follow a normal Gaussian distribution:
  $$P(x_j \mid y = c) = \frac{1}{\sqrt{2\pi \sigma_{c,j}^2}} \exp\left( -\frac{(x_j - \mu_{c,j})^2}{2\sigma_{c,j}^2} \right)$$
- **The "Naive" Assumption:** Features are conditionally independent given the class label ($x_i \perp x_j \mid y$).
- **Why it is suitable:** Provides an ultra-lightweight probabilistic benchmark.
- **Strengths:** Near-instantaneous training ($0.028$ seconds), very low memory footprint.
- **Why it fails on this dataset (64.98% Accuracy):**
  1. The conditional independence assumption is heavily violated: blood markers like `TT4`, `T4U`, and `FTI` are biologically co-dependent ($\text{FTI} \approx \text{TT4}/\text{T4U}$).
  2. Severe skewness in hormone values violates the Gaussian bell-curve assumption.
