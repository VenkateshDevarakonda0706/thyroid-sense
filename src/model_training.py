"""Model Training Module for ThyroidSense (Thyroid Disease Detection).

This module initializes, trains, and serializes the 6 traditional supervised
Machine Learning classifiers appropriate for a 2-1 semester B.Tech CSE AIML curriculum:
1. Logistic Regression (Baseline linear model)
2. Decision Tree Classifier (Interpretable tree model)
3. Random Forest Classifier (Ensemble bagging model)
4. K-Nearest Neighbors (KNN - k=3 from original 2-1 project)
5. Support Vector Classifier (SVC - RBF kernel with probability estimates)
6. Gaussian Naive Bayes (Probabilistic benchmark model)

Strict Academic Guidelines:
- Random seeds are fixed (random_state=42) for 100% reproducibility.
- Features are scaled strictly using parameters fitted on the training split.
- Models are persisted in models/ for inference and evaluation.
"""

import os
import time
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import joblib

from src.data_preprocessing import preprocess_pipeline

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def get_model_definitions(random_state: int = 42) -> Dict[str, Any]:
    """Define the supervised machine learning classifiers with hyperparameter specifications.

    Args:
        random_state: Random seed for deterministic reproducibility.

    Returns:
        Dictionary mapping model names to initialized Scikit-Learn estimators.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            random_state=random_state,
            criterion="gini",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=random_state,
            n_jobs=-1,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=3,
            metric="minkowski",
            p=2,  # Euclidean distance
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            probability=True,
            random_state=random_state,
        ),
        "Gaussian Naive Bayes": GaussianNB(),
    }


def train_single_model(
    model_name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    save_model: bool = True,
) -> Tuple[Any, float]:
    """Train an individual machine learning model and measure training latency.

    Args:
        model_name: Name of the model.
        model: Unfitted Scikit-Learn estimator.
        X_train: Standardized training feature matrix.
        y_train: Training labels array.
        save_model: Whether to save the trained model artifact to models/.

    Returns:
        Tuple of (fitted_model, training_time_seconds).
    """
    print(f"Training {model_name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    print(f"  -> Finished in {elapsed_time:.3f} seconds.")

    if save_model:
        slug = model_name.lower().replace(" ", "_").replace("-", "_")
        filepath = os.path.join(MODELS_DIR, f"{slug}.pkl")
        joblib.dump(model, filepath)
        print(f"  -> Saved artifact to {filepath}")

    return model, elapsed_time


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    save_models: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Train all defined classifiers on the standardized training data.

    Args:
        X_train: Standardized training feature matrix.
        y_train: Training labels array.
        random_state: Seed for reproducibility.
        save_models: Whether to serialize model artifacts.

    Returns:
        Dictionary mapping model names to fitted model instances and training metrics.
    """
    models_dict = get_model_definitions(random_state=random_state)
    trained_bundle = {}

    print("=" * 60)
    print("Beginning Training of 6 Machine Learning Classifiers")
    print("=" * 60)

    for name, model in models_dict.items():
        fitted_model, train_time = train_single_model(
            name, model, X_train, y_train, save_model=save_models
        )
        trained_bundle[name] = {
            "model": fitted_model,
            "train_time": train_time,
        }

    print("\nAll 6 models trained and persisted successfully!")
    return trained_bundle


if __name__ == "__main__":
    # Execute full pipeline
    data_bundle = preprocess_pipeline(save_scaler=True)
    X_train = data_bundle["X_train_scaled"]
    y_train = data_bundle["y_train"].values

    trained_bundle = train_all_models(X_train, y_train)
