"""Unit tests for the model training and evaluation modules."""

import os
import pytest
import numpy as np
import pandas as pd
import joblib

from src.data_preprocessing import preprocess_pipeline, CLASS_MAP, FEATURE_COLUMNS
from src.model_training import get_model_definitions, train_single_model
from src.model_evaluation import evaluate_single_model


@pytest.fixture(scope="module")
def prepared_data():
    """Fixture providing scaled train and test data from the authentic allhypo pipeline."""
    bundle = preprocess_pipeline(save_scaler=False, save_processed_csv=False)
    return {
        "X_train": bundle["X_train_resampled"],
        "y_train": bundle["y_train_resampled"],
        "X_test": bundle["X_test_scaled"],
        "y_test": bundle["y_test"].values,
    }


def test_model_definitions_presence():
    """Verify that all 6 required algorithms are defined."""
    models = get_model_definitions()
    expected = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "K-Nearest Neighbors",
        "Support Vector Machine",
        "Gaussian Naive Bayes",
    ]
    for m in expected:
        assert m in models


def test_fast_model_train_and_predict(prepared_data):
    """Test training and predicting with Decision Tree on the dataset."""
    models = get_model_definitions()
    dt = models["Decision Tree"]

    fitted_model, duration = train_single_model(
        "Decision Tree Test",
        dt,
        prepared_data["X_train"],
        prepared_data["y_train"],
        save_model=False,
    )
    assert duration > 0.0

    preds = fitted_model.predict(prepared_data["X_test"])
    assert len(preds) == len(prepared_data["y_test"])
    assert set(preds).issubset({1, 7, 9, 11})


def test_single_model_evaluation_metrics(prepared_data):
    """Verify evaluation metric calculations produce valid scores in [0, 1]."""
    models = get_model_definitions()
    dt = models["Decision Tree"]
    dt.fit(prepared_data["X_train"], prepared_data["y_train"])

    results = evaluate_single_model(
        "Decision Tree Test", dt, prepared_data["X_test"], prepared_data["y_test"]
    )

    for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        score = results[metric]
        assert 0.0 <= score <= 1.0
    assert results["accuracy"] > 0.90


def test_four_class_reports_handle_zero_test_support(prepared_data):
    """Verify evaluation report cleanly handles Class 11 having zero test support without crashing."""
    models = get_model_definitions()
    dt = models["Decision Tree"]
    dt.fit(prepared_data["X_train"], prepared_data["y_train"])

    results = evaluate_single_model(
        "Decision Tree Test", dt, prepared_data["X_test"], prepared_data["y_test"]
    )
    rep = results["classification_report"]

    assert "Secondary Hypothyroid" in rep
    assert rep["Secondary Hypothyroid"]["support"] == 0
    assert rep["Secondary Hypothyroid"]["f1-score"] == 0.0


def test_best_model_artifact_inference():
    """Verify that the persisted best model artifact and scaler can predict on a sample."""
    best_model_path = os.path.join("models", "best_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")

    assert os.path.exists(best_model_path), f"Missing {best_model_path}"
    assert os.path.exists(scaler_path), f"Missing {scaler_path}"

    best_model = joblib.load(best_model_path)
    scaler = joblib.load(scaler_path)

    # 14-feature sample patient vector: Severe Hypothyroid profile
    sample_df = pd.DataFrame(
        [[28, 0, 0, 0, 0, 0, 0, 0, 1, 35.0, 2.3, 42.0, 1.01, 41.0]],
        columns=FEATURE_COLUMNS,
    )
    scaled_sample = scaler.transform(sample_df)
    prediction = best_model.predict(scaled_sample)

    assert len(prediction) == 1
    assert prediction[0] in [1, 7, 9, 11]


def test_all_six_model_artifacts_loadable():
    """Verify that all six benchmark model artifacts exist and load with joblib."""
    model_files = [
        "decision_tree.pkl",
        "gaussian_naive_bayes.pkl",
        "k_nearest_neighbors.pkl",
        "logistic_regression.pkl",
        "random_forest.pkl",
        "support_vector_machine.pkl",
    ]
    for mf in model_files:
        path = os.path.join("models", mf)
        assert os.path.exists(path), f"Missing model file {path}"
        loaded = joblib.load(path)
        assert hasattr(loaded, "predict"), f"{mf} is missing predict method"


def test_batch_inference_pipeline():
    """Verify that batch inference functions across multiple records."""
    best_model = joblib.load(os.path.join("models", "best_model.pkl"))
    scaler = joblib.load(os.path.join("models", "scaler.pkl"))

    # Generate 5 diverse test samples
    samples = pd.DataFrame([
        [28, 0, 0, 0, 0, 0, 0, 0, 1, 35.0, 2.3, 42.0, 1.01, 41.0],
        [45, 1, 0, 0, 0, 0, 0, 0, 0, 1.5, 2.0, 100.0, 0.95, 105.0],
        [60, 0, 1, 0, 0, 0, 0, 0, 0, 8.2, 1.5, 70.0, 1.10, 63.6],
        [32, 0, 0, 0, 1, 0, 0, 0, 0, 0.2, 3.5, 140.0, 0.85, 164.7],
        [50, 1, 0, 0, 0, 1, 0, 0, 0, 22.0, 1.1, 35.0, 1.20, 29.2],
    ], columns=FEATURE_COLUMNS)

    scaled_samples = scaler.transform(samples)
    preds = best_model.predict(scaled_samples)

    assert len(preds) == 5
    for p in preds:
        assert p in [1, 7, 9, 11]
