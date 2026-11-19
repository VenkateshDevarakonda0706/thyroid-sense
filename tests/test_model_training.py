"""Unit tests for the model training and evaluation modules."""

import os
import pytest
import numpy as np
import pandas as pd
import joblib

from src.data_preprocessing import preprocess_pipeline, CLASS_MAP
from src.model_training import get_model_definitions, train_single_model
from src.model_evaluation import evaluate_single_model


@pytest.fixture(scope="module")
def prepared_data():
    """Fixture providing scaled train and test data."""
    bundle = preprocess_pipeline(save_scaler=False)
    return {
        "X_train": bundle["X_train_scaled"],
        "y_train": bundle["y_train"].values,
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
    knn = models["K-Nearest Neighbors"]
    knn.fit(prepared_data["X_train"], prepared_data["y_train"])

    results = evaluate_single_model(
        "KNN Test", knn, prepared_data["X_test"], prepared_data["y_test"]
    )

    for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        score = results[metric]
        assert 0.0 <= score <= 1.0
        assert score > 0.85  # KNN performs at >95% on this dataset


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
        columns=[
            "age", "sex", "on thyroxine", "on antithyroid medication",
            "pregnant", "thyroid surgery", "lithium", "goitre", "tumor",
            "TSH", "T3", "TT4", "T4U", "FTI"
        ]
    )
    scaled_sample = scaler.transform(sample_df)
    prediction = best_model.predict(scaled_sample)

    assert len(prediction) == 1
    assert prediction[0] in [1, 7, 9, 11]
