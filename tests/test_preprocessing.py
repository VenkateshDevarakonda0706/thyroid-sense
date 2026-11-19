"""Unit tests for the data preprocessing module."""

import os
import pytest
import numpy as np
import pandas as pd
from src.data_preprocessing import (
    load_processed_dataset,
    split_data,
    scale_features,
    clean_and_prepare_raw_data,
    FEATURE_COLUMNS,
    CLASS_MAP,
)


def test_load_processed_dataset():
    """Verify that the curated modeling dataset loads correctly with expected shape."""
    df = load_processed_dataset()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (20000, 15)
    assert "class" in df.columns
    for feature in FEATURE_COLUMNS:
        assert feature in df.columns


def test_no_missing_values():
    """Verify that the processed dataset contains zero null or NaN values."""
    df = load_processed_dataset()
    assert df.isnull().sum().sum() == 0


def test_stratified_split_proportions():
    """Verify that stratified splitting preserves equal 25% class distributions."""
    df = load_processed_dataset()
    X_train, X_test, y_train, y_test = split_data(
        df, target_column="class", test_size=0.20, random_state=42
    )

    assert len(X_train) == 16000
    assert len(X_test) == 4000
    assert len(y_train) == 16000
    assert len(y_test) == 4000

    # Each of the 4 classes must have exactly 4,000 train samples and 1,000 test samples
    for class_id in [1, 7, 9, 11]:
        assert (y_train == class_id).sum() == 4000
        assert (y_test == class_id).sum() == 1000


def test_scaling_prevents_data_leakage():
    """Verify that feature scaling computes mean and variance ONLY from X_train."""
    df = load_processed_dataset()
    X_train, X_test, _, _ = split_data(df, test_size=0.20, random_state=42)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, scaler_save_path=None)

    # Scaler mean must match X_train mean exactly
    expected_means = X_train.mean(axis=0).values
    np.testing.assert_allclose(scaler.mean_, expected_means, rtol=1e-5)

    # Standardized training data must have mean ~ 0 and std ~ 1
    assert np.all(np.abs(X_train_scaled.mean(axis=0)) < 1e-6)
    assert np.all(np.abs(X_train_scaled.std(axis=0) - 1.0) < 1e-2)


def test_clean_and_prepare_raw_data_removes_admin_columns():
    """Verify that raw data cleaner strips unneeded administrative columns."""
    sample_raw = pd.DataFrame({
        "patient_id": [1, 2],
        "referral source": ["SVHC", "SVI"],
        "dataset_name": ["allrep", "allrep"],
        "TSH measured": ["t", "t"],
        "age": [45, 60],
        "sex": ["F", "M"],
        "on thyroxine": ["f", "t"],
        "on antithyroid medication": ["f", "f"],
        "pregnant": ["f", "f"],
        "thyroid surgery": ["f", "f"],
        "lithium": ["f", "f"],
        "goitre": ["f", "f"],
        "tumor": ["f", "f"],
        "TSH": [1.5, 4.2],
        "T3": [2.1, 1.8],
        "TT4": [110, 85],
        "T4U": [1.02, 0.95],
        "FTI": [108, 89],
        "class": [7, 1],
    })

    cleaned = clean_and_prepare_raw_data(sample_raw)
    assert "patient_id" not in cleaned.columns
    assert "referral source" not in cleaned.columns
    assert "TSH measured" not in cleaned.columns
    assert cleaned["sex"].tolist() == [0, 1]
    assert cleaned["on thyroxine"].tolist() == [0, 1]
