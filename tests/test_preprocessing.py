"""Unit tests for the authentic UCI allhypo data preprocessing module."""

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_preprocessing import (
    parse_allhypo_file,
    clean_allhypo_features,
    load_allhypo_dataset,
    impute_missing_values,
    scale_features,
    oversample_train_data,
    preprocess_pipeline,
    FEATURE_COLUMNS,
    CONTINUOUS_COLUMNS,
    CLASS_MAP,
    NAME_TO_CLASS,
    DEFAULT_ALLHYPO_DATA_PATH,
    DEFAULT_ALLHYPO_TEST_PATH,
)


def test_parse_allhypo_files_shape_and_provenance():
    """Verify correct parsing of both source files from uci_thyroid_original."""
    df_train = parse_allhypo_file(DEFAULT_ALLHYPO_DATA_PATH)
    df_test = parse_allhypo_file(DEFAULT_ALLHYPO_TEST_PATH)

    assert len(df_train) == 2800, f"Expected 2800 train records, got {len(df_train)}"
    assert len(df_test) == 972, f"Expected 972 test records, got {len(df_test)}"
    assert len(df_train) + len(df_test) == 3772

    # Check patient ID extracted
    assert "patient_id" in df_train.columns
    assert "patient_id" in df_test.columns
    assert not df_train["patient_id"].isnull().all()


def test_allhypo_class_counts_and_target_mapping():
    """Verify class distributions in original partitions match expected counts."""
    loaded = load_allhypo_dataset()
    y_train = loaded["y_train"]
    y_test = loaded["y_test"]

    # Training class counts: 2580 Negative, 154 Compensated, 64 Primary, 2 Secondary
    train_counts = y_train.value_counts().to_dict()
    assert train_counts[7] == 2580
    assert train_counts[1] == 154
    assert train_counts[9] == 64
    assert train_counts[11] == 2

    # Testing class counts: 901 Negative, 40 Compensated, 31 Primary, 0 Secondary
    test_counts = y_test.value_counts().to_dict()
    assert test_counts[7] == 901
    assert test_counts[1] == 40
    assert test_counts[9] == 31
    assert 11 not in test_counts or test_counts[11] == 0


def test_patient_id_excluded_from_model_features():
    """Verify patient IDs and administrative flags are excluded from model features."""
    loaded = load_allhypo_dataset()
    X_train = loaded["X_train_raw"]
    X_test = loaded["X_test_raw"]

    assert "patient_id" not in X_train.columns
    assert "patient_id" not in X_test.columns
    assert "referral source" not in X_train.columns
    assert list(X_train.columns) == FEATURE_COLUMNS
    assert list(X_test.columns) == FEATURE_COLUMNS
    assert len(X_train.columns) == 14


def test_no_patient_id_overlap_between_train_and_test():
    """Verify that train and test partitions contain disjoint patient populations."""
    loaded = load_allhypo_dataset()
    train_pids = set(loaded["pids_train"].dropna())
    test_pids = set(loaded["pids_test"].dropna())

    overlap = train_pids.intersection(test_pids)
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping patient IDs between train and test!"


def test_imputation_statistics_learned_strictly_from_training_features():
    """Verify that imputation values are derived strictly from X_train and ignore X_test."""
    loaded = load_allhypo_dataset()
    X_tr = loaded["X_train_raw"]
    X_te = loaded["X_test_raw"]

    expected_tsh_median = float(X_tr["TSH"].dropna().median())
    expected_sex_mode = float(X_tr["sex"].dropna().mode().iloc[0])

    # Contaminate X_test with an extreme continuous outlier and inverted binary values
    X_te_polluted = X_te.copy()
    X_te_polluted["TSH"] = 9999.0
    X_te_polluted["sex"] = 1.0  # Force test mode to Male (1.0)

    X_tr_imp, X_te_imp, imp_vals = impute_missing_values(X_tr, X_te_polluted)

    # Imputation parameters must match pure X_train statistics
    assert imp_vals["TSH"] == expected_tsh_median
    assert imp_vals["TSH"] < 10.0
    assert imp_vals["sex"] == expected_sex_mode
    assert imp_vals["sex"] == 0.0  # Training mode is Female (0.0)


def test_mode_imputation_for_binary_features_ignores_test_pollution():
    """Verify that binary feature mode imputation uses training mode only and ignores test values."""
    train_data = pd.DataFrame({
        "age": [50.0, 60.0, 70.0],
        "sex": [0.0, 0.0, 1.0],  # Mode in train is 0.0
        "TSH": [1.0, 2.0, 3.0],
        "TT4": [100.0, 105.0, 110.0],
        "FTI": [100.0, 105.0, 110.0],
        "T3": [2.0, 2.1, 2.2],
        "T4U": [0.9, 1.0, 1.1],
    })
    test_data = pd.DataFrame({
        "age": [np.nan, 65.0],
        "sex": [np.nan, 1.0],  # In test, only 1.0 is observed!
        "TSH": [np.nan, 2.5],
        "TT4": [102.0, np.nan],
        "FTI": [102.0, np.nan],
        "T3": [2.0, np.nan],
        "T4U": [0.95, np.nan],
    })

    tr_imp, te_imp, imp_vals = impute_missing_values(
        train_data, test_data, continuous_cols=["age", "TSH", "TT4", "FTI", "T3", "T4U"]
    )

    # sex mode in train is 0.0, so missing sex in test must be imputed with 0.0 (train mode), NOT 1.0 (test mode)
    assert imp_vals["sex"] == 0.0
    assert te_imp["sex"].iloc[0] == 0.0
    # age median in train is 60.0
    assert imp_vals["age"] == 60.0
    assert te_imp["age"].iloc[0] == 60.0


def test_imputation_values_persisted_to_disk():
    """Verify that imputation values dictionary is persisted to models/imputation_values.json."""
    bundle = preprocess_pipeline(save_scaler=True, save_processed_csv=False)
    imp_path = os.path.join("models", "imputation_values.json")
    assert os.path.exists(imp_path)

    import json
    with open(imp_path, "r", encoding="utf-8") as f:
        saved_imp = json.load(f)

    assert saved_imp == bundle["imputation_values"]
    assert "TSH" in saved_imp
    assert "sex" in saved_imp
    assert saved_imp["sex"] == 0.0


def test_scaler_fitted_only_on_original_training_data():
    """Verify StandardScaler is fitted on imputed original X_train before oversampling."""
    bundle = preprocess_pipeline(save_scaler=False, save_processed_csv=False)
    X_tr_imp = bundle["X_train_imputed"]
    scaler = bundle["scaler"]

    expected_means = X_tr_imp.mean(axis=0).values
    np.testing.assert_allclose(scaler.mean_, expected_means, rtol=1e-5)


def test_oversampling_affects_training_data_only():
    """Verify RandomOverSampler balances training split while test set remains 972 untouched."""
    bundle = preprocess_pipeline(save_scaler=False, save_processed_csv=False)
    X_tr_res = bundle["X_train_resampled"]
    y_tr_res = bundle["y_train_resampled"]
    X_te_scaled = bundle["X_test_scaled"]
    y_te = bundle["y_test"]

    # Training set is balanced to 4 * 2,580 = 10,320 rows
    assert len(X_tr_res) == 10320
    assert len(y_tr_res) == 10320
    for c in [1, 7, 9, 11]:
        assert (y_tr_res == c).sum() == 2580

    # Test set remains exactly 972 records
    assert len(X_te_scaled) == 972
    assert len(y_te) == 972


def test_feature_names_and_ordering_uniform():
    """Verify uniform ordering of the 14 features across the entire pipeline."""
    bundle = preprocess_pipeline(save_scaler=False, save_processed_csv=False)
    assert bundle["feature_names"] == FEATURE_COLUMNS
    assert len(bundle["feature_names"]) == 14
    assert list(bundle["scaler"].feature_names_in_) == FEATURE_COLUMNS


def test_all_four_classes_handled():
    """Verify that all four target classes (1, 7, 9, 11) are mapped and handled."""
    assert set(CLASS_MAP.keys()) == {1, 7, 9, 11}
    assert set(NAME_TO_CLASS.values()) == {1, 7, 9, 11}
