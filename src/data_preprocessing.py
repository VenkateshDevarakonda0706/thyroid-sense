"""Data Preprocessing Module for ThyroidSense (Thyroid Disease Detection).

This module provides clear, reusable, beginner-friendly functions implementing
the authentic UCI allhypo dataset pipeline with strict prevention of data leakage.

Authentic UCI allhypo Single-Task Workflow Order:
1. Parse original allhypo.data (2,800 train records) and allhypo.test (972 test records).
2. Clean feature columns, convert binary/continuous attributes, and map 4 target classes:
   - 1: Compensated Hypothyroid
   - 7: Negative (Healthy)
   - 9: Primary Hypothyroid
   - 11: Secondary Hypothyroid
3. Extract original Garavan Institute patient IDs into metadata, strictly excluding them from features.
4. Separate features (X) and target (y) for train and test partitions.
5. Compute missing-value imputation statistics (medians for continuous labs, 0.0/mode for symptoms)
   STRICTLY using training features only.
6. Impute missing values in both training and testing partitions using those training statistics.
7. Fit StandardScaler on the imputed original training features ONLY.
8. Transform testing features using the fitted training scaler.
9. Apply RandomOverSampler ONLY to the scaled training features and training labels (test is untouched).
10. Persist artifacts (scaler, imputation dictionary, feature names, class mapping).
"""

import os
from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import joblib

# Default paths relative to project root
DEFAULT_ALLHYPO_DATA_PATH = os.path.join("data", "raw", "uci_thyroid_original", "allhypo.data")
DEFAULT_ALLHYPO_TEST_PATH = os.path.join("data", "raw", "uci_thyroid_original", "allhypo.test")
DEFAULT_ALLHYPO_NAMES_PATH = os.path.join("data", "raw", "uci_thyroid_original", "allhypo.names")

DEFAULT_RAW_PATH = os.path.join("data", "raw", "thyroid_raw.csv")
DEFAULT_PROCESSED_PATH = os.path.join("data", "processed", "allhypo_processed.csv")
DEFAULT_SCALER_PATH = os.path.join("models", "scaler.pkl")
DEFAULT_IMPUTATION_PATH = os.path.join("models", "imputation_values.json")

# Target class label mappings
CLASS_MAP: Dict[int, str] = {
    1: "Compensated Hypothyroid",
    7: "Negative",
    9: "Primary Hypothyroid",
    11: "Secondary Hypothyroid",
}

NAME_TO_CLASS: Dict[str, int] = {
    "compensated hypothyroid": 1,
    "negative": 7,
    "primary hypothyroid": 9,
    "secondary hypothyroid": 11,
}

# The 29 attributes in the original allhypo.names file
ALLHYPO_RAW_COLUMNS: List[str] = [
    "age",
    "sex",
    "on thyroxine",
    "query on thyroxine",
    "on antithyroid medication",
    "sick",
    "pregnant",
    "thyroid surgery",
    "I131 treatment",
    "query hypothyroid",
    "query hyperthyroid",
    "lithium",
    "goitre",
    "tumor",
    "hypopituitary",
    "psych",
    "TSH measured",
    "TSH",
    "T3 measured",
    "T3",
    "TT4 measured",
    "TT4",
    "T4U measured",
    "T4U",
    "FTI measured",
    "FTI",
    "TBG measured",
    "TBG",
    "referral source",
]

# The 14 clinical features used for modeling
FEATURE_COLUMNS: List[str] = [
    "age",
    "sex",
    "on thyroxine",
    "on antithyroid medication",
    "pregnant",
    "thyroid surgery",
    "lithium",
    "goitre",
    "tumor",
    "TSH",
    "T3",
    "TT4",
    "T4U",
    "FTI",
]

CONTINUOUS_COLUMNS: List[str] = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]


def parse_allhypo_file(filepath: str) -> pd.DataFrame:
    """Parse an authentic UCI allhypo data/test file according to its schema.

    Each record contains 29 comma-separated attributes followed by the class label,
    with an optional Garavan patient ID after a vertical bar ('|').

    Args:
        filepath: Path to the allhypo.data or allhypo.test file.

    Returns:
        pd.DataFrame containing the 29 raw attributes, 'raw_target', and 'patient_id'.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Source file not found at {filepath}")

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue

            parts = line.split("|")
            row_str = parts[0].strip()
            patient_id = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip().isdigit() else np.nan

            fields = [x.strip() for x in row_str.split(",")]
            if len(fields) != len(ALLHYPO_RAW_COLUMNS) + 1:
                raise ValueError(
                    f"Unexpected field count in {filepath}: expected {len(ALLHYPO_RAW_COLUMNS) + 1}, got {len(fields)}"
                )

            attrs = fields[:-1]
            raw_target = fields[-1].rstrip(".")

            rec = dict(zip(ALLHYPO_RAW_COLUMNS, attrs))
            rec["raw_target"] = raw_target
            rec["patient_id"] = patient_id
            records.append(rec)

    return pd.DataFrame(records)


def clean_allhypo_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Clean allhypo attributes into the 14-feature schema and integer target series.

    - Replaces '?' with np.nan.
    - Encodes binary features (f/t, F/M) to 0 and 1.
    - Casts continuous columns to float.
    - Extracts patient IDs separately (never used as model features).
    - Maps target strings to integer codes (1, 7, 9, 11).

    Args:
        df: Raw DataFrame returned by parse_allhypo_file.

    Returns:
        Tuple of (X_features, y_target, patient_ids).
    """
    cleaned = df.copy()
    cleaned = cleaned.replace("?", np.nan)

    binary_mapping = {"f": 0, "t": 1, "F": 0, "M": 1, "false": 0, "true": 1}
    for col in FEATURE_COLUMNS:
        cleaned[col] = cleaned[col].replace(binary_mapping)
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    X = cleaned[FEATURE_COLUMNS].copy()

    # Target mapping
    target_clean = cleaned["raw_target"].astype(str).str.strip().str.rstrip(".")
    if not set(target_clean.unique()).issubset(NAME_TO_CLASS.keys()):
        unexpected = set(target_clean.unique()) - set(NAME_TO_CLASS.keys())
        raise ValueError(f"Unexpected target labels found: {unexpected}")

    y = target_clean.map(NAME_TO_CLASS).astype(int)
    patient_ids = cleaned["patient_id"].copy()

    return X, y, patient_ids


def load_allhypo_dataset(
    data_path: str = DEFAULT_ALLHYPO_DATA_PATH,
    test_path: str = DEFAULT_ALLHYPO_TEST_PATH,
) -> Dict[str, Any]:
    """Load, parse, and validate the authentic UCI allhypo dataset.

    Validates:
    - Train record count: exactly 2,800
    - Test record count: exactly 972
    - Combined record count: exactly 3,772
    - Train class counts: 2,580 Negative, 154 Compensated, 64 Primary, 2 Secondary
    - Test class counts: 901 Negative, 40 Compensated, 31 Primary, 0 Secondary
    - Zero patient ID overlap between train and test partitions

    Returns:
        Dictionary containing X_train_raw, X_test_raw, y_train, y_test, and patient IDs.
    """
    df_train_raw = parse_allhypo_file(data_path)
    df_test_raw = parse_allhypo_file(test_path)

    # Validate row counts
    if len(df_train_raw) != 2800:
        raise ValueError(f"Expected exactly 2800 training records, found {len(df_train_raw)}")
    if len(df_test_raw) != 972:
        raise ValueError(f"Expected exactly 972 testing records, found {len(df_test_raw)}")

    X_train_raw, y_train, pids_train = clean_allhypo_features(df_train_raw)
    X_test_raw, y_test, pids_test = clean_allhypo_features(df_test_raw)

    # Validate train class counts
    train_counts = y_train.value_counts().to_dict()
    expected_train = {7: 2580, 1: 154, 9: 64, 11: 2}
    if train_counts != expected_train:
        raise ValueError(f"Unexpected train class counts: {train_counts} vs expected {expected_train}")

    # Validate test class counts
    test_counts = y_test.value_counts().to_dict()
    expected_test = {7: 901, 1: 40, 9: 31}  # class 11 has 0 support
    if test_counts != expected_test:
        raise ValueError(f"Unexpected test class counts: {test_counts} vs expected {expected_test}")

    # Validate zero patient ID overlap
    train_pid_set = set(pids_train.dropna())
    test_pid_set = set(pids_test.dropna())
    pid_overlap = train_pid_set.intersection(test_pid_set)
    if len(pid_overlap) > 0:
        raise ValueError(f"Data leakage detected! {len(pid_overlap)} patient IDs overlap between train and test!")

    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "pids_train": pids_train,
        "pids_test": pids_test,
        "feature_names": FEATURE_COLUMNS,
        "class_mapping": CLASS_MAP,
    }


def impute_missing_values(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    continuous_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Impute missing feature values using statistics learned STRICTLY from X_train.

    CRITICAL LEAKAGE PREVENTION & METHODOLOGICAL PRINCIPLE:
    - Continuous feature medians are calculated exclusively on X_train.
    - Binary/categorical feature modes are calculated exclusively on X_train.
    - Missing binary values are NOT assumed to represent absence; instead, they are imputed
      with the statistical mode calculated strictly from the training partition.
    - X_test is transformed using the training-derived medians and modes.
    - No information from X_test leaks into the imputation parameters.

    Args:
        X_train: Training features DataFrame.
        X_test: Test features DataFrame.
        continuous_cols: List of continuous laboratory columns to impute with median.

    Returns:
        Tuple of (X_train_imputed, X_test_imputed, imputation_values_dict).
    """
    if continuous_cols is None:
        continuous_cols = CONTINUOUS_COLUMNS

    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()
    imputation_values: Dict[str, float] = {}

    # 1. Impute continuous columns using training set median
    for col in continuous_cols:
        if col in X_train_clean.columns:
            valid_series = X_train_clean[col].dropna()
            median_val = float(valid_series.median()) if len(valid_series) > 0 else 0.0
            imputation_values[col] = median_val
            X_train_clean[col] = X_train_clean[col].fillna(median_val)
            X_test_clean[col] = X_test_clean[col].fillna(median_val)

    # 2. Impute binary/categorical features using training set statistical mode
    for col in X_train_clean.columns:
        if col not in continuous_cols:
            valid_series = X_train_clean[col].dropna()
            mode_series = valid_series.mode()
            mode_val = float(mode_series.iloc[0]) if len(mode_series) > 0 else 0.0
            imputation_values[col] = mode_val
            X_train_clean[col] = X_train_clean[col].fillna(mode_val)
            X_test_clean[col] = X_test_clean[col].fillna(mode_val)

    return X_train_clean, X_test_clean, imputation_values


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_save_path: Optional[str] = DEFAULT_SCALER_PATH,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Standardize features by fitting StandardScaler strictly on original training data.

    STATISTICAL RATIONALE:
    StandardScaler is fitted on the imputed ORIGINAL training features (X_train)
    BEFORE oversampling. This ensures that:
    1. Scaling parameters (mean and std) reflect the true natural population distribution.
    2. Over-replicating rare samples does not artificially compress or skew feature variance.
    3. The test partition is transformed using genuine empirical population metrics.

    Args:
        X_train: Imputed training features DataFrame.
        X_test: Imputed test features DataFrame.
        scaler_save_path: Optional path to save the fitted StandardScaler artifact.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, fitted_scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)

    return X_train_scaled, X_test_scaled, scaler


def oversample_train_data(
    X_train: np.ndarray,
    y_train: pd.Series,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply RandomOverSampler to the scaled training partition ONLY.

    CRITICAL LEAKAGE PREVENTION:
    - Oversampling is applied strictly to X_train and y_train.
    - The test partition is NEVER oversampled and remains untouched.
    - Note on Secondary Hypothyroid (Class 11): Oversampling replicates the 2 training
      samples to match majority class size (2,580 rows). This balances gradient updates
      during training, but does NOT create new independent biological observations.

    Args:
        X_train: Scaled training feature matrix.
        y_train: Training class labels.
        random_state: Random state for deterministic resampling.

    Returns:
        Tuple of (X_train_resampled, y_train_resampled).
    """
    ros = RandomOverSampler(random_state=random_state)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled


def preprocess_pipeline(
    data_path: str = DEFAULT_ALLHYPO_DATA_PATH,
    test_path: str = DEFAULT_ALLHYPO_TEST_PATH,
    random_state: int = 42,
    save_scaler: bool = True,
    save_processed_csv: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute the complete end-to-end authentic UCI allhypo preprocessing workflow.

    Primary Sequence:
    1. Load original training (2,800) and testing (972) partitions separately.
    2. Clean features and extract target labels (Classes 1, 7, 9, 11).
    3. Fit missing-value imputation statistics using training features only.
    4. Transform training and testing features using those same statistics.
    5. Fit StandardScaler on the imputed original training features only.
    6. Transform testing features using the fitted scaler.
    7. Apply RandomOverSampler only to the scaled training features and training labels.
    8. Persist artifacts and return bundle for model training and evaluation.

    Args:
        data_path: Path to allhypo.data.
        test_path: Path to allhypo.test.
        random_state: Random state for deterministic reproducibility.
        save_scaler: Whether to persist the fitted scaler to models/scaler.pkl.
        save_processed_csv: Whether to save cleaned processed CSVs to data/processed/.

    Returns:
        Dictionary containing all data partitions, scaler, imputation statistics, and metadata.
    """
    # 1. Load authentic allhypo data
    loaded = load_allhypo_dataset(data_path=data_path, test_path=test_path)
    X_train_raw = loaded["X_train_raw"]
    X_test_raw = loaded["X_test_raw"]
    y_train = loaded["y_train"]
    y_test = loaded["y_test"]

    # 2. Impute using statistics learned STRICTLY from X_train
    X_train_imp, X_test_imp, imputation_values = impute_missing_values(
        X_train_raw, X_test_raw
    )

    # 3. Fit scaler on imputed original training data ONLY (before oversampling)
    scaler_path = DEFAULT_SCALER_PATH if save_scaler else None
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train_imp, X_test_imp, scaler_save_path=scaler_path
    )

    # Persist fitted imputation values when save_scaler is True
    if save_scaler:
        import json
        os.makedirs(os.path.dirname(DEFAULT_IMPUTATION_PATH), exist_ok=True)
        with open(DEFAULT_IMPUTATION_PATH, "w", encoding="utf-8") as f:
            json.dump(imputation_values, f, indent=2)

    # 4. Oversample training data ONLY
    X_train_res, y_train_res = oversample_train_data(
        X_train_scaled, y_train, random_state=random_state
    )

    # 5. Optionally save processed clean CSVs for transparency and traceability
    if save_processed_csv:
        os.makedirs(os.path.dirname(DEFAULT_PROCESSED_PATH), exist_ok=True)
        # Create consolidated processed dataset with partition column
        df_train_proc = pd.DataFrame(X_train_scaled, columns=FEATURE_COLUMNS)
        df_train_proc["class"] = y_train.values
        df_train_proc["partition"] = "train"
        df_train_proc["patient_id"] = loaded["pids_train"].values

        df_test_proc = pd.DataFrame(X_test_scaled, columns=FEATURE_COLUMNS)
        df_test_proc["class"] = y_test.values
        df_test_proc["partition"] = "test"
        df_test_proc["patient_id"] = loaded["pids_test"].values

        df_all_proc = pd.concat([df_train_proc, df_test_proc], axis=0, ignore_index=True)
        df_all_proc.to_csv(DEFAULT_PROCESSED_PATH, index=False)

    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train_imputed": X_train_imp,
        "X_test_imputed": X_test_imp,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "X_train_resampled": X_train_res,
        "y_train_resampled": y_train_res,
        "y_train": y_train,
        "y_test": y_test,
        "pids_train": loaded["pids_train"],
        "pids_test": loaded["pids_test"],
        "scaler": scaler,
        "imputation_values": imputation_values,
        "feature_names": FEATURE_COLUMNS,
        "class_mapping": CLASS_MAP,
        "pipeline_type": "authentic_uci_allhypo",
    }


def preprocess_pipeline_leakage_safe(**kwargs: Any) -> Dict[str, Any]:
    """Alias for preprocess_pipeline to preserve backwards compatibility."""
    return preprocess_pipeline(**kwargs)


def preprocess_pipeline_legacy(
    filepath: str = os.path.join("data", "processed", "thyroid_processed.csv"),
    test_size: float = 0.20,
    random_state: int = 42,
    save_scaler: bool = False,
) -> Dict[str, Any]:
    """Execute the legacy/preliminary preprocessing workflow on the historical 20,000-sample balanced dataset.

    NOTE: Preserved for historical/academic comparison to demonstrate the effect
    of pre-split oversampling in preliminary semester 2-1 experiments.
    """
    if not os.path.exists(filepath):
        # Fallback if historical file moved
        filepath = os.path.join("archive", "legacy_csvs", "thyroidpro.csv")
    df = pd.read_csv(filepath)
    if "class" not in df.columns:
        raise KeyError("Target column 'class' not found in legacy dataset")

    from sklearn.model_selection import train_test_split
    X = df.drop(columns=["class"])
    y = df["class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "class_mapping": CLASS_MAP,
        "pipeline_type": "legacy_preliminary_balanced",
    }
