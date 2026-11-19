"""Data Preprocessing Module for ThyroidSense (Thyroid Disease Detection).

This module provides clear, reusable, beginner-friendly functions to:
1. Load raw or processed thyroid disease datasets.
2. Clean raw features, impute missing values, and encode categorical variables.
3. Separate features and target variables.
4. Perform stratified train-test splitting to maintain class distribution.
5. Standardize numerical features using StandardScaler (fitted ONLY on training data to prevent data leakage).
6. Save and load preprocessing artifacts (scaler, feature names, label encodings).
"""

import os
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Default paths relative to project root
DEFAULT_RAW_PATH = os.path.join("data", "raw", "thyroid_raw.csv")
DEFAULT_PROCESSED_PATH = os.path.join("data", "processed", "thyroid_processed.csv")
DEFAULT_SCALER_PATH = os.path.join("models", "scaler.pkl")

# Target class label human-readable mapping
CLASS_MAP: Dict[int, str] = {
    1: "Compensated Hypothyroid",
    7: "Negative (Healthy)",
    9: "Primary Hypothyroid",
    11: "Secondary Hypothyroid",
}

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


def load_raw_dataset(filepath: str = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """Load the raw UCI thyroid disease dataset.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        pd.DataFrame containing the raw dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    return df


def load_processed_dataset(filepath: str = DEFAULT_PROCESSED_PATH) -> pd.DataFrame:
    """Load the cleaned and curated modeling dataset.

    Args:
        filepath: Path to the processed CSV file.

    Returns:
        pd.DataFrame containing the processed dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    return df


def clean_and_prepare_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw 31-column UCI dataset into the 15-column modeling schema.

    Operations performed:
    1. Drop non-predictive administrative columns: patient_id, referral source, dataset_name.
    2. Drop redundant boolean measurement flags: TSH measured, TBG measured, etc.
    3. Replace '?' with NaN and cast features to numeric.
    4. Impute missing values in laboratory test columns using column median.
    5. Filter rows to the 4 target classes: Compensated Hypothyroid, Negative,
       Primary Hypothyroid, Secondary Hypothyroid.
    6. Ensure clean binary encoding for symptom and demographic columns.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned pd.DataFrame ready for splitting and modeling.
    """
    cleaned = df.copy()

    # Standardize column names
    cleaned.columns = [c.strip() for c in cleaned.columns]

    # Drop administrative and redundant measurement flag columns if present
    cols_to_drop = [
        "patient_id", "referral source", "dataset_name",
        "TSH measured", "T3 measured", "TT4 measured",
        "T4U measured", "FTI measured", "TBG measured", "TBG",
        "query on thyroxine", "sick", "I131 treatment",
        "query hypothyroid", "query hyperthyroid", "psych", "hypopituitary"
    ]
    present_drops = [c for c in cols_to_drop if c in cleaned.columns]
    cleaned = cleaned.drop(columns=present_drops, errors="ignore")

    # Replace '?' with NaN
    cleaned = cleaned.replace("?", np.nan)

    # Convert binary columns
    binary_mapping = {"f": 0, "t": 1, "F": 0, "M": 1, "false": 0, "true": 1}
    for col in cleaned.columns:
        if col != "class":
            cleaned[col] = cleaned[col].replace(binary_mapping)
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Impute missing continuous laboratory measurements with median
    continuous_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
    for col in continuous_cols:
        if col in cleaned.columns:
            median_val = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median_val)

    # Impute binary features with mode (0)
    for col in cleaned.columns:
        if col != "class":
            cleaned[col] = cleaned[col].fillna(0)

    return cleaned


def split_data(
    df: pd.DataFrame,
    target_column: str = "class",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into training and testing partitions using stratified sampling.

    Stratified sampling ensures the exact class proportions are maintained
    in both training and testing sets, preventing class distribution shift.

    Args:
        df: DataFrame containing features and target.
        target_column: Name of the target label column.
        test_size: Fraction of samples assigned to the test partition (default: 0.20).
        random_state: Seed for reproducible random sampling.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}")

    X = df.drop(columns=[target_column])
    y = df[target_column].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_save_path: Optional[str] = DEFAULT_SCALER_PATH,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Standardize features by removing the mean and scaling to unit variance.

    CRITICAL LEAKAGE PREVENTION:
    The scaler is fitted STRICTLY on X_train only (`scaler.fit_transform(X_train)`).
    The fitted scaler is then applied to X_test using `scaler.transform(X_test)`.
    X_test data is NEVER used to compute mean or standard deviation.

    Args:
        X_train: Training features DataFrame.
        X_test: Test features DataFrame.
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


def preprocess_pipeline(
    filepath: str = DEFAULT_PROCESSED_PATH,
    test_size: float = 0.20,
    random_state: int = 42,
    save_scaler: bool = True,
) -> Dict[str, object]:
    """Execute the complete end-to-end data preprocessing pipeline.

    Args:
        filepath: Path to the processed dataset CSV.
        test_size: Fraction of test split.
        random_state: Random state seed.
        save_scaler: Whether to persist the fitted scaler to models/scaler.pkl.

    Returns:
        Dictionary containing X_train, X_test, y_train, y_test, scaled arrays,
        fitted scaler, and feature names.
    """
    df = load_processed_dataset(filepath)
    X_train, X_test, y_train, y_test = split_data(
        df, target_column="class", test_size=test_size, random_state=random_state
    )

    scaler_path = DEFAULT_SCALER_PATH if save_scaler else None
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, scaler_save_path=scaler_path
    )

    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "feature_names": list(X_train.columns),
        "class_mapping": CLASS_MAP,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Executing Thyroid Disease Data Preprocessing Pipeline")
    print("=" * 60)

    data_bundle = preprocess_pipeline()
    print(f"Features count: {len(data_bundle['feature_names'])}")
    print(f"Features: {data_bundle['feature_names']}")
    print(f"X_train shape: {data_bundle['X_train_scaled'].shape}")
    print(f"X_test shape:  {data_bundle['X_test_scaled'].shape}")
    print(f"y_train class distribution:\n{data_bundle['y_train'].value_counts()}")
    print(f"y_test class distribution:\n{data_bundle['y_test'].value_counts()}")
    print("\nData preprocessing pipeline executed successfully!")
