"""
data_processing.py
------------------
Handles loading, cleaning, and preprocessing of cardiovascular dataset.
Provides feature scaling and train/test splitting utilities.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import os

logger = logging.getLogger(__name__)

# Feature columns used for training and prediction
FEATURE_COLUMNS = [
    "age", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active"
]
TARGET_COLUMN = "cardio"


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load cardiovascular dataset from a CSV file.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded dataset with {len(df)} records from {filepath}")

    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing outliers and invalid values.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    original_len = len(df)

    # Drop rows with nulls
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    # Remove physiologically impossible blood pressure values
    df = df[(df["ap_hi"] > 0) & (df["ap_hi"] < 300)]
    df = df[(df["ap_lo"] > 0) & (df["ap_lo"] < 200)]

    # Remove impossible height/weight values
    df = df[(df["height"] > 100) & (df["height"] < 250)]
    df = df[(df["weight"] > 30) & (df["weight"] < 250)]

    # Age sanity check (stored in years)
    df = df[(df["age"] > 0) & (df["age"] < 120)]

    logger.info(f"Cleaned data: {original_len} → {len(df)} records")
    return df.reset_index(drop=True)


def preprocess(df: pd.DataFrame, scaler: StandardScaler = None):
    """
    Extract features/target and apply standard scaling.

    Args:
        df: Cleaned DataFrame.
        scaler: Optional pre-fitted StandardScaler. If None, a new one is fitted.

    Returns:
        Tuple of (X_scaled, y, scaler).
    """
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info("Fitted new StandardScaler on training data")
    else:
        X_scaled = scaler.transform(X)
        logger.info("Applied existing StandardScaler")

    return X_scaled, y, scaler


def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into train and test sets, then preprocess.

    Args:
        df: Cleaned DataFrame.
        test_size: Fraction of data for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler).
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[TARGET_COLUMN]
    )

    X_train, y_train, scaler = preprocess(train_df)
    X_test, y_test, _ = preprocess(test_df, scaler=scaler)

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler


def prepare_patient_input(patient_data: dict, scaler: StandardScaler) -> np.ndarray:
    """
    Transform a single patient dict into a scaled feature array for prediction.

    Args:
        patient_data: Dict with patient feature values.
        scaler: Fitted StandardScaler from training.

    Returns:
        2D numpy array ready for model.predict().
    """
    row = [patient_data[col] for col in FEATURE_COLUMNS]
    X = np.array(row).reshape(1, -1)
    return scaler.transform(X)
