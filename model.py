"""
model.py
--------
Handles training, evaluation, saving, and loading of the
Logistic Regression model for cardiovascular risk prediction.
"""

import os
import pickle
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from data_processing import load_data, clean_data, get_train_test_split, FEATURE_COLUMNS

logger = logging.getLogger(__name__)

_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(_DIR, "cardiorisk_model.pkl")
SCALER_PATH = os.path.join(_DIR, "cardiorisk_scaler.pkl")
DATA_PATH   = os.path.join(_DIR, "data", "cardio_sample.csv")


def train_model() -> tuple:
    """
    Load data, train a Logistic Regression model, evaluate it, and persist artifacts.

    Returns:
        Tuple of (trained model, fitted scaler, metrics dict).
    """
    logger.info("Starting model training pipeline...")

    df = load_data(DATA_PATH)
    df = clean_data(df)

    X_train, X_test, y_train, y_test, scaler = get_train_test_split(df)

    model = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    model.fit(X_train, y_train)
    logger.info("Model training complete")

    metrics = evaluate_model(model, X_test, y_test)

    save_artifacts(model, scaler)
    return model, scaler, metrics


def evaluate_model(model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate model performance and log results.

    Args:
        model: Trained LogisticRegression model.
        X_test: Scaled test features.
        y_test: True labels.

    Returns:
        Dict containing accuracy, f1_score, and confusion_matrix.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, zero_division=0)

    logger.info(f"Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{report}")

    return {
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "classification_report": report
    }


def save_artifacts(model: LogisticRegression, scaler) -> None:
    """
    Persist the trained model and scaler to disk using pickle.

    Args:
        model: Trained model.
        scaler: Fitted StandardScaler.
    """
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Artifacts saved: {MODEL_PATH}, {SCALER_PATH}")


def load_artifacts() -> tuple:
    """
    Load persisted model and scaler from disk.
    If artifacts don't exist, trigger training first.

    Returns:
        Tuple of (model, scaler).
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        logger.warning("Model artifacts not found. Training now...")
        model, scaler, _ = train_model()
        return model, scaler

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    logger.info("Model and scaler loaded from disk")
    return model, scaler


def predict_risk(model: LogisticRegression, X: np.ndarray) -> dict:
    """
    Run prediction and return class label with probability.

    Args:
        model: Trained LogisticRegression model.
        X: Scaled feature array (shape: 1 x n_features).

    Returns:
        Dict with 'risk_label' (0/1) and 'risk_probability' (float).
    """
    label = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    return {
        "risk_label": label,
        "risk_probability": round(probability, 4),
        "risk_level": _risk_level(probability)
    }


def _risk_level(probability: float) -> str:
    """Map probability to a human-readable risk level."""
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Moderate"
    else:
        return "High"
