"""
utils.py
--------
Input validation, logging configuration, and shared helper utilities.
"""

import logging
import sys
from typing import Tuple

# Expected fields with their types and valid ranges
PATIENT_SCHEMA = {
    "age":         {"type": (int, float), "min": 1,   "max": 120},
    "gender":      {"type": (int,),       "min": 1,   "max": 2},
    "height":      {"type": (int, float), "min": 100, "max": 250},
    "weight":      {"type": (int, float), "min": 30,  "max": 250},
    "ap_hi":       {"type": (int, float), "min": 50,  "max": 300},
    "ap_lo":       {"type": (int, float), "min": 30,  "max": 200},
    "cholesterol": {"type": (int,),       "min": 1,   "max": 3},
    "gluc":        {"type": (int,),       "min": 1,   "max": 3},
    "smoke":       {"type": (int,),       "min": 0,   "max": 1},
    "alco":        {"type": (int,),       "min": 0,   "max": 1},
    "active":      {"type": (int,),       "min": 0,   "max": 1},
}


def configure_logging(level: int = logging.INFO) -> None:
    """
    Set up root logger with a consistent format for console output.

    Args:
        level: Logging level (default: INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def validate_patient_data(data: dict) -> Tuple[bool, str]:
    """
    Validate incoming patient JSON against the expected schema.

    Args:
        data: Dict parsed from request JSON.

    Returns:
        Tuple of (is_valid: bool, error_message: str).
        error_message is empty string when valid.
    """
    if not isinstance(data, dict):
        return False, "Request body must be a JSON object"

    for field, rules in PATIENT_SCHEMA.items():
        # Check presence
        if field not in data:
            return False, f"Missing required field: '{field}'"

        value = data[field]

        # Check type
        if not isinstance(value, rules["type"]):
            expected = " or ".join(t.__name__ for t in rules["type"])
            return False, f"Field '{field}' must be of type {expected}, got {type(value).__name__}"

        # Check range
        if not (rules["min"] <= value <= rules["max"]):
            return False, (
                f"Field '{field}' must be between {rules['min']} and {rules['max']}, "
                f"got {value}"
            )

    return True, ""


def build_error_response(message: str, status_code: int) -> dict:
    """
    Construct a standardised error response payload.

    Args:
        message: Human-readable error description.
        status_code: HTTP status code.

    Returns:
        Dict with 'error' and 'status' keys.
    """
    return {"error": message, "status": status_code}


def build_success_response(data: dict) -> dict:
    """
    Wrap prediction results in a standard success envelope.

    Args:
        data: Prediction result dict.

    Returns:
        Dict with 'status' and 'data' keys.
    """
    return {"status": 200, "data": data}
