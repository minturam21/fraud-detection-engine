import json
import joblib
import os
from typing import Any, Dict

from utils.logger import get_logger

logger = get_logger("loader")


def load_json(path: str) -> Dict[str, Any]:
    """
    Load JSON with clear error handling.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r") as f:
        return json.load(f)


def load_model(model_path: str):
    """
    Load trained model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    logger.info(f"Loading model from {model_path}")
    return joblib.load(model_path)


def load_thresholds(threshold_path: str) -> Dict[str, float]:
    """
    Load thresholds from JSON.
    Ensures required keys exist.
    """
    thresholds = load_json(threshold_path)

    required = {"low", "medium", "high"}
    if not required.issubset(thresholds.keys()):
        raise ValueError(
            f"Threshold file missing required keys {required}. Got: {thresholds.keys()}"
        )

    return thresholds
