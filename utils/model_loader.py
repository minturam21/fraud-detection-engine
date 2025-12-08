import joblib
from pathlib import Path
from functools import lru_cache


MODEL_PATH = Path("models/model.joblib")


class ModelNotFoundError(Exception):
    pass


@lru_cache(maxsize=1)
def get_model():
    """
    Load the model once and cache it.
    """
    if not MODEL_PATH.exists():
        raise ModelNotFoundError(f"Model file not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    return model
