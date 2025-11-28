import os
import json
import joblib
from datetime import datetime

def save_model(model, save_path):
    joblib.dump(model, save_path)
    return save_path

def save_thresholds(thresholds, save_path):
    """
    Save thresholds dictionary as JSON.
    """
    with open(save_path, "w") as f:
        json.dump(thresholds, f, indent=4)
    return save_path


def save_metadata(metadata, save_path):
    """
    Save metadata (model type, metrics, training date) as JSON.
    """
    with open(save_path, "w") as f:
        json.dump(metadata, f, indent=4)
    return save_path


def save_all(model, thresholds, metadata, model_dir="data/models/"):
    """
    Save:
      - trained model
      - thresholds
      - metadata
    Ensures directory exists and returns file paths.
    """

    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.pkl")
    thresholds_path = os.path.join(model_dir, "thresholds.json")
    metadata_path = os.path.join(model_dir, "metadata.json")

    # Add timestamp in metadata
    metadata["saved_at"] = datetime.utcnow().isoformat()

    save_model(model, model_path)
    save_thresholds(thresholds, thresholds_path)
    save_metadata(metadata, metadata_path)

    return {
        "model": model_path,
        "thresholds": thresholds_path,
        "metadata": metadata_path
    }