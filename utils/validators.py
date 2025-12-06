import pandas as pd
from typing import List

REQUIRED_COLUMNS = [
    "timestamp",
    "user_id",
    "event_type",
    "device_id",
    "ip",
    "receiver_id",
    "amount",
    "lat",
    "lon",
]


def validate_scoring_input(df: pd.DataFrame) -> None:
    """
    Validate input for scoring:
    - required columns exist
    - no entirely missing rows
    """

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns for scoring: {missing}")

    if df.empty:
        raise ValueError("Input dataframe for scoring is empty.")
