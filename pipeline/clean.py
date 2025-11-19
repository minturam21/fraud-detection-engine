import pandas as pd

def drop_missing_critical(df):
    df = df.dropna(subset=["user_id", "timestamp", "event-type"])
    return df

def normalize_timestamps(df):
    """
    Convert timestamp column to UTC and ensure valid datetime format.
    Invalid timestamps become NaT and will be dropped later.
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df

def remove_duplicates(df):
    """
    Remove exact duplicates and near-duplicates.
    Near duplicates: same user_id + timestamp + event_type.
    """
    # Remove exact duplicates
    df = df.drop_duplicates()

    # Remove near-duplicates
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["user_id", "timestamp", "event_type"],keep="first")

    return df

def fix_small_time_skew(df):
    """
    If event order is slightly incorrect (< 2 minutes difference),
    adjust timestamps to maintain logical sequence.
    """

    df = df.sort_values("timestamp")

    # shift small backward jumps
    df["timestamp_shift"] = df["timestamp"].shift(1)

    mask = (df["timestamp"] < df["timestamp_shift"]) & \
           ((df["timestamp_shift"]-df["timestamp"]).dt.total_seconds()<120)
    df.loc[mask, "timestamp"] = df.loc[mask,"timestamp_shift"]
    df = df.drop(colums = ["timestamp_shift"])
    return df

def drop_impossible_sequences(df):
    """
     Drop rows where event order is impossible:
      - transaction before login by a large time gap
      - reset password before login
      - label timestamp before transaction
      - large backward time jumps
    """
    df = df.sort_values("timestamp")

    # backward time jump > 5 minutes (300 seconds)
    df["pre_timestamp"] = df["timestamp"].shift(1)

    mask_backward_jump = ((df["timestamp"])<(df["pre_timestamp"])) & \
                         ((df["pre_timestamp"]-df["timestamp"]).dt.total_seconds()< 300)
    
    # drop such row
    df = df[~mask_backward_jump]
    df = df.drop(colums = ["pre_timestamp"])
    return df


def remove_invalid_amounts(df):
    """
    Remove invalid transaction rows (amount <= 0).
    Do NOT remove login/reset/OTP events.
    """
    mask = (df["event_type"] == "transaction")
    df = df[~(mask & (df["amount"] <=0))]

    return df

def validate_labels(df):
    """
    Ensure fraud_label_timestamp is AFTER the transaction_timestamp.
    If label timestamp is earlier, it's leakage or corrupted data.
    Such rows must be removed.
    """
    # Only check rows with a fraud label timestamp present
    mask_label = df["fraud_label_timestamp"].notna()

    # Valid rows: label_time > transaction_time
    mask_valid = df["fraud_label_timestamp"] > df["transaction_timestamp"]

    # Keep rows where:
    #  - no label (legit)
    #  - OR label is valid
    df = df[~mask_label | mask_valid]
    return df

def data_cleaned(df):
    """
    Run all cleaning steps in the correct order.
    Returns a fully cleaned DataFrame ready for feature engineering.
    """
    df = drop_missing_critical(df)
    df = normalize_timestamps(df)
    df = remove_duplicates(df)
    df = fix_small_time_skew(df)
    df = drop_impossible_sequences(df)
    df = remove_invalid_amounts(df)
    df = validate_labels(df)
    return df   




