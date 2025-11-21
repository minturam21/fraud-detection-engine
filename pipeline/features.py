import pandas as pd
import numpy as np

def failed_login_velocity(df):

    # Sort by time for rolling windows
    df = df.sort_values("timestamp")

    # Only failed login rows
    fails = df[df["event_type"] == "login_fail"].copy()

    # Group by user and apply rolling 10-minute window
    fails["failed_login_10min"] = (
        fails
        .groupby("user_id")["timestamp"]
        .rolling("10min")
        .count()
        .reset_index(level=0, drop=True)
    )

    # Merge back into main dataframe
    df = df.merge(
        fails[["failed_login_10min"]],
        left_index=True,
        right_index=True,
        how="left"
    )

    # Fill non-fail rows with 0
    df["failed_login_10min"] = df["failed_login_10min"].fillna(0)

    return df

def new_device_flag(df):
    """Uses groupby + cumcount logic to detect if a device is new for the user."""

    # Sort to ensure time order
    df = df.sort_values("timestamp")

    # Identify first time a user uses a device
    df["new_device"] = (
        df.groupby(["user_id", "device_id"]).cumcount ==0
    ).astype(int)

    return df
def new_ip_flag(df):
     # Ensure sorting by time (important for correctness)
     df = df.sort_values("timestamp")

     # Check if it's the first occurrence of this IP for the user
     df["new_ips"] = (
         df.groupby(["user_id", "ip"]).cumcount() ==0
     ).astype(int)

     return df

def amount_deviation(df):
    df = df.sort_values("timestamp")

    # Only consider transaction rows for this feature
    is_txn = df["event_type"] == "transaction"

    # Compute historical (past only) average amount per user
    df["user_avg_amount"] = (
        df[is_txn] 
        .groupby("user_id")["amount"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level = 0, drop = True)
    )
    # Compute deviation (difference from historical avg)
    df["amount_deviation"] = abs(df["amount"] - df["user_avg_amount"])

    # First transaction for each user gets NaN → fill with 0
    df["amount_deviation"] = df["amount_deviation"].fillna(0)

    # Cleanup temporary column
    df = df.dropna(colums = ["user_avg_amount"])
    return df

def distance_from_last_location():
    df = df.sort_values("timestamp")

    # Shift latitude/longitude to get previous coordinates per user
    df["pre_lat"] = df.groupby("user_id")["lat"].shift(1)
    df["pre_lon"] = df.groupby("user_id")["lon"].shift(1)

    # Haversine formula (vectorized)
    R = 6371 #km

    lat1 = np.radians(df["pre_lat"])
    lon1 = np.radians(df["pre_lon"])
    lat2 = np.radians(df["lat"])
    lon2 = np.radians(df["lon"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat/2)**2 +
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    )
    df["dist_from_last_loc"] = 2 * R * np.arcsin(np.sqrt(a))

    # First row of each user will be NaN → fill with 0
    df["dist_from_last_loc"] = df["dist_from_last_loc"].fillna(0)
    
     # Drop temporary columns
    df = df.drop(columns=["prev_lat", "prev_lon"])

    return df