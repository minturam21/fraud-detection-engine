import pandas as pd
import numpy as np

# Count failed login attempts in the last 10 minutes per user
def failed_login_velocity(df):
    df = df.sort_values("timestamp")

    fails = df[df["event_type"] == "login_fail"].copy()
    fails = fails.set_index("timestamp")  

    fails["failed_login_10min"] = (
        fails.groupby("user_id")["event_type"]
        .rolling("10min")
        .count()
    )

    fails = fails.reset_index()

    df = df.merge(
        fails[["timestamp", "user_id", "failed_login_10min"]],
        on=["timestamp", "user_id"],
        how="left"
    )

    df["failed_login_10min"] = df["failed_login_10min"].fillna(0)
    return df


# Detect if this device is new for the user
def new_device_flag(df):
    df = df.sort_values("timestamp")
    df["new_device"] = (df.groupby(["user_id", "device_id"]).cumcount() == 0).astype(int)
    return df


# Detect if this IP is new for the user
def new_ip_flag(df):
    df = df.sort_values("timestamp")
    df["new_ips"] = (df.groupby(["user_id", "ip"]).cumcount() == 0).astype(int)
    return df

# Compare transaction amount against user's historical average
def amount_deviation(df):
    df = df.sort_values("timestamp")

    is_txn = df["event_type"] == "transaction"

    df["user_avg_amount"] = (
        df[is_txn]
        .groupby("user_id")["amount"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    df["amount_deviation"] = abs(df["amount"] - df["user_avg_amount"])
    df["amount_deviation"] = df["amount_deviation"].fillna(0)

    df = df.dropna(subset=["user_avg_amount"])
    return df


# Distance between this event's location and the user's previous one
def distance_from_last_location(df):
    df = df.sort_values("timestamp")

    df["pre_lat"] = df.groupby("user_id")["lat"].shift(1)
    df["pre_lon"] = df.groupby("user_id")["lon"].shift(1)

    R = 6371  # earth radius in km

    lat1 = np.radians(df["pre_lat"])
    lon1 = np.radians(df["pre_lon"])
    lat2 = np.radians(df["lat"])
    lon2 = np.radians(df["lon"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    df["dist_from_last_loc"] = 2 * R * np.arcsin(np.sqrt(a))
    df["dist_from_last_loc"] = df["dist_from_last_loc"].fillna(0)

    df = df.drop(columns=["pre_lat", "pre_lon"])
    return df

# Time since last login, reset, and transaction for each user
def time_gaps(df):
    df = df.sort_values("timestamp")

    # Last login
    last_login = (
        df[df["event_type"] == "login"]
        .groupby("user_id")["timestamp"]
        .shift(1)
    )
    df["last_login_time"] = last_login
    df["time_since_last_login"] = (
        (df["timestamp"] - df["last_login_time"]).dt.total_seconds().fillna(0)
    )

    # Last password reset
    last_reset = (
        df[df["event_type"] == "reset_password"]
        .groupby("user_id")["timestamp"]
        .shift(1)
    )
    df["last_time_reset"] = last_reset
    df["time_since_last_reset"] = (
        (df["timestamp"] - df["last_time_reset"]).dt.total_seconds().fillna(0)
    )

    # Last transaction
    last_txn = (
        df[df["event_type"] == "transaction"]
        .groupby("user_id")["timestamp"]
        .shift(1)
    )
    df["last_txn_time"] = last_txn
    df["time_since_last_txn"] = (
        (df["timestamp"] - df["last_txn_time"]).dt.total_seconds().fillna(0)
    )

    df = df.drop(columns=["last_login_time", "last_time_reset", "last_txn_time"])
    return df


# Velocity features: login frequency, txn frequency, IP/device change speed
def velocity_features(df):
    df = df.sort_values("timestamp")

    # Login velocity
    login_events = df[df["event_type"] == "login"].copy().set_index("timestamp")
    login_events["login_velocity_10min"] = (
        login_events.groupby("user_id")["event_type"].rolling("10min").count()
    )
    login_events = login_events.reset_index()

    # Transaction velocity
    txn_events = df[df["event_type"] == "transaction"].copy().set_index("timestamp")
    txn_events["txn_velocity_10min"] = (
        txn_events.groupby("user_id")["event_type"].rolling("10min").count()
    )
    txn_events = txn_events.reset_index()

    # IP change
    df["prev_ip"] = df.groupby("user_id")["ip"].shift(1)
    df["ip_change"] = (df["ip"] != df["prev_ip"]).astype(int)

    df_ip = df.set_index("timestamp")
    df_ip["ip_change_velocity_10min"] = (
        df_ip.groupby("user_id")["ip_change"].rolling("10min").sum()
    )
    df_ip = df_ip.reset_index()

    # Device change
    df["prev_device"] = df.groupby("user_id")["device_id"].shift(1)
    df["device_change"] = (df["device_id"] != df["prev_device"]).astype(int)

    df_dev = df.set_index("timestamp")
    df_dev["device_change_velocity_10min"] = (
        df_dev.groupby("user_id")["device_change"].rolling("10min").sum()
    )
    df_dev = df_dev.reset_index()

    # Merge everything
    df = df.merge(
        login_events[["timestamp", "user_id", "login_velocity_10min"]],
        on=["timestamp", "user_id"], how="left"
    ).merge(
        txn_events[["timestamp", "user_id", "txn_velocity_10min"]],
        on=["timestamp", "user_id"], how="left"
    ).merge(
        df_ip[["timestamp", "user_id", "ip_change_velocity_10min"]],
        on=["timestamp", "user_id"], how="left"
    ).merge(
        df_dev[["timestamp", "user_id", "device_change_velocity_10min"]],
        on=["timestamp", "user_id"], how="left"
    )

    df = df.fillna({
        "login_velocity_10min": 0,
        "txn_velocity_10min": 0,
        "ip_change_velocity_10min": 0,
        "device_change_velocity_10min": 0
    })

    df = df.drop(columns=["prev_ip", "prev_device", "ip_change", "device_change"])
    return df


# First time this user sends to this receiver
def first_time_receiver_flag(df):
    df = df.sort_values("timestamp").reset_index(drop=True)

    txn_df = df[df["event_type"] == "transaction"].copy()

    txn_df["new_receiver"] = (
        txn_df.groupby(["user_id", "receiver_id"]).cumcount() == 0
    ).astype(int)

    df = df.merge(
        txn_df[["new_receiver"]],
        left_index=True, right_index=True, how="left"
    )

    df["new_receiver"] = df["new_receiver"].fillna(0).astype(int)
    return df

# Drop raw identifier columns BEFORE model training
def select_model_features(df):
    drop_cols = ["timestamp", "user_id", "device_id", "ip", "event_type", "receiver_id"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return df


# Master function to run all feature engineering steps
def assemble_features(df):
    df = df.sort_values("timestamp")

    df = failed_login_velocity(df)
    df = new_device_flag(df)
    df = new_ip_flag(df)
    df = amount_deviation(df)
    df = distance_from_last_location(df)
    df = time_gaps(df)
    df = velocity_features(df)
    df = first_time_receiver_flag(df)

    df = df.reset_index(drop=True)
    return df
