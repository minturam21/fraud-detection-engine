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

def time_gaps(df):
    """ Compute time gaps between important user actions using vectorized operations."""
    df = df.sort_values("timestamp")

    #prepare Column
    df["time_since_last_login"] = 0
    df["time_since_last_reset"] = 0
    df["time_since_last_txn"] = 0

    # last login
    last_login_time = (
        df[df["event_type"] == "login"]
        .groupby("user_id")["timestamp"]
        .shift(1)
    )

    df["last_login_time"] = last_login_time
    df["time_since_last_login"] = (
       (df["timestamp"] - df["last_login_time"])
       .dt.total_seconds()
       .fillna(0)
    )

    # last time password reset
    last_time_reset = (
        df[df["event_type"]=="reset_password"]
        .groupby("user_id")["timestamp"]
        .shift(1)
    )

    df["last_time_reset"] = last_time_reset
    df["time_since_last_reset"] = (
        (df["timestamp"] - df["last_time_reset"])
        .dt.total_seconds()
        .fillna(0)
    )

    # last transaction

    last_tnx_time = (
        [df["event_type"] =="transaction"]
        .groupby("user_id")["timestamp"]
        .shift(1)
    )

    df["last_txn_time"] = last_tnx_time
    df["time_since_last_txn"] = (
        (df["timestamp"] - df["last_txn_time"])
        .dt.total_seconds()
        .fillna(0)
    )

    # clean temporary column
    df = df.drop(columns =[
        "last_login_time",
        "last_rest_time",
        "last_txn_time"
    ] )

    return df

def velocity_features(df):
    # Compute multiple velocity features (login, txn, IP change, device change)

    df = df.sort_values("timestamp")

    # login Velocity
    login_events = df[df["event_type"]=="login"].copy
    login_events["login_velocity_10min"] = (
        login_events
        .groupby("user_id")["timestamp"]
        .rolling("10min")
        .count()
        .reset_index(level = 0, drop = True)
    )

    # transaction velocity
    txn_events = df[df["event_type"]=="transaction"].copy
    txn_events["txn_velocity_10min"]= (
        df.groupby("user_id")["timestamp"]
        .rolling("10min")
        .count()
        .resert_index(level=0, drop=True)
    )


    # ip change velocity
    df["prev_ip"] = df.groupby("user_id")["ips"].shift(1)
    df["ip_change"] = (df["ip"] != df["prev_ip"]).astype(int)

    df["ip_change_velocity_10min"] = (
        df.groupby("user_id")
        .rolling("10min", on = "timestamp")["ip_change"]
        .sum()
        .reset_index(level=0, drop = True)
    )

    # device change velocity
    df["prev_device"] = df.groupby("user_id")["device_id"].shift(1)
    df["device_change"] = (df["device_id"] != df["prev_device"]).astype(int)

    df["device_change_velocity_10min"] = (
        df.groupby("user_id")
        .rolling("10min", on = "timestamp")["device_change"]
        .sum()
        .reset_index(level=0, drop=True)
    )

    # merge login velocity and transaction velocity back into df
    df = df.merge(
        login_events =[["login_velocity_10min"]],
        left_index = True,
        right_index = True,
        how = "left"
    ).merge(
        txn_events = [["txn_velocity_10min"]],
        left_index = True,
        right_index = True,
        how = "left"
    )

    # fill Nan with 0
    df = df.fillna({
        "login_velocity_10min" : 0,
        "txn_velocity_10min" : 0,
        "ip_change_velocity_10min": 0,
        "device_change_velocity_10min" :0
    })

    # clean up temp columns
    df = df.dropna(columns= ["prev_ip", "prev_device", "ip_change", "device_change"])

    return df

def first_time_receiver_flag(df):
    """Flag if the payment receiver is new for this user.
    Uses groupby + cumcount for fast, scalable computation."""

    df = df.sort_values("timestamp")

    txn_df = df[df["event_type"] == "transaction"].copy()

    # First time a user send money to a receiver cumcount = 0
    df["new_receiver"] = (
        txn_df.groupby(["user_id", "receiver_id"]).cumcount() ==0
    ).astype(int)

    # merge back to new df
    df = df.merge(
        txn_df[["new_receiver"]],
        right_index = True,
        left_index = True,
        how = "left"
    )

    # fill non-transaction event with 0
    df["new_receiver"] = df["new_receiver"].fillna(0)

    return df

def assemble_features(df):

    df = df.sort_values("timestamp")

    # apply all features functions
    df = failed_login_velocity(df)
    df = new_device_flag(df)
    df = new_ip_flag(df)
    df = amount_deviation(df)
    df = distance_from_last_location(df)
    df = time_gaps(df)
    df = velocity_features(df)
    df = first_time_receiver_flag(df)

    # clean up
    df = df.reset_index(drop = True)

    return df


 
