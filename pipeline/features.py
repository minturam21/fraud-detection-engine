import pandas as pd

def failed_login_velocity(df):
    """
    Count the number of failed logins in the last 10 minutes for each user.
    """
    df = df.sort_values("timestamp")

    # create an empty column
    df["failed_login_10min"] = 0

    # Loop through failed login events only
    for i, row in df.itterrows():
        if row["event_type"] == "login_failed":
            user = row["user_id"]
            current_time = row["timestamp"]

    # find all failed logins for the same user in last 10 minutes
    mask = (
        (df["user_id"] ==user) & 
        (df["event_type"] =="login_failed") &
        (df["timestamp"] >= current_time - pd.Timedelta(minutes=10)) &
        (df["timestamp"] <= current_time)
    )        
    
    df.at[i, "failed_login_10min"] = mask.sum()

    return df