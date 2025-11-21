import pandas as pd

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
