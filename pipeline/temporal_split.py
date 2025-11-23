import pandas as pd

def ensure_datetime(df, time_col="timstamp"):

    df["time_col"] = pd.to_datetime(df[time_col], error = "coerce")
    return df

def sort_by_time(df, time_col = "timestamp"):

    df = df.sort_values(time_col).reset_index(drop=True)
    return df

def temporal_train_text_split(df, split_date, label_col = "label"):
    df = ensure_datetime(df, "timestamp")
    df = sort_by_time(df, "timestamp")
    split_date = pd.to_datetime(split_date)

    # split
    train_def = df[df["timestamp"]< split_date]
    test_df = df[df["timestamp"]>= split_date]

    # check
    if train_def.empty:
        raise ValueError("trainin data set is empty")
    if test_df.empty:
        raise ValueError("test data set is empty")
    if test_df[label_col].sum ==0:
        print("Warning: No fraud cases in test set.")

    # Separate features and labels
    X_train = train_def.drop(columns = [label_col])
    y_train = train_def[label_col]

    X_test = test_df.drop(coumns=[label_col])
    y_test = test_df[label_col]

    return X_train, X_test, y_train, y_test
