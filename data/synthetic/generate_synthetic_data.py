import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def random_ip():
    return ".".join(str(random.randint(1, 255)) for _ in range(4))

def generate_synthetic_data(
    n_users=500,
    n_events=60000,
    fraud_rate=0.03,
    seed=42
):
    np.random.seed(seed)
    random.seed(seed)

    users = [f"user_{i}" for i in range(n_users)]
    devices = [f"device_{i}" for i in range(300)]
    receivers = [f"recv_{i}" for i in range(500)]

    base_time = datetime(2024, 1, 1)

    rows = []

    # Store last known locations for impossible travel patterns
    last_lat = {u: 20 + np.random.randn() * 5 for u in users}
    last_lon = {u: 80 + np.random.randn() * 5 for u in users}

    for i in range(n_events):

        user = random.choice(users)
        event_time = base_time + timedelta(seconds=i * random.randint(1, 8))

        # Basic distribution of events
        if np.random.rand() < 0.55:
            event_type = "transaction"
        elif np.random.rand() < 0.15:
            event_type = "login_fail"
        elif np.random.rand() < 0.10:
            event_type = "reset_password"
        else:
            event_type = "login"

        amount = np.nan
        receiver = None

        # Fraud Pattern 1:
        # Sudden very large amounts
        if event_type == "transaction":
            if np.random.rand() < fraud_rate:
                # Fraudulent behavior: large spikes
                amount = round(np.random.uniform(10000, 50000), 2)
            else:
                # Normal behavior
                amount = round(np.random.exponential(scale=2000), 2)

            # Receiver
            receiver = random.choice(receivers)

        # Fraud Pattern 2:
        # New device + new IP + first-time receiver combination
        # Strong sign of fraud
        if np.random.rand() < fraud_rate / 2:
            device_id = f"fraud_device_{random.randint(1,50)}"
            ip = f"200.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}"
            receiver = f"fraud_recv_{random.randint(1,30)}"
        else:
            device_id = random.choice(devices)
            ip = random_ip()

        # Fraud Pattern 3:
        # Impossible travel (big jumps in location)
        if np.random.rand() < fraud_rate / 3:
            lat = last_lat[user] + np.random.uniform(30, 80)  # Big jump
            lon = last_lon[user] + np.random.uniform(30, 80)
        else:
            lat = last_lat[user] + np.random.randn() * 2
            lon = last_lon[user] + np.random.randn() * 2

        # Update last location
        last_lat[user] = lat
        last_lon[user] = lon

        # Fraud Pattern 4:
        # High txn velocity (multiple txns in short time)
        if event_type == "transaction" and np.random.rand() < fraud_rate / 3:
            # Duplication of very fast multiple events
            for v in range(random.randint(2, 5)):
                rows.append([
                    event_time + timedelta(seconds=v),
                    user,
                    device_id,
                    ip,
                    "transaction",
                    round(amount + np.random.randint(200, 1000), 2),
                    receiver,
                    lat,
                    lon,
                    1  # fraudulent
                ])

        # Fraud Pattern 5:
        # Night-time fraud (most fraud spikes between 2 AM and 5 AM
        if 2 <= event_time.hour <= 5 and event_type == "transaction":
            if np.random.rand() < fraud_rate * 2:
                label = 1  # fraud
        else:
            label = 1 if (event_type == "transaction" and np.random.rand() < fraud_rate) else 0

        # Base event record
        rows.append([
            event_time,
            user,
            device_id,
            ip,
            event_type,
            amount,
            receiver,
            lat,
            lon,
            label
        ])

    df = pd.DataFrame(rows, columns=[
        "timestamp",
        "user_id",
        "device_id",
        "ip",
        "event_type",
        "amount",
        "receiver_id",
        "lat",
        "lon",
        "label"
    ])

    df.to_csv("transactions.csv", index=False)
    print("Enhanced synthetic data generated → transactions.csv")
    print(df.head(), df.shape)
    return df


if __name__ == "__main__":
    generate_synthetic_data()
