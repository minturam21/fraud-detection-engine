import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Ensure folder exists
os.makedirs("data/models", exist_ok=True)

# Create simple fake training dataset
X = pd.DataFrame({
    "failed_login_10min": [0,1,3,5,2],
    "new_device_flag": [0,1,0,1,0],
    "new_ip_flag": [0,0,1,1,0],
    "z_amount": [0.1,0.5,3.0,2.5,0.2],
    "distance_from_last_location_km": [1,2,10,300,20],
    "first_time_receiver_flag": [0,1,0,1,0],
    "time_between_login_reset_sec": [500,30,100,5,200],
    "tx_count_5min": [0,1,2,6,1]
})

y = [0,0,1,1,0]  # fake labels

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "data/models/model.pkl")
print("✔ Fake model saved → data/models/model.pkl")
