import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import joblib

from scoring.rule_engine import apply_rule
from scoring.score import score_event
from scoring.decision_pipeline import decision_pipeline

# Load model + thresholds
model = joblib.load("data/models/model.pkl")

with open("data/models/thresholds.json", "r") as f:
    thresholds = json.load(f)


# Fake incoming event with engineered features
event = {
    "failed_login_10min": 4,
    "new_device_flag": 1,
    "new_ip_flag": 0,
    "z_amount": 2.8,
    "distance_from_last_location_km": 10,
    "first_time_receiver_flag": 1,
    "time_between_login_reset_sec": 8,
    "tx_count_5min": 3
}

df_row = pd.DataFrame([event])

#RULE ENGINE
rule_score, rule_flags = apply_rule(event)

# MODEL + RULE COMBINER
score_output = score_event(
    model=model,
    thresholds= thresholds,
    event_features_row=df_row,
    rule_score=rule_score,
    rule_flags=rule_flags
)

# DECISION ENGINE

decision = decision_pipeline(
    final_score=score_output["final_score"],
    model_score=score_output["model_score"],
    rule_score=score_output["rule_score"],
    rule_flags=score_output["rule_flags"],
    threshold=thresholds,
    extra_context={"user_id": 101, "amount": 7000}
)

print("\n FULL SYSTEM DECISION OUTPUT \n")
print(decision)
