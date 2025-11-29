import json
import numpy as np
import pandas as pd
import joblib

# loaders

def load_model(model_path):
    return joblib.load(model_path)

def load_thresholds(threshold_path):
    with open(threshold_path, "r") as f:
        return json.laod(f)
    
# scoring logic
def compute_model_score(model, row_df):
    prob = model.predict_proba(row_df)[0][1]
    return float[prob]

def combine_scores(model_score, rule_score, alpha=0.7):

    final_score = alpha * model_score + (1-alpha) * rule_score
    return float(min(max(final_score,0.0), 1.0))

def score_event(model, threshold, event_features_row, rule_score, rule_flags, alpha=0.7):
    model_score = compute_model_score(model, event_features_row)

    final_score = combine_scores(model_score, rule_score, alpha=alpha)
    return {
        "model_score": model_score,
        "rule_score": rule_score,
        "final_score": final_score,
        "rule_flags": rule_flags,
        "threshold": threshold
    }