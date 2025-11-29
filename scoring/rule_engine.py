import numpy as np
import pandas as pd

# rule definition
def rule_failed_login_velocity(row):
    if row.get("failed_login_10min",0)>=3:
        return 0.25, "failed_login_velocity"
    return 0.0, None

def rule_new_device(row):
    if row.get("new_device_flag",0) ==1:
        return 0.20, "new_device"
    return 0.0, None

def rule_new_ip(row):
    if row.get("new_ip_flag", 0)==1:
        return 0.15, "new_ip"
    return 0.0, None

def rule_amount_deviation(row):
    if row.get("z_amount", 0)>=3:
        return 0.30, "high_amount_deviation"
    return 0.0, None

def rule_first_time_receiver(row):
    if row.get("first_time_receiver_flag", 0) ==1:
        return 0.15, "first_time_receiver"
    return 0.0, None

def rule_transaction_velocity(row):
    if row.get("txn_count_5min",0)>=5:
        return 0.25, "high_txn_velocity"
    return 0.0, None

def rule_time_between_login_and_reset(row):
    if row.get("time_between_login_reset_sec", 999)<=10:
        return 0.20, "instant_password_reset"
    return 0.0, None

def rule_impossible_travel(row):
    if row.get("distance_from_last_location_km", 0)>=500:
        return 0.30, "impossible_travel"
    return 0.0, None

# main rule engine
rule = [
    rule_failed_login_velocity,
    rule_new_device,
    rule_new_device,
    rule_amount_deviation,
    rule_first_time_receiver,
    rule_impossible_travel,
    rule_first_time_receiver,
    rule_time_between_login_and_reset,
    rule_transaction_velocity
]

def apply_rule(row):
    total_score = 0.0
    flags=[]

    for rule_fn in rule:
        score, flag = rule_fn(row)
        total_score +=score
        if flag:
            flags.append(flag)
    
    # 1.0 is max capacity 
    total_score = min(total_score,1.0)
    return total_score, flags

def run_rule_engine(df):
    df = df.copy()
    rule_score =[]
    rule_flags = []

    for _, row in df.itterrow():
        score, flags = apply_rule(row)
        rule_score.append(score)
        rule_flags.append(flags)

        df["rule_score"] = rule_score
        df["rule_flags"] = rule_flags
        return df