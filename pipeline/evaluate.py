import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

def threshold_summary_df(y_true, y_prob, thresholds):
    rows = []
    y_true = np.array(y_true)

    for t in thresholds:
        preds = (y_prob >= t).astype(int)

        TP = int(((preds == 1) & (y_true == 1)).sum())
        FP = int(((preds == 1) & (y_true == 0)).sum())
        FN = int(((preds == 0) & (y_true == 1)).sum())
        TN = int(((preds == 0) & (y_true == 0)).sum())

        precision = TP / (TP + FP + 1e-9)
        recall = TP / (TP + FN + 1e-9)
        fpr = FP / (FP + TN + 1e-9)

        rows.append({
            "threshold": t,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "precision": precision,
            "recall": recall,
            "fpr": fpr
        })

    return pd.DataFrame(rows)

def precision_at_k(y_true, y_prob, k):
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df = df.sort_values("p", ascending=False)
    topk = df.head(k)
    if len(topk) == 0:
        return 0.0
    return float(topk["y"].sum() / len(topk))

def recall_at_fixed_fpr(y_true, y_prob, target_fpr=0.01):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # find first index where fpr >= target_fpr
    idx = np.searchsorted(fpr, target_fpr)

    if idx >= len(fpr):
        # fallback: use last available values
        return float(tpr[-1]), float(thresholds[-1])

    return float(tpr[idx]), float(thresholds[idx])

def fraud_capture_rate(y_true, y_prob, threshold):
    preds = (y_prob >= threshold).astype(int)
    y_true = np.array(y_true)
    TP = int(((preds == 1) & (y_true == 1)).sum())
    FN = int(((preds == 0) & (y_true == 1)).sum())
    return float(TP / (TP + FN + 1e-9))

def detection_latency_stats(df, prob_col="pred_prob", time_col="timestamp", label_col="label", threshold=0.5):
    df = df.copy()
    df["pred"] = (df[prob_col] >= threshold).astype(int)

    # fraud rows
    fraud_df = df[df[label_col] == 1]
    if fraud_df.empty:
        return None, None

    # detected fraud rows (where pred==1) and their detection times
    detected = fraud_df[fraud_df["pred"] == 1]
    if detected.empty:
        return None, None

    # latency: detection_time - event_time (seconds)
    latencies = (detected[time_col] - fraud_df[time_col]).dt.total_seconds()

    # keep only positive latencies
    latencies = latencies[latencies >= 0]
    if latencies.empty:
        return None, None

    return float(latencies.mean()), float(latencies.median())

def evaluate_pro(y_true, y_prob, top_k=100, target_fpr=0.01):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # ROC AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_value = auc(fpr, tpr)

    # Recall at target FPR and threshold at that FPR
    recall_fixed_fpr, best_threshold = recall_at_fixed_fpr(y_true, y_prob, target_fpr=target_fpr)

    # Precision at top_k
    prec_k = precision_at_k(y_true, y_prob, k=top_k)

    # Fraud capture rate at chosen threshold
    capture_rate = fraud_capture_rate(y_true, y_prob, threshold=best_threshold)

    return {
        "auc": float(auc_value),
        "recall_at_fpr": float(recall_fixed_fpr),
        "precision_at_k": float(prec_k),
        "best_threshold_at_fpr": float(best_threshold),
        "fraud_capture_rate": float(capture_rate)
    }