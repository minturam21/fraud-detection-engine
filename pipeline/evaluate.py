from typing import List, Tuple, Dict, Optional, Any
import numpy as np 
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

def _safe_series(x):
    return pd.Series(x).reset_index(drop=True)

def thresold_summary_df(y_true: pd.Series, y_score:pd.Series) -> pd.DataFrame:
    y_true = _safe_series(y_true).astype(int)
    y_score = _safe_series(y_score).astype(float)
    n= len(y_true)

    # sort by score decending
    order = np.argsort(-y_score.values)
    y_sorted = y_true.values[order]
    scores_sorted = y_score.values[order]

    # cumulative true positive at each cutoff
    cum_TP = np.cumsum(y_sorted)
    indices = np.arange(1, n+1)
    cum_FP = indices - cum_TP

    total_pos = int(y_true.sum())
    total_neg = n - total_pos

    TP = cum_TP
    FP = cum_FP
    FN = total_pos - TP
    TN = total_neg - FP

    # avoid division by zero
    precision = np.where(indices>0, TP/indices, 0.0)
    recall = np.where(total_pos>0, TP/total_pos, 0.0)
    fpr = np.where((FP+TN)>0,FP/(FP+TN), 0.0)

    df = pd.DataFrame({
        "score": scores_sorted,
        "TP": TP,
        "FP":FP,
        "FN":FN,
        "TN": TN,
        "precision": precision,
        "recall": recall,
        "fpr":fpr
    })
    return df

def detection_latency_stats(
        y_true: pd.Series,
        y_score: pd.Series,
        timestamps: Optional[pd.Series],
        thresold: float
) -> Optional[dict[str,float]]:
    """Compute latency stats for detected true positives.
    Assumes timestamps aligned with y_true and y_score (same length).
    latency = detection_time - event_time in seconds.
    For offline eval, detection_time is the same event timestamp (no delay),
    but this function is useful if you have separate detection timestamps.
    Here we will compute time difference only for true positives with score >= threshold."""
    
    if timestamps is None:
        return None
    
    y_true = _safe_series(y_true).astype(int)
    y_score = _safe_series(y_score).astype(float)
    timestamps = pd.to_datetime(_safe_series(timestamps))

    mask_detected_tp = (y_true==1) & (y_score>=thresold)
    if mask_detected_tp.sum()==0:
        return None
    
    # Assuming detection_time == event_time (no streaming delay). If you have a different detection timestamp,
    # pass it to this function in place of timestamps.
    # Here latency will be zero for each detected event but we keep the API for compatibility.
    
    lat_secs = np.zeros(int(mask_detected_tp.sum()))

    # if in a future version we have separate detection timestamps, compute differences here.
    stats = {
        "detected_true_positives": int(mask_detected_tp.sum()),
        "latency_median_seconds": float(np.median(lat_secs)),
        "latency_mean_seconds": float(np.mean(lat_secs)),
        "latency_p90_seconds": float(np.percentile(lat_secs,90))
    }
    return stats