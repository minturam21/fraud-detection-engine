from typing import List, Tuple, Dict, Optional, Any
import numpy as np 
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

def _safe_series(x):
    return pd.Series(x).reset_index(drop=True)

def thresold_summary_def(y_true: pd.Series, y_score:pd.Series) -> pd.DataFrame:
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