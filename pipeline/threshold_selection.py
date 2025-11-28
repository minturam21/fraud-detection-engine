import numpy as np
import pandas as pd

def select_thresholds(y_true, y_prob, eval_results, medium_factor=0.5, low_percentile=30):

     # HIGH THRESHOLD - block risk
    high_threshold = float(eval_results["best_threshold_at_fpr"])

    # MEDIUM THRESHOLD - OTP
    medium_threshold = high_threshold * medium_factor

    #  LOW THRESHOLD - allow
    #  Based on the distribution of predictions
    low_threshold = float(np.percentile(y_prob, low_percentile))

    return {
        "low": low_threshold,
        "medium": medium_threshold,
        "high": high_threshold
    }