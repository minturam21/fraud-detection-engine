import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score

def select_thresholds(y_true, y_prob, eval_results,
                      medium_strategy="f1",
                      low_percentile=10):
    """Improved 3-level thresholding:
      - HIGH: block threshold based on FPR target (from eval_results)
      - MEDIUM: F1-optimized threshold or precision-recall balance
      - LOW: low percentile of scores (events below this are very safe)"""

    # HIGH THRESHOLD — use the best threshold at target FPR
    high_threshold = float(eval_results["best_threshold_at_fpr"])

    # MEDIUM THRESHOLD — find best F1 threshold
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_f1_idx = np.argmax(f1_scores)

    # thresholds array is one shorter than P/R arrays
    if best_f1_idx == 0:
        medium_threshold = high_threshold * 0.7
    else:
        medium_threshold = float(thresholds[best_f1_idx - 1])

    # Ensure medium threshold is below high
    medium_threshold = min(medium_threshold, high_threshold * 0.95)

    # LOW THRESHOLD — score percentiles (for safe zone)
    low_threshold = float(np.percentile(y_prob, low_percentile))

    # Ensure ordering
    low_threshold = max(0.0, min(low_threshold, medium_threshold * 0.9))

    return {
        "low": low_threshold,
        "medium": medium_threshold,
        "high": high_threshold
    }
