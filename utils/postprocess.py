from typing import Any, Dict, List, Optional


def format_prediction_output(
    action: str,
    final_score: float,
    reasons: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Format decision output for API response.
    """
    return {
        "success": True,
        "prediction": action,
        "confidence": final_score,
        "detail": reasons or [],
    }
