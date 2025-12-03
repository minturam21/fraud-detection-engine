import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("scoring.rule_engine")

# Flags that force immediate block
FORCE_BLOCK = {"impossible_travel", "high_amount_deviation"}

# Flags that force OTP
FORCE_OTP = {"new_device", "first_time_receiver", "instant_password_reset"}

DEFAULT_POLICY = {
    "action_labels": {"allow": "ALLOW", "otp": "OTP", "block": "BLOCK"},
    "tolerate_missing_threshold": False
}


def validate_threshold(th: Dict[str, float]):
    required = {"low", "medium", "high"}
    if not required.issubset(th.keys()):
        raise ValueError(f"Threshold keys missing: required {required}")

    low, med, high = th["low"], th["medium"], th["high"]
    if not (0 <= low <= med <= high <= 1):
        raise ValueError("Invalid threshold ordering.")


def decision_from_score(score: float, threshold: Dict[str, float], policy=DEFAULT_POLICY):
    validate_threshold(threshold)

    if score >= threshold["high"]:
        return policy["action_labels"]["block"]
    if score >= threshold["medium"]:
        return policy["action_labels"]["otp"]
    return policy["action_labels"]["allow"]


def forced_decision(rule_flags: List[str], threshold, policy=DEFAULT_POLICY) -> Optional[str]:
    flags = set(rule_flags)

    if flags & FORCE_BLOCK:
        return policy["action_labels"]["block"]

    if flags & FORCE_OTP:
        return policy["action_labels"]["otp"]

    return None


def decision_pipeline(
    final_score: float,
    model_score: float,
    rule_score: float,
    rule_flags: List[str],
    threshold: Dict[str, float],
    policy: Dict[str, Any] = DEFAULT_POLICY,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    
    if context is None:
        context = {}

    if not policy.get("tolerate_missing_threshold", False):
        validate_threshold(threshold)

    reasons = []

    forced = forced_decision(rule_flags, threshold, policy)
    if forced:
        reasons.append("forced_flag_trigger")
        for flag in rule_flags:
            reasons.append(f"flag:{flag}")
        return {
            "action": forced,
            "final_score": final_score,
            "model_score": model_score,
            "rule_score": rule_score,
            "rule_flags": rule_flags,
            "threshold": threshold,
            "reasons": reasons,
            "context": context,
        }

    action = decision_from_score(final_score, threshold, policy)

    if action == "BLOCK":
        reasons.append("high_score_block")
    elif action == "OTP":
        reasons.append("medium_score_otp")
    else:
        reasons.append("allowed_score_low")

    for flag in rule_flags:
        reasons.append(f"flag:{flag}")

    return {
        "action": action,
        "final_score": final_score,
        "model_score": model_score,
        "rule_score": rule_score,
        "rule_flags": rule_flags,
        "threshold": threshold,
        "reasons": reasons,
        "context": context,
    }
