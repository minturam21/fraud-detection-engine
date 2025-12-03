from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Rules that force immediate BLOCK
force_block_flags = {"impossible_travel", "high_amount_deviation"}

# Rules that force OTP / step-up verification
force_otp_flags = {"new_device", "first_time_receiver", "instant_password_reset"}

# Default policy configuration
default_policy = {
    "action_labels": {
        "allow": "ALLOW",
        "otp": "OTP",
        "block": "BLOCK",
    },
    "tolerate_missing_threshold": False,
}


def _validate_threshold(threshold: Dict[str, float]) -> None:
    """
    Ensure threshold dict contains required keys and valid ordering.
    Required keys: low, medium, high
    Constraint: 0.0 <= low <= medium <= high <= 1.0
    """
    required = {"low", "medium", "high"}

    if not required.issubset(threshold.keys()):
        raise ValueError(
            f"threshold must contain keys {required}, got {list(threshold.keys())}"
        )

    low = threshold["low"]
    med = threshold["medium"]
    high = threshold["high"]

    if not (0.0 <= low <= med <= high <= 1.0):
        raise ValueError(
            "threshold order invalid: required 0.0 <= low <= medium <= high <= 1.0"
        )


def decision_from_score(
    final_score: float,
    threshold: Dict[str, float],
    policy: Dict[str, Any] = default_policy,
) -> str:
    """
    Map final_score into an action label based on threshold config.
    """
    _validate_threshold(threshold)

    if final_score >= threshold["high"]:
        return policy["action_labels"]["block"]
    if final_score >= threshold["medium"]:
        return policy["action_labels"]["otp"]
    return policy["action_labels"]["allow"]


def apply_forced_flags(
    rule_flags: List[str],
    threshold: Dict[str, float],
    policy: Dict[str, Any] = default_policy,
) -> Optional[str]:
    """
    If any hard rule flag is present, override with BLOCK or OTP.
    Returns action label or None.
    """
    flags = set(rule_flags or [])

    # Any hard block flags?
    if flags & force_block_flags:
        return policy["action_labels"]["block"]

    # Any OTP / step-up flags?
    if flags & force_otp_flags:
        return policy["action_labels"]["otp"]

    return None


def decision_pipeline(
    final_score: float,
    model_score: float,
    rule_score: float,
    rule_flags: Optional[List[str]],
    threshold: Dict[str, float],
    policy: Dict[str, Any] = default_policy,
    extra_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Combine model score + rule score + rule flags + thresholds
    to produce a final decision object.
    """
    if extra_context is None:
        extra_context = {}

    # Fail early if threshold is malformed (unless policy says to tolerate)
    if not policy.get("tolerate_missing_threshold", False):
        _validate_threshold(threshold)

    reasons: List[str] = []
    rule_flags = rule_flags or []

    # Forced flags always win first (BLOCK / OTP)
    forced_action = apply_forced_flags(rule_flags, threshold, policy)
    if forced_action:
        reasons.append("forced_by_rule_flag")
        for rf in rule_flags:
            reasons.append(f"rule:{rf}")

        decision = {
            "action": forced_action,
            "final_score": final_score,
            "model_score": model_score,
            "rule_score": rule_score,
            "rule_flags": rule_flags,
            "threshold": threshold,
            "reasons": reasons,
            "context": extra_context,
        }

        logger.debug("forced decision: %s", decision)
        return decision

    # Fallback to score-based policy
    action = decision_from_score(final_score, threshold, policy)

    if action == policy["action_labels"]["block"]:
        reasons.append("score>=high_threshold")
    elif action == policy["action_labels"]["otp"]:
        reasons.append("medium_threshold<=score<high_threshold")
    else:
        reasons.append("score<medium_threshold")

    for rf in rule_flags:
        reasons.append(f"rule:{rf}")

    decision = {
        "action": action,
        "final_score": final_score,
        "model_score": model_score,
        "rule_score": rule_score,
        "rule_flags": rule_flags,
        "threshold": threshold,
        "reasons": reasons,
        "context": extra_context,
    }

    logger.debug("decision made: %s", decision)
    return decision
