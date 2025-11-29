from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

force_block_flags = { "impossible_travel", "high_amount_deviation"}
force_otp_flags = {"new_device", "first_time_receiver", "instant_password_reset"}

default_policy = {
    "action_labels" :{ "allow":"ALLOW", "otp":"OTP", "block": "BLOCK"},
    "tolerate_missing_thresold": False
}

def _validate_threshold(threshold: Dict[str,float])-> None:

    required = {"low", "medium", "high"}
    if not required.issubset(set(threshold.keys())):
        raise ValueError(f"threshold must contain keys{required}, got {list(threshold.keys())}")
    
    low, med, high = threshold["low"], threshold["medium"], threshold["high"]

    if not (0.0<=low <=med <=high <=1.0):
        raise ValueError("threshold roder invalid: required 0.0<=low <= medium <= high <=1.0")

def decision_from_score(final_score: float, thresold: Dict[str,float], policy: Dict[str,Any]= default_policy):
    _validate_threshold(thresold)

    if final_score >= thresold["high"]:
        return policy["action_labels"]["block"]
    if final_score >= thresold["medium"]:
        return policy["action_labels"]["otp"]
    return policy["action_labels"]["allow"]

def apply_forced_flags(rule_flags: List[str], thresold: Dict[str,float], policy: Dict[str,Any]=default_policy)-> Optional[str]:
    flags = set(rule_flags or [])
    if flags & force_block_flags:
        return policy["action_labels"]["block"]
    if flags & force_block_flags:
        return policy["action_labels"]["otp"]
    return None

def decision_pipeline(
        final_score: float,
        model_score: float,
        rule_score: float,
        rule_flags: Optional[List[str]],
        threshold: Dict[str,float],
        policy: Dict[str, Any] = default_policy,
        extra_context: Optional[Dict[str, Any]] = None,
)-> Dict[str, Any]:
    if policy is None:
        policy = default_policy
    if extra_context is None:
        extra_context = {}

    # validate thresholds
    if not policy.get("tolerate_missing_threshold", False):
        _validate_threshold(threshold)
    
    reasons: List[str] = None
    action: Optional[str] = None

    # Forced rules: immediate override (BLOCK > OTP)
    forced = apply_forced_flags(rule_flags or [], threshold, policy=policy)
    if forced:
        action = forced
        reasons.append("forced_by_rule_flag")
    
    # Otherwise, use score-based decision
    if action is None:
        action = decision_from_score(final_score, threshold, policy=policy)
        if action == policy["action_labels"]["block"]:
            reasons.append("score>=high_threshold")
        elif action == policy["action_labels"]["otp"]:
            reasons.append("medium_threshold<=score<high_threshold")
        else:
            reasons.append("score<medium_threshold")
        

        # Add rule flags to reasons for traceability
        if rule_flags:
            reasons.append([f"rule:{r}" for r in rule_flags])

        decision = {
            "action" : action,
            "final_score": float(final_score),
            "model_score": float(model_score),
            "rule_score": float(rule_score),
            "rule_flags": rule_flags or [],
            "threshold" : threshold,
            "reasons" : reasons,
            "context" : extra_context
        }
    
        logger.debug("decision made : action=%s score=%.3f flags=%s reasons=%s", action, final_score, rule_flags, reasons)

        return decision