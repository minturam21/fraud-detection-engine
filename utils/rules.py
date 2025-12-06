
from typing import Tuple, List

def compute_rule_score_and_flags(model_score: float) -> Tuple[float, List[str]]:
    """
    Improved rule engine so high-risk inputs actually trigger BLOCK.
    """

    flags: List[str] = []

    # HIGH RISK RULE — triggers BLOCK
    if model_score >= 0.6:
        flags.append("high_amount_deviation")
        rule_score = 0.9
        return rule_score, flags

    # MEDIUM RISK RULE — triggers OTP
    if model_score >= 0.4:
        flags.append("new_device")
        rule_score = 0.6
        return rule_score, flags

    # LOW RISK RULE — ALLOW
    rule_score = 0.2
    return rule_score, flags
