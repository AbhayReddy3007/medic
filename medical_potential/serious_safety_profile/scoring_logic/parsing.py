"""Parsing and normalization helpers for scoring."""

from __future__ import annotations

from typing import Optional

from .constants import SEVERITY_LEVEL


def parse_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def severity_level(category: str) -> int:
    """Return numeric severity level (1-4) for a category string; 0 if unknown."""
    cat = (category or "").lower().strip()
    for key, level in SEVERITY_LEVEL.items():
        if key in cat:
            return level
    return 0


def derive_trial_delta(trial: dict) -> Optional[float]:
    explicit_delta = parse_float(trial.get("Delta SAE (%)"))
    if explicit_delta is not None:
        return explicit_delta

    drug = parse_float(trial.get("SAE Rate Drug (%)"))
    control = parse_float(trial.get("SAE Rate Control (%)"))
    if drug is None or control is None:
        return None
    return round(drug - control, 2)


def trial_band(delta_sae: Optional[float]) -> Optional[str]:
    if delta_sae is None:
        return None
    if delta_sae <= 0:
        return "S1"
    if delta_sae <= 0.5:
        return "S2"
    if delta_sae <= 2.0:
        return "S3"
    return "S4"


def calculate_presence_rate(event: dict, total_studies: int) -> Optional[float]:
    """
    Calculate presence rate as a percentage.
    presence_rate = (Number of Studies with respective SAE / Total number of studies) * 100
    """
    if total_studies <= 0:
        return None
    
    num_studies = int(event.get("number_of_studies") or 0)
    if num_studies <= 0:
        return None
    
    return round((num_studies / total_studies) * 100, 2)


def is_clinically_meaningful(event: dict, total_studies: int) -> bool:
    """
    Determine if an SAE is clinically meaningful based on severity level and presence rate.
    
    Returns True if the event is clinically meaningful based on the criteria.
    """
    sev = severity_level(event.get("severity_category", ""))
    presence_rate = calculate_presence_rate(event, total_studies)
    
    if presence_rate is None:
        return False
    
    # Severity levels 3 and 4 (life-threatening and death/irreversible)
    if sev in [3, 4]:
        if presence_rate >= 3.0:
            return True
        if 1.0 <= presence_rate < 3.0:
            # Meaningful only if unexpected
            expectedness = str(event.get("expectedness_classification", "")).strip().lower()
            return expectedness == "unexpected"
        # < 1%: Not meaningful unless regulatory concern exists
        return False
    
    # Severity level 2 (organ-specific)
    if sev == 2:
        if presence_rate >= 10.0:
            return True
        if 3.0 <= presence_rate < 10.0:
            # Meaningful only if unexpected
            expectedness = str(event.get("expectedness_classification", "")).strip().lower()
            return expectedness == "unexpected"
        # < 3%: Not meaningful
        return False
    
    # Severity level 1 (manageable)
    if sev == 1:
        if presence_rate >= 10.0:
            # Meaningful only if unexpected
            expectedness = str(event.get("expectedness_classification", "")).strip().lower()
            return expectedness == "unexpected"
        # < 10%: Not meaningful
        return False
    
    return False
