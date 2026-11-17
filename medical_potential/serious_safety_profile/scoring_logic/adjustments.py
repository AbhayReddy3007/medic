"""Scoring adjustment helpers (pattern, expectedness, caps)."""

from __future__ import annotations

from typing import Optional


def get_regulatory_cap(regulatory_impact: dict) -> tuple[Optional[int], str]:
    consequence = str(regulatory_impact.get("regulatory_consequence", "")).lower().strip()
    
    if "market_threatening" in consequence:
        return 1, "Regulatory type: Market Threatening (Withdrawal/ Rejection) (cap 1)"
    if "established_human_risk" in consequence or "human_risk" in consequence or "human risk" in consequence:
        return 2, "Regulatory type: Established serious human risk (cap 2)"
    if "monitoring" in consequence or "rems" in consequence:
        return 3, "Regulatory Monitoring or REMS (cap 3)"
    return None, "No regulatory cap"


def get_post_marketing_cap(post_marketing_safety: dict) -> tuple[Optional[int], str]:
    if not post_marketing_safety or not post_marketing_safety.get("evaluated"):
        return None, "Post-marketing not evaluated"

    if post_marketing_safety.get("boxed_warning_added") or post_marketing_safety.get("withdrawal_from_market"):
        return 2, "Post-marketing boxed warning added/ withdrawal (cap 2)"
    if post_marketing_safety.get("rems_added") or post_marketing_safety.get("major_label_restriction"):
        return 3, "Post-marketing REMS/restriction added (cap 3)"


    return None, "No post-marketing cap"
