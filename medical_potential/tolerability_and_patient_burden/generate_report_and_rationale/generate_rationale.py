"""Prompt-based rationale generator for tolerability scoring."""

from __future__ import annotations

import asyncio
import logging
import json
from typing import Any

from medical_potential.gemini_utils import gemini_api_call

logger = logging.getLogger(__name__)

NOT_GENERATED_MESSAGE = "Rationale has not been generated."


def _prepare_prompt_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Select and reduce input data to minimal fields needed for a concise rationale."""
    if not isinstance(data, dict):
        return {}

    molecule = data.get("molecule")
    tolerability_score = data.get("score_numeric")

    # Pull top signals: discontinuation rates, key AEs, severity, guardrail, trials
    discontinuation_rate_data = data.get('supporting_data', {}) 
    side_effect_summary = data.get('side_effect_summary', [])
    trials = data.get("_scoring_breakdown", {}).get("total_trials_extracted")

    return {
        "molecule": molecule,
        "tolerability_score": tolerability_score,
        "discontinuation_rate": discontinuation_rate_data,
        "side_effect_summary": side_effect_summary,
        "num_trials": trials,
        "justification": (data.get("justification") or ""),
    }


async def generate_rationale_for_score(
    data: dict[str, Any],
    model_name: str = "gemini-3-flash-preview",
) -> tuple[str, dict[str, Any]]:
    """Generate a concise qualitative rationale for tolerability score.

    Returns a tuple: (rationale_text, payload_used_for_prompt).
    """
    payload = _prepare_prompt_payload(data)

    prompt = f"""
You are a concise medical insights writer. Produce a short, plain-text rationale (one sentence or a very short paragraph) that explains the main driving forces behind the Tolerability & Patient Burden score for a drug.

Score ranges from 1 to 5, 1 indicating poor performance and while 5 indicating favourable performance in the context of unmet needs solving
The score and supporting data and justifications are provided below, based on that write a concise rationale on what is driving the score

Input (JSON):
{json.dumps(payload, indent=2)}

Requirements:
- Explain only the primary drivers that push the score higher or lower (e.g., elevated discontinuation relative to comparators, frequent/severe AEs, guardrail triggers, or very limited trial evidence).
- Mention uncertainty when trial data are limited or rates are missing.
- Avoid methodology, scoring framework names, internal field names, or tables.
- Do not mention or cite specific trial identifiers.
- Keep this evaluation more qualitative. Provide commentary and assessment without mentioning any specific numerical values.
- Base your assessment entirely on the provided input data. Do not reference external facts, assumptions, or outside data, as doing so will violate the constraints.
- Use plain language appropriate for a Medical Affairs audience.
- Return plain text only (max ~50 words).
""".strip()

    try:
        rationale = (
            await asyncio.to_thread(
                gemini_api_call,
                prompt=prompt,
                model=model_name,
                temperature=0,
            )
        ).strip()
        if rationale:
            return rationale, payload
        logger.warning("[TOLERABILITY][RATIONALE] Prompt output was empty; rationale was not generated.")
    except Exception:
        logger.exception("[TOLERABILITY][RATIONALE] Prompt generation failed; rationale was not generated.")

    return NOT_GENERATED_MESSAGE, payload
