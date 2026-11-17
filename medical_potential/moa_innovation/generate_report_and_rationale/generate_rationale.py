"""Prompt-based rationale generator for MoA Innovation scoring.

Creates a short, plain-text rationale explaining the main drivers behind the
MoA Innovation `score` using the generated `output_payload` as input.
"""

from __future__ import annotations

import asyncio
import logging
import os
import json
from typing import Any

from medical_potential.gemini_utils import gemini_api_call

logger = logging.getLogger(__name__)

NOT_GENERATED_MESSAGE = "Rationale has not been generated."

# Directional scoring rubric to guide the rationale (used directionally only)
SCORING_RUBRIC = """
Score 5 — Exceptional

Used when:
- true FIC mechanism
- strong biological rationale
- meaningful potential to create a new treatment class

Score 4 — Strong

Used when:
- known validated mechanism
- clearly superior implementation
- real mechanistic differentiation

Typical case: strong BIC candidate

Score 3 — Moderate

Used when:
- known validated mechanism
- limited mechanism-level differentiation
- competes mainly on non-MoA factors

This is the standard me-too / fast follower score.

Score 2 — Weak

Used when:
- mechanism is older or strategically outdated
- class has been replaced by better mechanisms

Score 1 — Poor

Used when:
- weak or speculative biology
- repeated clinical failures for same mechanism in same indication
- no convincing rescue hypothesis
"""


def _prepare_prompt_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Select and reduce input data to minimal fields needed for a concise rationale."""
    if not isinstance(data, dict):
        return {}

    molecule = data.get("drug_name") or data.get("molecule")
    moa = data.get("mechanism_statement")
    score = data.get("score")
    classification = data.get("moa_classification")

    justification = data.get("justification") or {}
    narrative_rationale = data.get("narrative_rationale") or {}


    return {
        "molecule": molecule,
        "mechanism": moa,
        "moa_score": score,
        "classification": classification,
        "justification": justification,
        "narrative_rationale": narrative_rationale,
    }


async def generate_rationale_for_score(
    data: dict[str, Any],
    model_name: str = os.getenv("GEMINI_FLASH_PREVIEW_MODEL", "gemini-3-flash-preview"),
) -> tuple[str, dict[str, Any]]:
    """Generate a concise qualitative rationale for MoA Innovation score.

    Returns a tuple: (rationale_text, payload_used_for_prompt).
    """
    payload = _prepare_prompt_payload(data)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return NOT_GENERATED_MESSAGE, payload

    prompt = f"""
You are a concise medical insights writer. Produce a short, plain-text rationale (one sentence or a very short paragraph) that explains the main driving forces behind the MoA Innovation score for a drug.

Focus only on the provided input. Explain primary drivers that push the score higher or lower (e.g., First-in-Class novelty with strong biology, meaningful mechanistic differentiation, prior mechanism-level failures, or the mechanism being outdated). Mention uncertainty when evidence is limited.

Scoring Rubric (use directionally only):
{SCORING_RUBRIC}

Instruction: Use the rubric directionally to interpret the score. Explain briefly how the input data (mechanism summary, novelty vs SOC, number of pipeline/validated examples, and primary sources) supports or contradicts the rubric criteria for the assigned score. The explanation should reference the rubric directionally to understand the score better (e.g., "supports BIC features:..."), not restate it verbatim or mention it.

Input (JSON):
{json.dumps(payload, indent=2)}

Requirements:
- Explain only the primary drivers that push the score higher or lower.
- Mention uncertainty when trial or evidence data are limited.
- Use the rubric directionally and explain how the input supports the rubric-based decision.
- Avoid methodology, scoring framework names, internal field names, or tables.
- Do not mention or cite specific trial identifiers.
- Keep this evaluation qualitative and plain-language for a Medical Affairs audience.
- Return plain text only (max ~50 words).
""".strip()

    try:
        rationale = await asyncio.to_thread(
            gemini_api_call,
            prompt=prompt,
            model=model_name,
            temperature=0,
            google_search=False,
        )
        rationale = rationale.strip()
        if rationale:
            return rationale, payload
        logger.warning("[MoA][RATIONALE] Prompt output was empty; rationale was not generated.")
    except Exception:
        logger.exception("[MoA][RATIONALE] Prompt generation failed; rationale was not generated.")

    return NOT_GENERATED_MESSAGE, payload
