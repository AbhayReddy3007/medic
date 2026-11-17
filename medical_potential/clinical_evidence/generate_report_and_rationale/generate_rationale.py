"""Rationale generator for Clinical Evidence dimension.

Produces a concise rationale (plain text) explaining the final clinical
evidence score based on the provided `output_payload` using a prompt
to the Gemini model. Returns the rationale text and the reduced
prompt payload used for generation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from medical_potential.gemini_utils import gemini_api_call

logger = logging.getLogger(__name__)

NOT_GENERATED_MESSAGE = "Rationale has not been generated."

# Scoring methodology to include in the prompt (directional guidance)
SCORING_METHODOLOGY = """
Score Calculation

The final score combines the highest clinical development phase with the
overall strength of study design, geography, and evidence support, while
incorporating an adjustment for consistency of findings.

First, calculate the representative evidence score:

Representative Evidence Score = mode(D, G, E)

Then calculate the raw score:

Raw Score = ((P + Representative Evidence Score) / 2) + C

Finally, constrain the score to a minimum of 1 and a maximum of 5:

Final Score = min(5, max(1, ((P + mode(D, G, E)) / 2) + C))

Where:
- P = Phase Score
- D = Study Design Score
- G = Geography Score
- E = Evidence Support Score
- C = Consistency Adjustment
- mode(D, G, E) = the most frequently occurring value among D, G, and E. If no
  mode exists, use the median of D, G, and E.

Interpretation:
- The Phase Score reflects the maturity of clinical development and is given
  equal weight to the overall quality of supporting evidence.
- Study Design, Geography, and Evidence Support are combined into a single
  representative evidence score using their mode, ensuring the final score
  reflects the predominant strength of the evidence base.
- The Consistency Adjustment rewards replicated positive findings and
  penalizes conflicting or unavailable evidence.
- The final score is capped between 1 and 5 to maintain comparability across
  molecules.
"""


def _prepare_prompt_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Reduce the full output payload to the minimal fields needed by the LLM.

    Mirrors the pattern used in other dimensions: pick the molecule name,
    phase/score components (P, D, G, E, C), counts, and a short list of
    primary sources and a plain-language evidence summary when present.
    """
    if not isinstance(data, dict):
        return {}

    molecule = data.get("molecule_name") or data.get("drug_name") or data.get("generic_name")
    final_score = data.get("final_score") or data.get("final_score") or data.get("final_score")

    P = data.get("score_p")
    D = data.get("score_d")
    G = data.get("score_g")
    E = data.get("score_e")
    C = data.get("score_c")

    total_trials = data.get("total_trials")
    rct_count = data.get("rct_count")
    evidence_level = data.get("evidence_level")

    primary_sources = []
    try:
        sources = data.get("report", {})
        primary_sources = sources.get("primary_sources", []) if isinstance(sources, dict) else []
    except Exception:
        primary_sources = []

    summary = data.get("evidence_gaps") or data.get("llm_consistency_report") or ""

    return {
        "molecule": molecule,
        "final_score": final_score,
        "P": P,
        "D": D,
        "G": G,
        "E": E,
        "C": C,
        "total_trials": total_trials,
        "rct_count": rct_count,
        "evidence_level": evidence_level,
        "primary_sources": primary_sources,
        "summary": summary,
    }


async def generate_rationale_for_score(
    data: dict[str, Any],
    model_name: str = "gemini-3-flash-preview",
) -> tuple[str, dict[str, Any]]:
    """Generate a concise rationale for the Clinical Evidence final score.

    Returns a tuple `(rationale_text, payload_used_for_prompt)`.
    """
    payload = _prepare_prompt_payload(data)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return NOT_GENERATED_MESSAGE, payload

    prompt = f"""
You are a concise clinical-evidence writer. Produce a short, plain-text rationale (one sentence or very short paragraph, ~40 words) that explains why the *final clinical evidence score* was assigned for this molecule.

Use the scoring methodology below directionally to interpret the score and explicitly state how the available inputs (phase, study design, geography, evidence support, consistency adjustment, and trial counts) support that final score.

Scoring Methodology:
{SCORING_METHODOLOGY}

Input (JSON):
{json.dumps(payload, indent=2)}

Requirements:
- Explain the score in plain Medical Affairs language.
- Use the P/D/G/E/C inputs only as reasoning signals and DO NOT mention the letters, numerical values, component scores, methodology dimensions, calculations, weights, or rubric categories.
- Describe only the underlying evidence, development maturity, breadth of clinical evaluation, consistency of findings, and evidence limitations.
- If evidence is limited, explicitly state the uncertainty.
- Do not explain how the score was calculated.
- Keep this evaluation more qualitative. Provide commentary and assessment without mentioning any specific numerical values.
- Avoid any wording such as "P=5", "E=3", "C=0.25", "development phase", "evidence score", "consistency score", "rubric", "methodology", or similar scoring terminology.
- Return plain text only.
- Maximum 50 words.
""".strip()

    try:
        response = await asyncio.to_thread(
            gemini_api_call,
            prompt=prompt,
            model=model_name,
            temperature=0,
        )
        rationale = response.strip()
        if rationale:
            return rationale, payload
        logger.warning("[Clinical Evidence][RATIONALE] Prompt output empty; no rationale generated.")
    except Exception:
        logger.exception("[Clinical Evidence][RATIONALE] Prompt generation failed.")

    return NOT_GENERATED_MESSAGE, payload
