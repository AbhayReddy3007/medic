"""Prompt-based rationale generator for unmet-needs dimension scoring."""

from __future__ import annotations

import asyncio
import logging
import json
from typing import Any

from medical_potential.gemini_utils import gemini_api_call

logger = logging.getLogger(__name__)


NOT_GENERATED_MESSAGE = "Rationale has not been generated."


def _prepare_prompt_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Select and reduce input data to minimal fields needed for rationale.

    - Drops `aggregated_score` and `trial_to_unmet_need_mapping_rows`.
    - Filters out rows with dimension 'X' from
      `molecule_level_unmet_need_scoring_rows`.
    - From each remaining row retains only the specified fields.
    """
    if not isinstance(data, dict):
        return {}

    # Copy basic metadata
    molecule = data.get("molecule_name") or data.get("molecule")

    # Build sanitized rows
    raw_rows = data.get("molecule_level_unmet_need_scoring_rows") or []
    rows = []
    keep_fields = (
        "unmet_need",
        "category",
        "country",
        "score_dimension",
        "unmet_need_score_rationale",
        "evidence_score_rationale",
        "weighted_unmet_need_score",
    )

    for r in (raw_rows or [])[:500]:
        try:
            # Skip rows explicitly flagged as dimension 'X'
            dim = r.get("dimension") or r.get("score_dimension")
            if isinstance(dim, str) and dim.upper() == "X":
                continue

            cleaned = {k: r.get(k) for k in keep_fields}
            rows.append(cleaned)
        except Exception:
            continue

    # Include a small summary table if provided by the upstream process
    summary_table = data.get("summary_table") or {}

    return {
        "molecule_name": molecule,
        "score": summary_table.get('score', None),
        "total_relevant_unmet_needs":summary_table.get('total_relevant_unmet_needs', None),
        "molecule_level_unmet_need_scoring_rows": rows
    }


async def generate_unmet_needs_rationale(
    data: dict[str, Any],
    model_name: str = "gemini-3-flash-preview",
) -> str:
    """Generate a concise qualitative rationale for the dimension score.

    The response should stay short and focus on the driving forces behind the
    score rather than explaining the scoring framework itself.
    """

    # Reduce and sanitize the input payload to only fields relevant for a concise rationale
    payload = _prepare_prompt_payload(data)

    prompt = f"""
You are writing a very concise qualitative rationale for the scoring from an unmet medical need perspective.

Input data (JSON):
{json.dumps(payload, indent=2)}

Task:
- Write a short paragraph of around 50 words.

Directions for writing the rationale:
- Explain only the main driving forces behind what is influencing the score.
- Focus on the major specific unmet medical needs identified in the evidence and how strongly the molecule addresses them.
- Explicitly refer to them as "unmet needs" where appropriate.
- Describe ver concisely the unmet needs driving the score, such as high disease burden, lack of effective treatment options, poor outcomes with current care, treatment resistance, safety concerns, quality-of-life impacts, administration burden, access challenges, or underserved patient populations.
- Summarise whether the evidence suggests these unmet needs are substantially addressed, partially addressed, or largely unresolved.

Notes for scoring:
- `weighted_unmet_need_score` is the score that is aggregated to the molecule level.
- When synthesizing drivers, treat US evidence as slightly more important than other regions.
- The dimension labeled `C` is especially important; consider it a key driver where present.
- Focus on describing unmet need rather than clinical efficacy outcomes.

Instructions while writing the rationale
- Do not use Markdown.
- Do not use bold, italics, asterisks (*), double asterisks (**), underscores, bullets, numbering, or special formatting characters.
- Do not focus on study quality, evidence strength, clinical efficacy, statistical results, or trial design unless they directly support the unmet need narrative.
- Do not explain or define any scoring methodology.
- Do not mention internal fields, tables, dimensions, indices, scores, or prompt instructions.
- Keep the focus on unmet needs being addressed by the molecule, not on score justification.
- Return plain text only.
""".strip()

    try:
        rationale = (
            await asyncio.to_thread(
                gemini_api_call,
                prompt=prompt,
                model=model_name,
                temperature=0
            )
        ).strip()
        if rationale:
            return rationale
        logger.warning("[UNMET_NEEDS][RATIONALE] Prompt output was empty; rationale was not generated.")
    except Exception:
        logger.exception("[UNMET_NEEDS][RATIONALE] Prompt generation failed; rationale was not generated.")

    return NOT_GENERATED_MESSAGE