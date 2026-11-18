from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import logging

from medical_potential.config import CLAUDE_SONNET_MODEL
from .utils import JUSTIFICATION_FIELDS, call_claude_json, source_key, text_value

logger = logging.getLogger(__name__)

RUBRIC = """SCORING RULES (from documentation - follow EXACTLY):

Score 5 = Exceptional (First-in-Class):
- True FIC mechanism (target/pathway is NEW to this disease setting)
- No prior approved or clearly established class exists
- Strong biological rationale
- Meaningful potential to create a new treatment class
- Do NOT give 5 for novelty alone - biology must be convincing

Score 4 = Strong (Best-in-Class):
- Known VALIDATED mechanism (approved drugs exist using similar mechanism)
- Clearly SUPERIOR implementation with real mechanistic differentiation
- Examples: dual/triple/quad agonism vs single target, better selectivity, better pathway modulation
- Overcomes known mechanistic limitations of the class
- This is the typical BIC score

Score 3 = Moderate (Me-too / Fast Follower):
- Known validated mechanism
- LIMITED mechanism-level differentiation
- Competes mainly on non-MoA factors (dosing, formulation, convenience)
- Standard fast follower in established class

Score 2 = Weak (Outdated):
- Mechanism is OLDER or strategically OUTDATED
- Class has been REPLACED by better mechanisms
- Market and guidelines have shifted toward better alternatives
- Even if mechanism still works, it is no longer a strong source of innovation

Score 1 = Poor (Invalid):
- WEAK or SPECULATIVE biology (loose association with disease, mostly theoretical)
- OR repeated clinical FAILURES for same mechanism in same indication
- OR mechanism has been clinically INVALIDATED
- No convincing rescue hypothesis

CLASSIFICATION DECISION TREE:
1. Is the mechanism NEW to this indication (no prior approved class)?
    → If YES + strong biology = Score 5 (FIC)
    → If YES + weak biology = Score 1 (Poor - speculative)

2. Is the mechanism VALIDATED but with CLEAR SUPERIOR implementation?
    → If YES (dual/triple/quad agonism, better selectivity, etc.) = Score 4 (BIC)

3. Is the mechanism VALIDATED but with LIMITED differentiation?
    → If YES = Score 3 (Me-too)

4. Is the mechanism OLDER/OUTDATED (replaced by better mechanisms)?
    → If YES = Score 2 (Weak/Outdated)

5. Is the biology WEAK/SPECULATIVE or has mechanism been INVALIDATED?
    → If YES = Score 1 (Poor)

IMPORTANT FOR MULTI-AGONISTS (dual/triple/quad):
- A quad agonist that improves on triple/dual agonism IS mechanistic differentiation
- If each component pathway is individually validated, the combination has STRONG rationale
- Multi-agonism is EXACTLY what BIC (Score 4) describes: "dual agonism instead of single target"
"""


def _justification_text(row: pd.Series) -> str:
    return "\n\n".join(f"{field}: {text_value(row, field)}" for field in JUSTIFICATION_FIELDS)


def _faithfulness_prompt(row: pd.Series) -> str:
    return f"""Judge whether this MoA Innovation output is internally supported by its own justification.
Assigned classification: {row.get('moa_classification')}
Assigned score: {row.get('score')}

Justification:
{_justification_text(row)}

Rubric:
{RUBRIC}

Return JSON only using this schema:
- score_supported: boolean; true when the justification provides sufficient evidence for the assigned score, otherwise false.
- score_supported_rationale: string; brief explanation of the evidence supporting or failing to support the assigned score.
- why_not_higher_valid: boolean; true when the justification appropriately explains why the next higher category does not apply, otherwise false.
- why_not_higher_valid_rationale: string; brief explanation of whether the justification rules out the next higher category.
- internal_consistency: boolean; true when the claims and rationale in the justification do not materially contradict one another, otherwise false.
- internal_consistency_rationale: string; brief explanation of any contradiction or lack of contradiction in the justification.
- overall_rating: integer from 1 to 5; overall faithfulness of the assigned assessment, where 1 is poorly supported and 5 is strongly supported.
- overall_rating_rationale: string; 2-4 sentences explaining how the boolean decisions and evidence support the final overall rating.

Example:
{{"score_supported": true, "score_supported_rationale": "The justification identifies a validated mechanism and limited differentiation, supporting the assigned score.", "why_not_higher_valid": true, "why_not_higher_valid_rationale": "It explains that the evidence does not show the superior implementation required for the next higher category.", "internal_consistency": true, "internal_consistency_rationale": "The mechanism, differentiation, and classification claims are consistent with one another.", "overall_rating": 4, "overall_rating_rationale": "The assessment is strongly supported overall, with clear evidence for the assigned score and no material contradiction, although the differentiation is limited."}}"""


def judge_row(row: pd.Series, model: str = CLAUDE_SONNET_MODEL) -> dict:
    faithfulness, _ = call_claude_json(_faithfulness_prompt(row), model=model)
    return {
        "source_key": source_key(row),
        "drug_name": row.get("drug_name"),
        "score_supported": faithfulness.get("score_supported"),
        "score_supported_rationale": faithfulness.get("score_supported_rationale"),
        "why_not_higher_valid": faithfulness.get("why_not_higher_valid"),
        "why_not_higher_valid_rationale": faithfulness.get("why_not_higher_valid_rationale"),
        "internal_consistency": faithfulness.get("internal_consistency"),
        "internal_consistency_rationale": faithfulness.get("internal_consistency_rationale"),
        "faithfulness_score": faithfulness.get("overall_rating"),
        "faithfulness_score_rationale": faithfulness.get("overall_rating_rationale"),
        "judge_details": {"faithfulness": faithfulness},
    }


def _safe_judge_row(row: pd.Series, model: str) -> dict:
    try:
        return judge_row(row, model)
    except Exception as error:
        logger.exception("LLM judge failed for %s", row.get("drug_name"))
        return {"source_key": source_key(row), "drug_name": row.get("drug_name"), "judge_error": str(error)}


def run_llm_judge(rows: pd.DataFrame, model: str = CLAUDE_SONNET_MODEL) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=min(5, max(1, len(rows)))) as executor:
        results = list(executor.map(lambda item: _safe_judge_row(item[1], model), rows.iterrows()))
    return pd.DataFrame(results)
