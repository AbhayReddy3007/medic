"""Judge faithfulness: is the assigned clinical-evidence score supported by the text and recorded reasoning?"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from medical_potential.config import CLAUDE_SONNET_MODEL
from .utils import REPORT_FIELDS, call_claude_json, source_key, text_value

logger = logging.getLogger(__name__)

RUBRIC = """CLINICAL EVIDENCE SCORING RULES (higher score = stronger clinical evidence; follow EXACTLY):

Final = min(5, max(1, (P + mode(D, G, E)) / 2 + C)); if D, G and E are all different the median is used.

- P (phase): Phase 3/4 = 5, Phase 2 = 4, Phase 1 = 2, none = 0.
- D (design): best study design - RCT = 5, non-randomised = 3, single-arm = 2, observational = 2, none = 0.
- G (geography): US/EU/UK = 5; Canada, Switzerland, Australia, Japan = 4; other or unknown = 3.
- E (evidence support, 0/1/3/4/5): strength of post-launch real-world evidence; 5 = approved with strong multi-source
  real-world evidence; 1 = trials registered but no results.
- C (consistency, +0.5 / +0.25 / -0.5 / -1.0): rewards replicated, agreeing findings; penalises conflicting or
  unavailable evidence.

Evidence level: STRONG = Phase 3/4 trial, at least 2 trials with results and at least 2 RCTs or an FDA approval;
MODERATE = at least one RCT with results; LIMITED = only Phase 1/2 studies; INSUFFICIENT = nothing with results.

Score meaning: 5 = mature, well-replicated, well-designed evidence; 3 = partial or immature evidence;
1 = little or no usable clinical evidence."""


def _output_text(row: pd.Series) -> str:
    return "\n\n".join(f"{field}: {text_value(row, field)}" for field in REPORT_FIELDS)


def _faithfulness_prompt(row: pd.Series) -> str:
    return f"""Judge whether this Clinical Evidence output is internally supported by its own text, rationale and scoring evidence.
Assigned final score: {row.get('score')} (1-5, higher is stronger evidence)
Components: P={row.get('score_p')}, D={row.get('score_d')}, G={row.get('score_g')}, E={row.get('score_e')}, C={row.get('score_c')}
Evidence level: {row.get('evidence_level')} | FDA approved: {row.get('fda_approved')} | Trials: {row.get('total_trials')} | RCTs: {row.get('rct_count')}

Text and rationale:
{_output_text(row)}

Scoring reasoning recorded by the pipeline (per component):
{text_value(row, 'score_reasoning')}

Rubric:
{RUBRIC}

Return JSON only using this schema:
- score_supported: boolean; true when the text, rationale and recorded reasoning provide sufficient evidence for the assigned score and its components, otherwise false.
- score_supported_rationale: string; brief explanation of the evidence supporting or failing to support the assigned score.
- why_not_higher_valid: boolean; true when the output appropriately explains what keeps the score from being higher (immature phase, missing results or publications, no approval, inconsistent findings, weak real-world evidence), or, for a score of 5, that nothing material lowers it; otherwise false.
- why_not_higher_valid_rationale: string; brief explanation of whether the output rules out a higher score.
- internal_consistency: boolean; true when the claims in the text, the rationale and the recorded reasoning do not materially contradict one another, otherwise false.
- internal_consistency_rationale: string; brief explanation of any contradiction or lack of contradiction.
- overall_rating: integer from 1 to 5; overall faithfulness of the assigned assessment, where 1 is poorly supported and 5 is strongly supported.
- overall_rating_rationale: string; 2-4 sentences explaining how the boolean decisions and evidence support the final overall rating.

Example:
{{"score_supported": true, "score_supported_rationale": "Phase 3 trials with results and RCT designs support P=5 and D=5.", "why_not_higher_valid": true, "why_not_higher_valid_rationale": "The output notes that no peer-reviewed publications or FDA approval exist, which limits E and C.", "internal_consistency": true, "internal_consistency_rationale": "The text, rationale and recorded reasoning agree.", "overall_rating": 4, "overall_rating_rationale": "The score is strongly supported, although the consistency estimate is based on few studies with results."}}"""


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
