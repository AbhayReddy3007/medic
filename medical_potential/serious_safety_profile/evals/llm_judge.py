"""Judge faithfulness: is the assigned safety score supported by the report and rationale itself?"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from medical_potential.config import CLAUDE_SONNET_MODEL
from .utils import REPORT_FIELDS, call_claude_json, source_key, text_value

logger = logging.getLogger(__name__)

RUBRIC = """SERIOUS SAFETY PROFILE SCORING RULES (higher score = safer; follow EXACTLY):

Base score, from trial-level delta SAE (drug SAE rate minus control SAE rate, in percentage points):
- S1 (delta <= 0) -> 5; S2 (0 < delta <= 0.5) -> 4; S3 (0.5 < delta <= 2.0) -> 3; S4 (delta > 2.0) -> 2.
- Trials are weighted by phase, geography, dosage and size. If S4 trials carry more than 20% of the normalised weight
  the base score is 2. Otherwise S4 trials are treated as S3 and the dominant band among the highest-weight trials
  sets the base score; ties go to the lower score.
- If no trial has a usable delta, or there are no classified SAE events, the default score is 1.

Adjustments applied after the base score:
- At least two trials with positive delta SAE and drug-related deaths from severity-4 events force the score to 1.
- Life-threatening events together with an S4 trial reduce the score by 1.
- Repeated organ-specific events reduce the score by 1; repeated life-threatening events reduce it by 1.
- Each clinically meaningful unexpected event, and each clinically meaningful expected-but-rare event, reduces it by 1.

Caps:
- Regulatory: market-threatening action, withdrawal or rejection caps at 1; established serious human risk caps at 2;
  monitoring or REMS caps at 3; informational or precautionary action, or none, does not cap.
- Post-marketing (marketed molecules only): a new serious risk reduces the score by 1; a boxed warning or withdrawal
  caps at 2; a REMS or major label restriction caps at 3.
- The final score is clamped to 1-5.

Score meaning: 5 = no excess serious risk; 4 = small excess; 3 = moderate excess or some signals; 2 = substantial excess
or a serious class signal; 1 = severe, repeated or regulatory-level serious risk."""


def _output_text(row: pd.Series) -> str:
    return "\n\n".join(f"{field}: {text_value(row, field)}" for field in REPORT_FIELDS)


def _faithfulness_prompt(row: pd.Series) -> str:
    return f"""Judge whether this Serious Safety Profile output is internally supported by its own report, rationale and scoring evidence.
Assigned score: {row.get('score')} (1-5, higher is safer)
Approval status: {row.get('status_summary')}

Report and rationale:
{_output_text(row)}

Scoring reasoning recorded by the pipeline (step-by-step):
{text_value(row, 'score_reasoning')}

Key drivers: {text_value(row, 'key_drivers')}
Adjustments applied: {text_value(row, 'adjustments_applied')}

Rubric:
{RUBRIC}

Return JSON only using this schema:
- score_supported: boolean; true when the report, rationale and recorded reasoning provide sufficient evidence for the assigned score, otherwise false.
- score_supported_rationale: string; brief explanation of the evidence supporting or failing to support the assigned score.
- why_not_higher_valid: boolean; true when the output appropriately explains what keeps the score from being higher (the excess SAE rates, repeated or severe events, unexpected events, regulatory or post-marketing limits), or, for a score of 5, that nothing material lowers it; otherwise false.
- why_not_higher_valid_rationale: string; brief explanation of whether the output rules out a higher score.
- internal_consistency: boolean; true when the claims in the report, the rationale and the recorded reasoning do not materially contradict one another, otherwise false.
- internal_consistency_rationale: string; brief explanation of any contradiction or lack of contradiction.
- overall_rating: integer from 1 to 5; overall faithfulness of the assigned assessment, where 1 is poorly supported and 5 is strongly supported.
- overall_rating_rationale: string; 2-4 sentences explaining how the boolean decisions and evidence support the final overall rating.

Example:
{{"score_supported": true, "score_supported_rationale": "Repeated positive delta SAE and a recorded unexpected event support a score of 2.", "why_not_higher_valid": true, "why_not_higher_valid_rationale": "The output explains the excess SAE burden and repeated organ-specific events that hold the score down.", "internal_consistency": true, "internal_consistency_rationale": "The report, rationale and recorded reasoning agree.", "overall_rating": 4, "overall_rating_rationale": "The score is strongly supported, although missing SAE rates in some trials add uncertainty."}}"""


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
