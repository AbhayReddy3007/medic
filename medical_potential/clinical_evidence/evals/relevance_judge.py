"""Relevance evals for Clinical Evidence outputs.

* Field relevance: does each judged text field answer the question it exists to answer for this drug
  (1-5, judged against a per-field rubric)?
* Claim relevance: of the factual claims extracted from the text, how many bear on the assigned score,
  versus generic padding?

Source relevance is measured separately in `context_precision.py`.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from medical_potential.config import CLAUDE_SONNET_MODEL
from medical_potential.moa_innovation.evals.relevance_judge import summarize_claim_relevance  # noqa: F401
from .utils import REPORT_FIELDS, call_claude_json, field_short_name, source_key, text_value

logger = logging.getLogger(__name__)

FIELD_QUESTIONS = {
    "rwe_synthesis": "What does real-world and post-launch evidence say about this drug's outcomes, how consistent and strong is it, and how well does it align with the trials?",
    "llm_consistency_report": "How consistent are the findings across this drug's trials, and what share of studies agree?",
    "evidence_gaps": "What evidence is missing or weak for this drug (results, publications, approval, replication)?",
    "rationale": "What are the main drivers that push this drug's clinical evidence score up or down?",
}
CLAIM_BATCH = 40
CLAIM_LABELS = ("direct", "contextual", "irrelevant")


def _field_prompt(row: pd.Series) -> str:
    sections = "\n\n".join(
        f"[{field}]\nQuestion this field must answer: {FIELD_QUESTIONS[field]}\nText: {text_value(row, field) or '(empty)'}"
        for field in REPORT_FIELDS
    )
    return f"""Judge the RELEVANCE of each field in a Clinical Evidence assessment.
Relevance means the text answers its own question for THIS drug. It does not mean the text is true.
Drug: {row.get('drug_name')}
Evidence level: {row.get('evidence_level')} | FDA approved: {row.get('fda_approved')}
Assigned score: {row.get('score')} (1-5, higher is stronger evidence)

{sections}

Rate each field 1-5:
5 = directly and specifically answers the question for this drug, with no padding.
4 = answers the question, with minor generic or off-topic content.
3 = partly answers it; substantial generic class-level content or a missing sub-part.
2 = mostly off-topic, generic, or about a different drug or indication.
1 = does not answer the question, or is empty.
Also mark `drug_specific` true only when the text makes statements about this drug itself, not just the class.

Return JSON only as {{"fields": [{{"field": "rwe_synthesis", "score": 4, "drug_specific": true, "rationale": "one or two sentences"}}]}}.
Return one entry for every field listed above."""


def judge_field_relevance(row: pd.Series, model: str = CLAUDE_SONNET_MODEL) -> dict:
    result, _ = call_claude_json(_field_prompt(row), model=model)
    entries = result.get("fields", []) if isinstance(result, dict) else []
    by_field = {entry.get("field"): entry for entry in entries if isinstance(entry, dict)}

    record: dict = {"source_key": source_key(row), "drug_name": row.get("drug_name")}
    scores = []
    for field in REPORT_FIELDS:
        entry = by_field.get(field, {})
        try:
            score = min(5, max(1, int(entry.get("score"))))
        except (TypeError, ValueError):
            score = None
        record[f"relevance_{field_short_name(field)}"] = score
        if score is not None:
            scores.append(score)
    record["field_relevance_mean"] = sum(scores) / len(scores) if scores else None
    record["field_relevance_score"] = (record["field_relevance_mean"] - 1) / 4 if scores else None
    record["fields_not_drug_specific"] = sum(
        1 for field in REPORT_FIELDS if by_field.get(field, {}).get("drug_specific") is False
    )
    record["relevance_details"] = {"fields": entries}
    return record


def _safe_field_relevance(row: pd.Series, model: str) -> dict:
    try:
        return judge_field_relevance(row, model)
    except Exception as error:
        logger.exception("Field relevance failed for %s", row.get("drug_name"))
        return {"source_key": source_key(row), "drug_name": row.get("drug_name"), "relevance_error": str(error)}


def run_field_relevance(rows: pd.DataFrame, model: str = CLAUDE_SONNET_MODEL) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    with ThreadPoolExecutor(max_workers=min(5, len(rows))) as executor:
        results = list(executor.map(lambda item: _safe_field_relevance(item[1], model), rows.iterrows()))
    return pd.DataFrame(results)


# --- claim relevance --------------------------------------------------------------


def _claim_relevance_prompt(row: pd.Series, batch: list[dict]) -> str:
    listing = "\n".join(f"{item['id']}. [{item.get('field')}] {item.get('claim')}" for item in batch)
    return f"""Judge whether each factual claim bears on the Clinical Evidence assessment of this drug.
Drug: {row.get('drug_name')}
Evidence level: {row.get('evidence_level')} | FDA approved: {row.get('fda_approved')}
Assigned score: {row.get('score')} (1-5, higher is stronger evidence)

Labels:
- direct: the claim is about this drug's trial phases, designs, sizes, results, consistency, regulatory approval or real-world outcomes, and is evidence for or against the assigned score.
- contextual: the claim is about the drug class, the indication, the sponsor or the development programme, and is needed to interpret the score, but is not specific to this drug's evidence.
- irrelevant: the claim is generic background, trivia, or does not affect the evidence score.

Claims:
{listing}

Return JSON only as {{"claims": [{{"id": 1, "label": "direct|contextual|irrelevant"}}]}}. Return one entry for every claim id."""


def run_claim_relevance(
    claim_rows: pd.DataFrame, rows: pd.DataFrame, model: str = CLAUDE_SONNET_MODEL
) -> pd.DataFrame:
    """Label every extracted claim as direct / contextual / irrelevant. Returns one row per claim."""
    if claim_rows.empty or "claim" not in claim_rows.columns:
        return pd.DataFrame()
    rows_by_key = {source_key(row): row for _, row in rows.iterrows()}
    labelled = []
    for key, group in claim_rows.groupby("source_key"):
        row = rows_by_key.get(key)
        claims = [
            {"id": index + 1, "field": claim.get("field"), "claim": claim.get("claim"), "_position": position}
            for index, (position, claim) in enumerate(group.iterrows())
        ]
        for start in range(0, len(claims), CLAIM_BATCH):
            batch = claims[start : start + CLAIM_BATCH]
            labels: dict = {}
            error = None
            try:
                if row is None:
                    raise ValueError(f"no source row found for {key}")
                result, _ = call_claude_json(_claim_relevance_prompt(row, batch), model=model)
                entries = result.get("claims", []) if isinstance(result, dict) else []
                labels = {e.get("id"): str(e.get("label", "")).strip().lower() for e in entries if isinstance(e, dict)}
            except Exception as exc:
                logger.exception("Claim relevance batch failed for %s", key)
                error = str(exc)
            for item in batch:
                label = labels.get(item["id"])
                labelled.append({
                    "source_key": key,
                    "drug_name": group.loc[item["_position"]].get("drug_name"),
                    "field": item["field"],
                    "claim": item["claim"],
                    "claim_relevance": label if label in CLAIM_LABELS else "error",
                    "claim_relevance_error": error,
                })
    return pd.DataFrame(labelled)
