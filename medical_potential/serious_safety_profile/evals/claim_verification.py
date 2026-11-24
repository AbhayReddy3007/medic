"""Claim precision (faithfulness): are the factual claims in the safety report supported by evidence?

Claims are extracted from the report sections and the rationale, then verified by Claude against
  1. the web sources the run retrieved (fetched with web_fetch), and
  2. `pipeline_data`, the structured evidence the pipeline itself held (trial rates, SAE events,
     regulatory and post-marketing findings), which is the source for many figures in the report.
Unverifiable and errored claims count against the score, exactly as in the MoA evals.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from medical_potential.config import CLAUDE_SONNET_MODEL
from medical_potential.moa_innovation.evals.claim_verification import summarize_claims  # noqa: F401
from .utils import REPORT_FIELDS, call_claude_json, load_saved_sources, source_key, text_value

logger = logging.getLogger(__name__)

CLAIMS_PER_CALL = 6
MAX_SOURCES_IN_PROMPT = 60


def _claim_prompt(row: pd.Series) -> str:
    sections = "\n\n".join(
        f"{field}: {text_value(row, field)}" for field in REPORT_FIELDS if text_value(row, field)
    )
    return f"""Extract atomic, independently checkable factual claims from this serious safety assessment of a drug.
Drug: {row.get('drug_name')}
Approval status: {row.get('status_summary')}

{sections}

Include claims about trial safety results, serious adverse events, regulatory actions, approval status and post-marketing findings.
Exclude opinions, recommendations, score judgements, statements about how the score was derived, and claims that cannot be checked externally.
Return JSON only as {{"claims": [{{"claim_id": "c1", "field": "source field", "claim": "one factual claim"}}]}}."""


def extract_claims(row: pd.Series, model: str = CLAUDE_SONNET_MODEL) -> list[dict]:
    result, _ = call_claude_json(_claim_prompt(row), model=model)
    claims = result.get("claims", []) if isinstance(result, dict) else []
    return [dict(claim, source_key=source_key(row), drug_name=row.get("drug_name")) for claim in claims]


def _sources_block(sources: list[dict]) -> str:
    lines = [
        f"- {source['url']} | {source.get('title') or 'untitled'} | used in: {', '.join(source['steps'])}"
        for source in sources[:MAX_SOURCES_IN_PROMPT]
    ]
    return "\n".join(lines) if lines else "(no saved web sources for this run; use the pipeline data only)"


def _verify_prompt(claims: list[dict], row: pd.Series, sources: list[dict]) -> str:
    fetch_note = (
        "Verify each claim by fetching the web sources listed below with the web_fetch tool and by checking the pipeline data."
        if sources
        else "No web sources are available; check each claim against the pipeline data only."
    )
    return f"""You are an evidence reviewer for a Serious Safety Profile assessment of a drug. {fetch_note} Do not accept the generated report as evidence.

Drug: {row.get('drug_name')}
Approval status: {row.get('status_summary')}

Pipeline data (structured trial, adverse-event, regulatory and post-marketing evidence the assessment was built from):
{text_value(row, 'pipeline_data')}

Web sources the assessment drew on (the only URLs web_fetch can open):
{_sources_block(sources)}

You cannot search. A claim that neither the pipeline data nor a fetched page addresses is "unverifiable".

Claims:
{json.dumps(claims, indent=2, default=str)}

Source-type guidance by report field:
- `report_serious_safety_profile`: overall safety picture; pipeline data, labels and regulatory records.
- `report_clinical_trial_safety`: trial results; ClinicalTrials.gov and other registries, primary publications, the pipeline trial data.
- `report_serious_adverse_events_found`: specific events and their frequency; trial results, product labels, regulatory safety communications.
- `report_regulatory_assessment`: FDA/EMA records, labels, boxed warnings, REMS, safety communications.
- `report_post_marketing_safety`: FDA/EMA post-marketing records, label changes, withdrawals, pharmacovigilance publications.
- `rationale`: the factual statements it makes, checked as above.

Rules:
1. Check that the evidence concerns this drug, its dose and indication, and the right time period. Evidence about a related molecule is not direct evidence.
2. Absence of evidence is "unverifiable", not "contradicted". Mark "contradicted" only when reliable evidence directly conflicts with the claim.
3. Separate what the evidence reports from your interpretation. Do not infer causality or approval from an indirect association.
4. Cite only pages you actually fetched. Never construct, shorten or guess a URL.
5. For each claim write 2-4 sentences naming the specific finding and the authority of the evidence; for unverifiable claims say what was checked and what was missing.
6. Set evidence_basis to "web_source" when a fetched page supports or contradicts the claim, "pipeline_data" when only the pipeline data does, otherwise "none".

Return JSON only as {{"verifications": [{{"claim_id": "c1", "verdict": "supported|contradicted|unverifiable", "evidence_basis": "web_source|pipeline_data|none", "evidence": "2-4 sentences", "sources": [{{"title": "title or organisation", "url": "retrieved URL"}}]}}]}}.
Return one verification for every input claim, preserving each claim_id. Use an empty sources list when no page was fetched."""


def _norm_url(url: str | None) -> str:
    return (url or "").strip().rstrip("/").lower()


def _verify_batch(batch: list[dict], row: pd.Series, saved: dict, model: str) -> list[dict]:
    kwargs = {}
    if saved["sources"]:
        kwargs = {
            "fetch_urls": [source["url"] for source in saved["sources"]],
            "max_web_fetches": max(6, len(batch) * 2),
        }
    result, citations = call_claude_json(_verify_prompt(batch, row, saved["sources"]), model=model, **kwargs)
    retrieved = {_norm_url(c["url"]) for c in citations}
    saved_urls = {_norm_url(s["url"]) for s in saved["sources"]}
    verifications = result.get("verifications", []) if isinstance(result, dict) else []
    by_id = {}
    for item in verifications:
        cited = [src for src in (item.get("sources") or []) if isinstance(src, dict)]
        by_id[item.get("claim_id")] = {
            **item,
            # Only URLs the fetch tool actually retrieved count as real web evidence.
            "verified_sources": [
                {**src, "in_saved_sources": _norm_url(src.get("url")) in saved_urls}
                for src in cited
                if _norm_url(src.get("url")) in retrieved
            ],
            "unretrieved_cited_urls": [
                src.get("url") for src in cited if _norm_url(src.get("url")) not in retrieved
            ],
            "search_scope": "saved_sources_fetch_and_pipeline_data" if saved["sources"] else "pipeline_data",
        }
    missing = {"verdict": "error", "evidence": "Verifier returned no result for this claim", "verified_sources": []}
    return [{**claim, **by_id.get(claim.get("claim_id"), missing)} for claim in batch]


def verify_claims(claims: list[dict], row: pd.Series, model: str = CLAUDE_SONNET_MODEL) -> list[dict]:
    """Verify claims against the run's saved web sources and its pipeline data."""
    if not claims:
        return []
    saved = load_saved_sources(row)
    if not saved["sources"]:
        logger.warning("No saved sources found for %s; claims are checked against pipeline data only", row.get("drug_name"))
    verified: list[dict] = []
    for start in range(0, len(claims), CLAIMS_PER_CALL):
        batch = claims[start : start + CLAIMS_PER_CALL]
        try:
            verified.extend(_verify_batch(batch, row, saved, model))
        except Exception as error:
            logger.exception("Claim verification batch failed for %s", row.get("drug_name"))
            verified.extend(
                {**claim, "verdict": "error", "evidence": f"Verification failed: {error}", "verified_sources": []}
                for claim in batch
            )
    return [{**claim, "sources_file": saved["file"]} for claim in verified]


def _safe_extract(row: pd.Series, model: str) -> list[dict]:
    try:
        return extract_claims(row, model)
    except Exception:
        logger.exception("Claim extraction failed for %s", row.get("drug_name"))
        return []


def run_claim_verification(rows: pd.DataFrame, model: str = CLAUDE_SONNET_MODEL) -> pd.DataFrame:
    records = []
    with ThreadPoolExecutor(max_workers=min(5, max(1, len(rows)))) as executor:
        extracted = list(executor.map(lambda item: _safe_extract(item[1], model), rows.iterrows()))
    for (_, row), claims in zip(rows.iterrows(), extracted):
        records.extend(verify_claims(claims, row, model))
    return pd.DataFrame(records)
