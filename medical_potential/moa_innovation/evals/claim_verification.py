from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from medical_potential.config import CLAUDE_SONNET_MODEL
from .utils import JUSTIFICATION_FIELDS, call_claude_json, load_saved_sources, source_key, text_value

logger = logging.getLogger(__name__)

CLAIMS_PER_CALL = 6
MAX_SOURCES_IN_PROMPT = 60


def _claim_prompt(row: pd.Series) -> str:
    sections = "\n\n".join(
        f"{field}: {text_value(row, field)}" for field in JUSTIFICATION_FIELDS if text_value(row, field)
    )
    return f"""Extract atomic, independently checkable factual claims from these MoA justifications.
Drug: {row.get('drug_name')}
Indication: {row.get('indication')}
Analysis date: {row.get('analysis_date')}

{sections}

Exclude opinions, recommendations, score judgements, and claims that cannot be checked externally.
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
    return "\n".join(lines) if lines else "(no saved sources found for this run)"


def _verify_prompt(claims: list[dict], row: pd.Series, sources: list[dict]) -> str:
    return f"""You are an evidence reviewer for a Mechanism-of-Action Innovation assessment. The assessment was produced from the web sources listed below. Verify each factual claim independently by fetching those sources with the web_fetch tool; do not accept the generated justification as evidence.

Drug: {row.get('drug_name')}
Indication: {row.get('indication')}
Analysis date: {row.get('analysis_date')}

Sources the assessment drew on (the only URLs web_fetch can open):
{_sources_block(sources)}

Fetch the pages most relevant to the claims, using the "used in" step hints. You cannot search; claims that the fetched pages do not address are "unverifiable".

Claims:
{json.dumps(claims, indent=2, default=str)}

Source-type guidance by justification field:
- `justification_mechanism_summary` and `justification_biological_rationale`: peer-reviewed research and authoritative reviews.
- `justification_novelty_vs_soc`: FDA/EMA records and labels, current treatment guidelines.
- `justification_competitor_comparison`: regulatory records, ClinicalTrials.gov, primary publications, official company pipeline disclosures (company claims are secondary evidence).
- `justification_prior_validation_failure`: FDA/EMA records, ClinicalTrials.gov, peer-reviewed publications. Distinguish mechanism-level failure from failure of a single molecule caused by safety, PK, formulation, trial design or strategy.
- `justification_why_not_higher_score`: the source types above that directly support the comparison made.

Rules:
1. Check that the source concerns this drug, mechanism, indication and time period. Evidence about a related molecule is not direct evidence.
2. Absence of evidence is "unverifiable", not "contradicted". Mark "contradicted" only when a reliable source directly conflicts with the claim.
3. Separate what the source reports from your interpretation. Do not infer approval, efficacy, validation or failure from an indirect association.
4. Cite only pages you actually fetched. Never construct, shorten or guess a URL.
5. For each claim write 2-4 sentences naming the specific finding and the source's authority; for unverifiable claims say what was fetched and what was missing.

Return JSON only as {{"verifications": [{{"claim_id": "c1", "verdict": "supported|contradicted|unverifiable", "evidence": "2-4 sentences", "sources": [{{"title": "title or organisation", "url": "retrieved URL"}}]}}]}}.
Return one verification for every input claim, preserving each claim_id. Use an empty sources list when no reliable source was fetched."""


def _norm_url(url: str | None) -> str:
    return (url or "").strip().rstrip("/").lower()


def _verify_batch(batch: list[dict], row: pd.Series, saved: dict, model: str) -> list[dict]:
    if not saved["sources"]:
        raise ValueError("no saved sources to fetch")
    result, citations = call_claude_json(
        _verify_prompt(batch, row, saved["sources"]),
        model=model,
        fetch_urls=[source["url"] for source in saved["sources"]],
        max_web_fetches=max(6, len(batch) * 2),
    )
    retrieved = {_norm_url(c["url"]) for c in citations}
    saved_urls = {_norm_url(s["url"]) for s in saved["sources"]}
    verifications = result.get("verifications", []) if isinstance(result, dict) else []
    by_id = {}
    for item in verifications:
        cited = [src for src in (item.get("sources") or []) if isinstance(src, dict)]
        by_id[item.get("claim_id")] = {
            **item,
            # Only URLs the fetch tool actually retrieved count as real evidence.
            "verified_sources": [
                {**src, "in_saved_sources": _norm_url(src.get("url")) in saved_urls}
                for src in cited
                if _norm_url(src.get("url")) in retrieved
            ],
            "unretrieved_cited_urls": [
                src.get("url") for src in cited if _norm_url(src.get("url")) not in retrieved
            ],
            "search_scope": "saved_sources_fetch",
        }
    missing = {"verdict": "error", "evidence": "Verifier returned no result for this claim", "verified_sources": []}
    return [{**claim, **by_id.get(claim.get("claim_id"), missing)} for claim in batch]


def verify_claims(
    claims: list[dict],
    row: pd.Series,
    model: str = CLAUDE_SONNET_MODEL,
) -> list[dict]:
    """Verify claims by letting Claude fetch the source pages saved for this run."""
    if not claims:
        return []
    saved = load_saved_sources(row)
    if not saved["sources"]:
        logger.warning("No saved sources found for %s; claims will be marked as errors", row.get("drug_name"))
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


def run_claim_verification(
    rows: pd.DataFrame,
    model: str = CLAUDE_SONNET_MODEL,
) -> pd.DataFrame:
    records = []
    with ThreadPoolExecutor(max_workers=min(5, max(1, len(rows)))) as executor:
        extracted = list(executor.map(lambda item: _safe_extract(item[1], model), rows.iterrows()))
    for (_, row), claims in zip(rows.iterrows(), extracted):
        records.extend(verify_claims(claims, row, model))
    return pd.DataFrame(records)


def summarize_claims(claims: pd.DataFrame) -> pd.DataFrame:
    if claims.empty:
        return pd.DataFrame(columns=["source_key", "claims_total", "claims_supported", "claims_contradicted", "claims_unverifiable", "claims_error", "claim_details"])
    grouped = []
    for key, group in claims.groupby("source_key"):
        counts = group["verdict"].value_counts()
        grouped.append({
            "source_key": key,
            "claims_total": len(group),
            "claims_supported": int(counts.get("supported", 0)),
            "claims_contradicted": int(counts.get("contradicted", 0)),
            "claims_unverifiable": int(counts.get("unverifiable", 0)),
            "claims_error": int(counts.get("error", 0)),
            "claim_details": group.to_dict("records"),
        })
    return pd.DataFrame(grouped)
