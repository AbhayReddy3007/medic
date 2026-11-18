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


def _verify_prompt(claims: list[dict], row: pd.Series, sources: list[dict], restricted: bool) -> str:
    scope = (
        "Web search is restricted to the domains of the sources below. Search for and open the specific pages listed, "
        "and use other pages on those domains only when needed to confirm or refute a claim."
        if restricted
        else "No saved sources were found, so search the open web, preferring authoritative primary sources."
    )
    return f"""You are an evidence reviewer for a Mechanism-of-Action Innovation assessment. The assessment was produced from the web sources listed below. Verify each factual claim independently by searching those sources; do not accept the generated justification as evidence.

Drug: {row.get('drug_name')}
Indication: {row.get('indication')}
Analysis date: {row.get('analysis_date')}

Sources the assessment drew on:
{_sources_block(sources)}

{scope}

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
4. Cite only pages your search actually retrieved. Never construct, shorten or guess a URL.
5. For each claim write 2-4 sentences naming the specific finding and the source's authority; for unverifiable claims say what was searched and what was missing.

Return JSON only as {{"verifications": [{{"claim_id": "c1", "verdict": "supported|contradicted|unverifiable", "evidence": "2-4 sentences", "sources": [{{"title": "title or organisation", "url": "retrieved URL"}}]}}]}}.
Return one verification for every input claim, preserving each claim_id. Use an empty sources list when no reliable source was retrieved."""


def _norm_url(url: str | None) -> str:
    return (url or "").strip().rstrip("/").lower()


def _verify_batch(batch: list[dict], row: pd.Series, saved: dict, model: str, restrict: bool) -> list[dict]:
    domains = saved["domains"] if restrict and saved["domains"] else None
    result, citations = call_claude_json(
        _verify_prompt(batch, row, saved["sources"], restricted=bool(domains)),
        model=model,
        web_search=True,
        allowed_domains=domains,
        max_web_searches=max(6, len(batch) * 2),
    )
    retrieved = {_norm_url(c["url"]) for c in citations}
    saved_urls = {_norm_url(s["url"]) for s in saved["sources"]}
    verifications = result.get("verifications", []) if isinstance(result, dict) else []
    by_id = {}
    for item in verifications:
        cited = [src for src in (item.get("sources") or []) if isinstance(src, dict)]
        by_id[item.get("claim_id")] = {
            **item,
            # Only URLs the search tool actually returned count as real evidence.
            "verified_sources": [
                {**src, "in_saved_sources": _norm_url(src.get("url")) in saved_urls}
                for src in cited
                if _norm_url(src.get("url")) in retrieved
            ],
            "unretrieved_cited_urls": [
                src.get("url") for src in cited if _norm_url(src.get("url")) not in retrieved
            ],
            "search_scope": "saved_source_domains" if domains else "open_web",
        }
    missing = {"verdict": "error", "evidence": "Verifier returned no result for this claim", "verified_sources": []}
    return [{**claim, **by_id.get(claim.get("claim_id"), missing)} for claim in batch]


def verify_claims(
    claims: list[dict],
    row: pd.Series,
    model: str = CLAUDE_SONNET_MODEL,
    restrict_to_saved_sources: bool = True,
) -> list[dict]:
    """Verify claims with Claude web search, scoped to the domains of the sources saved for this run."""
    if not claims:
        return []
    saved = load_saved_sources(row)
    if not saved["sources"]:
        logger.warning("No saved sources found for %s; verifying against the open web", row.get("drug_name"))
    verified: list[dict] = []
    for start in range(0, len(claims), CLAIMS_PER_CALL):
        batch = claims[start : start + CLAIMS_PER_CALL]
        try:
            verified.extend(_verify_batch(batch, row, saved, model, restrict_to_saved_sources))
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
    restrict_to_saved_sources: bool = True,
) -> pd.DataFrame:
    records = []
    with ThreadPoolExecutor(max_workers=min(5, max(1, len(rows)))) as executor:
        extracted = list(executor.map(lambda item: _safe_extract(item[1], model), rows.iterrows()))
    for (_, row), claims in zip(rows.iterrows(), extracted):
        records.extend(verify_claims(claims, row, model, restrict_to_saved_sources))
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
