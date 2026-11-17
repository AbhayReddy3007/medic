from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import pandas as pd
import requests

from medical_potential.config import GEMINI_PRO_MODEL
from .utils import JUSTIFICATION_FIELDS, call_gemini_json, call_gemini_json_grounded, source_key, text_value

VERTEX_REDIRECT_HOST = "vertexaisearch.cloud.google.com"


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


def extract_claims(row: pd.Series, model: str = GEMINI_PRO_MODEL) -> list[dict]:
    result = call_gemini_json(_claim_prompt(row), model=model)
    claims = result.get("claims", []) if isinstance(result, dict) else []
    return [dict(claim, source_key=source_key(row), drug_name=row.get("drug_name")) for claim in claims]


def _verify_prompt(claims: list[dict], row: pd.Series) -> str:
    return f"""You are an evidence reviewer for a Mechanism-of-Action Innovation assessment. Verify each factual claim independently rather than accepting the generated justification or its listed citations as evidence.

Drug: {row.get('drug_name')}
Indication: {row.get('indication')}
Analysis date: {row.get('analysis_date')}

Claims:
{json.dumps(claims, indent=2, default=str)}

Use the most relevant source type for the justification field:
- `justification_mechanism_summary`: peer-reviewed primary research and authoritative reviews indexed in PubMed, Nature, NEJM, or The Lancet for receptor/pathway activity and physiological effects.
- `justification_novelty_vs_soc`: FDA Orange Book, FDA pages and labels, EMA records, and current treatment guidelines for approvals, approved indications, and standard-of-care status.
- `justification_competitor_comparison`: regulatory records, ClinicalTrials.gov, primary publications, and official company pipeline disclosures for approved and pipeline comparators. Treat company claims as secondary evidence.
- `justification_biological_rationale`: peer-reviewed primary research and authoritative reviews indexed in PubMed, Nature, NEJM, or The Lancet for disease biology and target relevance.
- `justification_prior_validation_failure`: FDA/EMA records, ClinicalTrials.gov, and peer-reviewed publications. Distinguish mechanism-level failure from failure of an individual molecule caused by safety, pharmacokinetics, formulation, trial design, or strategy.
- `justification_why_not_higher_score`: use the source types above that directly support the comparison being made; prioritize regulatory records, primary publications, guidelines, and ClinicalTrials.gov over commentary.

Source rules:
1. Prefer direct, authoritative, primary sources over aggregators, commercial databases, press articles, or uncited summaries.
2. Use at least one strong source when available and add a second independent source for consequential claims about approval, clinical outcomes, mechanism failure, or score-driving differentiation.
3. Check that the source actually concerns this drug, mechanism, indication, and time period. Do not use evidence about a related molecule as direct evidence for the claim.
4. Treat an absence of search results as unverifiable, not contradicted. Mark a claim contradicted only when a reliable source directly conflicts with it.
5. Separate what the source directly reports from your interpretation. Do not infer approval, efficacy, mechanism validation, or mechanism failure from an indirect association.
6. Cite only pages your search actually retrieved, and prefer the specific article, record, or trial page over a homepage, search listing, or PDF index. Never construct, shorten, or guess a URL; omit a source rather than cite one you did not retrieve.
7. Cite the same source name that the retrieved page shows (publisher or journal plus article title) so the link can be matched to the search results.

Evidence requirements:
- Write 2-4 substantive sentences explaining why the verdict is supported, contradicted, or unverifiable.
- Do not copy or lightly paraphrase the claim. Name the relevant finding, approval record, trial result, publication result, or source limitation and connect it explicitly to the claim.
- For supported claims, explain the specific evidence that establishes the claim and identify the source's authority.
- For contradicted claims, state the conflicting finding and explain exactly which part of the claim it disproves.
- For unverifiable claims, state what was searched and what required evidence was missing or too ambiguous to confirm.
- Include source title or organisation and a direct URL whenever available. Do not invent URLs, titles, or trial results.

Return JSON only as {{"verifications": [{{"claim_id": "c1", "verdict": "supported|contradicted|unverifiable", "evidence": "2-4 sentence explanatory assessment citing the relevant finding and source authority", "sources": [{{"title": "source title or organisation", "url": "direct URL if available"}}]}}]}}.
Return one verification for every input claim, preserving each claim_id. Use an empty sources list when no reliable source can be identified."""


def _resolve_url(url: str | None) -> tuple[str | None, bool]:
    """Follow a Vertex grounding redirect to the real page; other hosts are left untouched."""
    if not url:
        return None, False
    if urlparse(url).hostname != VERTEX_REDIRECT_HOST:
        return url, True
    try:
        with requests.get(
            url,
            allow_redirects=True,
            stream=True,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        ) as response:
            final_url = response.url
    except requests.RequestException:
        return url, False
    return (final_url, True) if urlparse(final_url).hostname != VERTEX_REDIRECT_HOST else (url, False)


def _claim_spans(raw: str, claim_ids: list[str]) -> dict[str, tuple[int, int]]:
    """Locate each claim's block in the raw response: from its claim_id to the next claim's."""
    starts = {}
    for claim_id in claim_ids:
        match = re.search(rf'"claim_id"\s*:\s*"{re.escape(str(claim_id))}"', raw)
        if match:
            starts[claim_id] = match.start()
    ordered = sorted(starts.items(), key=lambda item: item[1])
    ends = [start for _, start in ordered[1:]] + [len(raw)]
    return {claim_id: (start, end) for (claim_id, start), end in zip(ordered, ends)}


def _grounded_chunks(span: tuple[int, int] | None, model_sources: list, grounding: dict) -> list[dict]:
    chunks = grounding.get("chunks", [])
    raw = grounding.get("text", "")
    indices: list[int] = []
    for support in grounding.get("supports", []):
        segment = support["text"]
        start = raw.find(segment) if segment else -1
        if span and start >= 0 and start < span[1] and start + len(segment) > span[0]:
            indices.extend(support["chunk_indices"])
    overlapping = [chunks[i] for i in dict.fromkeys(indices) if i < len(chunks) and chunks[i]]

    cited = [str(s.get("title", "")).lower() for s in model_sources if isinstance(s, dict)]

    def is_cited(chunk: dict) -> bool:
        name = str(chunk.get("title") or "").lower()
        return bool(name) and any(name in title or title in name for title in cited if title)

    # Prefer the overlapping chunks the model also named; with no overlap, match by cited name across all chunks.
    pool = overlapping or [chunk for chunk in chunks if chunk]
    return [chunk for chunk in pool if is_cited(chunk)] or overlapping


def _attach_grounded_sources(verification: dict, grounding: dict, span: tuple[int, int] | None) -> dict:
    model_sources = verification.get("sources") or []
    chunks = _grounded_chunks(span, model_sources, grounding)
    with ThreadPoolExecutor(max_workers=min(5, max(1, len(chunks)))) as executor:
        resolved = list(executor.map(lambda chunk: _resolve_url(chunk["url"]), chunks))
    grounded_sources = [
        {"title": chunk["title"], "url": url, "link_resolved": ok, "redirect_url": chunk["url"]}
        for chunk, (url, ok) in zip(chunks, resolved)
    ]
    grounded_source_urls = ", ".join(
        dict.fromkeys(source["url"] for source in grounded_sources if source.get("url"))
    )
    return {
        **verification,
        "grounded_sources": grounded_sources,
        "grounded_source_urls": grounded_source_urls,
    }


def verify_claims(claims: list[dict], row: pd.Series, model: str = GEMINI_PRO_MODEL) -> list[dict]:
    if not claims:
        return []
    result, grounding = call_gemini_json_grounded(_verify_prompt(claims, row), model=model)
    verifications = result.get("verifications", []) if isinstance(result, dict) else []
    spans = _claim_spans(grounding.get("text", ""), [item.get("claim_id") for item in verifications])
    by_id = {
        item.get("claim_id"): _attach_grounded_sources(item, grounding, spans.get(item.get("claim_id")))
        for item in verifications
    }
    return [
        {**claim, **by_id.get(claim.get("claim_id"), {"verdict": "unverifiable", "evidence": "No verifier result", "sources": [], "grounded_sources": []})}
        for claim in claims
    ]


def run_claim_verification(rows: pd.DataFrame, model: str = GEMINI_PRO_MODEL) -> pd.DataFrame:
    records = []
    with ThreadPoolExecutor(max_workers=min(5, max(1, len(rows)))) as executor:
        extracted = list(executor.map(lambda item: extract_claims(item[1], model), rows.iterrows()))
    for (_, row), claims in zip(rows.iterrows(), extracted):
        records.extend(verify_claims(claims, row, model))
    return pd.DataFrame(records)


def summarize_claims(claims: pd.DataFrame) -> pd.DataFrame:
    if claims.empty:
        return pd.DataFrame(columns=["source_key", "claims_total", "claims_supported", "claims_contradicted", "claims_unverifiable", "claim_details"])
    grouped = []
    for key, group in claims.groupby("source_key"):
        counts = group["verdict"].value_counts()
        grouped.append({
            "source_key": key,
            "claims_total": len(group),
            "claims_supported": int(counts.get("supported", 0)),
            "claims_contradicted": int(counts.get("contradicted", 0)),
            "claims_unverifiable": int(counts.get("unverifiable", 0)),
            "claim_details": group.to_dict("records"),
        })
    return pd.DataFrame(grouped)
