"""Context precision: how much of the web context the safety run retrieved is actually relevant.

The Serious Safety pipeline stores every grounding source behind its search-enabled calls in the cached
payload (`grounding_sources_by_step`). Claude judges whether each source is relevant to the molecule's
serious safety, and the share that is relevant is reported (overall and per pipeline step).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from medical_potential.config import CLAUDE_SONNET_MODEL
from .utils import call_claude_json, load_saved_sources, source_key, text_value

logger = logging.getLogger(__name__)

SOURCES_PER_CALL = 40
VERDICTS = ("relevant", "irrelevant", "cannot_tell")


def _prompt(row: pd.Series, batch: list[dict], fetch_pages: bool) -> str:
    listing = "\n".join(
        f"{item['id']}. title: {item.get('title') or 'untitled'} | domain: {item.get('domain') or 'unknown'} "
        f"| url: {item['url']} | used in: {', '.join(item.get('steps') or [])}"
        for item in batch
    )
    how = (
        "Fetch each page with the web_fetch tool when the title and URL are not enough to decide."
        if fetch_pages
        else "Judge from the title, domain, URL and the step it was used in. You cannot open the pages."
    )
    return f"""You are auditing the web sources retrieved for a Serious Safety Profile assessment of a drug.
Drug: {row.get('drug_name')}
Approval status: {row.get('status_summary')}
Assigned safety score: {row.get('score')} (1-5, higher is safer)

A source is "relevant" when it is likely to contain evidence about this drug's serious adverse events, clinical
trial safety results, deaths, regulatory actions (labels, boxed warnings, REMS, safety communications, withdrawals),
approval or marketing status, or post-marketing safety findings, or about the safety of its drug class where that
informs how expected an event is.
A source is "irrelevant" when it concerns an unrelated drug or disease, is generic or promotional content with no
safety evidence value, or is an unrelated website. Use "cannot_tell" only when the information given is not enough to decide.

{how}

Sources:
{listing}

Return JSON only as {{"sources": [{{"id": 1, "verdict": "relevant|irrelevant|cannot_tell", "reason": "one short sentence"}}]}}.
Return one entry for every source id."""


def _judge_batch(row: pd.Series, batch: list[dict], model: str, fetch_pages: bool) -> list[dict]:
    kwargs = {}
    if fetch_pages:
        kwargs = {"fetch_urls": [item["url"] for item in batch], "max_web_fetches": min(20, max(6, len(batch)))}
    result, _ = call_claude_json(_prompt(row, batch, fetch_pages), model=model, **kwargs)
    entries = result.get("sources", []) if isinstance(result, dict) else []
    by_id = {entry.get("id"): entry for entry in entries if isinstance(entry, dict)}
    judged = []
    for item in batch:
        entry = by_id.get(item["id"], {})
        verdict = str(entry.get("verdict", "")).strip().lower()
        judged.append({
            **item,
            "verdict": verdict if verdict in VERDICTS else "cannot_tell",
            "reason": entry.get("reason", "Judge returned no result for this source"),
        })
    return judged


def judge_context_precision(row: pd.Series, model: str = CLAUDE_SONNET_MODEL, fetch_pages: bool = False) -> dict:
    saved = load_saved_sources(row)
    base = {"source_key": source_key(row), "drug_name": row.get("drug_name"), "sources_file": saved["file"]}
    if not saved["sources"]:
        return {**base, "sources_total": 0, "context_error": "no saved sources found for this run"}

    sources = [{"id": index + 1, **source} for index, source in enumerate(saved["sources"])]
    judged: list[dict] = []
    for start in range(0, len(sources), SOURCES_PER_CALL):
        batch = sources[start : start + SOURCES_PER_CALL]
        try:
            judged.extend(_judge_batch(row, batch, model, fetch_pages))
        except Exception as error:
            logger.exception("Context precision batch failed for %s", row.get("drug_name"))
            judged.extend({**item, "verdict": "error", "reason": f"Judging failed: {error}"} for item in batch)

    # Failed batches are excluded from the denominator so an API error does not read as irrelevance.
    scored = [item for item in judged if item["verdict"] != "error"]
    relevant = [item for item in scored if item["verdict"] == "relevant"]

    by_step: dict[str, dict] = {}
    for item in scored:
        for step in item.get("steps") or []:
            stats = by_step.setdefault(step, {"total": 0, "relevant": 0})
            stats["total"] += 1
            stats["relevant"] += item["verdict"] == "relevant"
    for stats in by_step.values():
        stats["precision"] = stats["relevant"] / stats["total"] if stats["total"] else None

    return {
        **base,
        "sources_total": len(judged),
        "sources_relevant": len(relevant),
        "sources_irrelevant": sum(item["verdict"] == "irrelevant" for item in scored),
        "sources_cannot_tell": sum(item["verdict"] == "cannot_tell" for item in scored),
        "sources_error": len(judged) - len(scored),
        "context_precision_score": len(relevant) / len(scored) if scored else None,
        "context_precision_by_step": by_step,
        "context_details": judged,
    }


def _safe_judge(row: pd.Series, model: str, fetch_pages: bool) -> dict:
    try:
        return judge_context_precision(row, model, fetch_pages)
    except Exception as error:
        logger.exception("Context precision failed for %s", row.get("drug_name"))
        return {"source_key": source_key(row), "drug_name": row.get("drug_name"), "context_error": str(error)}


def run_context_precision(
    rows: pd.DataFrame, model: str = CLAUDE_SONNET_MODEL, fetch_pages: bool = False
) -> pd.DataFrame:
    """Context precision per run. `fetch_pages=True` lets Claude open each source (slower, costlier)."""
    if rows.empty:
        return pd.DataFrame()
    with ThreadPoolExecutor(max_workers=min(5, len(rows))) as executor:
        results = list(executor.map(lambda item: _safe_judge(item[1], model, fetch_pages), rows.iterrows()))
    return pd.DataFrame(results)
