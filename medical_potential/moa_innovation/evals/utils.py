from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from json_repair import repair_json

from medical_potential.claude_utils import claude_api_call
from medical_potential.config import CLAUDE_SONNET_MODEL, GEMINI_PRO_MODEL
from medical_potential.gemini_utils import gemini_api_call

JUSTIFICATION_FIELDS = (
    "justification_mechanism_summary",
    "justification_novelty_vs_soc",
    "justification_competitor_comparison",
    "justification_biological_rationale",
    "justification_prior_validation_failure",
    "justification_why_not_higher_score",
)


def parse_json_response(text: str) -> Any:
    cleaned = (text or "").strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    if cleaned and cleaned[0] not in "{[":
        # Tool-using responses often put prose before the JSON; keep the outermost object.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    repaired = repair_json(cleaned)
    return json.loads(repaired) if isinstance(repaired, str) else repaired


def call_gemini_json(
    prompt: str,
    *,
    model: str = GEMINI_PRO_MODEL,
    google_search: bool = False,
) -> Any:
    response = gemini_api_call(
        prompt=prompt,
        model=model,
        temperature=0,
        google_search=google_search,
        thinking_level="high",
    )
    return parse_json_response(response)


def call_gemini_json_grounded(prompt: str, *, model: str = GEMINI_PRO_MODEL) -> tuple[Any, dict]:
    text, grounding = gemini_api_call(
        prompt=prompt,
        model=model,
        temperature=0,
        google_search=True,
        return_grounding=True,
        thinking_level="high",
    )
    return parse_json_response(text), grounding


def latest_rows(df: pd.DataFrame, molecules: list[str]) -> pd.DataFrame:
    selected = df[df["drug_name"].astype(str).isin(molecules)].copy()
    if selected.empty:
        return selected
    selected["_created_at_sort"] = pd.to_datetime(selected["created_at"], errors="coerce", utc=True)
    selected = selected.sort_values("_created_at_sort").drop_duplicates("drug_name", keep="last")
    return selected.drop(columns=["_created_at_sort"])


def source_key(row: pd.Series) -> str:
    return f"{row.get('drug_name', '')}|{row.get('created_at', '')}"


def eval_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def text_value(row: pd.Series, field: str) -> str:
    value = row.get(field, "")
    return "" if pd.isna(value) else str(value).strip()


# --- Claude helpers (web fetch over the sources saved by the MoA run) -------------

SOURCES_DIR = Path(__file__).resolve().parent.parent / "sources"
_VERTEX_REDIRECT_HOST = "vertexaisearch.cloud.google.com"


def call_claude_json(
    prompt: str,
    *,
    model: str = CLAUDE_SONNET_MODEL,
    fetch_urls: list[str] | None = None,
    max_web_fetches: int = 10,
    max_output_tokens: int = 16_000,
    web_search: bool = False,
    max_web_searches: int = 5,
) -> tuple[Any, list[dict]]:
    """Call Claude and parse the JSON answer; returns (data, web_citations)."""
    text, citations = claude_api_call(
        prompt=prompt,
        model=model,
        temperature=0,
        max_output_tokens=max_output_tokens,
        fetch_urls=fetch_urls,
        max_web_fetches=max_web_fetches,
        web_search=web_search,
        max_web_searches=max_web_searches,
        return_citations=True,
    )
    return parse_json_response(text), citations


def _sources_folder(drug_name: str) -> Path:
    # Must match _save_sources_json in moa_innovation.py.
    return SOURCES_DIR / (re.sub(r"[^\w.-]+", "_", str(drug_name).strip()).strip("._") or "unknown")


def load_saved_sources(row: pd.Series) -> dict:
    """Load the sources JSON saved for this row's run (closest in time to row['created_at'])."""
    folder = _sources_folder(row.get("drug_name", ""))
    files = sorted(folder.glob("sources_*.json")) if folder.is_dir() else []
    empty = {"file": None, "sources": [], "domains": []}
    if not files:
        return empty

    created = pd.to_datetime(row.get("created_at"), errors="coerce", utc=True)

    def distance(path: Path) -> float:
        try:
            ran_at = datetime.fromisoformat(json.loads(path.read_text(encoding="utf-8"))["run_timestamp"])
            ran_at = ran_at.astimezone(timezone.utc)  # naive timestamps are local time
            return abs((ran_at - created).total_seconds()) if pd.notna(created) else 0.0
        except Exception:
            return float("inf")

    chosen = min(files, key=distance) if pd.notna(created) else files[-1]
    try:
        data = json.loads(chosen.read_text(encoding="utf-8"))
    except Exception:
        return empty

    by_url: dict[str, dict] = {}
    for step, records in (data.get("sources_by_step") or {}).items():
        for record in records or []:
            url = record.get("url") if isinstance(record, dict) else None
            if not url:
                continue
            entry = by_url.setdefault(
                url, {"url": url, "title": record.get("title"), "domain": record.get("domain"), "steps": []}
            )
            entry["steps"].append(step)

    domains: list[str] = []
    for url in by_url:
        host = (urlparse(url).hostname or "").removeprefix("www.")
        if host and host != _VERTEX_REDIRECT_HOST and host not in domains:
            domains.append(host)
    return {"file": str(chosen), "sources": list(by_url.values()), "domains": domains}
