"""Shared helpers for the serious-safety evals.

Generic helpers (Claude calls, JSON parsing, row selection) are reused from the MoA evals.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from medical_potential.moa_innovation.evals.utils import (  # noqa: F401  (re-exported)
    call_claude_json,
    eval_timestamp,
    latest_rows,
    parse_json_response,
    source_key,
    text_value,
)

# Text columns of an eval row that are judged (the safety analogue of the MoA `justification_*` fields).
REPORT_FIELDS = (
    "report_serious_safety_profile",
    "report_clinical_trial_safety",
    "report_serious_adverse_events_found",
    "report_regulatory_assessment",
    "report_post_marketing_safety",
    "rationale",
)

_VERTEX_REDIRECT_HOST = "vertexaisearch.cloud.google.com"


def field_short_name(field: str) -> str:
    return field.removeprefix("report_")


def as_object(value: Any) -> Any:
    """Return dict/list values as is and decode JSON strings; anything else becomes None."""
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except ValueError:
            return None
    return None


SOURCES_DIR = Path(__file__).resolve().parent.parent / "sources"


def _sources_from_local_file(row: pd.Series) -> tuple[dict[str, dict], str | None]:
    """Fallback: the sources_*.json the run saved locally, closest in time to the row's `created_at`."""
    folder = SOURCES_DIR / (re.sub(r"[^\w.-]+", "_", str(row.get("drug_name", "")).strip()).strip("._") or "unknown")
    files = sorted(folder.glob("sources_*.json")) if folder.is_dir() else []
    if not files:
        return {}, None
    created = pd.to_datetime(row.get("created_at"), errors="coerce", utc=True)

    def distance(path: Path) -> float:
        try:
            ran_at = datetime.fromisoformat(json.loads(path.read_text(encoding="utf-8"))["run_timestamp"])
            return abs((ran_at.astimezone(timezone.utc) - created).total_seconds()) if pd.notna(created) else 0.0
        except Exception:
            return float("inf")

    chosen = min(files, key=distance) if pd.notna(created) else files[-1]
    try:
        data = json.loads(chosen.read_text(encoding="utf-8"))
    except Exception:
        return {}, None
    by_url: dict[str, dict] = {}
    for step, records in (data.get("sources_by_step") or {}).items():
        for record in records or []:
            url = record.get("url") if isinstance(record, dict) else None
            if url:
                entry = by_url.setdefault(
                    url, {"url": url, "title": record.get("title"), "domain": record.get("domain"), "steps": []}
                )
                if step not in entry["steps"]:
                    entry["steps"].append(step)
    return by_url, str(chosen)


def load_saved_sources(row: pd.Series) -> dict:
    """Sources the run retrieved, read from the `grounding_sources_by_step` stored in the payload."""
    by_step = as_object(row.get("grounding_sources_by_step")) or {}
    by_url: dict[str, dict] = {}
    for step, records in by_step.items():
        for record in records or []:
            url = record.get("url") if isinstance(record, dict) else None
            if not url:
                continue
            entry = by_url.setdefault(
                url, {"url": url, "title": record.get("title"), "domain": record.get("domain"), "steps": []}
            )
            if step not in entry["steps"]:
                entry["steps"].append(step)

    local_file = None
    if not by_url:
        by_url, local_file = _sources_from_local_file(row)

    domains: list[str] = []
    for url in by_url:
        host = (urlparse(url).hostname or "").removeprefix("www.")
        if host and host != _VERTEX_REDIRECT_HOST and host not in domains:
            domains.append(host)
    file_ref = local_file or row.get("cache_gcs_uri") or ("payload" if by_url else None)
    return {"file": file_ref, "sources": list(by_url.values()), "domains": domains}
