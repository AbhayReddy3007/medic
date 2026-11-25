"""Shared helpers for the clinical-evidence evals.

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

# Text columns of an eval row that are judged. The pipeline's PDF narrative is not stored, so the judged text
# is what the payload keeps: the grounded real-world-evidence synthesis, the consistency commentary, the
# evidence gaps and the score rationale.
REPORT_FIELDS = (
    "rwe_synthesis",
    "llm_consistency_report",
    "evidence_gaps",
    "rationale",
)

_VERTEX_REDIRECT_HOST = "vertexaisearch.cloud.google.com"
SOURCES_DIR = Path(__file__).resolve().parent.parent / "sources"


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


def load_saved_sources(row: pd.Series) -> dict:
    """Sources the run retrieved, read only from the `sources/<molecule>/sources_*.json` files.

    Picks the file whose `run_timestamp` is closest to the row's `created_at` (latest file if none).
    """
    folder = SOURCES_DIR / (re.sub(r"[^\w.-]+", "_", str(row.get("drug_name", "")).strip()).strip("._") or "unknown")
    files = sorted(folder.glob("sources_*.json")) if folder.is_dir() else []
    by_url: dict[str, dict] = {}
    chosen = None
    if files:
        created = pd.to_datetime(row.get("created_at"), errors="coerce", utc=True)

        def distance(path: Path) -> float:
            try:
                ran_at = datetime.fromisoformat(json.loads(path.read_text(encoding="utf-8"))["run_timestamp"])
                return abs((ran_at.astimezone(timezone.utc) - created).total_seconds())
            except Exception:
                return float("inf")

        chosen = min(files, key=distance) if pd.notna(created) else files[-1]
        try:
            data = json.loads(chosen.read_text(encoding="utf-8"))
        except Exception:
            data, chosen = {}, None
        for step, records in (data.get("sources_by_step") or {}).items():
            for record in records or []:
                url = record.get("url") if isinstance(record, dict) else None
                if not url:
                    continue
                entry = by_url.setdefault(
                    url, {"url": url, "title": record.get("title"), "domain": record.get("domain"), "steps": []}
                )
                if step not in entry["steps"]:
                    entry["steps"].append(step)

    domains: list[str] = []
    for url in by_url:
        host = (urlparse(url).hostname or "").removeprefix("www.")
        if host and host != _VERTEX_REDIRECT_HOST and host not in domains:
            domains.append(host)
    return {"file": str(chosen) if chosen else None, "sources": list(by_url.values()), "domains": domains}
