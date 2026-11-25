"""Collect Google Search grounding URLs during a dimension run and save them to disk.

Same output layout as moa_innovation / serious_safety_profile / clinical_evidence:
`<dimension_dir>/sources/<molecule>/sources_<timestamp>.json` with `drug_name`, `run_timestamp`,
`total_unique_urls` and `sources_by_step`.

Usage:
    calls, token = start_grounding_collection()
    try:
        ...  # search-enabled Gemini calls use gemini_search_call(..., step_name="...")
    finally:
        save_sources_json(DIMENSION_DIR, molecule_name, calls)
        stop_grounding_collection(token)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

from .gemini_utils import gemini_api_call

logger = logging.getLogger(__name__)

_GROUNDING_CALLS: ContextVar[list[dict] | None] = ContextVar("grounding_calls", default=None)


def start_grounding_collection() -> tuple[list[dict], object]:
    """Begin collecting grounding sources for this run. Returns (calls, reset_token)."""
    calls: list[dict] = []
    return calls, _GROUNDING_CALLS.set(calls)


def stop_grounding_collection(token) -> None:
    _GROUNDING_CALLS.reset(token)


def record_grounding_urls(step_name: str, grounding_urls: list[dict]) -> None:
    """Add one call's grounding records to the active collector (no-op when none is active)."""
    calls = _GROUNDING_CALLS.get()
    if calls is not None:
        calls.append({"step_name": step_name, "grounding_urls": grounding_urls})


def gemini_search_call(step_name: str, **kwargs) -> str:
    """Synchronous search-grounded Gemini call that records its sources under `step_name`."""
    kwargs.pop("google_search", None)
    kwargs.pop("get_grounding_urls", None)
    text, grounding_urls = gemini_api_call(google_search=True, get_grounding_urls=True, **kwargs)
    record_grounding_urls(step_name, grounding_urls)
    return text


async def gemini_search_call_async(step_name: str, **kwargs) -> str:
    """Async wrapper of `gemini_search_call` (runs in a worker thread)."""
    text, grounding_urls = await asyncio.to_thread(
        gemini_api_call, google_search=True, get_grounding_urls=True, **kwargs
    )
    # Recorded here, on the event-loop side, where the collector ContextVar is visible.
    record_grounding_urls(step_name, grounding_urls)
    return text


def _sources_by_step(grounding_calls: list[dict]) -> dict[str, list[dict]]:
    by_step: dict[str, dict[str, dict]] = {}
    for call in grounding_calls:
        bucket = by_step.setdefault(str(call.get("step_name")), {})
        for record in call.get("grounding_urls") or []:
            if isinstance(record, dict) and record.get("url"):
                bucket.setdefault(record["url"], record)
    return {step: list(records.values()) for step, records in by_step.items()}


def save_sources_json(dimension_dir: Path, molecule_name: str, grounding_calls: list[dict],
                      extra: dict | None = None) -> Path | None:
    """Write every web source used in this run. Best-effort: failures are logged, never raised."""
    try:
        now = datetime.now()
        folder = re.sub(r"[^\w.-]+", "_", str(molecule_name).strip()).strip("._") or "unknown"
        directory = Path(dimension_dir) / "sources" / folder
        directory.mkdir(parents=True, exist_ok=True)
        by_step = _sources_by_step(grounding_calls)
        unique_urls = {r["url"] for records in by_step.values() for r in records}
        payload = {
            "drug_name": molecule_name,
            "run_timestamp": now.isoformat(timespec="seconds"),
            "total_unique_urls": len(unique_urls),
            "sources_by_step": by_step,
            **(extra or {}),
        }
        path = directory / f"sources_{now.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        logger.info("Saved %s sources to %s", payload["total_unique_urls"], path)
        return path
    except Exception as error:
        logger.warning("Could not save sources JSON for %r: %s", molecule_name, error)
        return None
