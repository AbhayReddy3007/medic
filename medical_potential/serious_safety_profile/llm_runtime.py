"""Two-Step Gemini Clinical Trial Extraction - Core Function

This module provides the core extraction logic used by the market potential agent.
"""
import os
import json
import asyncio
from contextvars import ContextVar
from dotenv import load_dotenv
from json_repair import repair_json
from medical_potential.gemini_utils import gemini_api_call

# Load environment variables
load_dotenv(override=True)


def safe_json_parse(json_str: str, context: str = "") -> dict | list | None:
    """Parse JSON with automatic repair using json_repair library.

    Args:
        json_str: The JSON string to parse
        context: Optional context for error messages

    Returns:
        Parsed JSON object/array, or None if parsing fails
    """
    if not json_str or not json_str.strip():
        return None

    # Clean up common issues
    json_str = json_str.strip()

    # Remove markdown code blocks if present
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        parts = json_str.split("```")
        if len(parts) >= 2:
            json_str = parts[1].strip()

    # Find first { or [ if there's text before JSON
    json_start = -1
    for i, c in enumerate(json_str):
        if c in '{[':
            json_start = i
            break
    if json_start > 0:
        json_str = json_str[json_start:]

    # Try json_repair first (handles most issues automatically)
    try:
        repaired = repair_json(json_str, return_objects=True)
        if repaired is not None:
            return repaired
    except Exception as e:
        if context:
            print(f"  ⚠️ json_repair failed for {context}: {e}")

    # Fallback to standard json.loads
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    return None

# Per-run collector of grounding sources. Set by run_serious_safety_assessment();
# None (default) means source collection is off and calls behave as before.
_GROUNDING_CALLS: ContextVar[list[dict] | None] = ContextVar(
    "serious_safety_grounding_calls", default=None
)


def start_grounding_collection() -> tuple[list[dict], object]:
    """Begin collecting grounding sources for this run. Returns (calls, reset_token)."""
    calls: list[dict] = []
    return calls, _GROUNDING_CALLS.set(calls)


def stop_grounding_collection(token) -> None:
    _GROUNDING_CALLS.reset(token)


def _record_urls(records) -> list[str]:
    urls: list[str] = []
    for record in records or []:
        url = record.get("url") if isinstance(record, dict) else None
        if isinstance(url, str) and url and url not in urls:
            urls.append(url)
    return urls


def dedupe_grounding_urls(grounding_calls: list[dict]) -> list[str]:
    """Unique source URLs across all calls, in first-seen order."""
    urls: list[str] = []
    for call in grounding_calls:
        for url in _record_urls(call.get("grounding_urls")):
            if url not in urls:
                urls.append(url)
    return urls


def grounding_urls_by_step(grounding_calls: list[dict]) -> dict[str, list[str]]:
    """Unique source URLs per pipeline step."""
    by_step: dict[str, list[str]] = {}
    for call in grounding_calls:
        bucket = by_step.setdefault(str(call.get("step_name")), [])
        for url in _record_urls(call.get("grounding_urls")):
            if url not in bucket:
                bucket.append(url)
    return by_step


def grounding_sources_by_step(grounding_calls: list[dict]) -> dict[str, list[dict]]:
    """Full source records (url, title, domain, supported_text) per step, de-duplicated by URL."""
    by_step: dict[str, dict[str, dict]] = {}
    for call in grounding_calls:
        bucket = by_step.setdefault(str(call.get("step_name")), {})
        for record in call.get("grounding_urls") or []:
            if isinstance(record, dict) and record.get("url"):
                bucket.setdefault(record["url"], record)
    return {step: list(records.values()) for step, records in by_step.items()}


# Configuration
BATCH_SIZE = 8
MAX_WORKERS = 8
RATE_LIMIT_DELAY = 5.0
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0

# Models
SEARCH_MODEL = "gemini-3-flash-preview"
EXTRACTION_MODEL = "gemini-3-flash-preview"


async def gemini_call_with_search(
    prompt: str,
    model: str,
    response_mime_type: str = None,
    temperature: float = 0,
    step_name: str = "unspecified",
) -> str:
    """Make a Gemini call with Google Search enabled.

    Grounding sources are recorded under `step_name` when a collector is active
    (see start_grounding_collection); the return value is still just the text.

    Args:
        prompt: The prompt to send to Gemini
        model: Model name (e.g., "gemini-2.5-flash")
        response_mime_type: Optional mime type (e.g., "application/json").
                           Note: Using response_mime_type with Google Search tools may cause errors.
        step_name: Pipeline step label used to group the collected sources.
    """
    text, grounding_urls = await asyncio.to_thread(
        gemini_api_call,
        prompt=prompt,
        model=model,
        temperature=temperature,
        google_search=True,
        response_mime_type=response_mime_type,
        get_grounding_urls=True,
    )
    calls = _GROUNDING_CALLS.get()
    if calls is not None:
        calls.append({"step_name": step_name, "grounding_urls": grounding_urls})
    return text


async def gemini_call_without_search(
    prompt: str,
    model: str,
    response_mime_type: str = None,
    temperature: float = 0,
) -> str:
    """Make a Gemini call without Google Search enabled.

    Args:
        prompt: The prompt to send to Gemini
        model: Model name (e.g., "gemini-2.5-flash")
        response_mime_type: Optional mime type (e.g., "application/json").
    """
    return await asyncio.to_thread(
        gemini_api_call,
        prompt=prompt,
        model=model,
        temperature=temperature,
        google_search=False,
        response_mime_type=response_mime_type,
    )
