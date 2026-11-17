"""Shared utilities for unmet-needs extraction workflows."""

from __future__ import annotations

import json
from typing import Any

from json_repair import repair_json


def _extract_token_usage(response: Any) -> dict[str, int | None]:
    """Extract token usage from Gemini response metadata."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}

    def _to_int(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "input_tokens": _to_int(getattr(usage, "prompt_token_count", None)),
        "output_tokens": _to_int(getattr(usage, "candidates_token_count", None)),
        "total_tokens": _to_int(getattr(usage, "total_token_count", None)),
    }


def _safe_json_parse(json_str: str) -> dict | list | None:
    """Parse JSON with lightweight repair for model responses."""
    if not json_str or not json_str.strip():
        return None

    cleaned = json_str.strip()

    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) >= 2:
            cleaned = parts[1].strip()

    start_index = -1
    for i, ch in enumerate(cleaned):
        if ch in "[{":
            start_index = i
            break
    if start_index > 0:
        cleaned = cleaned[start_index:]

    try:
        repaired = repair_json(cleaned, return_objects=True)
        if repaired is not None:
            return repaired
    except Exception:
        pass

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _normalize_unmet_need_output(parsed: Any) -> list[dict[str, Any]]:
    """Normalize model output into a flat unmet need list."""
    flat_unmet_needs: list[dict[str, Any]] = []

    if not isinstance(parsed, dict):
        return flat_unmet_needs

    def _build_entry(item: dict[str, Any], category_override: str | None = None) -> dict[str, Any] | None:
        statement = str(item.get("unmet_need", "")).strip()
        if not statement:
            return None

        category = category_override if category_override is not None else str(item.get("category", "other")).strip()

        return {
            "unmet_need": statement,
            "category": category or "other",
            "evidence": str(item.get("evidence", "")).strip()
        }

    items = parsed.get("unmet_needs", [])
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            entry = _build_entry(item)
            if entry is not None:
                flat_unmet_needs.append(entry)
        if flat_unmet_needs:
            return flat_unmet_needs

    grouped = parsed.get("unmet_needs_by_category")
    if isinstance(grouped, list):
        for group in grouped:
            if not isinstance(group, dict):
                continue
            category = str(group.get("category", "other")).strip() or "other"
            group_items = group.get("unmet_needs", [])
            if not isinstance(group_items, list):
                continue
            for item in group_items:
                if not isinstance(item, dict):
                    continue
                entry = _build_entry(item, category_override=category)
                if entry is not None:
                    flat_unmet_needs.append(entry)

    return flat_unmet_needs
