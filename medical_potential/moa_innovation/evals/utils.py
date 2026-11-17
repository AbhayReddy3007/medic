from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from json_repair import repair_json

from medical_potential.config import GEMINI_PRO_MODEL
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
