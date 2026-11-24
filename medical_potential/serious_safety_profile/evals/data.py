"""Load Serious Safety Profile runs for evaluation.

A run is the `output_payload.json` that `run_serious_safety_assessment` caches in GCS (Step 15). It is
flattened to one row per molecule with the judged text columns, the score and its reasoning, a compact
`pipeline_data` summary of the evidence behind the score. Grounding sources are read separately from `sources/<molecule>/*.json`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from medical_potential.config import (
    GCS_BUCKET,
    GCS_MEDICAL_POTENTIAL_SUBFOLDER,
    GCS_PIPELINE_CACHE_BASE_PATH,
    SERIOUS_SAFETY_PROFILE_DIMENSION_NAME,
)

SECTION_KEYS = (
    "serious_safety_profile",
    "clinical_trial_safety",
    "serious_adverse_events_found",
    "regulatory_assessment",
    "post_marketing_safety",
)


def _clean(value: Any) -> Any:
    """NaN is not valid JSON; keep the structure but turn NaN into None."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def _number(value: Any) -> float:
    try:
        number = float(value)
        return 0.0 if math.isnan(number) else number
    except (TypeError, ValueError):
        return 0.0


def build_pipeline_data(payload: dict) -> dict:
    """Compact evidence summary: what the pipeline itself knew when it wrote the report."""
    score = payload.get("safety_score") or {}
    aggregation = payload.get("sae_aggregation") or {}
    trials = aggregation.get("trials") or []
    top_trials = sorted(trials, key=lambda trial: _number(trial.get("normalized_weight")), reverse=True)[:10]
    approval = payload.get("approval_market_status") or {}
    regulatory = payload.get("regulatory_impact") or {}
    post_marketing = payload.get("post_marketing_safety") or {}
    return _clean({
        "approval": {key: approval.get(key) for key in ("is_approved", "is_marketed", "status_summary", "major_markets")},
        "total_trials": payload.get("total_trials"),
        "weighted_rates": {key: aggregation.get(key) for key in ("sae_rate", "control_rate", "delta_rate", "trials_used")},
        "score": score.get("score"),
        "score_reasoning": score.get("reasoning"),
        "key_drivers": score.get("key_drivers"),
        "adjustments_applied": score.get("adjustments_applied"),
        "sae_events": [
            {
                "sae_name": event.get("sae_name"),
                "severity_category": event.get("severity_category"),
                "expectedness": event.get("expectedness_classification"),
                "number_of_studies": event.get("number_of_studies"),
            }
            for event in (payload.get("sae_event_categorization") or [])
        ],
        "regulatory_impact": {
            key: regulatory.get(key) for key in ("regulatory_consequence", "rationale", "key_safety_drivers")
        },
        "post_marketing": {
            key: post_marketing.get(key)
            for key in (
                "evaluated", "risk_level", "new_serious_risks", "boxed_warning_added", "rems_added",
                "major_label_restriction", "withdrawal_from_market", "key_findings",
            )
        },
        "top_trials_by_weight": [
            {
                "trial_id": trial.get("Trial ID"),
                "phase": trial.get("Phase"),
                "size": trial.get("Size"),
                "dosage": trial.get("Dosage"),
                "sae_rate_drug_pct": trial.get("SAE Rate Drug (%)"),
                "sae_rate_control_pct": trial.get("SAE Rate Control (%)"),
                "delta_sae_pct": trial.get("Delta SAE (%)"),
                "deaths_reported": trial.get("Deaths Reported"),
                "band": trial.get("band"),
                "normalized_weight": trial.get("normalized_weight"),
            }
            for trial in top_trials
        ],
    })


def payload_to_row(payload: dict, created_at: Any = None) -> dict:
    """Flatten one cached output payload into an eval row."""
    sections = (payload.get("report") or {}).get("sections") or {}
    score = payload.get("safety_score") or {}
    approval = payload.get("approval_market_status") or {}
    row: dict[str, Any] = {
        "drug_name": payload.get("molecule_name"),
        "created_at": created_at,
        "score": score.get("score"),
        "status_summary": approval.get("status_summary"),
        "is_approved": approval.get("is_approved"),
        "is_marketed": approval.get("is_marketed"),
        "total_trials": payload.get("total_trials"),
        "rationale": payload.get("rationale"),
        "score_reasoning": json.dumps(_clean(score.get("reasoning")), default=str),
        "key_drivers": json.dumps(_clean(score.get("key_drivers")), default=str),
        "adjustments_applied": json.dumps(_clean(score.get("adjustments_applied")), default=str),
        "pipeline_data": json.dumps(build_pipeline_data(payload), default=str),
        "grounding_urls": payload.get("grounding_urls") or [],
        "cache_gcs_uri": payload.get("cache_gcs_uri"),
        "report_gcs_uri": payload.get("report_gcs_uri"),
    }
    for key in SECTION_KEYS:
        row[f"report_{key}"] = sections.get(key)
    return row


def payload_gcs_path(molecule_name: str) -> str:
    return (
        f"{GCS_PIPELINE_CACHE_BASE_PATH}/{molecule_name.strip()}/{GCS_MEDICAL_POTENTIAL_SUBFOLDER}/"
        f"{SERIOUS_SAFETY_PROFILE_DIMENSION_NAME}/output_payload.json"
    )


def load_serious_safety_rows(molecules: list[str]) -> pd.DataFrame:
    """Latest cached payload per molecule from GCS, one eval row each."""
    from medical_potential.gcp_utils import get_gcs_client

    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET)

    # Molecule folders under the cache root, so names can be matched case-insensitively.
    cache_prefix = f"{GCS_PIPELINE_CACHE_BASE_PATH.strip('/')}/"
    listing = client.list_blobs(GCS_BUCKET, prefix=cache_prefix, delimiter="/")
    list(listing)  # prefixes are only populated once the listing has been consumed
    folders = {
        prefix[len(cache_prefix):].strip("/").casefold(): prefix[len(cache_prefix):].strip("/")
        for prefix in listing.prefixes
    }

    rows, missing = [], []
    for molecule in molecules:
        folder = folders.get(molecule.strip().casefold(), molecule.strip())
        blob = bucket.get_blob(payload_gcs_path(folder))
        if blob is None:
            missing.append(f"{molecule!r} (tried gs://{GCS_BUCKET}/{payload_gcs_path(folder)})")
            continue
        payload = json.loads(blob.download_as_bytes())
        payload.setdefault("molecule_name", folder)
        payload.setdefault("cache_gcs_uri", f"gs://{GCS_BUCKET}/{blob.name}")
        rows.append(payload_to_row(payload, created_at=blob.updated.isoformat() if blob.updated else None))

    if missing:
        available = ", ".join(sorted(folders.values())[:30]) or "none found"
        message = (
            "No cached serious-safety payload for: " + "; ".join(missing)
            + f". Molecule folders under gs://{GCS_BUCKET}/{cache_prefix}: {available}. "
            "The payload is written by Step 15 of run_serious_safety_assessment, so the molecule must have been "
            "run with the same GCS_BUCKET / GCS_PIPELINE_CACHE_BASE_PATH / GCS_MEDICAL_POTENTIAL_SUBFOLDER settings."
        )
        if not rows:
            raise FileNotFoundError(message)
        print(message)
    return pd.DataFrame(rows)


def load_rows_from_payload_files(paths: list[str | Path]) -> pd.DataFrame:
    """Offline alternative: build eval rows from payload JSON files on disk."""
    rows = []
    for path in paths:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        modified = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat()
        rows.append(payload_to_row(payload, created_at=modified))
    return pd.DataFrame(rows)
