"""Load Clinical Evidence runs for evaluation.

A run is the `output_payload.json` that `clinical_evidence_scoring` caches in GCS. It is flattened to one row
per molecule with the judged text columns, the P/D/G/E/C scores and their reasoning, and a compact
`pipeline_data` summary of the evidence behind the score. Grounding sources are read separately from
`sources/<molecule>/*.json`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from medical_potential.config import (
    CLINICAL_EVIDENCE_DIMENSION_NAME,
    GCS_BUCKET,
    GCS_MEDICAL_POTENTIAL_SUBFOLDER,
    GCS_PIPELINE_CACHE_BASE_PATH,
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


def build_pipeline_data(payload: dict) -> dict:
    """Compact evidence summary: what the pipeline itself knew when it wrote the score and rationale."""
    inputs = payload.get("report_input_payload") or {}
    scoring = inputs.get("scoring") or {}
    assessment = inputs.get("evidence_assessment") or {}
    consolidation = inputs.get("evidence_consolidation") or {}
    molecule = inputs.get("molecule") or {}
    trials = inputs.get("clinical_trials") or []
    return _clean({
        "molecule": {key: molecule.get(key) for key in ("generic_name", "brand_names", "sponsor", "indications")},
        "scores": {key: scoring.get(key) for key in ("P", "D", "G", "E", "C", "final")},
        "score_reasoning": scoring.get("reasoning"),
        "consistency_detail": scoring.get("consistency_llm_detail"),
        "evidence_assessment": {
            key: assessment.get(key)
            for key in (
                "evidence_level", "evidence_detail", "total_trials", "trials_with_results", "total_publications",
                "total_patients", "largest_trial", "rct_count", "has_regulatory_approval", "phase_distribution",
                "design_distribution",
            )
        },
        "evidence_consolidation": {
            key: consolidation.get(key)
            for key in ("findings_consistency", "replication", "peer_reviewed_count", "results_published")
        },
        "evidence_gaps": inputs.get("evidence_gaps"),
        "fda_product_entries": (inputs.get("sponsor_info") or {}).get("fda_product_entries"),
        "trials_sample": [
            {
                key: trial.get(key)
                for key in ("nct_id", "trial_id", "phase", "study_design", "status", "enrollment", "results_available")
                if key in trial
            }
            for trial in trials[:15]
        ],
    })


def payload_to_row(payload: dict, created_at: Any = None) -> dict:
    """Flatten one cached output payload into an eval row."""
    inputs = payload.get("report_input_payload") or {}
    scoring = inputs.get("scoring") or {}
    return {
        "drug_name": payload.get("molecule_name"),
        "created_at": created_at,
        "score": payload.get("final_score"),
        "score_p": payload.get("score_p"),
        "score_d": payload.get("score_d"),
        "score_g": payload.get("score_g"),
        "score_e": payload.get("score_e"),
        "score_c": payload.get("score_c"),
        "evidence_level": payload.get("evidence_level"),
        "fda_approved": payload.get("fda_approved"),
        "total_trials": payload.get("total_trials"),
        "rct_count": payload.get("rct_count"),
        "rationale": payload.get("generated_rationale"),
        "rwe_synthesis": inputs.get("rwe_synthesis"),
        "llm_consistency_report": payload.get("llm_consistency_report"),
        "evidence_gaps": payload.get("evidence_gaps"),
        "score_reasoning": json.dumps(_clean(scoring.get("reasoning")), default=str),
        "pipeline_data": json.dumps(build_pipeline_data(payload), default=str),
        "cache_gcs_uri": payload.get("cache_gcs_uri"),
        "report_gcs_uri": (payload.get("report") or {}).get("base_report_gcs_uri"),
    }


def payload_gcs_path(molecule_name: str) -> str:
    return (
        f"{GCS_PIPELINE_CACHE_BASE_PATH}/{molecule_name.strip()}/{GCS_MEDICAL_POTENTIAL_SUBFOLDER}/"
        f"{CLINICAL_EVIDENCE_DIMENSION_NAME}/output_payload.json"
    )


def load_clinical_evidence_rows(molecules: list[str]) -> pd.DataFrame:
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
        payload.setdefault("cache_gcs_uri", f"gs://{GCS_BUCKET}/{blob.name}")
        row = payload_to_row(payload, created_at=blob.updated.isoformat() if blob.updated else None)
        # The payload stores the normalised generic name, but the cache folder and the sources folder use the
        # name the run was started with, so that is the name the evals select and look up sources by.
        row["generic_name"] = row["drug_name"]
        row["drug_name"] = folder
        rows.append(row)

    if missing:
        available = ", ".join(sorted(folders.values())[:30]) or "none found"
        message = (
            "No cached clinical-evidence payload for: " + "; ".join(missing)
            + f". Molecule folders under gs://{GCS_BUCKET}/{cache_prefix}: {available}. "
            "The payload is written at the end of clinical_evidence_scoring, so the molecule must have been "
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
