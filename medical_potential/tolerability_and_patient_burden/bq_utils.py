"""BigQuery helpers for tolerability data push.

Provides a simple append-only loader that creates the table if needed.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from google.cloud import bigquery

from medical_potential.gcp_utils import get_bq_client
from medical_potential.config import PROJECT_ID, BQ_DATASET_ID, TOLERABILITY_TABLE
from medical_potential.config import CLINICAL_TRIALS_TOLERABILITY_DATA_TABLE

logger = logging.getLogger(__name__)


# Explicit schema for tolerability table (no `updated_at` field; append-only)
EXPLICIT_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("molecule_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("tolerability_score", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("score_numeric", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("discontinuation_rate_drug", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("discontinuation_rate_soc", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("soc_source", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("difference", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("key_aes", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("ae_frequency", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("ae_severity", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("vs_placebo", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("vs_soc", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("guardrail", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("guardrail_reason", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("justification", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("base_score", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("soc_adjustment", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("burden_adjustment", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("trials_used", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("total_trials_extracted", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
]

CLINICAL_TRIALS_TOLERABILITY_DATA_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("molecule_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("trial_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("dosage", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("phase", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("trial_location", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("secondary_location", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("size", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("drug_arm_size", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("placebo_arm_size", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("discontinued_ae_drug_n", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("discontinued_ae_placebo_n", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("comparator_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("comparator_arm_size_n", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("discontinued_ae_comparator_n", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("common_aes", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("ae_severity", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("ae_persistence", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("management_required", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("discontinuation_rate_drug_percent", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("discontinuation_rate_placebo_percent", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("discontinuation_rate_comparator_percent", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="NULLABLE"),
]



def append_tolerability_to_bq(molecule_name: str, score_result: Dict) -> None:
    """Append a tolerability row to BigQuery, creating table if needed.

    This is intentionally append-only; callers should provide the final payload
    to be stored as a single row per call.
    """
    client = get_bq_client()
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{TOLERABILITY_TABLE}"

    supporting = score_result.get("supporting_data", {})
    side_effects = score_result.get("side_effect_summary", {})
    comparison = score_result.get("comparison", {})
    breakdown = score_result.get("_scoring_breakdown", {})

    new_score_numeric = score_result.get("score_numeric")
    if isinstance(new_score_numeric, (int, float)):
        new_score_numeric = int(new_score_numeric)
    else:
        new_score_numeric = None

    row = {
        "molecule_name": molecule_name,
        "tolerability_score": score_result.get("tolerability_score", ""),
        "score_numeric": new_score_numeric,
        "discontinuation_rate_drug": supporting.get("discontinuation_rate_drug", ""),
        "discontinuation_rate_soc": supporting.get("discontinuation_rate_soc", ""),
        "soc_source": supporting.get("soc_source", ""),
        "difference": supporting.get("difference", ""),
        "key_aes": side_effects.get("key_aes", ""),
        "ae_frequency": side_effects.get("frequency", ""),
        "ae_severity": side_effects.get("severity", ""),
        "vs_placebo": comparison.get("vs_placebo", ""),
        "vs_soc": comparison.get("vs_soc", ""),
        "guardrail": score_result.get("guardrail", ""),
        "guardrail_reason": score_result.get("guardrail_reason", "") or "",
        "justification": score_result.get("justification", ""),
        "base_score": int(breakdown.get("base_score", 0)),
        "soc_adjustment": int(breakdown.get("soc_adjustment", 0)),
        "burden_adjustment": int(breakdown.get("burden_adjustment", 0)),
        "trials_used": int(breakdown.get("trials_used", 0)),
        "total_trials_extracted": int(breakdown.get("total_trials_extracted", 0)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    job_config = bigquery.LoadJobConfig(
        schema=EXPLICIT_SCHEMA,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
    )

    load_job = client.load_table_from_json([row], table_id, job_config=job_config)
    load_job.result()


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        s = str(value).strip()
        if not s or s.lower() in ("n/a", "na", "null", "none"):
            return None
        # remove commas and percent signs if present
        s = s.replace(",", "").replace("%", "")
        # allow floats that represent whole numbers
        f = float(s)
        return int(f)
    except Exception:
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    try:
        s = str(value).strip()
        if not s or s.lower() in ("n/a", "na", "null", "none"):
            return None
        s = s.replace(",", "").replace("%", "")
        return float(s)
    except Exception:
        return None


def _normalize_tolerability_trial_row(trial: Dict[str, Any], created_at: datetime) -> dict[str, Any]:
    # Use safe .get() access and coerce numeric fields to the expected types
    return {
        "molecule_name": trial.get("Molecule", ""),
        "trial_id": trial.get("Trial ID", ""),
        "dosage": trial.get("Dosage", ""),
        "phase": trial.get("Phase", ""),
        "trial_location": trial.get("Primary Region", ""),
        "secondary_location": trial.get("Secondary Countries", ""),
        "size": _coerce_int(trial.get("Size", "")),
        "drug_arm_size": _coerce_int(trial.get("Drug Arm Size (N)", "")),
        "placebo_arm_size": _coerce_int(trial.get("Placebo Arm Size (N)", "")),
        "discontinued_ae_drug_n": _coerce_int(trial.get("Discontinued AE Drug (N)", "")),
        "discontinued_ae_placebo_n": _coerce_int(trial.get("Discontinued AE Placebo (N)", "")),
        "comparator_name": trial.get("Comparator Name", ""),
        "comparator_arm_size_n": _coerce_int(trial.get("Comparator Arm Size (N)", "")),
        "discontinued_ae_comparator_n": _coerce_int(trial.get("Discontinued AE Comparator (N)", "")),
        "common_aes": trial.get("Common AEs", ""),
        "ae_severity": trial.get("AE Severity", ""),
        "ae_persistence": trial.get("AE Persistence", ""),
        "management_required": trial.get("Management Required", ""),
        "discontinuation_rate_drug_percent": _coerce_float(trial.get("Discontinuation Rate Drug (%)", "")),
        "discontinuation_rate_placebo_percent": _coerce_float(trial.get("Discontinuation Rate Placebo (%)", "")),
        "discontinuation_rate_comparator_percent": _coerce_float(trial.get("Discontinuation Rate Comparator (%)", "")),
        "created_at": created_at.isoformat(),
    }


def push_tolerability_trials_to_bq(
    molecule_name: str,
    extraction_result: Dict[str, Any],
    incremental_mode: bool = True,
) -> None:
    """Load refreshed tolerability trial rows into the clinical trials table."""
    trials = extraction_result.get("trials", []) if isinstance(extraction_result, dict) else []
    if not trials:
        logger.warning("push_tolerability_trials_to_bq: no tolerability trial rows for %s", molecule_name)
        return

    client = get_bq_client()
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{CLINICAL_TRIALS_TOLERABILITY_DATA_TABLE}"
    created_at = datetime.now(timezone.utc)
    rows = [_normalize_tolerability_trial_row(trial, created_at) for trial in trials]

    job_config = bigquery.LoadJobConfig(
        schema=CLINICAL_TRIALS_TOLERABILITY_DATA_SCHEMA,
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
        write_disposition=(
            bigquery.WriteDisposition.WRITE_APPEND
            if incremental_mode
            else bigquery.WriteDisposition.WRITE_TRUNCATE
        ),
    )

    load_job = client.load_table_from_json(rows, table_id, job_config=job_config)
    load_job.result()
    logger.info("Loaded %d tolerability trial row(s) into %s", load_job.output_rows, table_id)


def load_tolerability_trials_from_bq(molecule_name: str) -> list[dict[str, Any]]:
    """Load tolerability trial rows for a molecule from the persisted table."""
    if not molecule_name or not molecule_name.strip():
        return []

    client = get_bq_client()
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{CLINICAL_TRIALS_TOLERABILITY_DATA_TABLE}"
    sql = f"""
        SELECT *
        FROM `{table_id}`
        WHERE LOWER(molecule_name) = @molecule_name
        ORDER BY created_at DESC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("molecule_name", "STRING", molecule_name.strip().lower())
        ]
    )
    rows = client.query(sql, job_config=job_config).result().to_dataframe().to_dict(orient="records")
    logger.info("Loaded %d tolerability row(s) from %s", len(rows), table_id)
    return rows