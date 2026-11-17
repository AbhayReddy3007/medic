"""BigQuery helpers for Serious Safety Profile data push."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from google.cloud import bigquery

from medical_potential.gcp_utils import get_bq_client
from medical_potential.config import (PROJECT_ID, 
                                      BQ_DATASET_ID, 
                                      CLINICAL_TRIALS_SERIOUS_SAFETY_DATA,
                                      SERIOUS_SAFETY_PROFILE_SAE_SUMMARY)


# coercion helpers
def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        s = str(value).strip()
        if not s or s.lower() in ("n/a", "na", "null", "none"):
            return None
        s = s.replace(",", "").replace("%", "")
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

logger = logging.getLogger(__name__)

# Bigquery table schema for serious safety data
SERIOUS_SAFETY_TABLE_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("molecule_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("trial_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("dosage", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("phase", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("primary_region", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("secondary_countries", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("size", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("drug_arm_size_n", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("control_arm_size_n", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("sae_drug_n", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("sae_control_n", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("sae_rate_drug_percent", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("sae_rate_control_percent", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("serious_events_reported", "STRING", mode="REPEATED"),
    bigquery.SchemaField("deaths_reported", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("drug_related_deaths", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("drug_related_deaths_rationale", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("delta_sae_percent", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
]

# Uploading serious safety data to bq table

def _normalize_serious_safety_trial_row(trial: Dict[str, Any], created_at: datetime) -> dict[str, Any]:
    """Normalize an extracted trial dict into the serious safety table shape."""
    return {
        "molecule_name": trial.get("Molecule", ""),
        "trial_id": trial.get("Trial ID") or trial.get("NCT_ID") or "",
        "dosage": trial.get("Dosage", ""),
        "phase": trial.get("Phase", ""),
        "primary_region": trial.get("Primary Region", ""),
        "secondary_countries": trial.get("Secondary Countries", ""),
        "size": _coerce_int(trial.get("Size", "")),
        "drug_arm_size_n": _coerce_int(trial.get("Drug Arm Size (N)", "")),
        "control_arm_size_n": _coerce_int(trial.get("Control Arm Size (N)", "")),
        "sae_drug_n": _coerce_int(trial.get("SAE Drug (N)", "")),
        "sae_control_n": _coerce_int(trial.get("SAE Control (N)", "")),
        "sae_rate_drug_percent": _coerce_float(trial.get("SAE Rate Drug (%)", "")),
        "sae_rate_control_percent": _coerce_float(trial.get("SAE Rate Control (%)", "")),
        "serious_events_reported": trial.get("Serious Events Reported", []),
        "deaths_reported": _coerce_int(trial.get("Deaths Reported", "")),
        "drug_related_deaths": _coerce_int(trial.get("Drug-Related Deaths", "")),
        "drug_related_deaths_rationale": trial.get("Drug-Related Deaths Rationale", ""),
        "delta_sae_percent": _coerce_float(trial.get("Delta SAE (%)", "")),
        "created_at": created_at.isoformat(),
    }

def push_serious_safety_trials_to_bq(
    molecule_name: str,
    extraction_result: Dict[str, Any],
    incremental_mode: bool = True,
) -> None:
    """Load refreshed serious-safety trial rows into the clinical trials table."""
    trials = extraction_result.get("trials", []) if isinstance(extraction_result, dict) else []
    if not trials:
        logger.warning("push_serious_safety_trials_to_bq: no serious-safety trial rows for %s", molecule_name)
        return

    client = get_bq_client()
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{CLINICAL_TRIALS_SERIOUS_SAFETY_DATA}"
    created_at = datetime.now(timezone.utc)
    rows = [_normalize_serious_safety_trial_row(trial, created_at) for trial in trials]

    job_config = bigquery.LoadJobConfig(
        schema=SERIOUS_SAFETY_TABLE_SCHEMA,
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
        write_disposition=(
            bigquery.WriteDisposition.WRITE_APPEND
            if incremental_mode
            else bigquery.WriteDisposition.WRITE_TRUNCATE
        ),
    )

    load_job = client.load_table_from_json(rows, table_id, job_config=job_config)
    load_job.result()
    logger.info("Loaded %d serious-safety trial row(s) into %s", load_job.output_rows, table_id)

# Deleting any existing trials to avoid duplication
def delete_existing_trials(molecule_name: str, extraction_result: dict) -> int:
    """Delete existing serious-safety rows for a molecule before reloading them.

    Deletes rows for the molecule and the refreshed trial IDs so reruns only
    replace the trial rows that are being reprocessed.
    """
    trials = extraction_result.get("trials", []) if isinstance(extraction_result, dict) else []
    incremental_trial_ids = [
        str(trial.get("trial_id") or trial.get("Trial ID") or trial.get("NCT_ID") or "").strip()
        for trial in trials
    ]
    incremental_trial_ids = [trial_id for trial_id in incremental_trial_ids if trial_id]
    if not molecule_name or not molecule_name.strip() or not incremental_trial_ids:
        return 0

    try:
        client = get_bq_client()
    except Exception as e:
        logger.exception("BigQuery auth failed while deleting serious-safety rows for %s", molecule_name)
        return 0

    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{CLINICAL_TRIALS_SERIOUS_SAFETY_DATA}"
    sql = f"""
        DELETE FROM `{table_id}`
        WHERE LOWER(molecule_name) = @molecule_name
          AND trial_id IN UNNEST(@trial_ids)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("molecule_name", "STRING", molecule_name.strip().lower()),
            bigquery.ArrayQueryParameter("trial_ids", "STRING", incremental_trial_ids),
        ]
    )

    delete_job = client.query(sql, job_config=job_config)
    delete_job.result()
    affected_rows = int(getattr(delete_job, "num_dml_affected_rows", 0) or 0)
    logger.info(
        "Deleted %d existing serious-safety row(s) for molecule %s from %s",
        affected_rows,
        molecule_name,
        table_id,
    )
    return affected_rows


def load_serious_safety_sae_summary(molecule_name: str) -> list[dict]:
    """Load SAE event categorization rows for a molecule from the SAE summary BQ table.

    Returns a list of dict rows (empty list on error or if no rows found).
    """
    if not molecule_name:
        return []
    try:
        client = get_bq_client()
    except Exception:
        logger.exception("BigQuery auth failed while loading SAE summary for %s", molecule_name)
        return []

    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{SERIOUS_SAFETY_PROFILE_SAE_SUMMARY}"
    sql = f"""
        SELECT *
        FROM `{table_id}`
        WHERE LOWER(molecule_name) = @molecule
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("molecule", "STRING", molecule_name.strip().lower())
        ]
    )
    try:
        job = client.query(sql, job_config=job_config)
        rows = job.result()
        # Convert BigQuery Row objects to plain dicts
        return [dict(r) for r in rows] if rows is not None else []
    except Exception:
        logger.exception("Failed to load SAE summary rows from %s for %s", table_id, molecule_name)
        return []
