"""Shared utilities for Serious Safety web data extraction."""
from __future__ import annotations

from typing import List, Dict
from google.cloud import bigquery

from medical_potential.gcp_utils import get_bq_client
from medical_potential.config import PROJECT_ID, BQ_DATASET_ID, CLINICAL_TRIALS_SERIOUS_SAFETY_DATA


def load_serious_safety_trials_data_from_bq(molecule_name: str) -> List[Dict]:
    """Load serious safety trial rows for a molecule from the persisted table.

    Returns a list of dicts matching the table column names.
    """
    if not molecule_name or not molecule_name.strip():
        return []

    client = get_bq_client()
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{CLINICAL_TRIALS_SERIOUS_SAFETY_DATA}"
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
    return rows


def _normalize_trial_for_scoring(trial: dict) -> dict:
    """Remap persisted table column names to the scoring display keys expected by the weighting logic.

    The returned dict uses keys like `Molecule`, `Trial ID`, `Dosage`, `Phase`,
    `Primary Region`, `SAE Drug (N)`, `SAE Rate Drug (%)`, etc., matching the
    inputs expected by the weighting and summarization routines.
    """
    return {
        "Molecule": trial.get("molecule_name", ""),
        "Trial ID": trial.get("trial_id", ""),
        "NCT_ID": trial.get("trial_id", ""),
        "Dosage": trial.get("dosage") or "N/A",
        "Phase": trial.get("phase") or "N/A",
        "Primary Region": trial.get("primary_region") or "N/A",
        "Secondary Countries": trial.get("secondary_countries") or "N/A",
        "Size": trial.get("size"),
        "Drug Arm Size (N)": trial.get("drug_arm_size_n"),
        "Control Arm Size (N)": trial.get("control_arm_size_n"),
        "SAE Drug (N)": trial.get("sae_drug_n"),
        "SAE Control (N)": trial.get("sae_control_n"),
        "SAE Rate Drug (%)": trial.get("sae_rate_drug_percent"),
        "SAE Rate Control (%)": trial.get("sae_rate_control_percent"),
        "Serious Events Reported": trial.get("serious_events_reported", ""),
        "Deaths Reported": trial.get("deaths_reported"),
        "Drug-Related Deaths": trial.get("drug_related_deaths"),
        "Drug-Related Deaths Rationale": trial.get("drug_related_deaths_rationale", ""),
        "Delta SAE (%)": trial.get("delta_sae_percent"),
    }
