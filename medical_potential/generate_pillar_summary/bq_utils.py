"""BigQuery helpers for pillar summary generation."""

from __future__ import annotations

import re
import logging
from datetime import datetime, timezone
from typing import Any

from google.cloud import bigquery

from medical_potential.config import (
    BQ_DATASET_ID,
    CLINICAL_EFFICACY_TABLE,
    DRUG_DETAILS_FULL_TABLE_ID,
    PILLAR_SUMMARY_TABLE,
    PILLAR_NAME,
    PROJECT_ID,
    TRIAL_LLM_CONFIDENCE_THRESHOLD,
)
from medical_potential.gcp_utils import get_bq_client


logger = logging.getLogger(__name__)


GET_PRODUCT_DETAILS_QUERY = f"""
SELECT DISTINCT
    Cleaned_generic_name,
    Highest_Development_Stage,
    Cleaned_Indication,
    Mechanism_of_Action
FROM `{DRUG_DETAILS_FULL_TABLE_ID}`
WHERE
Mechanism_of_Action IS NOT NULL
AND LOWER(Cleaned_generic_name) = LOWER(@molecule_name)
"""

_HIGHEST_DEVELOPMENT_STAGE_ORDER = {
    "MARKETED": 4,
    "PRE-REGISTRATION": 3,
    "PHASE III": 2,
    "PHASE II": 1,
    "PHASE I": 0,
}

_PRIMARY_INDICATION = ["Type 2 Diabetes", "Obesity"]
_PRIMARY_INDICATION_MATCHER = re.compile(r"type\s*2\s*diabet(?:e|es)?|obesit(?:y|ies)", re.IGNORECASE)


CLINICAL_EFFICACY_TABLE_ID = f"{PROJECT_ID}.{BQ_DATASET_ID}.{CLINICAL_EFFICACY_TABLE}"
PILLAR_SUMMARY_TABLE_ID = f"{PROJECT_ID}.{BQ_DATASET_ID}.{PILLAR_SUMMARY_TABLE}"

GET_DOSAGE_DETAILS_QUERY = f"""
SELECT 
    DISTINCT dosage
FROM 
    `{CLINICAL_EFFICACY_TABLE_ID}`
WHERE 
    LOWER(molecule_name) = LOWER(@molecule_name)
    AND llm_confidence > {TRIAL_LLM_CONFIDENCE_THRESHOLD}
    AND dosage IS NOT NULL
    AND TRIM(dosage) != ''
ORDER BY 
    dosage
"""

PILLAR_SUMMARY_TABLE_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("molecule_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("Pillar", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("Summary", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
]


def _normalize_stage(stage: str | None) -> str:
    if not stage:
        return ""
    return " ".join(stage.strip().upper().split())


def get_highest_development_stage(stages: list[str | None]) -> str | None:
    """Return the most advanced stage using the requested priority order."""

    best_stage = None
    best_rank = -1

    for stage in stages:
        normalized_stage = _normalize_stage(stage)
        rank = _HIGHEST_DEVELOPMENT_STAGE_ORDER.get(normalized_stage)
        if rank is None:
            continue
        if rank > best_rank:
            best_rank = rank
            best_stage = stage.strip() if stage else None

    return best_stage


def get_drug_detials(molecule_name: str) -> dict:
    """Return formatted product details for a single molecule."""

    molecule = (molecule_name or "").strip()
    if not molecule:
        raise ValueError("molecule_name is required")

    client = get_bq_client()
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("molecule_name", "STRING", molecule),
        ]
    )
    query_job = client.query(GET_PRODUCT_DETAILS_QUERY, job_config=job_config)
    rows = [dict(row) for row in query_job.result()]

    dosage_query_job = client.query(GET_DOSAGE_DETAILS_QUERY, job_config=job_config)
    dosage_rows = [dict(row) for row in dosage_query_job.result()]
    unique_dosages: list[str] = []
    seen_dosages: set[str] = set()
    for row in dosage_rows:
        dosage = (row.get("dosage") or "").strip()
        if not dosage or dosage.lower() in {"n/a", "none"}:
            continue

        dosage_key = dosage.lower()
        if dosage_key in seen_dosages:
            continue

        seen_dosages.add(dosage_key)
        unique_dosages.append(dosage)

    cleaned_indications: list[str] = []
    for row in rows:
        indication_value = (row.get("Cleaned_Indication") or "").strip()
        if not indication_value:
            continue

        for indication in (item.strip() for item in indication_value.split(",")):
            if indication and indication not in cleaned_indications:
                cleaned_indications.append(indication)

    secondary_indications = [
        indication
        for indication in cleaned_indications
        if not _PRIMARY_INDICATION_MATCHER.search(indication)
    ]

    return {
        "drug_name": molecule,
        "Highest Development Stage": get_highest_development_stage(
            [row.get("Highest_Development_Stage") for row in rows]
        ),
        "Primary indication": _PRIMARY_INDICATION,
        "Secondary Indication": secondary_indications,
        "Dosage Strength details": unique_dosages,
    }


def push_pillar_summary_to_bq(molecule_name: str, summary: str) -> None:
    """Append a generated Medical Potential pillar summary to BigQuery."""

    molecule = (molecule_name or "").strip()
    summary_text = (summary or "").strip()

    if not molecule:
        raise ValueError("molecule_name is required")
    if not summary_text:
        raise ValueError("summary is required")

    client = get_bq_client()
    created_at = datetime.now(timezone.utc).isoformat()
    row: dict[str, Any] = {
        "molecule_name": molecule,
        "Pillar": PILLAR_NAME,
        "Summary": summary_text,
        "created_at": created_at,
    }

    job_config = bigquery.LoadJobConfig(
        schema=PILLAR_SUMMARY_TABLE_SCHEMA,
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )

    load_job = client.load_table_from_json([row], PILLAR_SUMMARY_TABLE_ID, job_config=job_config)
    load_job.result()
    logger.info("Loaded 1 %s pillar summary row for %s into %s", PILLAR_NAME, molecule, PILLAR_SUMMARY_TABLE_ID)