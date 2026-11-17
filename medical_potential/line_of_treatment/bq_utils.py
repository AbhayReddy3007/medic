"""BigQuery persistence helpers for Lines-of-Treatment results."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from google.cloud import bigquery

from medical_potential.config import BQ_DATASET_ID, LOT_TABLE, PROJECT_ID
from medical_potential.gcp_utils import get_bq_client


LOT_RESULTS_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("drug_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("country", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("lot_score", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("lot_type", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("rationale", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("confidence", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("final_lot_score", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="NULLABLE"),
]


def push_results_to_bigquery(rows: Sequence[Any]) -> None:
    """Append LOT result rows to BigQuery, creating the table when needed."""
    if not rows:
        return

    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{LOT_TABLE}"
    created_at = datetime.now(timezone.utc).isoformat()
    payload = [
        {
            "drug_name": row.drug_name,
            "country": row.country,
            "lot_score": row.lot_score,
            "lot_type": row.lot_type,
            "rationale": row.rationale,
            "confidence": row.confidence,
            "final_lot_score": row.final_lot_score,
            "created_at": created_at,
        }
        for row in rows
    ]

    job_config = bigquery.LoadJobConfig(
        schema=LOT_RESULTS_SCHEMA,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
    )
    load_job = get_bq_client().load_table_from_json(
        payload,
        table_id,
        job_config=job_config,
    )
    load_job.result()
