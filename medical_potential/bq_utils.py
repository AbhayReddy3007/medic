"""BigQuery persistence helpers for medical-potential pipeline data."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from google.cloud import bigquery

from medical_potential.config import (
    BQ_DATASET_ID,
    GEMINI_STATS_TABLE,
    PROJECT_ID,
)
from medical_potential.gcp_utils import get_bq_client

logger = logging.getLogger(__name__)

GEMINI_STATS_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("molecule_name", "STRING"),
    bigquery.SchemaField("dimension", "STRING"),
    bigquery.SchemaField("model_used", "STRING"),
    bigquery.SchemaField("service_tier", "STRING"),
    bigquery.SchemaField("execution_time_seconds", "FLOAT64"),
    bigquery.SchemaField("input_token_count", "INT64"),
    bigquery.SchemaField("output_token_count", "INT64"),
    bigquery.SchemaField("thought_token_count", "INT64"),
    bigquery.SchemaField("tool_token_count", "INT64"),
    bigquery.SchemaField("total_token_count", "INT64"),
    bigquery.SchemaField("total_num_of_calls", "INT64"),
    bigquery.SchemaField("num_of_search_enabled_calls", "INT64"),
    bigquery.SchemaField("num_of_search_queries", "INT64"),
    bigquery.SchemaField("run_status", "STRING"),
    bigquery.SchemaField("error_message", "STRING"),
    bigquery.SchemaField("cloud_run_job", "STRING"),
    bigquery.SchemaField("cloud_run_task_index", "INT64"),
    bigquery.SchemaField("created_at", "TIMESTAMP"),
]


def append_gemini_stats_to_bigquery(
    molecule_name: str,
    dimension_name: str,
    stats_by_model: list[dict[str, Any]],
    run_status: str,
    error_message: str | None = None,
    execution_time_seconds: float | None = None,
) -> None:
    """Append one row per model used by a dimension-molecule run."""
    if not stats_by_model:
        logger.info(
            "No Gemini calls recorded for %s/%s; no stats row written",
            molecule_name,
            dimension_name,
        )
        return

    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{GEMINI_STATS_TABLE}"
    common_fields = {
        "molecule_name": molecule_name.strip(),
        "dimension": dimension_name.strip(),
        "execution_time_seconds": execution_time_seconds,
        "run_status": run_status,
        "error_message": error_message,
        "cloud_run_job": os.getenv("CLOUD_RUN_EXECUTION",""),
        "cloud_run_task_index": os.getenv("CLOUD_RUN_TASK_INDEX", None),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    rows = [{**common_fields, **model_stats} for model_stats in stats_by_model]

    client = get_bq_client()
    job_config = bigquery.LoadJobConfig(
        schema=GEMINI_STATS_SCHEMA,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
    )
    load_job = client.load_table_from_json(rows, table_id, job_config=job_config)
    load_job.result()
    logger.info(
        "Appended %d Gemini stats row(s) for %s/%s to %s",
        len(rows),
        molecule_name,
        dimension_name,
        table_id,
    )
