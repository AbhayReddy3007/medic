"""Shared BigQuery utilities for the unmet-needs assessment pipeline."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from google.cloud import bigquery
from medical_potential.config import (
    BQ_DATASET_ID,
    PROJECT_ID,
)
from medical_potential.gcp_utils import get_bq_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

TRIAL_MAPPING_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("trial_id",              "STRING"),
    bigquery.SchemaField("molecule_name",          "STRING"),
    bigquery.SchemaField("unmet_need",             "STRING"),
    bigquery.SchemaField("category",               "STRING"),
    bigquery.SchemaField("country",                "STRING"),
    bigquery.SchemaField("addresses_unmet_need",   "STRING"),
    bigquery.SchemaField("evidence_strength",      "STRING"),
    bigquery.SchemaField("relevant_endpoint",      "STRING"),
    bigquery.SchemaField("explanation",            "STRING"),
    bigquery.SchemaField("search_sources_used",    "STRING"),
    bigquery.SchemaField("phase",                  "STRING"),
    bigquery.SchemaField("phase_status",           "STRING"),
    bigquery.SchemaField("trial_size",             "FLOAT64"),
    bigquery.SchemaField("trial_location",         "STRING"),
    bigquery.SchemaField("created_at_utc",         "TIMESTAMP"),
]

UNMET_NEED_SCORING_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("molecule_name",               "STRING"),
    bigquery.SchemaField("unmet_need",                  "STRING"),
    bigquery.SchemaField("category",                    "STRING"),
    bigquery.SchemaField("country",                     "STRING"),
    bigquery.SchemaField("score_dimension",             "STRING"),
    bigquery.SchemaField("unmet_need_score",            "FLOAT64"),
    bigquery.SchemaField("evidence_impact_score",       "FLOAT64"),
    bigquery.SchemaField("unmet_need_score_rationale",  "STRING"),
    bigquery.SchemaField("evidence_score_rationale",    "STRING"),
    bigquery.SchemaField("confidence",                  "STRING"),
    bigquery.SchemaField("num_of_trials_assessed",      "INT64"),
    bigquery.SchemaField("evidence_impact_weight",      "FLOAT64"),
    bigquery.SchemaField("weighted_unmet_need_score",   "FLOAT64"),
    bigquery.SchemaField("created_at_utc",              "TIMESTAMP"),
]


# ---------------------------------------------------------------------------
# Generic export helper
# ---------------------------------------------------------------------------

def _coerce_dataframe(df: pd.DataFrame, schema: list[bigquery.SchemaField]) -> pd.DataFrame:
    """Reorder and coerce a DataFrame to match the given BigQuery schema."""
    schema_cols = [f.name for f in schema]
    missing = [c for c in schema_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    df = df[schema_cols].copy()

    for field in schema:
        col = field.name
        if field.field_type == "STRING":
            df[col] = df[col].astype("string")
            df[col] = df[col].where(df[col].notna(), None)
        elif field.field_type == "FLOAT64":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif field.field_type == "INT64":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif field.field_type == "TIMESTAMP":
            # TIMESTAMP fields are already datetime objects, ensure timezone-aware
            df[col] = pd.to_datetime(df[col], utc=True)

    return df


def export_to_gbq(
    rows: list[dict[str, Any]],
    table_name: str,
    schema: list[bigquery.SchemaField],
    write_disposition: str = bigquery.WriteDisposition.WRITE_APPEND,
) -> None:
    """Export a list of row dicts to a BigQuery table.

    Args:
        rows: Records to upload (list of dicts).
        table_name: Unqualified table name within ``BQ_DATASET_ID``.
        schema: BigQuery schema field list that defines column order and types.
        write_disposition: WRITE_APPEND (default) or WRITE_TRUNCATE.
    """
    if not rows:
        logger.warning("export_to_gbq: no rows to export for table '%s'. Skipping.", table_name)
        return

    # Add UTC timestamp to each row
    utc_now = datetime.now(timezone.utc)
    enriched_rows = []
    for row in rows:
        enriched_row = dict(row)
        enriched_row["created_at_utc"] = utc_now
        enriched_rows.append(enriched_row)

    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{table_name}"
    df = _coerce_dataframe(pd.DataFrame(enriched_rows), schema)

    client = get_bq_client()
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
        write_disposition=write_disposition,
    )

    load_job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    load_job.result()
    logger.info(
        "Exported %d rows to %s (job: %s)",
        load_job.output_rows,
        table_id,
        load_job.job_id,
    )


# ---------------------------------------------------------------------------
# Generic import helper
# ---------------------------------------------------------------------------

def import_from_gbq(
    table_name: str,
    columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Import rows from a BigQuery table into a list of dictionaries.

    Args:
        table_name: Unqualified table name within ``BQ_DATASET_ID``.
        columns: Optional explicit column list. Defaults to all columns.
    """
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{table_name}"
    selected_columns = ", ".join(columns) if columns else "*"
    query = f"SELECT {selected_columns} FROM `{table_id}`"

    client = get_bq_client()
    query_job = client.query(query)
    rows = query_job.result().to_dataframe().to_dict(orient="records")
    logger.info("Imported %d rows from %s", len(rows), table_id)
    return rows
