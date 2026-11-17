"""BigQuery helpers and MoA innovation query definitions for mechanism of action."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from google.api_core.exceptions import NotFound
from google.cloud import bigquery

from medical_potential.config import (BQ_DATASET_ID, 
                                      PROJECT_ID,
                                      MOA_INNOVATION_TABLE,
                                      DRUG_DETAILS_FULL_TABLE_ID)

from medical_potential.gcp_utils import get_bq_client


MOA_INNOVATION_QUERY = f"""
SELECT DISTINCT cleaned_generic_name, Mechanism_of_Action, Cleaned_Indication
FROM `{DRUG_DETAILS_FULL_TABLE_ID}`
WHERE
    ( UPPER(cleaned_Target) LIKE '%GLUCAGON LIKE PEPTIDE 1%' OR UPPER(cleaned_Target) LIKE '%GLP-1%' OR UPPER(cleaned_Target) LIKE '%GLUCAGON LIKE PEPTIDE-1%'  )
    OR
 (data_source = 'IPD' AND Mechanism_of_Action = 'Glucagon-like peptide-1 (GLP-1) agonist'
    )
"""


def run_query(query: str):
    """Run a BigQuery query and return the results as a list of dicts."""
    client = get_bq_client()
    query_job = client.query(query)
    results = query_job.result()
    return [dict(row) for row in results]


MOA_INNOVATION_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("drug_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("indication", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("mechanism_statement", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("moa_classification", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("score", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("guardrail", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("confidence_score", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("confidence_tier", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("justification_mechanism_summary", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("justification_novelty_vs_soc", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("justification_competitor_comparison", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("justification_biological_rationale", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("justification_prior_validation_failure", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("justification_why_not_higher_score", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("grounding_urls", "STRING", mode="REPEATED"),
    bigquery.SchemaField("grounding_urls_by_step", "JSON", mode="NULLABLE"),
    bigquery.SchemaField("narrative_rationale", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("sources_primary", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("sources_secondary", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("sources_tertiary", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("analysis_date", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("processing_time_seconds", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
]


def _ensure_moa_innovation_table(client, table_id: str) -> None:
    """Create the table or append missing nullable/repeated fields."""
    try:
        table = client.get_table(table_id)
    except NotFound:
        client.create_table(bigquery.Table(table_id, schema=MOA_INNOVATION_SCHEMA))
        return

    existing_names = {field.name for field in table.schema}
    missing_fields = [
        field for field in MOA_INNOVATION_SCHEMA if field.name not in existing_names
    ]
    if missing_fields:
        table.schema = list(table.schema) + missing_fields
        client.update_table(table, ["schema"])


def append_moa_innovation_payload_to_bigquery(output_payload: dict) -> None:
    """Append one MoA innovation output payload row to BigQuery.

    Uses append-only semantics and relies on BigQuery CREATE_IF_NEEDED behavior
    for table creation.
    """
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{MOA_INNOVATION_TABLE}"

    justification = output_payload.get("justification", {}) or {}
    sources = output_payload.get("sources_used", {}) or {}

    grounding_urls_by_step = output_payload.get("grounding_urls_by_step", {}) or {}
    if isinstance(grounding_urls_by_step, dict):
        grounding_urls_by_step = {
            step_name: [
                value
                for value in urls
                if isinstance(value, str) and value
            ]
            for step_name, urls in grounding_urls_by_step.items()
            if isinstance(step_name, str) and isinstance(urls, list)
        }
    else:
        grounding_urls_by_step = {}

    score_val = output_payload.get("score")
    if isinstance(score_val, (int, float)):
        score = int(score_val)
    elif isinstance(score_val, str):
        try:
            score = int(float(score_val.strip()))
        except (TypeError, ValueError):
            score = None
    else:
        score = None

    confidence_val = output_payload.get("confidence_score", 0.0)
    try:
        confidence_score = float(confidence_val)
    except (TypeError, ValueError):
        confidence_score = None

    processing_time_val = output_payload.get("processing_time_seconds", 0.0)
    try:
        processing_time_seconds = float(processing_time_val)
    except (TypeError, ValueError):
        processing_time_seconds = None

    row = {
        "drug_name": (output_payload.get("drug_name") or "").strip(),
        "indication": output_payload.get("indication", ""),
        "mechanism_statement": output_payload.get("mechanism_statement", ""),
        "moa_classification": output_payload.get("moa_classification", ""),
        "score": score,
        "guardrail": output_payload.get("guardrail", ""),
        "confidence_score": confidence_score,
        "confidence_tier": output_payload.get("confidence_tier", ""),
        "justification_mechanism_summary": justification.get("mechanism_summary", ""),
        "justification_novelty_vs_soc": justification.get("novelty_vs_soc", ""),
        "justification_competitor_comparison": justification.get("competitor_comparison", ""),
        "justification_biological_rationale": justification.get("biological_rationale", ""),
        "justification_prior_validation_failure": justification.get("prior_validation_failure", ""),
        "justification_why_not_higher_score": justification.get("why_not_higher_score", ""),
        "grounding_urls": [
            value
            for value in output_payload.get("grounding_urls", [])
            if isinstance(value, str) and value
        ],
        "grounding_urls_by_step": grounding_urls_by_step,
        "narrative_rationale": output_payload.get("narrative_rationale", ""),
        "sources_primary": json.dumps(sources.get("primary", []), default=str),
        "sources_secondary": json.dumps(sources.get("secondary", []), default=str),
        "sources_tertiary": json.dumps(sources.get("tertiary", []), default=str),
        "analysis_date": output_payload.get("analysis_date", ""),
        "processing_time_seconds": processing_time_seconds,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    client = get_bq_client()
    _ensure_moa_innovation_table(client, table_id)
    job_config = bigquery.LoadJobConfig(
        schema=MOA_INNOVATION_SCHEMA,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
    )
    load_job = client.load_table_from_json([row], table_id, job_config=job_config)
    load_job.result()
