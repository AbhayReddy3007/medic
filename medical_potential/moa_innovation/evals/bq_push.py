from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from google.cloud import bigquery

from medical_potential.config import BQ_DATASET_ID, PROJECT_ID
from medical_potential.gcp_utils import get_bq_client

EVAL_TABLE = os.getenv("MOA_INNOVATION_EVAL_TABLE", "moa_innovation_eval_results")


def _json_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def build_bq_rows(results: list[dict[str, Any]], eval_run_id: str, model: str) -> list[dict[str, Any]]:
    timestamp = datetime.now(timezone.utc).isoformat()
    rows = []
    for result in results:
        row = dict(result)
        row.update({"eval_run_id": eval_run_id, "eval_timestamp": timestamp, "judge_model": model})
        for key in ("claim_details", "judge_details"):
            row[key] = _json_value(row.get(key))
        rows.append(row)
    return rows


def push_eval_rows(rows: list[dict[str, Any]], table_name: str = EVAL_TABLE) -> None:
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{table_name}"
    client = get_bq_client()
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
        autodetect=True,
    )
    client.load_table_from_json(rows, table_id, job_config=job_config).result()
