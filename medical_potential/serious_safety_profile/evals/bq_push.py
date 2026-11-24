from __future__ import annotations

import os
from typing import Any

from medical_potential.moa_innovation.evals.bq_push import build_bq_rows, push_eval_rows as _push  # noqa: F401

EVAL_TABLE = os.getenv("SERIOUS_SAFETY_EVAL_TABLE", "serious_safety_profile_eval_results")


def push_eval_rows(rows: list[dict[str, Any]], table_name: str = EVAL_TABLE) -> None:
    _push(rows, table_name=table_name)
