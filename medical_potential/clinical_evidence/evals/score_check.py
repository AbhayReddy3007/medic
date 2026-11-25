"""Deterministic score checks (the clinical-evidence analogue of the MoA classification check).

The Clinical Evidence final score is computed by a formula, so the stored score can be checked without an LLM:
  Final = min(5, max(1, (P + mode(D, G, E)) / 2 + C))
Each component must also be inside its allowed range.
"""

from __future__ import annotations

import pandas as pd

from medical_potential.clinical_evidence.scoring_logic.deterministic_scores import compute_final_score
from .utils import source_key

# Allowed (min, max) per component.
RANGES = {
    "score_p": (0, 5),
    "score_d": (0, 5),
    "score_g": (0, 5),
    "score_e": (0, 5),
    "score_c": (-1.0, 0.5),
}
TOLERANCE = 0.01


def _float_or_none(value):
    try:
        number = float(value)
        return None if pd.isna(number) else number
    except (TypeError, ValueError):
        return None


def _check_row(row: pd.Series) -> tuple[bool, str, float | None]:
    score = _float_or_none(row.get("score"))
    if score is None or not 1 <= score <= 5:
        return False, f"final score {row.get('score')!r} is not a number from 1 to 5", None

    problems: list[str] = []
    components = {}
    for field, (low, high) in RANGES.items():
        value = _float_or_none(row.get(field))
        components[field] = value
        if value is None:
            problems.append(f"{field} is missing")
        elif not low <= value <= high:
            problems.append(f"{field}={value:g} is outside {low:g} to {high:g}")

    expected = None
    if all(value is not None for value in components.values()):
        expected = compute_final_score(
            components["score_p"], components["score_d"], components["score_g"],
            components["score_e"], components["score_c"],
        )
        if abs(expected - score) > TOLERANCE:
            problems.append(f"final score {score:g} differs from formula result {expected:g}")

    return not problems, "pass" if not problems else "; ".join(problems), expected


def check_clinical_evidence_score(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for index, row in df.iterrows():
        passed, reason, expected = _check_row(row)
        results.append({
            "source_row_index": index,
            "drug_name": row.get("drug_name"),
            "source_created_at": row.get("created_at"),
            "source_key": source_key(row),
            "source_score": row.get("score"),
            "recomputed_score": expected,
            "score_check_pass": passed,
            "score_check_reason": reason,
        })
    return pd.DataFrame(results)


def summarize_score_check(results: pd.DataFrame) -> dict:
    if results.empty:
        return {"total_rows": 0, "passed_rows": 0, "failed_rows": 0, "pass_rate": None}
    return {
        "total_rows": len(results),
        "passed_rows": int(results["score_check_pass"].sum()),
        "failed_rows": int((~results["score_check_pass"]).sum()),
        "pass_rate": float(results["score_check_pass"].mean()),
        "failure_reasons": results.loc[~results["score_check_pass"], "score_check_reason"].tolist(),
        "score_counts": results["source_score"].value_counts().sort_index().to_dict(),
    }
