"""Deterministic score checks (the safety analogue of the MoA classification check).

The Serious Safety score is computed by rules, so the stored score can be checked without an LLM:
it must be an integer 1-5, equal the final step of its own reasoning, and respect the regulatory and
post-marketing caps recorded in that reasoning.
"""

from __future__ import annotations

import pandas as pd

from .utils import as_object, source_key


def _int_or_none(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _check_row(row: pd.Series) -> tuple[bool, str, int | None]:
    raw_score = row.get("score")
    score = _int_or_none(raw_score)
    if score is None or not 1 <= score <= 5:
        return False, f"score {raw_score!r} is not an integer from 1 to 5", None

    reasoning = as_object(row.get("score_reasoning")) or {}
    problems: list[str] = []

    final = _int_or_none((reasoning.get("step7_final") or {}).get("score"))
    if final is None:
        problems.append("no final step in score reasoning")
    elif final != score:
        problems.append(f"score {score} differs from reasoning final score {final}")

    regulatory_cap = _int_or_none((reasoning.get("step5_regulatory_adjustment") or {}).get("cap"))
    if regulatory_cap is not None and score > regulatory_cap:
        problems.append(f"score {score} exceeds regulatory cap {regulatory_cap}")

    post_marketing_cap = _int_or_none(
        (reasoning.get("step6_post_marketing_adjustment") or {}).get("post_marketing_cap")
    )
    if post_marketing_cap is not None and score > post_marketing_cap:
        problems.append(f"score {score} exceeds post-marketing cap {post_marketing_cap}")

    return not problems, "pass" if not problems else "; ".join(problems), final


def check_safety_score(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for index, row in df.iterrows():
        passed, reason, final = _check_row(row)
        results.append({
            "source_row_index": index,
            "drug_name": row.get("drug_name"),
            "source_created_at": row.get("created_at"),
            "source_key": source_key(row),
            "source_score": row.get("score"),
            "reasoning_final_score": final,
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
