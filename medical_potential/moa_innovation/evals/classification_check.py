from __future__ import annotations

import pandas as pd

EXPECTED_SCORE_BY_CLASS = {
    "first-in-class": 5,
    "best-in-class": 4,
    "me-too": 3,
    "weak/outdated": 2,
    "poor/invalid": 1,
}


def check_classification_score(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for index, row in df.iterrows():
        classification = str(row.get("moa_classification", "")).strip()
        normalized = classification.casefold()
        expected = EXPECTED_SCORE_BY_CLASS.get(normalized)
        raw_score = row.get("score")
        try:
            actual = int(float(raw_score))
        except (TypeError, ValueError):
            actual = None
        passed = expected is not None and actual == expected
        reason = "pass" if passed else (
            f"unknown classification: {classification}" if expected is None else
            f"expected score {expected}, found {raw_score}"
        )
        results.append({
            "source_row_index": index,
            "drug_name": row.get("drug_name"),
            "source_created_at": row.get("created_at"),
            "source_key": f"{row.get('drug_name', '')}|{row.get('created_at', '')}",
            "source_classification": classification,
            "source_score": raw_score,
            "expected_score": expected,
            "classification_check_pass": passed,
            "classification_check_reason": reason,
        })
    return pd.DataFrame(results)


def summarize_classification_check(results: pd.DataFrame) -> dict:
    if results.empty:
        return {"total_rows": 0, "passed_rows": 0, "failed_rows": 0, "pass_rate": None}
    return {
        "total_rows": len(results),
        "passed_rows": int(results["classification_check_pass"].sum()),
        "failed_rows": int((~results["classification_check_pass"]).sum()),
        "pass_rate": float(results["classification_check_pass"].mean()),
        "failures_by_class": results.loc[~results["classification_check_pass"]]
        .groupby("source_classification").size().to_dict(),
        "score_crosstab": pd.crosstab(
            results["source_classification"], results["source_score"]
        ).to_dict(),
    }
