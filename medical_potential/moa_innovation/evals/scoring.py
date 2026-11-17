from __future__ import annotations

import pandas as pd


def _claim_verification_score(row: pd.Series) -> float | None:
    supported_value = row.get("claims_supported")
    contradicted_value = row.get("claims_contradicted")
    if pd.isna(supported_value) or pd.isna(contradicted_value):
        return None
    supported = int(supported_value)
    contradicted = int(contradicted_value)
    denominator = supported + contradicted
    return supported / denominator if denominator else None


def _faithfulness_score(row: pd.Series) -> float | None:
    faithfulness_rating = row.get("faithfulness_score")
    if pd.isna(faithfulness_rating):
        return None
    return (float(faithfulness_rating) - 1) / 4


def build_eval_scores(
    classification_results: pd.DataFrame,
    claim_results: pd.DataFrame | None = None,
    judge_results: pd.DataFrame | None = None,
    *,
    amber_threshold: float = 65.0,
    green_threshold: float = 85.0,
    claim_verification_weight: float = 0.5,
    faithfulness_weight: float = 0.5,
) -> pd.DataFrame:
    total_weight = claim_verification_weight + faithfulness_weight
    if claim_verification_weight < 0 or faithfulness_weight < 0 or total_weight <= 0:
        raise ValueError("Score weights must be non-negative and have a positive total")

    result = classification_results.copy()
    for frame in (claim_results, judge_results):
        if frame is not None and not frame.empty:
            result = result.merge(frame, on="source_key", how="left", suffixes=("", "_eval"))
    result["claim_verification_score"] = result.apply(_claim_verification_score, axis=1)
    result["faithfulness_score_normalized"] = result.apply(_faithfulness_score, axis=1)
    result["composite_score"] = (
        (
            result["claim_verification_score"] * claim_verification_weight
            + result["faithfulness_score_normalized"] * faithfulness_weight
        )
        / total_weight
        * 100
    ).where(
        result["claim_verification_score"].notna()
        & result["faithfulness_score_normalized"].notna()
    )
    result["eval_status"] = result["composite_score"].map(
        lambda value: None if pd.isna(value) else (
            "Green" if value >= green_threshold else "Amber" if value >= amber_threshold else "Red"
        )
    )
    return result
