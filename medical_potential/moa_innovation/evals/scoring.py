from __future__ import annotations

import pandas as pd

# Every component is normalised to 0-1 before weighting.
DEFAULT_WEIGHTS: dict[str, float] = {
    "claim_precision": 0.25,     # claims supported by the sources the run retrieved / ALL claims
    "factual_accuracy": 0.15,    # claims supported by those sources or by open web / ALL claims
    "judge_faithfulness": 0.10,  # LLM judge: is the score supported by its own justification
    "context_precision": 0.15,   # retrieved sources that are relevant / all sources
    "field_relevance": 0.20,     # does each justification field answer its own question
    "claim_relevance": 0.15,     # claims that bear on the classification / all claims
}


def _ratio(numerator, denominator) -> float | None:
    if pd.isna(numerator) or pd.isna(denominator) or not denominator:
        return None
    return float(numerator) / float(denominator)


def _claim_precision(row: pd.Series) -> float | None:
    """Supported / ALL claims. Unverifiable and errored claims count against the score."""
    return _ratio(row.get("claims_supported"), row.get("claims_total"))


def _factual_accuracy(row: pd.Series) -> float | None:
    checked = row.get("open_web_checked")
    if pd.isna(checked) or not checked:
        return None
    supported = row.get("claims_supported")
    extra = row.get("claims_supported_open_web")
    if pd.isna(supported) or pd.isna(extra):
        return None
    return _ratio(int(supported) + int(extra), row.get("claims_total"))


def _judge_faithfulness(row: pd.Series) -> float | None:
    rating = row.get("faithfulness_score")
    return None if pd.isna(rating) else (float(rating) - 1) / 4


def _merge(result: pd.DataFrame, frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty or "source_key" not in frame.columns:
        return result
    return result.merge(frame, on="source_key", how="left", suffixes=("", "_eval"))


def build_eval_scores(
    classification_results: pd.DataFrame,
    claim_results: pd.DataFrame | None = None,
    judge_results: pd.DataFrame | None = None,
    context_precision_results: pd.DataFrame | None = None,
    field_relevance_results: pd.DataFrame | None = None,
    claim_relevance_results: pd.DataFrame | None = None,
    *,
    amber_threshold: float = 65.0,
    green_threshold: float = 85.0,
    weights: dict[str, float] | None = None,
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    """Combine all eval components into one composite score (0-100) and a Red/Amber/Green status.

    Components that were not run (or failed) for a molecule are dropped and the remaining weights are
    renormalised; `composite_coverage` is the share of total weight actually used. No composite or status is
    given when coverage is below `min_coverage`, so a molecule is never rated on one thin signal.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Score weights must be non-negative and have a positive total")

    result = classification_results.copy()
    for frame in (
        claim_results, judge_results, context_precision_results, field_relevance_results, claim_relevance_results
    ):
        result = _merge(result, frame)

    nan = pd.Series(float("nan"), index=result.index)

    def column(name: str) -> pd.Series:
        return pd.to_numeric(result[name], errors="coerce") if name in result else nan

    components = {
        "claim_precision": result.apply(_claim_precision, axis=1) if len(result) else nan,
        "factual_accuracy": result.apply(_factual_accuracy, axis=1) if len(result) else nan,
        "judge_faithfulness": result.apply(_judge_faithfulness, axis=1) if len(result) else nan,
        "context_precision": column("context_precision_score"),
        "field_relevance": column("field_relevance_score"),
        "claim_relevance": column("claim_relevance_score"),
    }
    for name, series in components.items():
        result[f"{name}_score"] = pd.to_numeric(series, errors="coerce")

    total_weight = sum(weights.values())
    weighted = pd.Series(0.0, index=result.index)
    used = pd.Series(0.0, index=result.index)
    for name, weight in weights.items():
        values = result[f"{name}_score"]
        weighted += values.fillna(0) * weight
        used += values.notna() * weight

    result["composite_coverage"] = used / total_weight
    result["composite_score"] = (weighted / used.where(used > 0) * 100).where(
        result["composite_coverage"] >= min_coverage
    )
    result["eval_status"] = result["composite_score"].map(
        lambda value: None if pd.isna(value) else (
            "Green" if value >= green_threshold else "Amber" if value >= amber_threshold else "Red"
        )
    )
    return result
