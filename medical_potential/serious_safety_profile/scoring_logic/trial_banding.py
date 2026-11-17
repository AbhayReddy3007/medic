"""Trial-weight and trial-banding helpers."""

from __future__ import annotations

from typing import Optional

from .parsing import parse_float


def trial_band(delta_sae: Optional[float]) -> Optional[str]:
    """Classify trial into band (S1-S4) based on delta SAE."""
    if delta_sae is None:
        return None
    if delta_sae <= 0:
        return "S1"
    if delta_sae <= 0.5:
        return "S2"
    if delta_sae <= 2.0:
        return "S3"
    return "S4"

def assign_weight_tiers(trials: list[dict]) -> dict[int, str]:

    if not trials:
        return {}

    # Use normalized_weight directly from each trial
    indexed = []
    for i, trial in enumerate(trials):
        w = parse_float(trial.get("normalized_weight"))
        if w is None or w < 0:
            w = 0.0
        indexed.append({"idx": i, "weight": w})
    indexed.sort(key=lambda x: x["weight"], reverse=True)

    tiers: dict[int, str] = {}
    cumulative = 0.0
    i = 0
    n = len(indexed)

    while i < n and cumulative <= 0.5:
        item = indexed[i]
        tiers[item["idx"]] = "high"
        cumulative += item["weight"]
        i += 1

    if i > 0 and i < n:
        tie_w = indexed[i - 1]["weight"]
        while i < n and indexed[i]["weight"] == tie_w:
            tiers[indexed[i]["idx"]] = "high"
            cumulative += indexed[i]["weight"]
            i += 1

    medium_threshold = cumulative + 0.3

    while i < n and cumulative <= medium_threshold:
        item = indexed[i]
        tiers[item["idx"]] = "medium"
        cumulative += item["weight"]
        i += 1

    if i > 0 and i < n:
        tie_w = indexed[i - 1]["weight"]
        while i < n and indexed[i]["weight"] == tie_w:
            tiers[indexed[i]["idx"]] = "medium"
            i += 1

    while i < n:
        tiers[indexed[i]["idx"]] = "low"
        i += 1

    return tiers


def build_trial_banding(trials: list[dict], derive_trial_delta) -> tuple[list[dict], dict[str, int], dict[str, int], bool]:
    step1_trials: list[dict] = []
    for idx, trial in enumerate(trials):
        step1_trials.append(
            {
                "idx": idx,
                "trial_id": trial.get("Trial ID") or f"Trial {idx + 1}",
                "delta_sae": derive_trial_delta(trial),
                "control_rate": parse_float(trial.get("SAE Rate Control (%)")),
            }
        )

    tiers = assign_weight_tiers(trials)
    band_counts = {"S1": 0, "S2": 0, "S3": 0, "S4": 0}
    high_tier_band_counts = {"S1": 0, "S2": 0, "S3": 0, "S4": 0}
    has_s4 = False

    # First pass: classify all trials into bands
    for t in step1_trials:
        band = trial_band(t["delta_sae"])
        t["band"] = band
        tier = tiers.get(t["idx"], "low")
        t["weight_tier"] = tier
        idx = t["idx"]
        
        # Add weight_tier and band to the original trial
        trials[idx]["weight_tier"] = tier
        trials[idx]["band"] = band
        if band is None:
            continue
        band_counts[band] += 1
        if tier == "high":
            high_tier_band_counts[band] += 1
        if band == "S4":
            has_s4 = True

    # Second pass: Calculate total normalized_weight for all S4 trials
    # If total <= 0.2, reclassify all S4 trials as S3
    if has_s4:
        s4_total_weight = 0.0
        for t in step1_trials:
            if t["band"] == "S4":
                normalized_weight = parse_float(trials[t["idx"]].get("normalized_weight"))
                if normalized_weight is not None and normalized_weight >= 0:
                    s4_total_weight += normalized_weight
        
        if s4_total_weight <= 0.2:
            # Reclassify all S4 trials as S3
            for t in step1_trials:
                if t["band"] == "S4":
                    idx = t["idx"]
                    trials[idx]["band"] = "S3"
                    t["band"] = "S3"
            
            # Update band counts
            band_counts["S3"] += band_counts["S4"]
            high_tier_band_counts["S3"] += high_tier_band_counts["S4"]
            band_counts["S4"] = 0
            high_tier_band_counts["S4"] = 0
            has_s4 = False

    return trials, band_counts, high_tier_band_counts, has_s4
