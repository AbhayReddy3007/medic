"""Serious Safety Profile - Dimension 5: Weighted SAE Rate Aggregator.

This module aggregates Serious Adverse Event (SAE) rates across multiple
clinical trials for a given molecule, producing a single weighted average
that reflects the relative evidential value of each trial.

Weighting logic
---------------
Not all trials carry equal scientific weight.  Each trial's contribution is
scaled by four factors multiplied together:

  Trial Weight = Phase Weight × Geography Weight × Dosage Weight × Sample Size

  Phase Weight
      Later-phase trials (Phase 3/4) provide more robust safety evidence than
      early-phase studies.  Weights range from 0.40 (Phase 1) to 1.00 (Phase 3+).

  Geography Weight
      Trials conducted in regulatory Tier-1 regions (US, EU, UK) are weighted
      at 1.00 because those data sets meet the highest regulatory standards.
      Tier-2 regions (Canada, Japan, Australia) receive 0.85, and all others 0.65.

  Dosage Weight
      For approved drugs, data from the approved dose is weighted at 1.00;
      off-label and exploratory doses are penalised.  For clinical-stage drugs,
      the lead registrational dose is weighted at 1.00, with alternative and
      exploratory doses receiving lower weights.

  Sample Size
      Larger trials produce more reliable rate estimates, so the arm sample size
      is included as a direct multiplier (not square-rooted).

SAE rate derivation
-------------------
When a trial reports only raw event counts (SAE Drug (N), SAE Control (N))
rather than pre-computed percentages, rates are derived on the fly:

  SAE Rate (%) = (Events (N) / Arm Size (N)) x 100

The weighted aggregate is then:

  Weighted Rate = Σ(Trial_Weight x Rate) / Σ(Trial_Weight)

This ensures that larger, later-phase, well-geographied trials dominate the
final figure rather than a simple unweighted mean.
"""

import re
from typing import Optional
from .utils import load_serious_safety_trials_data_from_bq, _normalize_trial_for_scoring

# Phase weights - higher phase = more reliable
PHASE_WEIGHTS = {
    # Phase 3+ gets full weight
    "4": 1.00,
    "3b": 1.00,
    "3": 1.00,
    "3a": 1.00,
    # Phase 2 gets 0.75
    "2b": 0.75,
    "2": 0.75,
    "2a": 0.75,
    # Phase 1/2 transitional
    "1/2": 0.60,
    "1/2a": 0.60,
    "1/2b": 0.60,
    # Phase 1 - exploratory (low weight)
    "1b": 0.40,
    "1": 0.40,
    "1a": 0.40,
}

# Geography weights - regulatory acceptance tiers
GEOGRAPHY_WEIGHTS = {
    # Tier 1: US, EU, UK = 1.00
    "US": 1.00, "EU": 1.00, "UK": 1.00,
    "United States": 1.00, "Europe": 1.00, "United Kingdom": 1.00,
    # Tier 2: High acceptance = 0.85
    "Canada": 0.85, "Switzerland": 0.85, "Australia": 0.85, "Japan": 0.85,
    # Tier 3: All others = 0.65
    "China": 0.65, "India": 0.65, "Korea": 0.65, "South Korea": 0.65,
    "Latin America": 0.65, "Middle East": 0.65, "Asia-Pacific": 0.65,
    "Brazil": 0.65, "Mexico": 0.65, "Russia": 0.65, "Taiwan": 0.65,
}

DEFAULT_GEOGRAPHY_WEIGHT = 0.65

# Dosage weights - commercial drugs (approved)
DOSAGE_WEIGHTS_COMMERCIAL = {
    "approved": 1.00,           # Approved dosage
    "off_label": 0.70,          # Common off-label / secondary dose
    "exploratory": 0.40,        # Non-commercial / exploratory dose
}

# Dosage weights - clinical-stage drugs (not yet approved)
DOSAGE_WEIGHTS_CLINICAL = {
    "target": 1.00,             # Lead / target registrational dose
    "alternative": 0.75,        # Clinically relevant alternative dose
    "exploratory": 0.50,        # Early exploratory dose
}


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _parse_int(value) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.upper() in {"N/A", "NONE", "NULL", "UNKNOWN"}:
        return None
    try:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        cleaned = re.sub(r"[^\d]", "", s)
        return int(cleaned) if cleaned else None
    except (TypeError, ValueError):
        return None


def _parse_float(value) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.upper() in {"N/A", "NONE", "NULL", "UNKNOWN"}:
        return None
    try:
        if isinstance(value, (int, float)):
            return float(value)
        cleaned = re.sub(r"[^\d.]", "", s)
        return float(cleaned) if cleaned else None
    except (TypeError, ValueError):
        return None


def _calculate_sae_rates(trial: dict) -> None:
    """Compute SAE Rate (%) = (SAE Count / Arm Size) × 100 for drug and control arms.

    Explicit rates take precedence; counts are used as a fallback.
    Writes SAE Rate Drug (%), SAE Rate Control (%), and Delta SAE (%) into the trial dict.
    """
    drug_rate = _parse_float(trial.get("SAE Rate Drug (%)"))
    control_rate = _parse_float(trial.get("SAE Rate Control (%)"))

    drug_n = _parse_int(trial.get("SAE Drug (N)"))
    control_n = _parse_int(trial.get("SAE Control (N)"))
    drug_arm_n = _parse_int(trial.get("Drug Arm Size (N)"))
    control_arm_n = _parse_int(trial.get("Control Arm Size (N)"))

    if drug_rate is None and drug_n is not None and drug_arm_n and drug_arm_n > 0:
        drug_rate = round((drug_n / drug_arm_n) * 100, 2)
    if control_rate is None and control_n is not None and control_arm_n and control_arm_n > 0:
        control_rate = round((control_n / control_arm_n) * 100, 2)

    trial["SAE Rate Drug (%)"] = drug_rate
    trial["SAE Rate Control (%)"] = control_rate
    trial["Delta SAE (%)"] = (
        round(drug_rate - control_rate, 2)
        if (drug_rate is not None and control_rate is not None)
        else None
    )


# =============================================================================
# WEIGHTING HELPERS
# Resolve each factor of: Trial Weight = Phase × Geography × Dosage × Sample Size
# =============================================================================

def normalize_phase(phase: str) -> str:
    """Normalise a phase string (Roman numeral, prefixed, or numeric) to a lowercase key."""
    if not phase or str(phase).upper() in ["N/A", "NONE", ""]:
        return ""

    phase = str(phase).strip().upper().replace("PHASE", "").strip()

    # Convert Roman numerals
    roman = {"IV": "4", "III": "3", "II": "2", "I": "1"}
    if phase in roman:
        return roman[phase]

    return phase.lower()


def get_phase_weight(phase: str) -> float:
    """Return evidential weight for a trial phase: Phase 3/4 → 1.00, Phase 2 → 0.75, Phase 1 → 0.40."""
    normalized = normalize_phase(phase)
    if not normalized:
        return 0.40  # Conservative default for missing phase
    return PHASE_WEIGHTS.get(normalized, 0.40)


def get_geography_weight(region: str) -> float:
    """Return the regulatory-acceptance weight for a trial region.

    Tier-1 (US, EU, UK) → 1.00 | Tier-2 (Canada, Japan, Australia) → 0.85 | others → 0.65.
    Multi-region strings are split on '/' or ',' and the highest weight is used.
    """
    if not region or str(region).upper() in ["N/A", "NONE", ""]:
        return DEFAULT_GEOGRAPHY_WEIGHT

    # Map countries to regions
    country_map = {
        "united states": "US", "usa": "US", "us": "US",
        "germany": "EU", "france": "EU", "spain": "EU", "italy": "EU",
        "netherlands": "EU", "belgium": "EU", "poland": "EU",
        "united kingdom": "UK", "uk": "UK",
        "canada": "Canada", "australia": "Australia", "japan": "Japan",
        "china": "China", "india": "India", "korea": "Korea",
        "brazil": "Latin America", "mexico": "Latin America",
    }

    # Split by / and ,
    parts = re.split(r"[/,]", region.lower())
    weights = []

    for part in parts:
        part = part.strip()
        if part in country_map:
            r = country_map[part]
            weights.append(GEOGRAPHY_WEIGHTS.get(r, DEFAULT_GEOGRAPHY_WEIGHT))
        elif part.upper() in GEOGRAPHY_WEIGHTS:
            weights.append(GEOGRAPHY_WEIGHTS[part.upper()])

    return max(weights) if weights else DEFAULT_GEOGRAPHY_WEIGHT


def get_dosage_weight(dosage: str, is_approved: bool = False, dosage_type: str = None) -> float:
    """Return the clinical-relevance weight for a trial dosage.

    Approved drugs: approved → 1.00, off-label → 0.70, exploratory → 0.40.
    Clinical-stage drugs: target → 1.00, alternative → 0.75, exploratory → 0.50.
    Missing dosage defaults to the lowest tier weight.
    """
    if not dosage or str(dosage).upper() in ["N/A", "NONE", ""]:
        # Conservative weight for missing dosage
        return DOSAGE_WEIGHTS_CLINICAL["exploratory"] if not is_approved else DOSAGE_WEIGHTS_COMMERCIAL["exploratory"]

    # Use explicit dosage type if provided
    if dosage_type:
        if is_approved:
            return DOSAGE_WEIGHTS_COMMERCIAL.get(dosage_type, 0.70)
        else:
            return DOSAGE_WEIGHTS_CLINICAL.get(dosage_type, 0.75)

    # Default: approved drug = approved dose, clinical = target dose
    if is_approved:
        return DOSAGE_WEIGHTS_COMMERCIAL["approved"]
    else:
        return DOSAGE_WEIGHTS_CLINICAL["target"]


def calculate_trial_weight(trial: dict, is_approved: bool = False) -> float:
    """Return composite trial weight: Phase × Geography × Dosage × Sample Size."""
    phase_w = get_phase_weight(trial.get("Phase", ""))
    geo_w = get_geography_weight(trial.get("Primary Region", ""))
    dosage_w = get_dosage_weight(trial.get("Dosage", ""), is_approved)
    size = _parse_int(trial.get("Size")) or 1
    return phase_w * geo_w * dosage_w * size


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def calculate_weighted_sae_rate(trials: list[dict], is_approved: bool = False) -> dict:
    """Return weighted SAE rates across all trials using Σ(weight × rate) / Σ(weight).

    Drug and control arms are aggregated independently; trials with partial data
    still contribute to whichever arm has a valid rate.

    Returns:
        drug_rate    - weighted SAE rate for the drug arm   (float | None)
        control_rate - weighted SAE rate for the control arm (float | None)
        delta_rate   - drug_rate - control_rate; positive = excess drug SAE burden
        trials_used  - trials with valid drug-arm SAE data
    """
    drug_weighted_sum = 0.0
    drug_total_weight = 0.0
    control_weighted_sum = 0.0
    control_total_weight = 0.0
    trials_used = 0

    total_weight = 0.0
    
    for trial in trials:
        _calculate_sae_rates(trial)
        weight = calculate_trial_weight(trial, is_approved)
        trial["weight_not_normalised"] = weight
        total_weight += weight
        drug_rate = trial.get("SAE Rate Drug (%)")
        if drug_rate is not None and isinstance(drug_rate, (int, float)):
            drug_weighted_sum += weight * drug_rate
            drug_total_weight += weight
            trials_used += 1
        control_rate = trial.get("SAE Rate Control (%)")
        if control_rate is not None and isinstance(control_rate, (int, float)):
            control_weighted_sum += weight * control_rate
            control_total_weight += weight

    # Calculate normalised weight for each trial
    for trial in trials:
        weight = trial.get("weight_not_normalised", 0)
        if total_weight > 0:
            trial["normalized_weight"] = round(weight / total_weight, 4)

    weighted_drug = round(drug_weighted_sum / drug_total_weight, 2) if drug_total_weight > 0 else None
    weighted_control = round(control_weighted_sum / control_total_weight, 2) if control_total_weight > 0 else None

    delta = (
        round(weighted_drug - weighted_control, 2)
        if (weighted_drug is not None and weighted_control is not None)
        else None
    )

    return {
        "sae_rate": weighted_drug,
        "control_rate": weighted_control,
        "delta_rate": delta,
        "trials_used": trials_used,
        "trials": trials,
    }


def calculate_trial_weights_and_agg_sae(molecule_name: str, is_approved: bool = False) -> dict:
    """Public entry point: load persisted trials for `molecule_name`, derive missing SAE rates,

    then return the weighted aggregate. Returns all fields from calculate_weighted_sae_rate()
    plus total_trials and trials.
    """
    # Load persisted trials from BigQuery and normalize to scoring keys
    loaded = load_serious_safety_trials_data_from_bq(molecule_name)
    loaded_trials = [_normalize_trial_for_scoring(row) for row in loaded]

    for trial in loaded_trials:
        _calculate_sae_rates(trial)

    result = calculate_weighted_sae_rate(loaded_trials, is_approved)
    result["total_trials"] = len(loaded_trials)

    return result



