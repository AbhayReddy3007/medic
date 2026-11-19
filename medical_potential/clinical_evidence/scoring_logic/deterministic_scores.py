"""Rule-based PDGEC components (P, D, G) and the final score formula."""

from statistics import mode as _stat_mode

TIER1_COUNTRIES = {
    "United States", "United Kingdom", "Germany", "France", "Italy", "Spain",
    "Netherlands", "Belgium", "Austria", "Sweden", "Denmark", "Finland",
    "Norway", "Ireland", "Portugal", "Poland", "Czech Republic", "Greece",
    "Hungary", "Romania", "Bulgaria", "Croatia", "Slovakia", "Slovenia",
    "Estonia", "Latvia", "Lithuania", "Luxembourg", "Malta", "Cyprus", "Iceland",
    # Common abbreviation forms from ClinicalTrials.gov
    "US", "USA", "UK",
}

TIER2_COUNTRIES = {"Canada", "Switzerland", "Australia", "Japan"}


def _get_highest_phase(trials: list) -> int:
    """Return numeric highest phase across all trials: 4,3,2,1 or 0."""
    best = 0
    for t in trials:
        p = str(t.get("phase", "")).upper()
        if "4" in p: best = max(best, 4)
        elif "3" in p: best = max(best, 3)
        elif "2" in p: best = max(best, 2)
        elif "1" in p: best = max(best, 1)
    return best


def score_phase(trials: list) -> tuple:
    """P score based on highest phase completed trial. Returns (score, reasoning)."""
    highest = _get_highest_phase(trials)
    phase_counts = {}
    for t in trials:
        p = str(t.get("phase", "")).upper()
        if "3" in p or "4" in p:   phase_counts["Phase 3/4"] = phase_counts.get("Phase 3/4", 0) + 1
        elif "2" in p:             phase_counts["Phase 2"] = phase_counts.get("Phase 2", 0) + 1
        elif "1" in p:             phase_counts["Phase 1"] = phase_counts.get("Phase 1", 0) + 1

    score = {4: 5, 3: 5, 2: 4, 1: 2}.get(highest, 0)

    if highest >= 3:
        dist = "; ".join(f"{k}: {v}" for k, v in phase_counts.items() if v > 0)
        reasoning = (f"Highest completed phase is Phase {highest}, indicating late-stage clinical development.\n"
                     f"Phase distribution: {dist}.")
    elif highest == 2:
        reasoning = (f"Highest completed phase is Phase 2, indicating mid-stage clinical development.\n"
                     f"No Phase 3 trials found; efficacy signals are preliminary.")
    elif highest == 1:
        reasoning = (f"Only Phase 1 trials identified, indicating early-stage clinical exploration.\n"
                     f"Safety and dosing data may exist but no efficacy confirmation yet.")
    else:
        reasoning = (f"No phased clinical trials identified for this molecule.\n"
                     f"The molecule may be preclinical or the trials are not yet registered.")

    return score, reasoning


def score_design(trials: list) -> tuple:
    """D score based on best study design found. Returns (score, reasoning)."""
    designs = [t.get("study_design", "") for t in trials]
    rct_count = sum(1 for d in designs if "RCT" in d or "Randomized" in d)
    nonrand_count = sum(1 for d in designs if "Non-Randomized" in d)
    single_count = sum(1 for d in designs if "Single-arm" in d)
    obs_count = sum(1 for d in designs if "Observational" in d)

    if rct_count > 0:
        score = 5
        reasoning = (f"{rct_count} randomized controlled trial(s) found, the gold standard for clinical evidence.\n"
                     f"RCT design minimizes bias and provides the strongest causal inference.")
    elif nonrand_count > 0:
        score = 3
        reasoning = (f"{nonrand_count} non-randomized controlled or multi-center study(ies) found.\n"
                     f"Controlled but without randomization; moderate risk of selection bias.")
    elif single_count > 0:
        score = 2
        reasoning = (f"{single_count} single-arm exploratory study(ies) found, with no control group.\n"
                     f"Useful for early signal detection but cannot establish comparative efficacy.")
    elif obs_count > 0:
        score = 2
        reasoning = (f"{obs_count} observational/real-world study(ies) found.\n"
                     f"Provides real-world context but is subject to confounding and selection bias.")
    else:
        score = 0
        reasoning = (f"No meaningful study design information could be identified.\n"
                     f"Trials may be registered without design details or may not exist.")

    return score, reasoning


def score_geography(trials: list) -> tuple:
    """G score based on geography tier of trial locations. Returns (score, reasoning)."""
    all_countries = set()
    for t in trials:
        countries = t.get("trial_countries", [])
        if not countries:
            loc = t.get("trial_location", "")
            if loc and loc != "Not specified":
                countries = [c.strip() for c in loc.split(",")]
        all_countries.update(countries)

    tier1_found = all_countries & TIER1_COUNTRIES
    tier2_found = all_countries & TIER2_COUNTRIES
    tier3_found = all_countries - TIER1_COUNTRIES - TIER2_COUNTRIES

    if tier1_found:
        score = 5
        t1_sample = ", ".join(sorted(tier1_found)[:5])
        reasoning = (f"Trials conducted in Tier 1 regions (US/EU/UK): {t1_sample}.\n"
                     f"These geographies have the most rigorous regulatory oversight and data quality standards.")
    elif tier2_found:
        score = 4
        t2_sample = ", ".join(sorted(tier2_found)[:5])
        reasoning = (f"Trials conducted in Tier 2 regions: {t2_sample}.\n"
                     f"Strong regulatory frameworks but not the primary markets (US/EU) for global approval.")
    elif all_countries:
        score = 3
        t3_sample = ", ".join(sorted(tier3_found)[:5]) if tier3_found else ", ".join(sorted(all_countries)[:5])
        reasoning = (f"Trials conducted in Tier 3 regions only: {t3_sample}.\n"
                     f"Results may have limited generalizability to US/EU populations and regulatory pathways.")
    else:
        score = 3
        reasoning = (f"No geographic location data available for any registered trial.\n"
                     f"Defaulting to Tier 3 score; location could not be verified from registry data.")

    return score, reasoning


def compute_final_score(P: float, D: float, G: float, E: float, C: float) -> float:
    """
    Final Score = min(5, max(1, ((P + mode(D, G, E)) / 2) + C))

    P            — Phase score (development maturity)
    mode(D, G, E) — Modal score across Study Design (D), Geography (G),
                    and Evidence Support (E), representing the overall
                    quality of the evidence body. If all three are distinct
                    (no mode), the median is used as a fallback.
    C            — Consistency adjustment
    """
    try:
        dge_mode = _stat_mode([D, G, E])
    except Exception:
        # No unique mode — use median as fallback
        dge_mode = sorted([D, G, E])[1]
    raw = ((P + dge_mode) / 2.0) + C
    return round(min(5.0, max(1.0, raw)), 2)
