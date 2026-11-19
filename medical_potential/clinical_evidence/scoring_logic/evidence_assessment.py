"""Steps 4-5: evidence quality assessment, consolidation and gap identification."""

import re

def assess_all_evidence(trials, publications, fda_entries):
    phase_counts = {"Phase 1": 0, "Phase 2": 0, "Phase 3/4": 0, "Other": 0}
    for t in trials:
        p = str(t.get("phase", "")).upper()
        if "3" in p or "4" in p:   phase_counts["Phase 3/4"] += 1
        elif "2" in p:             phase_counts["Phase 2"] += 1
        elif "1" in p:             phase_counts["Phase 1"] += 1
        else:                      phase_counts["Other"] += 1

    design_counts = {}
    for t in trials:
        d = t.get("study_design", "Unknown")
        design_counts[d] = design_counts.get(d, 0) + 1

    rct_count = design_counts.get("Randomized Controlled Trial (RCT)", 0)
    trials_with_results = sum(1 for t in trials if t.get("results_available"))
    sizes = []
    for t in trials:
        try:
            s = int(t.get("trial_size") or 0)
            if s > 0: sizes.append(s)
        except (ValueError, TypeError): pass
    total_patients = sum(sizes)
    max_size = max(sizes) if sizes else 0

    has_fda = len(fda_entries) > 0
    has_p3 = phase_counts["Phase 3/4"] > 0

    if has_p3 and trials_with_results >= 2 and (rct_count >= 2 or has_fda):
        level, detail = "STRONG", "Multiple late-stage studies with published results"
    elif trials_with_results >= 1 and rct_count >= 1:
        level, detail = "MODERATE", "At least one controlled study with results"
    elif len(trials) > 0 and (phase_counts["Phase 1"] + phase_counts["Phase 2"]) > 0:
        level, detail = "LIMITED", "Early-stage studies only"
    else:
        level, detail = "INSUFFICIENT", "No available results"

    return {
        "evidence_level": level, "evidence_detail": detail,
        "total_trials": len(trials), "trials_with_results": trials_with_results,
        "total_publications": len(publications), "total_patients": total_patients,
        "largest_trial": max_size, "rct_count": rct_count,
        "has_regulatory_approval": has_fda,
        "phase_distribution": phase_counts, "design_distribution": design_counts,
    }


# ===================================================================
# STEP 5 -- Evidence Consolidation
# ===================================================================
def consolidation_summary(trials, publications, assessment):
    """
    Step 5: Consolidation based on TRIALS and PEER-REVIEWED PUBLICATIONS only.
    Tracks exact sources for consistency assessment.
    """
    def _to_float_pair(key):
        """Return list of (trial_id, float_value) for given endpoint key."""
        pairs = []
        for t in trials:
            raw = t.get(key)
            if raw is None: continue
            try: pairs.append((t.get("trial_id", "Unknown"), float(raw)))
            except (ValueError, TypeError): pass
        return pairs

    hba1c_pairs = _to_float_pair("hba1c_change_pct")
    weight_pairs = _to_float_pair("weight_change_pct")

    consistency = "Not assessable (insufficient endpoint data from BQ)"
    consistency_sources = []
    parts = []

    if len(hba1c_pairs) >= 2:
        ids = [p[0] for p in hba1c_pairs]
        vals = [p[1] for p in hba1c_pairs]
        if all(v < 0 for v in vals) or all(v > 0 for v in vals):
            parts.append("HbA1c: CONSISTENT")
            consistency_sources.append(f"HbA1c consistent across trials: {', '.join(ids)}")
        else:
            parts.append("HbA1c: MIXED")
            pos = [p[0] for p in hba1c_pairs if p[1] > 0]
            neg = [p[0] for p in hba1c_pairs if p[1] < 0]
            consistency_sources.append(
                f"HbA1c mixed -- positive in [{', '.join(pos)}], negative in [{', '.join(neg)}]"
            )

    if len(weight_pairs) >= 2:
        ids = [p[0] for p in weight_pairs]
        vals = [p[1] for p in weight_pairs]
        if all(v < 0 for v in vals) or all(v > 0 for v in vals):
            parts.append("Weight: CONSISTENT")
            consistency_sources.append(f"Weight consistent across trials: {', '.join(ids)}")
        else:
            parts.append("Weight: MIXED")
            pos = [p[0] for p in weight_pairs if p[1] > 0]
            neg = [p[0] for p in weight_pairs if p[1] < 0]
            consistency_sources.append(
                f"Weight mixed -- positive in [{', '.join(pos)}], negative in [{', '.join(neg)}]"
            )

    # Publication-based consistency: check if multiple publications report same phase outcomes
    pub_phase_outcomes = {}
    for p in publications:
        phase = p.get("study_phase", "")
        outcomes = p.get("key_outcomes", "")
        pmid = p.get("pmid", "") or p.get("doi", "") or "unknown"
        if phase and "Not specified" not in phase:
            if phase not in pub_phase_outcomes:
                pub_phase_outcomes[phase] = []
            pub_phase_outcomes[phase].append(pmid)

    for phase, pmids in pub_phase_outcomes.items():
        if len(pmids) >= 2:
            parts.append(f"Publications: {phase} replicated")
            consistency_sources.append(
                f"{len(pmids)} publications for {phase}: PMIDs/DOIs [{', '.join(pmids[:5])}]"
            )

    if parts:
        consistency = "; ".join(parts)

    # Replication tracking
    phase_result_counts = {}
    phase_result_ids = {}
    for t in trials:
        if t.get("results_available"):
            p = str(t.get("phase", "")).upper()
            k = "Phase 3/4" if ("3" in p or "4" in p) else ("Phase 2" if "2" in p else "Other")
            phase_result_counts[k] = phase_result_counts.get(k, 0) + 1
            phase_result_ids.setdefault(k, []).append(t.get("trial_id", ""))
    replicated = []
    for k, v in phase_result_counts.items():
        if v >= 2:
            ids = ", ".join(phase_result_ids[k][:5])
            replicated.append(f"{k} ({v} studies: {ids})")

    return {
        "total_studies": assessment["total_trials"],
        "studies_with_results": assessment["trials_with_results"],
        "results_published": len(publications) > 0,
        "peer_reviewed_count": len(publications),
        "findings_consistency": consistency,
        "consistency_sources": consistency_sources,
        "replication": ", ".join(replicated) if replicated else "No replicated phases",
        "evidence_level": assessment["evidence_level"],
        "evidence_detail": assessment["evidence_detail"],
    }


def identify_evidence_gaps(trials, publications, assessment, consolidation) -> list:
    """Identify gaps like a medical analyst would."""
    gaps = []
    if assessment.get("rct_count", 0) == 0:
        gaps.append("No randomized controlled trials identified")
    if assessment.get("total_publications", 0) == 0:
        gaps.append("No peer-reviewed publications found")
    if not assessment.get("has_regulatory_approval"):
        gaps.append("No FDA regulatory approval on record")

    # Long-term data
    has_long = False
    for p in publications:
        dur = (p.get("duration", "") or "").lower()
        if "year" in dur:
            try:
                yrs = int(re.search(r"(\d+)", dur).group(1))
                if yrs >= 2: has_long = True
            except Exception: pass
    if not has_long:
        gaps.append("Limited long-term safety data (no studies >= 2 years found)")

    # Head-to-head
    comps = set()
    for t in trials:
        for c in t.get("comparators", []):
            if "ACTIVE" in c.get("arm_type", "").upper():
                comps.add(c.get("arm_label", ""))
    if not comps:
        gaps.append("No head-to-head comparisons with active therapies")

    # Geographic diversity
    all_countries = set()
    for t in trials:
        cc = t.get("trial_countries", [])
        all_countries.update(cc)
    if len(all_countries) <= 2:
        gaps.append(f"Limited geographic diversity (only {len(all_countries)} countries)")

    # Replication
    if "No replicated" in consolidation.get("replication", ""):
        gaps.append("No replication of results across independent studies")

    return gaps if gaps else ["No major evidence gaps identified"]
