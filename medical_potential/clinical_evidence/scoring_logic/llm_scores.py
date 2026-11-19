"""LLM-assisted PDGEC components (E, C) with rule-based fallbacks."""

from medical_potential.config import GEMINI_FLASH_PREVIEW_MODEL
from medical_potential.gemini_utils import gemini_api_call

from ..llm_runtime import _gemini_semaphore
from ..utils import log
def score_evidence(trials: list, fda_entries: list, rwe_synthesis: str = "") -> tuple:
    """
    E score — Evidence Support, assessed by Gemini 2.5 Flash using RWE synthesis.

    Scoring tiers:
      5  — Strong post-launch real-world evidence (approved + meaningful RWE)
      4  — FDA-approved, limited RWE or RWE still maturing
      3  — Trial results available (registry / BQ endpoints) but no post-launch RWE
      1  — Trials registered but no results yet
      0  — No clinical evidence of any kind

    Gemini reads the RWE synthesis from drug_research_summarizer and assigns the
    tier with a plain-text justification. Falls back to rule-based logic if the
    API is unavailable.
    """
    has_fda  = len(fda_entries) > 0
    trials_with_results = sum(1 for t in trials if t.get("results_available"))
    has_bq_endpoints    = any(
        t.get("hba1c_change_pct") is not None or t.get("weight_change_pct") is not None
        for t in trials
    )

    # ── Gemini path ────────────────────────────────────────────────────────
    if not rwe_synthesis:
        log.warn("RWE synthesis is empty — using rule-based E scoring.")
    else:
        prompt = f"""You are a senior pharmaceutical evidence analyst.

Your task is to assign an Evidence Support score (E) for a drug based on the
real-world evidence (RWE) synthesis below, which was generated from post-launch
real-world databases (MIMIC-IV, CMS Medicare, ResDAC, CDC, All of Us, etc.) and
the innovator's own clinical data.

SCORING RULES — assign EXACTLY ONE integer from this list:
  5 = Strong post-launch RWE: the drug is approved and multiple independent
      real-world databases show consistent evidence of efficacy and/or safety
      post-launch (e.g. HbA1c reduction, weight loss, CV benefit confirmed in
      Medicare claims, EHR cohorts, or NIH All of Us data).
  4 = FDA-approved but RWE is limited or still maturing — fewer than 2
      independent post-launch databases with quantitative outcome data, OR
      RWE data are mostly from a single source.
  3 = Clinical trial results exist in registries or sponsor-reported endpoint
      data (HbA1c change, weight change) but no meaningful post-launch
      real-world data is available yet.
  1 = Trials are registered but no results have been reported yet.
  0 = No clinical evidence of any kind found.

CONTEXT FLAGS (from registry data):
  FDA-approved product entries found: {has_fda}
  Trials with reported results: {trials_with_results}
  Sponsor-reported endpoints available: {has_bq_endpoints}

REAL-WORLD EVIDENCE SYNTHESIS (from 10 post-launch data sources):
{rwe_synthesis[:4000]}

REQUIRED OUTPUT FORMAT — two lines only, no JSON, no markdown, no preamble:
SCORE: <integer: 0, 1, 3, 4, or 5>
REASONING: <2-3 sentences explaining the score, citing specific RWE sources>"""

        try:
            with _gemini_semaphore:
                raw = gemini_api_call(
                    prompt=prompt,
                    model=GEMINI_FLASH_PREVIEW_MODEL,
                    temperature=0.1,
                    max_output_tokens=512,
                )
            raw = raw.strip()
            score = None
            reasoning_text = ""
            for line in raw.splitlines():
                line = line.strip()
                if line.upper().startswith("SCORE:"):
                    try:
                        score = int(float(line.split(":", 1)[1].strip()))
                    except ValueError:
                        pass
                elif line.upper().startswith("REASONING:"):
                    reasoning_text = line.split(":", 1)[1].strip()
            if score is None:
                raise ValueError(f"Could not parse SCORE from response: {raw[:200]}")
            if score not in (0, 1, 3, 4, 5):
                score = min([0, 1, 3, 4, 5], key=lambda x: abs(x - score))
            log.ok(f"Gemini E score: {score}")
            return score, f"[LLM-assessed] {reasoning_text}"
        except Exception as e:
            log.warn(f"Gemini E scoring failed ({e}), falling back to rule-based.")

    # ── Rule-based fallback ────────────────────────────────────────────────
    if has_fda and rwe_synthesis:
        score = 5
        reasoning = ("FDA approval on record and RWE synthesis available from post-launch sources.\n"
                     "Indicates strong post-launch real-world evidence base.")
    elif has_fda:
        score = 4
        reasoning = ("FDA-approved product found but no RWE synthesis was generated.\n"
                     "Approval implies regulatory review; RWE depth could not be assessed.")
    elif trials_with_results > 0 or has_bq_endpoints:
        score = 3
        reasoning = (f"{trials_with_results} trial(s) with registry results or BQ endpoint data.\n"
                     f"Results exist but no post-launch real-world evidence available.")
    elif len(trials) > 0:
        score = 1
        reasoning = (f"{len(trials)} trial(s) registered but none report results yet.\n"
                     f"Clinical development underway; no efficacy or safety data available.")
    else:
        score = 0
        reasoning = ("No clinical evidence of any kind found.\n"
                     "Molecule may be preclinical, withdrawn, or not yet in human trials.")

    return score, reasoning


def _call_gemini(prompt: str, max_tokens: int = 1024) -> str:
    """Call Gemini through the shared helper function."""
    try:
        with _gemini_semaphore:
            return gemini_api_call(
                prompt=prompt,
                model=GEMINI_FLASH_PREVIEW_MODEL,
                temperature=0.2,
                max_output_tokens=max_tokens,
            )
    except Exception as e:
        log.warn(f"Gemini API call failed: {e}")

    return ""


def score_consistency(trials: list, rwe_synthesis: str, consolidation: dict) -> tuple:
    """
    C score — Consistency Adjustment, assessed by Gemini 2.5 Flash.

    Scoring thresholds:
      +0.5  — >= 20% of studies show consistent positive findings
      +0.25 — >= 10% but < 20% of studies are strong with no replication
      -0.5  — Mixed or conflicting evidence across studies
      -1.0  — Results unavailable (no results reported in any trial)

    Gemini reads trial data + the RWE synthesis and assigns a score.
    Falls back to rule-based logic if the API is unavailable.
    """
    molecule_name = next(
        (t["molecule_name"] for t in trials if t.get("molecule_name")), "Unknown"
    )

    n_trials      = len(trials)
    n_results     = sum(1 for t in trials if t.get("results_available"))
    n_rct         = sum(1 for t in trials if "randomized" in str(t.get("study_design", "")).lower())
    cons_text     = consolidation.get("findings_consistency", "Not assessed")
    replication   = consolidation.get("replication", "")
    hba1c_vals    = [t["hba1c_change_pct"] for t in trials if t.get("hba1c_change_pct") is not None]
    weight_vals   = [t["weight_change_pct"] for t in trials if t.get("weight_change_pct") is not None]
    terminated    = [t.get("trial_id", "?") for t in trials
                     if "terminat" in str(t.get("status", "")).lower()]

    # ── Gemini path ────────────────────────────────────────────────────────
    if True:  # Gemini model available — proceed
        prompt = f"""You are a senior pharmaceutical evidence analyst.

Assign a Consistency Adjustment score (C) for the molecule "{molecule_name}".

SCORING RULES — assign EXACTLY ONE value from this list:
  +0.5  = >= 20% of all registered studies show consistent POSITIVE findings
          (e.g. HbA1c reduction, weight loss confirmed in multiple independent
          trials AND corroborated by real-world data sources)
  +0.25 = >= 10% but < 20% of studies are strong/positive with no replication
          (a single high-quality study with clear positive signal, not yet
          independently replicated)
  -0.5  = Mixed or conflicting evidence: some studies positive, others negative
          or null; OR real-world data contradicts trial results
  -1.0  = Results unavailable: no trial has reported any results yet, or data
          is entirely absent from registry and real-world sources

CRITICAL RULES:
- Base the percentage on TOTAL registered trials (n={n_trials}), not just those
  with results.
- Routine adverse events (nausea, GI events, injection-site reactions) are NOT
  evidence of inconsistency. Do NOT use them to justify -0.5 or -1.0.
- -1.0 means truly NO results available, not just limited results.
- +0.25 is the appropriate score when there is exactly 1 strong positive study
  and no contradicting evidence.

TRIAL EVIDENCE:
- Total trials: {n_trials}  |  With results: {n_results}  |  RCTs: {n_rct}
- Rule-based consistency flag: {cons_text}
- Replication summary: {replication}
- HbA1c values across trials: {hba1c_vals[:20] if hba1c_vals else 'Not available'}
- Weight change values across trials: {weight_vals[:20] if weight_vals else 'Not available'}
- Terminated trials: {terminated if terminated else 'None'}

REAL-WORLD EVIDENCE SYNTHESIS (post-launch databases):
{rwe_synthesis[:3000] if rwe_synthesis else 'Not available'}

REQUIRED OUTPUT FORMAT — three lines only, no JSON, no markdown, no preamble:
SCORE: <number: one of 0.5, 0.25, -0.5, -1.0>
PCT: <float: estimated % of total trials with consistent positive findings>
REASONING: <3-4 sentences citing specific trials, endpoints, and RWE sources that drove the score>"""

        try:
            with _gemini_semaphore:
                raw = gemini_api_call(
                    prompt=prompt,
                    model=GEMINI_FLASH_PREVIEW_MODEL,
                    temperature=0.1,
                    max_output_tokens=1024,
                )
            raw = raw.strip()
            score = None
            pct = 0.0
            rationale = ""
            for line in raw.splitlines():
                line = line.strip()
                if line.upper().startswith("SCORE:"):
                    try:
                        score = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.upper().startswith("PCT:"):
                    try:
                        pct = float(line.split(":", 1)[1].strip().rstrip("%"))
                    except ValueError:
                        pass
                elif line.upper().startswith("REASONING:"):
                    rationale = line.split(":", 1)[1].strip()
            if score is None:
                raise ValueError(f"Could not parse SCORE from response: {raw[:200]}")
            valid_scores = [0.5, 0.25, -0.5, -1.0]
            if score not in valid_scores:
                score = min(valid_scores, key=lambda x: abs(x - score))
            reasoning_text = (
                f"[LLM-assessed] ~{pct:.1f}% of {n_trials} studies consistent positive.\n"
                f"{rationale}"
            )
            score_consistency._last_llm_detail = {"summary": rationale, "score": score, "pct": pct}
            log.ok(f"Gemini C score: {score}  ({pct:.1f}% consistent)")
            return score, reasoning_text
        except Exception as e:
            log.warn(f"Gemini C scoring failed ({e}), falling back to rule-based.")

    # ── Rule-based fallback ────────────────────────────────────────────────
    return _rule_based_consistency(trials, consolidation)


def _rule_based_consistency(trials: list, consolidation: dict) -> tuple:
    """Fallback rule-based consistency scoring using the new C thresholds."""
    n_total  = len(trials)
    n_results = sum(1 for t in trials if t.get("results_available"))
    cons     = consolidation.get("findings_consistency", "")
    result_trials = [t for t in trials if t.get("results_available")]

    # No results at all → -1.0
    if n_results == 0:
        return -1.0, (
            f"[Rule-based] No trial results available across {n_total} registered studies.\n"
            f"Consistency cannot be assessed; score is -1.0 (results unavailable)."
        )

    # Mixed signals → -0.5
    if "MIXED" in cons.upper():
        return -0.5, (
            f"[Rule-based] Mixed/conflicting endpoint signals detected across trials.\n"
            f"Consistency flag: {cons}"
        )

    # Calculate % of total trials with consistent positive results
    pct = (n_results / n_total * 100) if n_total > 0 else 0

    if pct >= 20 and "CONSISTENT" in cons.upper():
        ids = ", ".join(t.get("trial_id", "?") for t in result_trials[:5])
        return 0.5, (
            f"[Rule-based] {pct:.1f}% of {n_total} studies ({n_results}) show consistent positive findings.\n"
            f"Threshold >= 20% met. Trials: [{ids}]"
        )
    elif pct >= 10:
        ids = ", ".join(t.get("trial_id", "?") for t in result_trials[:3])
        return 0.25, (
            f"[Rule-based] {pct:.1f}% of {n_total} studies ({n_results}) are strong/positive, no replication yet.\n"
            f"Threshold >= 10% < 20% met. Trials: [{ids}]"
        )
    else:
        # Results exist but below 10% — treat as insufficient for positive adjustment
        return -0.5, (
            f"[Rule-based] Only {pct:.1f}% of {n_total} studies ({n_results}) report results.\n"
            f"Insufficient consistent evidence; treating as mixed/uncertain."
        )
