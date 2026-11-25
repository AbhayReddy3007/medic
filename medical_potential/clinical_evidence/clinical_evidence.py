"""Clinical Evidence: Main Orchestrator.

Six-step pharmaceutical analyst workflow with PDGEC scoring. Exposes the single
entry point `clinical_evidence_scoring` for external callers.

Step 3 (real-world evidence) uses Google-Search-grounded Gemini calls; the grounding
URLs are returned in the output payload and the report input payload.

Concurrency notes:
    - Trial-source and RWE-source fetches run in worker threads.
    - All Gemini calls are gated by one global semaphore (see llm_runtime.py).
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from medical_potential.config import CLINICAL_EVIDENCE_DIMENSION_NAME
from medical_potential.gcp_utils import (
    append_dimension_score_to_bigquery,
    upload_dimension_payload_cache_to_gcs,
    upload_dimension_report_pdf_to_gcs,
)

from .bq_utils import (
    build_pub_rows,
    build_scoring_row,
    build_trial_rows,
    query_bigquery,
    write_bq_to_bigquery,
)
from .generate_report_and_rationale import generate_pdf_report, generate_rationale_for_score
from .scoring_logic import (
    assess_all_evidence,
    compute_final_score,
    consolidation_summary,
    identify_evidence_gaps,
    score_consistency,
    score_design,
    score_evidence,
    score_geography,
    score_phase,
)
from .utils import get_innovator, log, submit_with_ctx
from .web_data_extraction import (
    enrich_bq_trials,
    global_trial_scan,
    innovator_deep_dive,
    merge_trial_lists,
    normalize_molecule,
    run_rwe_summarizer,
    search_eu_ctr,
    search_who_ictrp,
)

load_dotenv()

DIMENSION_NAME = CLINICAL_EVIDENCE_DIMENSION_NAME

SOURCES_DIR = Path(__file__).resolve().parent / "sources"


def _save_sources_json(molecule_name: str, rwe: dict) -> Path | None:
    """Write every grounding source used in this run to sources/<molecule>/sources_<timestamp>.json.

    Same layout as moa_innovation / serious_safety_profile (`sources_by_step`, `total_unique_urls`,
    `run_timestamp`). Steps are the RWE data sources. Best-effort: a failure is logged and never
    affects the assessment.
    """
    try:
        now = datetime.now()
        folder = re.sub(r"[^\w.-]+", "_", str(molecule_name).strip()).strip("._") or "unknown"
        directory = SOURCES_DIR / folder
        directory.mkdir(parents=True, exist_ok=True)

        sources_by_step: dict[str, list] = {}
        unique_urls: set[str] = set()
        for step, records in (rwe.get("grounding_urls_by_source") or {}).items():
            bucket: dict[str, dict] = {}
            for record in records or []:
                if isinstance(record, dict) and record.get("url"):
                    bucket.setdefault(record["url"], record)
            sources_by_step[f"rwe_{step}"] = list(bucket.values())
            unique_urls.update(bucket)

        payload = {
            "drug_name": molecule_name,
            "run_timestamp": now.isoformat(timespec="seconds"),
            "total_unique_urls": len(unique_urls),
            "sources_by_step": sources_by_step,
        }
        path = directory / f"sources_{now.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        log.ok(f"Saved {payload['total_unique_urls']} sources to {path}")
        return path
    except Exception as e:
        log.warn(f"Could not save sources JSON for {molecule_name!r}: {e}")
        return None


async def clinical_evidence_scoring(molecule_name: str, max_trials: int = 20,
                                     write_bq: bool = False) -> dict:
    """
    Run the full clinical evidence scoring pipeline for one molecule.

    Args:
        molecule_name: Molecule to analyze.

    Returns:
        A dict containing PDGEC scores, summary metrics, and consistency text.

    All log.* calls made in here (and in every function this calls,
    including nested thread pools submitted via submit_with_ctx) write to
    whatever buffer is currently set in _log_buffer_var for this thread's
    context. With no buffer set, logs print immediately.
    """
    if not molecule_name or not str(molecule_name).strip():
        raise ValueError("molecule_name must be provided")

    if max_trials <= 0:
        raise ValueError("max_trials must be > 0")

    log.info(f"[DIMENSION START] Running dimension '{DIMENSION_NAME}' for molecule '{str(molecule_name).strip()}'")
    log.header(f"CLINICAL EVIDENCE ANALYZER v3 -- {str(molecule_name).upper()}")
    t0 = time.time()

    # Step 0
    log.step(0, "MOLECULE NORMALIZATION")
    molecule = normalize_molecule(str(molecule_name.strip()))
    log.ok(f"Generic : {molecule['generic_name']}")
    log.info(f"Brands  : {', '.join(molecule.get('brand_names', [])) or 'N/A'}")
    log.info(f"Sponsor : {molecule.get('sponsor', 'Unknown')}")

    # Step 1
    log.step(1, "GLOBAL CLINICAL TRIAL EVIDENCE SCAN")
    log.info("Running trial source searches in parallel...")

    bq_trials = []
    ctgov_trials = []
    eu_trials = []
    who_trials = []

    def _fetch_bq():
        log.sub("1A: BigQuery")
        return query_bigquery(molecule)

    def _fetch_ctgov():
        log.sub("1B: ClinicalTrials.gov")
        return global_trial_scan(molecule, max_per_query=max_trials)

    def _fetch_eu():
        log.sub("1C: EU Clinical Trials Register")
        return search_eu_ctr(molecule, max_results=max_trials)

    def _fetch_who():
        log.sub("1D: WHO ICTRP")
        return search_who_ictrp(molecule, max_results=max_trials)

    source_fns = {
        "bq": _fetch_bq,
        "ctgov": _fetch_ctgov,
        "eu": _fetch_eu,
        "who": _fetch_who,
    }

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {submit_with_ctx(executor, fn): name for name, fn in source_fns.items()}
        results_map = {}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results_map[name] = future.result()
            except Exception as e:
                log.warn(f"Trial source '{name}' failed: {e}")
                results_map[name] = []

    bq_trials = results_map.get("bq", [])
    ctgov_trials = results_map.get("ctgov", [])
    eu_trials = results_map.get("eu", [])
    who_trials = results_map.get("who", [])

    if bq_trials:
        log.sub("1E: Enriching BQ trials")
        bq_trials = enrich_bq_trials(bq_trials)
    trials = merge_trial_lists(bq_trials, ctgov_trials, eu_trials, who_trials)
    log.ok(f"Total unique trials: {len(trials)}")
    log.info(f"  BQ: {len(bq_trials)} | CTgov: {len(ctgov_trials)} | EU CTR: {len(eu_trials)} | WHO: {len(who_trials)}")
    log.info(f"  With results: {sum(1 for t in trials if t.get('results_available'))}")

    # Step 2
    log.step(2, "INNOVATOR WEBSITE DEEP DIVE")
    sponsor_info = innovator_deep_dive(molecule)

    # Step 3 — Real-World Evidence Summarizer
    log.step(3, "REAL-WORLD EVIDENCE SYNTHESIS")
    rwe = run_rwe_summarizer(molecule["generic_name"])
    rwe_synthesis = rwe["synthesis"]
    # Saved right after the grounded step so the sources are kept even if a later step fails.
    _save_sources_json(molecule_name.strip(), rwe)
    publications = []   # journal fetching disabled; RWE synthesis used instead

    # Step 4
    log.step(4, "EVIDENCE QUALITY ASSESSMENT")
    assessment = assess_all_evidence(trials, publications, sponsor_info.get("fda_product_entries", []))
    log.ok(f"Evidence Level: {assessment['evidence_level']}")

    # Step 5
    log.step(5, "EVIDENCE CONSOLIDATION")
    consolidation = consolidation_summary(trials, publications, assessment)
    log.ok(f"Studies: {consolidation['total_studies']} | Results: {consolidation['studies_with_results']}")
    log.ok(f"Consistency: {consolidation['findings_consistency']}")

    # Step 6
    log.step(6, "PDGEC SCORING")
    score_consistency._last_llm_detail = None  # reset before each run
    P, P_reason = score_phase(trials)
    D, D_reason = score_design(trials)
    G, G_reason = score_geography(trials)
    E, E_reason = score_evidence(trials, sponsor_info.get("fda_product_entries", []), rwe_synthesis)
    C, C_reason = score_consistency(trials, rwe_synthesis, consolidation)
    final = compute_final_score(P, D, G, E, C)
    gaps = identify_evidence_gaps(trials, publications, assessment, consolidation)
    reasoning = {"P": P_reason, "D": D_reason, "G": G_reason, "E": E_reason, "C": C_reason}

    log.ok(f"P={P}  D={D}  G={G}  E={E}  C={C}")
    log.ok(f"FINAL SCORE = {final} / 5.0")
    log.info(f"Evidence Gaps: {'; '.join(gaps)}")

    # Step 7 - Generate PDF Report
    log.header("GENERATING OUTPUTS")
    ts_bq = datetime.now().isoformat()

    llm_consistency_detail = getattr(score_consistency, '_last_llm_detail', None)
    report_input_payload = {
        "molecule": molecule,
        "scoring": {"P": P, "D": D, "G": G, "E": E, "C": C, "final": final,
                    "reasoning": reasoning,
                    "consistency_llm_detail": llm_consistency_detail},
        "evidence_gaps": gaps,
        "evidence_consolidation": consolidation, "evidence_assessment": assessment,
        "clinical_trials": trials, "publications": publications,
        "sponsor_info": sponsor_info,
        "rwe_synthesis": rwe_synthesis,
        "rwe_grounding_urls": rwe["grounding_urls"],
        "rwe_grounding_urls_by_source": rwe["grounding_urls_by_source"],
    }

    report_metadata = {
        "report_generated": False,
        "base_report_gcs_uri": None,
        "archive_report_gcs_uri": None,
    }
    try:
        pdf_bytes = generate_pdf_report(
            json_data=report_input_payload,
            molecule_name=molecule_name.strip(),
        )
        base_report_gcs_uri, archive_report_gcs_uri = upload_dimension_report_pdf_to_gcs(
            pdf_bytes=pdf_bytes,
            molecule_name=molecule["generic_name"],
            dimension_name=DIMENSION_NAME,
        )
        report_metadata = {
            "report_generated": True,
            "base_report_gcs_uri": base_report_gcs_uri,
            "archive_report_gcs_uri": archive_report_gcs_uri,
        }
    except Exception as e:
        log.warn(f"PDF report generation/upload failed: {e}")

    scoring_row = build_scoring_row(molecule, trials, publications, assessment,
                                     consolidation, P, D, G, E, C, final, gaps, reasoning, ts_bq)
    trial_rows  = build_trial_rows(molecule, trials, ts_bq)
    pub_rows    = build_pub_rows(molecule, publications, ts_bq)

    if write_bq:
        log.sub("Writing to BigQuery")
        write_bq_to_bigquery(scoring_row, trial_rows, pub_rows)

    elapsed = time.time() - t0
    log.header(f"COMPLETE -- {elapsed:.1f}s elapsed")

    # Extract clean LLM consistency narrative directly from the stored detail
    llm_detail = getattr(score_consistency, '_last_llm_detail', None)
    if llm_detail:
        pct = llm_detail.get("pct", 0)
        summary = llm_detail.get("summary", "")
        llm_consistency_text = f"~{pct:.1f}% of {len(trials)} studies consistent positive.\n{summary}".strip()
    else:
        # Rule-based was used — surface the C reasoning as the report text
        llm_consistency_text = reasoning.get("C", "").replace("[Rule-based]", "").strip()

    output_payload = {
        "molecule_name":          molecule["generic_name"],
        "brand_names":            ", ".join(molecule.get("brand_names", [])),
        "innovator":              get_innovator(trials, molecule),
        "score_p":                P,
        "score_d":                D,
        "score_g":                G,
        "score_e":                E,
        "score_c":                C,
        "final_score":            final,
        "evidence_level":         assessment.get("evidence_level", ""),
        "total_trials":           len(trials),
        "rct_count":              sum(1 for t in trials if "randomized" in str(t.get("study_design","")).lower()),
        "publications":           len(publications),
        "fda_approved":           "Yes" if assessment.get("regulatory_approval") else "No",
        "evidence_gaps":          "; ".join(gaps),
        "consistency":            consolidation.get("findings_consistency", ""),
        "llm_consistency_report": llm_consistency_text,
        "rwe_grounding_urls": rwe["grounding_urls"],
        "report_input_payload":   report_input_payload,
        "report":                 report_metadata,
    }

    # Step 8 - Generate concise rationale (LLM) then push the score to BQ
    rationale_text = ""
    rationale_payload = {}
    try:
        rationale_text, rationale_payload = await generate_rationale_for_score(output_payload)
    except Exception:
        log.err("[Clinical Evidence] Generating rationale failed; continuing without rationale.")
        rationale_text = "Rationale has not been generated."
        rationale_payload = {}

    # Attach rationale to output payload for caching
    output_payload["generated_rationale"] = rationale_text
    output_payload["rationale_payload"] = rationale_payload

    append_dimension_score_to_bigquery(
        molecule_name=molecule_name.strip(),
        dimension_name=DIMENSION_NAME,
        score=output_payload.get("final_score"),
        rationale=rationale_text,
    )

    # Step 9 - Push the output payload to the pipeline Cache
    upload_dimension_payload_cache_to_gcs(
        payload=output_payload,
        molecule_name=molecule_name.strip(),
        dimension_name=DIMENSION_NAME,
    )

    log.info(f"[DIMENSION COMPLETE] Dimension '{DIMENSION_NAME}' complete for molecule '{molecule_name.strip()}'")
