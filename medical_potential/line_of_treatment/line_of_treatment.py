"""Public entry point for the Lines-of-Therapy (LOT) assessment.

Behaviour is controlled by two flags in ``medical_potential/config.py``:

    BENCHMARK_RERUN: bool         — True  → rebuild benchmarks only, skip pipeline.
                                    False → run the full LOT scoring pipeline.
    COUNTRIES_TO_RERUN: list[str] — countries to rebuild (only used when
                                    BENCHMARK_RERUN=True).

Module responsibilities:
  - ``lot_scoring.py``               — core scoring logic, MOA lookup, overlay
                                       analysis, BigQuery writes
  - ``generate_country_lot.py``      — SoC benchmark creation (DOCX + JSON),
                                       GCS cache management
  - ``generate_report_and_rationale/generate_lot_report.py``
                                     — PDF report generation
  - ``generate_report_and_rationale/generate_rationale.py``
                                     — final LOT rationale generation
"""

from __future__ import annotations

import logging

from medical_potential.config import (
    GCS_BUCKET,
    GCS_SOC_BASE_PATH,
    LINE_OF_TREATMENT_DIMENSION_NAME,
    BENCHMARK_RERUN,
    COUNTRIES_TO_RERUN,
)
from medical_potential.gcp_utils import (
    append_dimension_score_to_bigquery,
    upload_dimension_payload_cache_to_gcs,
    upload_dimension_report_pdf_to_gcs,
)
from medical_potential.line_of_treatment.bq_utils import push_results_to_bigquery
from medical_potential.line_of_treatment.generate_report_and_rationale import (
    generate_lot_rationale_for_molecule,
    generate_lot_report,
)
from medical_potential.line_of_treatment.lot_scoring import (
    LotRow,
    build_overlay_prompt,
    compute_final_lot_score,
    discover_countries_gcs,
    lookup_moa,
    process_country,
)
from medical_potential.line_of_treatment.generate_country_lot import (
    rerun_countries,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def line_of_treatment_assessment(
    molecule_name: str,
    report_generate: bool = True,
) -> None:
    """Run the LOT assessment for one molecule.

    Behaviour is controlled by two variables in ``medical_potential/config.py``:

    - ``BENCHMARK_RERUN = True``: delete and rebuild the benchmark DOCX + JSON
      for every country in ``COUNTRIES_TO_RERUN``, then return immediately.
      The LOT scoring pipeline is NOT run.  Use this to refresh SoC benchmarks
      without triggering a full re-score.

    - ``BENCHMARK_RERUN = False`` (default): run the full LOT scoring pipeline
      for *molecule_name*, using whatever benchmarks are already cached in GCS.
      ``COUNTRIES_TO_RERUN`` is ignored entirely — no benchmarks are regenerated
      during a normal scoring run, so calling this function repeatedly is safe
      and idempotent with respect to the benchmarks.

    Args:
        molecule_name: Generic name of the molecule to assess.
        report_generate: Whether to generate the PDF report after scoring.
                         Only consulted when BENCHMARK_RERUN is False.
    """
    # ── molecule_name is mandatory when BENCHMARK_RERUN is False ─────────────
    if not BENCHMARK_RERUN:
        if not molecule_name or not molecule_name.strip():
            logger.error(
                "[LOT] molecule_name is mandatory when BENCHMARK_RERUN=False. "
                "Provide a molecule name to run the LOT pipeline."
            )
            raise ValueError(
                "molecule_name is mandatory when BENCHMARK_RERUN=False."
            )
        molecule_name = molecule_name.strip()

    # ── BENCHMARK_RERUN mode: rebuild benchmarks only, then stop ─────────────
    if BENCHMARK_RERUN:
        if not COUNTRIES_TO_RERUN:
            logger.info(
                "[LOT] BENCHMARK_RERUN=True but COUNTRIES_TO_RERUN is empty — "
                "nothing to rebuild. Set COUNTRIES_TO_RERUN in config.py."
            )
            return
        logger.info(
            "[LOT] BENCHMARK_RERUN=True — rebuilding benchmarks for: %s. "
            "LOT pipeline will NOT run.",
            ", ".join(COUNTRIES_TO_RERUN),
        )
        rerun_countries(COUNTRIES_TO_RERUN)
        logger.info("[LOT] Benchmark rebuild complete. Exiting without running LOT pipeline.")
        return

    # ── Normal LOT pipeline (BENCHMARK_RERUN=False) ───────────────────────────
    # molecule_name already validated and stripped above
    drugs = [molecule_name]
    logger.info("[LOT] Step 1/10: Starting assessment for molecule '%s'", molecule_name)

    logger.info("[LOT] Step 2/10: Looking up mechanism of action from BigQuery")
    drug_moa = lookup_moa(drugs)
    for drug, records in drug_moa.items():
        logger.info("[SOC_LOT] %s -> %s", drug, records)

    overlay_prompt = build_overlay_prompt(drugs, drug_moa)

    logger.info("[LOT] Step 3/10: Discovering country SOC documents in gs://%s/%s", GCS_BUCKET, GCS_SOC_BASE_PATH)
    countries = discover_countries_gcs()
    logger.info("[LOT] Found %d country folder(s): %s", len(countries), ", ".join(countries.keys()))

    logger.info("[LOT] Step 4/10: Processing country documents")
    all_rows: list[LotRow] = []
    for country, blob_names in countries.items():
        try:
            all_rows.extend(process_country(country, blob_names, drugs, overlay_prompt))
        except Exception:
            logger.exception("[SOC_LOT] Failed processing country '%s'", country)

    logger.info("[LOT] Step 5/10: Computing final LOT score across all countries")
    final_lot_score = compute_final_lot_score(all_rows)
    for r in all_rows:
        r.final_lot_score = final_lot_score

    logger.info("[LOT] Step 6/10: Persisting country-level LOT results to BigQuery")
    push_results_to_bigquery(all_rows)

    logger.info("[LOT] Step 7/10: Summarizing country-level results")
    logger.info("[LOT] SUMMARY")
    for country in countries:
        n = sum(1 for r in all_rows if r.country == country)
        logger.info("[SOC_LOT]   %s: %d row(s)", country, n)

    report_gcs_uri: str | None = None
    archived_report_gcs_uri: str | None = None
    if report_generate:
        logger.info("[LOT] Step 8/10: Generating and uploading PDF report for '%s'", molecule_name)
        report_bytes = generate_lot_report(molecule_name)
        report_gcs_uri, archived_report_gcs_uri = upload_dimension_report_pdf_to_gcs(
            pdf_bytes=report_bytes,
            molecule_name=molecule_name,
            dimension_name=LINE_OF_TREATMENT_DIMENSION_NAME,
        )

    logger.info("[LOT] Step 9/10: Generating rationale and persisting final dimension score")
    rationale, rationale_prompt_payload = await generate_lot_rationale_for_molecule(molecule_name)

    append_dimension_score_to_bigquery(
        molecule_name=molecule_name,
        dimension_name=LINE_OF_TREATMENT_DIMENSION_NAME,
        score=final_lot_score,
        rationale=rationale,
    )

    output_payload = {
        "molecule_name": molecule_name,
        "country_results": [
            {
                "drug_name": row.drug_name,
                "country": row.country,
                "lot_score": row.lot_score,
                "lot_type": row.lot_type,
                "rationale": row.rationale,
                "confidence": row.confidence,
                "final_lot_score": row.final_lot_score,
            }
            for row in all_rows
        ],
        "final_lot_score": final_lot_score,
        "rationale": rationale,
        "rationale_prompt_payload": rationale_prompt_payload,
        "report_gcs_uri": report_gcs_uri,
        "archived_report_gcs_uri": archived_report_gcs_uri,
    }

    logger.info("[LOT] Step 10/10: Uploading final output payload to GCS")
    cache_gcs_uri = upload_dimension_payload_cache_to_gcs(
        payload=output_payload,
        molecule_name=molecule_name,
        dimension_name=LINE_OF_TREATMENT_DIMENSION_NAME,
    )
    output_payload["cache_gcs_uri"] = cache_gcs_uri