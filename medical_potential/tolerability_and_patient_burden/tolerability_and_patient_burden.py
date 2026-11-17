"""Patient tolerability helper script

This module extracts the `get_dimension_vi_tolerability` run function along with
the minimal helper functions and constants required to run it independently
within the `tolerability_and_patient_burden` package.
"""

from dotenv import load_dotenv
import logging
from medical_potential.config import (
    GEMINI_FLASH_PREVIEW_MODEL,
    TOLERABILITY_AND_PATIENT_BURDEN_DIMENSION_NAME,
)

from .tolerability_tools import (extract_incremental_trials,
                                 extract_tolerability_data,
                                 delete_existing_trials,
                                 compute_tolerability_score)
from .bq_utils import append_tolerability_to_bq, push_tolerability_trials_to_bq
from medical_potential.gcp_utils import (
    append_dimension_score_to_bigquery,
    upload_dimension_payload_cache_to_gcs,
    upload_dimension_report_pdf_to_gcs,
)

from .generate_report_and_rationale import generate_tolerability_report, generate_rationale_for_score

# Load env variables
load_dotenv()

# Configuration
DIMENSION_NAME = TOLERABILITY_AND_PATIENT_BURDEN_DIMENSION_NAME

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


async def tolerability_and_patient_burden_assessment(molecule_name: str, 
                                                     drug_class: str = "GLP-1", 
                                                     indication: str = None,
                                                     incremental_mode: bool = True) -> None:
    """Assess Patient Tolerability & Burden for the given molecule.

    Note: This function now logs progress and persists:
      - full tolerability payload to the tolerability table (append)
      - the dimension score to the central dim-scores table
      - the output payload to GCS cache
    The function intentionally does not return the payload (per request).
    """

    logger.info("[DIMENSION START] Tolerability & Patient Burden dimension start for %s", molecule_name)

    # Step 0: Check for incremental clinical trial data
    incremental_trials = extract_incremental_trials(molecule_name,
                                                    incremental_mode)

    if len(incremental_trials) == 0:
        logger.warning("[Tolerability] No incremental trial rows available for %s; skipping remaining steps", molecule_name)
        skip_payload = {
            "molecule": molecule_name,
            "dimension": DIMENSION_NAME,
            "incremental_mode": incremental_mode,
            "status": "skipped_no_incremental_trials",
            "message": "Incremental trials were None, so extraction, scoring, and BQ writes were skipped.",
        }
        try:
            upload_dimension_payload_cache_to_gcs(skip_payload, molecule_name, DIMENSION_NAME)
        except Exception as e:
            logger.warning("[Tolerability] Skip payload upload failed (non-fatal): %s", e)
        return

    # Step 1: Extract trial data with tolerability fields
    logger.info("[Tolerability Step 1] Extracting trial data for %s...", molecule_name)
    result = await extract_tolerability_data(molecule_name, 
                                             incremental_trials)
    trials = result.get("trials", [])
    logger.info("[Tolerability Step 1] Extracted %s trials", len(trials))

    if not trials:
        logger.warning("[Tolerability] No trials found for %s - aborting dimension", molecule_name)
        return

    # Step 2: Replace molecule rows in the tolerability trial table.
    try:
        logger.info("[Tolerability Step 2] Deleting existing tolerability trials for %s...", molecule_name)
        if incremental_mode:
            delete_existing_trials(molecule_name, result)
        logger.info("[Tolerability Step 2] Loading refreshed tolerability trials into BQ...")
        push_tolerability_trials_to_bq(molecule_name, result, incremental_mode=incremental_mode)
        logger.info("[Tolerability Step 2] Tolerability trials reloaded into BQ")
    except Exception as e:
        logger.exception("[Tolerability] Failed to refresh tolerability trial rows for %s", molecule_name)
        return

    # Step 3: Apply weightage framework and compute score from persisted trials.
    logger.info("[Tolerability Step 3] Computing tolerability score from persisted trials...")
    score_result = await compute_tolerability_score(
        molecule=molecule_name, drug_class=drug_class, indication=indication
    )

    # Add molecule name and trial count to result
    score_result["molecule"] = molecule_name
    score_result.setdefault("_scoring_breakdown", {})["total_trials_extracted"] = len(trials)

    logger.info("[Tolerability Step 3] Score computed: %s", score_result.get("score_numeric"))

    # Persist full tolerability payload to tolerability_table (append)
    try:
        logger.info("[Tolerability Step 4] Appending tolerability payload to BQ table...")
        append_tolerability_to_bq(molecule_name, score_result)
        logger.info("[Tolerability Step 4] Tolerability payload appended to BQ")
    except Exception as e:
        logger.warning("[Tolerability] Tolerability payload BQ append failed (non-fatal): %s", e)

    # Generate a concise rationale for the score
    rationale_text = None
    rationale_prompt_payload = None
    try:
        logger.info("[Tolerability Step 5] Generating rationale for dimension score...")
        rationale_text, rationale_prompt_payload = await generate_rationale_for_score(score_result, 
                                                                                      model_name=GEMINI_FLASH_PREVIEW_MODEL)
        score_result.setdefault("_generated_rationale", rationale_text)
    except Exception as e:
        logger.warning("[Tolerability] Rationale generation failed (non-fatal): %s", e)

    # Determine what to store in BigQuery: None if rationale wasn't generated
    try:
        logger.info("[Tolerability Step 6] Appending dimension score to dim-scores table...")
        rationale_for_bq = None
        if isinstance(rationale_text, str) and rationale_text:
            rationale_for_bq = rationale_text

        append_dimension_score_to_bigquery(
            molecule_name=molecule_name,
            dimension_name=DIMENSION_NAME,
            score=score_result.get("score_numeric"),
            rationale=rationale_for_bq,
        )
        logger.info("[Tolerability Step 6] Dimension score appended to dim-scores table")
    except Exception as e:
        logger.warning("[Tolerability] Dimension score append failed (non-fatal): %s", e)

    # Attach rationale prompt payload and final rationale to the payload cached to GCS
    try:
        score_result["rationale"] = rationale_for_bq
        score_result["rationale_prompt_payload"] = rationale_prompt_payload
    except Exception:
        pass

    # Step 6: Generate PDF report and upload it to GCS (parity with MoA flow)
    try:
        logger.info("[Tolerability Step 7] Generating PDF report for %s...", molecule_name)
        pdf_bytes = generate_tolerability_report(molecule_name)
        logger.info("[Tolerability Step 7] PDF generated (%s bytes). Uploading to GCS...", len(pdf_bytes) if pdf_bytes else 0)
        base_gcs_uri, archive_gcs_uri = upload_dimension_report_pdf_to_gcs(pdf_bytes, molecule_name, DIMENSION_NAME)
        logger.info("[Tolerability Step 7] Report uploaded to GCS: %s", base_gcs_uri)
    except Exception as e:
        logger.warning("[Tolerability] Report generation/upload failed (non-fatal): %s", e)

    report = {
        "report_generated": True,
        "base_report_gcs_uri": base_gcs_uri,
        "archive_report_gcs_uri": archive_gcs_uri
    }

    # Storing 
    score_result['report_metadata'] = report

    # Upload the full payload to GCS cache for reproducibility
    try:
        logger.info("[Tolerability Step 8] Uploading payload to GCS cache...")
        gcs_uri = upload_dimension_payload_cache_to_gcs(score_result, molecule_name, DIMENSION_NAME)
        logger.info("[Tolerability Step 8] Payload uploaded to GCS: %s", gcs_uri)
    except Exception as e:
        logger.warning("[Tolerability] Payload upload to GCS failed (non-fatal): %s", e)

    logger.info("[DIMENSION COMPLETE] Tolerability and Patient burden dimension end for %s", molecule_name)
