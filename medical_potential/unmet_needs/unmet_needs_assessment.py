"""Reusable unmet-needs assessment workflow.

This module exposes one orchestrator function:
- unmet_needs_assesment(molecule_name, incremental_mode)

It runs:
1) Trial-to-unmet-need mapping
2) Unmet-need scoring
3) Prompt-based unmet-needs report generation
4) Generate a concise rationale for the score
5) Persist the dimension score to BigQuery
6) Cache the final payload to GCS
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from medical_potential.config import GEMINI_FLASH_PREVIEW_MODEL, UNMET_NEEDS_DIMENSION_NAME
from medical_potential.gcp_utils import (
    append_dimension_score_to_bigquery,
    upload_dimension_payload_cache_to_gcs,
    upload_dimension_report_pdf_to_gcs,
)
from .compute_unmet_need_score_from_trial_data import (
    map_clinical_trial_data_with_unmet_needs,
    score_unmet_needs,
)
from .generate_report_and_rationale import (
    build_unmet_needs_report_payload,
    generate_unmet_needs_rationale,
    generate_prompt_unmet_needs_report_bytes,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DIMENSION_NAME = UNMET_NEEDS_DIMENSION_NAME


TRIAL_MAPPING_MODEL = GEMINI_FLASH_PREVIEW_MODEL
UNMET_NEED_SCORING_MODEL = GEMINI_FLASH_PREVIEW_MODEL
REPORT_MODEL = GEMINI_FLASH_PREVIEW_MODEL


async def unmet_needs_assesment(molecule_name: str, incremental_mode: bool = True) -> None:
    """Run unmet-needs assessment workflow and cache the final result."""
    if not molecule_name or not molecule_name.strip():
        raise ValueError("molecule_name must be provided")

    molecule = molecule_name.strip()
    logger.info("[DIMENSION START] Running dimension '%s' for molecule '%s'", DIMENSION_NAME, molecule)

    logger.info("Step 1/3 - Trial to unmet-need mapping")
    mapping_payload = await map_clinical_trial_data_with_unmet_needs(
        model=TRIAL_MAPPING_MODEL,
        molecule_name=molecule,
        incremental=incremental_mode,
    )

    clinical_trials_assessed = bool(mapping_payload.get("clinical_trials_assessed", False))
    clinical_trials_assessed_count = int(mapping_payload.get("clinical_trials_assessed_count", 0) or 0)

    if not clinical_trials_assessed:
        logger.warning("[Unmet Needs] No incremental trial rows available for %s; skipping remaining steps", molecule)
        skip_payload = {
            "molecule": molecule,
            "dimension": DIMENSION_NAME,
            "incremental_mode": incremental_mode,
            "status": "skipped_no_incremental_trials",
            "message": "Incremental trials were None, so extraction, scoring, and BQ writes were skipped.",
            "clinical_trials_details": {
                "incremental_clinical_trials_assessed": clinical_trials_assessed,
                "incremental_clinical_trials_assessed_count": clinical_trials_assessed_count,
            },
            "mapping_payload": mapping_payload,
        }
        try:
            upload_dimension_payload_cache_to_gcs(skip_payload, molecule, DIMENSION_NAME)
        except Exception as e:
            logger.warning("[Unmet Needs] Skip payload upload failed (non-fatal): %s", e)
        return

    logger.info("Step 2/3 - Unmet-need scoring")
    score_summary = await score_unmet_needs(
        molecule_name=molecule,
        model=UNMET_NEED_SCORING_MODEL,
    )
    score_summary["scoring_executed"] = True

    logger.info(
        "Unmet need aggregate score: %s (unique unmet needs: %d)",
        score_summary.get("unmet_need_score"),
        score_summary.get("molecule_level_unmet_needs_count", 0),
    )

    report = {
        "report_generated": False,
        "base_report_gcs_uri": None,
        "archive_report_gcs_uri": None,
        "summary_table": {
            "score": score_summary.get("unmet_need_score"),
            "total_trials": 0,
            "total_relevant_unmet_needs": 0,
        },
        "content": {},
    }

    logger.info("Step 3/3 - Unmet-needs report generation")
    report_input_payload = build_unmet_needs_report_payload(
        molecule_name=molecule,
        aggregated_score=score_summary.get("unmet_need_score")
    )

    try:
        pdf_result = await generate_prompt_unmet_needs_report_bytes(
            data=report_input_payload,
            model_name=REPORT_MODEL,
        )
        pdf_bytes, report_content = pdf_result
        base_report_gcs_uri, archive_report_gcs_uri = upload_dimension_report_pdf_to_gcs(
            pdf_bytes=pdf_bytes,
            molecule_name=molecule,
            dimension_name=DIMENSION_NAME,
        )

        report = {
            "report_generated": True,
            "base_report_gcs_uri": base_report_gcs_uri,
            "archive_report_gcs_uri": archive_report_gcs_uri,
            "summary_table": report_input_payload.get("summary_table", {}),
            "content": report_content,
        }
    except Exception:
        logger.exception("Step 3/3 - Unmet-needs report generation failed")

    logger.info("Step 4/5 - Generate concise score rationale")
    rationale = await generate_unmet_needs_rationale(
        data=report_input_payload,
        model_name=REPORT_MODEL,
    )

    logger.info("Step 5/5 - Upload unmet need dimension score & rationale to BQ")
    append_dimension_score_to_bigquery(
        molecule_name=molecule,
        dimension_name=DIMENSION_NAME,
        score=score_summary.get("unmet_need_score"),
        rationale=rationale,
    )

    final_payload = {
        "molecule_name": molecule,
        "incremental_mode": incremental_mode,
        "clinical_trials_details": {
            "incremental_clinical_trials_assessed": clinical_trials_assessed,
            "incremental_clinical_trials_assessed_count": clinical_trials_assessed_count,
        },
        "scoring": score_summary,
        "unmet_need_score": score_summary.get("unmet_need_score"),
        "report_input_data": report_input_payload,
        "report": report,
        "rationale": rationale,
    }

    logger.info("Step 6/6 - Cache final payload")
    output_payload = dict(final_payload)
    cache_gcs_uri = upload_dimension_payload_cache_to_gcs(
        payload=output_payload,
        molecule_name=molecule,
        dimension_name=DIMENSION_NAME,
    )

    output_payload["cache_gcs_uri"] = cache_gcs_uri

    logger.info("[DIMENSION COMPLETE] Dimension '%s' complete for molecule '%s'", DIMENSION_NAME, molecule)
