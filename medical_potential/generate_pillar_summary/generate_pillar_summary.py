"""Build and generate Medical Potential pillar summaries from product metadata and reports."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv

from medical_potential.config import GEMINI_FLASH_PREVIEW_MODEL, PILLAR_SUMMARY_TABLE
from medical_potential.gemini_utils import gemini_api_call
from medical_potential.generate_pillar_summary.bq_utils import get_drug_detials, push_pillar_summary_to_bq
from medical_potential.generate_pillar_summary.dimension_report_extraction import get_dimension_reports_from_gcs
from medical_potential.generate_pillar_summary.prompts import (
    PROMPT_FOR_CLINICAL_MOLECULES,
    PROMPT_FOR_MARKETED_MOLECULE,
)

load_dotenv()

logger = logging.getLogger(__name__)


def generate_pillar_summary(molecule_name: str, model_name: str = GEMINI_FLASH_PREVIEW_MODEL) -> str:
    """Generate a concise Medical Potential pillar summary for one molecule.

    Args:
        molecule_name: Molecule to summarize.
        model_name: Gemini model to use for the summary generation call.

    Returns:
        The model-generated single-paragraph pillar summary.
    """

    molecule = (molecule_name or "").strip()
    if not molecule:
        raise ValueError("molecule_name must be provided")

    logger.info("Step 1/5: fetching product details for molecule=%s", molecule)
    product_details = get_drug_detials(molecule)
    logger.info("Step 1/5 complete: product details retrieved | stage=%s", product_details.get("Highest Development Stage"))

    logger.info("Step 2/5: selecting prompt template for molecule=%s", molecule)
    stage = product_details.get("Highest Development Stage")
    is_marketed = _is_marketed_or_preregistered(stage)
    logger.info("Step 2/5 complete: using %s prompt template", "marketed" if is_marketed else "clinical")

    logger.info("Step 3/5: building prompt with product details for molecule=%s", molecule)
    prompt_text = _build_prompt(molecule, product_details)
    logger.info("Step 3/5 complete: prompt built")

    logger.info("Step 4/5: fetching and trimming dimension PDFs from GCS for molecule=%s", molecule)
    pdf_first_page_payloads, missing_reports = get_dimension_reports_from_gcs(molecule_name=molecule)
    logger.info(
        "Step 4/5 complete: fetched %d report(s); %d missing or unreadable",
        len(pdf_first_page_payloads),
        len(missing_reports),
    )

    if missing_reports:
        logger.warning("Missing or unreadable pillar reports for %s: %d", molecule, len(missing_reports))
    if not pdf_first_page_payloads:
        logger.warning("No pillar report PDFs found for %s; generating from metadata only", molecule)

    logger.info("Step 5/5: calling Gemini model=%s with temperature=0", model_name)
    summary_text = _call_gemini_with_pdfs(prompt_text, pdf_first_page_payloads, model_name=model_name)
    logger.info("Step 5/5 complete: pillar summary generated")

    logger.info("Step 6/6: pushing pillar summary to BigQuery table=%s", PILLAR_SUMMARY_TABLE)
    push_pillar_summary_to_bq(molecule, summary_text)
    logger.info("Step 6/6 complete: pillar summary stored in BigQuery")
    return summary_text


def _extract_token_usage(response: Any) -> dict[str, int]:
    """Extract Gemini token counts from the response metadata."""

    usage_metadata = getattr(response, "usage_metadata", None) or getattr(response, "metadata", None)

    def _to_int(value: Any) -> int:
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0

    return {
        "input_tokens": _to_int(getattr(usage_metadata, "prompt_token_count", None)),
        "output_tokens": _to_int(getattr(usage_metadata, "candidates_token_count", None)),
        "thought_tokens": _to_int(getattr(usage_metadata, "thoughts_token_count", None)),
        "total_tokens": _to_int(getattr(usage_metadata, "total_token_count", None)),
    }

def _format_product_details(product_details: dict) -> str:
    return json.dumps(product_details, indent=2, ensure_ascii=True, sort_keys=True)


def _normalize_stage(stage: str | None) -> str:
    return " ".join((stage or "").strip().upper().split())


def _is_marketed_or_preregistered(stage: str | None) -> bool:
    normalized_stage = _normalize_stage(stage)
    return (
        normalized_stage.startswith("MARKETED")
        or normalized_stage.startswith("PRE-REGISTRATION")
        or normalized_stage.startswith("PRE-REGISTERED")
    )


def _build_prompt(molecule_name: str, product_details: dict) -> str:
    stage = product_details.get("Highest Development Stage")
    prompt_template = (
        PROMPT_FOR_MARKETED_MOLECULE
        if _is_marketed_or_preregistered(stage)
        else PROMPT_FOR_CLINICAL_MOLECULES
    )

    return prompt_template.format(
        molecule_name=molecule_name,
        product_details=_format_product_details(product_details),
    )


def _call_gemini_with_pdfs(
    prompt_text: str,
    pdf_first_page_payloads: list[tuple[str, bytes]],
    model_name: str,
) -> str:
    return gemini_api_call(
        prompt=prompt_text,
        model=model_name,
        temperature=0,
        pdf_attachments=pdf_first_page_payloads,
    )
