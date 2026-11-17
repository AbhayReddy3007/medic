"""Utilities for fetching and trimming pillar dimension reports from GCS."""

from __future__ import annotations

import logging
from io import BytesIO

from dotenv import load_dotenv
from google.api_core.exceptions import NotFound
from pypdf import PdfReader, PdfWriter

from medical_potential.config import GCS_BUCKET, GCS_REPORT_BASE_PATH, MEDICAL_POTENTIAL_DIMENSION_NAMES, PILLAR_NAME
from medical_potential.gcp_utils import get_gcs_client

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_DIMENSION_NAMES = MEDICAL_POTENTIAL_DIMENSION_NAMES


def extract_first_page_pdf(pdf_bytes: bytes) -> bytes:
    """Return a new PDF containing only the first page of the input PDF."""

    reader = PdfReader(BytesIO(pdf_bytes))
    if len(reader.pages) == 0:
        raise ValueError("PDF has no pages")

    writer = PdfWriter()
    writer.add_page(reader.pages[0])

    output_buffer = BytesIO()
    writer.write(output_buffer)
    return output_buffer.getvalue()


def _normalize_dimension_name(dimension_name: str) -> str:
    normalized = (dimension_name or "").strip()
    if not normalized:
        raise ValueError("dimension_names must not contain empty values")
    return normalized


def get_dimension_reports_from_gcs(
    molecule_name: str,
    dimension_names: list[str] = list(DEFAULT_DIMENSION_NAMES),
    bucket_name: str = GCS_BUCKET,
) -> tuple[list[tuple[str, bytes]], list[str]]:
    """Fetch dimension reports from GCS and keep only the first page of each PDF.

    Returns:
    - a list of ``(dimension_name, first_page_pdf_bytes)`` items
    - a list of missing or unreadable GCS blob paths
    """

    molecule = (molecule_name or "").strip()
    if not molecule:
        raise ValueError("molecule_name must be provided")

    if not dimension_names:
        raise ValueError("dimension_names must not be empty")

    bucket = get_gcs_client().bucket(bucket_name)

    pdf_first_page_payloads: list[tuple[str, bytes]] = []
    missing_reports: list[str] = []

    for dimension_name in dimension_names:
        normalized_dimension_name = _normalize_dimension_name(dimension_name)
        blob_path = f"{GCS_REPORT_BASE_PATH}/{molecule}/{PILLAR_NAME}/{normalized_dimension_name}.pdf"
        blob = bucket.blob(blob_path)

        logger.info("Step: fetching dimension report from GCS | molecule=%s | dimension=%s", molecule, normalized_dimension_name)

        try:
            full_pdf_bytes = blob.download_as_bytes()
        except NotFound:
            missing_reports.append(blob_path)
            logger.warning("Report missing, skipping: gs://%s/%s", bucket_name, blob_path)
            continue

        try:
            first_page_pdf_bytes = extract_first_page_pdf(full_pdf_bytes)
            pdf_first_page_payloads.append((normalized_dimension_name, first_page_pdf_bytes))
            logger.info("Attached first page only for report: gs://%s/%s", bucket_name, blob_path)
        except Exception as exc:
            missing_reports.append(blob_path)
            logger.warning("Could not extract first page from gs://%s/%s: %s", bucket_name, blob_path, exc)

    return pdf_first_page_payloads, missing_reports