"""generate_country_lot.py
──────────────────────────
Selective per-country SoC benchmark regeneration.

Behaviour is controlled by two variables in ``medical_potential/config.py``:

    BENCHMARK_RERUN: bool        — when True, the LOT pipeline is skipped entirely;
                                   only the benchmarks for COUNTRIES_TO_RERUN are
                                   rebuilt, then the process exits.  When False, the
                                   normal LOT scoring pipeline runs, using whatever
                                   benchmarks are already cached (no regeneration).

    COUNTRIES_TO_RERUN: list[str] — country names whose cached DOCX + JSON artefacts
                                    should be deleted and rebuilt from source PDFs.
                                    Only consulted when BENCHMARK_RERUN is True.
                                    An empty list is a no-op.

Example config.py settings:

    # Rebuild US and UK benchmarks, skip the LOT pipeline:
    BENCHMARK_RERUN    = True
    COUNTRIES_TO_RERUN = ["US", "UK"]

    # Normal LOT pipeline run (benchmarks used as-is):
    BENCHMARK_RERUN    = False
    COUNTRIES_TO_RERUN = []   # ignored when BENCHMARK_RERUN is False

This module also owns all DOCX / JSON creation helpers so that both
``lot_scoring.py`` and ``line_of_treatment.py`` import from a single source
of truth.

GCS layout (under GCS_BUCKET):
  {LOT_BENCHMARK_PATH}/docs/{country_slug}.docx   ← human-readable benchmark
  {LOT_BENCHMARK_PATH}/json/{country_slug}.json   ← machine-readable benchmark
                                                     (used for LOT classification)
"""

from __future__ import annotations

import io
import json
import logging
import re

import fitz  # PyMuPDF
from docx import Document
from medical_potential.config import (
    GCS_BUCKET,
    GCS_SOC_BASE_PATH,
    GEMINI_FLASH_PREVIEW_MODEL,
    LOT_BENCHMARK_PATH,
    COUNTRIES_TO_RERUN,
    BENCHMARK_RERUN,
)
from medical_potential.gcp_utils import get_gcs_client
from medical_potential.gemini_utils import gemini_api_call

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_JSON_MARKER = "---JSON---"

SYSTEM_INSTRUCTION = """You are a Clinical Pharmacist and Diabetes/Obesity Treatment Pathway Analyst.
Analyze the provided Standard of Care (SoC) document(s) and create a country-level Lines of Therapy (LOT) benchmark.

LANGUAGE HANDLING:
- The source documents may be in English, French, or a mix of languages.
- Regardless of the source language, ALL output must be in English.
- Translate any non-English clinical terms, drug names, and therapy descriptions into their standard English equivalents.

PURPOSE:
Neutral, evidence-based extraction for downstream LOT comparison.

SCOPE:
- Type 2 Diabetes Mellitus (T2DM)
- Obesity / Weight Management

NOTE:
- You may receive text extracted from multiple SoC PDFs for the same country.
- Synthesise all documents into a single unified LOT benchmark for that country.
- If documents conflict, prefer the most recent or most authoritative guidance.

OUTPUT FORMAT STRICT:
Produce TWO sections separated by exactly the marker line "---JSON---".

SECTION 1 — Plain-text benchmark (for human readability / Word document):
Country: [Country]
Country-Level SoC LOT Benchmark

1L:
Recommended therapies/interventions:
Patient segment or trigger:
Treatment goal or rationale:

2L:
Recommended therapies/interventions:
Patient segment or trigger:
Treatment goal or rationale:

3L:
Recommended therapies/interventions:
Patient segment or trigger:
Treatment goal or rationale:

Salvage:
Recommended therapies/interventions:
Patient segment or trigger:
Treatment goal or rationale:

Therapy Classes Explicitly Mentioned:
- List all major classes

SECTION 2 — Structured JSON (for machine consumption / LOT classification):
After the "---JSON---" marker output a single valid JSON object with this exact schema:
{
  "country": "<country name>",
  "lines_of_therapy": {
    "1L": {
      "recommended_therapies": "<string>",
      "patient_segment": "<string>",
      "treatment_goal": "<string>"
    },
    "2L": {
      "recommended_therapies": "<string>",
      "patient_segment": "<string>",
      "treatment_goal": "<string>"
    },
    "3L": {
      "recommended_therapies": "<string>",
      "patient_segment": "<string>",
      "treatment_goal": "<string>"
    },
    "Salvage": {
      "recommended_therapies": "<string>",
      "patient_segment": "<string>",
      "treatment_goal": "<string>"
    }
  },
  "therapy_classes_mentioned": ["<class1>", "<class2>"]
}

FORMATTING RULES (plain-text section only):
- Output plain business English.
- Do NOT use markdown.
- Do NOT use LaTeX.
- Use Unicode symbols directly: ≥ ≤ kg/m2.
- Do NOT use *, **, #, or $.
- Produce plain text.

FORMATTING RULES (JSON section):
- Output valid JSON only — no markdown fences, no extra text before or after the JSON object.
- Use Unicode symbols directly in string values: ≥ ≤ kg/m2.
- Empty or unavailable fields should be empty strings "", not null."""


# ══════════════════════════════════════════════════════════════════════════════
# GCS PATH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _slugify_country(country: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", country.strip().lower())
    return slug.strip("_") or "unknown_country"


def get_soc_benchmark_docx_blob_name(country: str) -> str:
    """GCS blob path for the country's DOCX benchmark."""
    base = LOT_BENCHMARK_PATH.strip("/")
    return f"{base}/docs/{_slugify_country(country)}.docx"


def get_soc_benchmark_json_blob_name(country: str) -> str:
    """GCS blob path for the country's JSON benchmark."""
    base = LOT_BENCHMARK_PATH.strip("/")
    return f"{base}/json/{_slugify_country(country)}.json"


# ══════════════════════════════════════════════════════════════════════════════
# PDF READING
# ══════════════════════════════════════════════════════════════════════════════

def _extract_pdf_text_from_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return "".join(page.get_text("text") for page in doc)
    finally:
        doc.close()


def extract_country_pdfs_text_gcs(blob_names: list[str]) -> str:
    """Download and concatenate text from all PDFs for a single country."""
    gcs_client = get_gcs_client()
    bucket = gcs_client.bucket(GCS_BUCKET)
    sections = []
    for i, blob_name in enumerate(blob_names, 1):
        blob = bucket.blob(blob_name)
        pdf_bytes = blob.download_as_bytes()
        text = _extract_pdf_text_from_bytes(pdf_bytes)
        filename = blob_name.rsplit("/", 1)[-1]
        sections.append(f"--- Document {i}: {filename} ---\n{text}")
    return "\n\n".join(sections)


def discover_country_pdfs(country: str, base_path: str = GCS_SOC_BASE_PATH) -> list[str]:
    """Return sorted GCS blob names for all PDFs belonging to *country*."""
    gcs_client = get_gcs_client()
    bucket = gcs_client.bucket(GCS_BUCKET)
    prefix = f"{base_path.rstrip('/')}/{country}/"
    blobs = [
        b.name
        for b in gcs_client.list_blobs(bucket, prefix=prefix)
        if b.name.lower().endswith(".pdf")
    ]
    if not blobs:
        raise FileNotFoundError(
            f"No PDFs found for country '{country}' under gs://{GCS_BUCKET}/{prefix}"
        )
    return sorted(blobs)


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def run_soc_extraction(pdf_text: str, country: str) -> str:
    """Call Gemini to extract the SoC benchmark from raw PDF text."""
    prompt = (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Country: {country}\n\n"
        f"Below are the full SoC document(s) for {country}:\n\n{pdf_text}"
    )
    return gemini_api_call(
        prompt=prompt,
        model=GEMINI_FLASH_PREVIEW_MODEL,
        temperature=0.1,
    ) or "No response received."


def split_soc_extraction_output(raw_output: str) -> tuple[str, dict | None]:
    """Split Gemini's dual-section output into ``(plain_text, json_dict)``.

    Returns ``(full_output, None)`` when the ``---JSON---`` marker is absent
    or the JSON fails to parse.
    """
    if _JSON_MARKER in raw_output:
        plain_part, json_part = raw_output.split(_JSON_MARKER, 1)
        plain_part = plain_part.strip()
        json_part = json_part.strip()
        json_part = re.sub(r"^```(?:json)?", "", json_part).strip()
        json_part = re.sub(r"```$", "", json_part).strip()
        try:
            return plain_part, json.loads(json_part)
        except json.JSONDecodeError:
            logger.warning(
                "[COUNTRY_LOT] JSON section present but failed to parse for — "
                "falling back to plain text only"
            )
            return plain_part, None
    return raw_output.strip(), None


# ══════════════════════════════════════════════════════════════════════════════
# DOCX CREATION
# ══════════════════════════════════════════════════════════════════════════════

def _clean_text(text: str) -> str:
    replacements = {
        "$\\ge$": "≥", "$\\le$": "≤",
        "\\ge": "≥", "\\le": "≤",
        "**": "", "*": "", "#": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\$+", "", text)
    return text.strip()


def _add_soc_benchmark_section(doc: Document, text: str) -> None:
    for line in text.split("\n"):
        line = _clean_text(line)
        if not line:
            continue
        line_lower = line.lower()
        if line.startswith("Country:"):
            doc.add_heading(line, level=1)
        elif line in ["1L:", "2L:", "3L:", "Salvage:"]:
            doc.add_heading(line.replace(":", ""), level=2)
        elif (
            line_lower.startswith("recommended therapies")
            or line_lower.startswith("patient segment")
            or line_lower.startswith("treatment goal")
        ):
            p = doc.add_paragraph()
            p.add_run(line).bold = True
        elif line.startswith("-"):
            doc.add_paragraph(line[1:].strip(), style="List Bullet")
        else:
            doc.add_paragraph(line)


def build_soc_benchmark_docx_bytes(country: str, plain_text: str) -> bytes:
    """Build a DOCX benchmark document for *country* and return its bytes."""
    doc = Document()
    doc.add_heading(f"{country} — Country-Level SoC LOT Benchmark", level=1)
    _add_soc_benchmark_section(doc, plain_text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# GCS SAVE / DELETE / LOAD
# ══════════════════════════════════════════════════════════════════════════════

def save_soc_benchmark_docx(country: str, docx_bytes: bytes) -> str:
    """Upload DOCX benchmark bytes to GCS. Returns the blob name."""
    blob_name = get_soc_benchmark_docx_blob_name(country)
    gcs_uri = f"gs://{GCS_BUCKET}/{blob_name}"
    try:
        gcs_client = get_gcs_client()
        blob = gcs_client.bucket(GCS_BUCKET).blob(blob_name)
        blob.upload_from_string(
            docx_bytes,
            content_type=(
                "application/vnd.openxmlformats-officedocument"
                ".wordprocessingml.document"
            ),
        )
        logger.info("[COUNTRY_LOT] Saved DOCX for '%s' → %s", country, gcs_uri)
    except Exception:
        logger.exception("[COUNTRY_LOT] Failed to save DOCX for '%s'", country)
    return blob_name


def save_soc_benchmark_json(country: str, benchmark_dict: dict) -> str:
    """Upload structured JSON benchmark to GCS. Returns the blob name."""
    blob_name = get_soc_benchmark_json_blob_name(country)
    gcs_uri = f"gs://{GCS_BUCKET}/{blob_name}"
    try:
        gcs_client = get_gcs_client()
        blob = gcs_client.bucket(GCS_BUCKET).blob(blob_name)
        blob.upload_from_string(
            json.dumps(benchmark_dict, ensure_ascii=False, indent=2),
            content_type="application/json",
        )
        logger.info("[COUNTRY_LOT] Saved JSON for '%s' → %s", country, gcs_uri)
    except Exception:
        logger.exception("[COUNTRY_LOT] Failed to save JSON for '%s'", country)
    return blob_name


def _delete_blob_if_exists(blob_name: str) -> None:
    """Delete a single GCS blob silently if it exists."""
    try:
        gcs_client = get_gcs_client()
        blob = gcs_client.bucket(GCS_BUCKET).blob(blob_name)
        if blob.exists():
            blob.delete()
            logger.info("[COUNTRY_LOT] Deleted gs://%s/%s", GCS_BUCKET, blob_name)
        else:
            logger.info("[COUNTRY_LOT] Nothing to delete at gs://%s/%s", GCS_BUCKET, blob_name)
    except Exception:
        logger.exception("[COUNTRY_LOT] Failed to delete gs://%s/%s", GCS_BUCKET, blob_name)


def delete_country_benchmarks(country: str) -> None:
    """Delete both the DOCX and JSON benchmark files for *country* from GCS."""
    logger.info("[COUNTRY_LOT] Deleting cached benchmarks for '%s'...", country)
    _delete_blob_if_exists(get_soc_benchmark_docx_blob_name(country))
    _delete_blob_if_exists(get_soc_benchmark_json_blob_name(country))


def load_soc_benchmark_json_if_exists(country: str) -> dict | None:
    """Return the parsed JSON benchmark for *country*, or ``None`` if absent."""
    blob_name = get_soc_benchmark_json_blob_name(country)
    try:
        gcs_client = get_gcs_client()
        blob = gcs_client.bucket(GCS_BUCKET).blob(blob_name)
        if not blob.exists():
            return None
        raw = blob.download_as_text(encoding="utf-8")
        data = json.loads(raw)
        logger.info("[COUNTRY_LOT] Loaded existing JSON for '%s'", country)
        return data
    except Exception:
        logger.exception("[COUNTRY_LOT] Failed to load JSON for '%s'", country)
        return None


def extract_text_from_docx_bytes(docx_bytes: bytes) -> str:
    """Return plain text from a DOCX stored as bytes (one paragraph per line)."""
    doc = Document(io.BytesIO(docx_bytes))
    return "\n".join(p.text for p in doc.paragraphs)


def load_soc_benchmark_from_docx(country: str, docx_bytes: bytes | None = None) -> str:
    """Return plain text of the DOCX benchmark for *country*.

    Downloads from GCS when ``docx_bytes`` is not provided.
    """
    if docx_bytes is None:
        blob_name = get_soc_benchmark_docx_blob_name(country)
        gcs_client = get_gcs_client()
        blob = gcs_client.bucket(GCS_BUCKET).blob(blob_name)
        docx_bytes = blob.download_as_bytes()
    return extract_text_from_docx_bytes(docx_bytes)


def soc_benchmark_json_to_text(benchmark_dict: dict) -> str:
    """Convert a structured JSON benchmark dict to overlay-ready plain text."""
    lines: list[str] = []
    country = benchmark_dict.get("country", "")
    if country:
        lines.append(f"Country: {country}")
        lines.append("Country-Level SoC LOT Benchmark")
        lines.append("")
    for key in ("1L", "2L", "3L", "Salvage"):
        lot_data = benchmark_dict.get("lines_of_therapy", {}).get(key, {})
        lines.append(f"{key}:")
        lines.append(
            f"Recommended therapies/interventions: {lot_data.get('recommended_therapies', '')}"
        )
        lines.append(f"Patient segment or trigger: {lot_data.get('patient_segment', '')}")
        lines.append(f"Treatment goal or rationale: {lot_data.get('treatment_goal', '')}")
        lines.append("")
    therapy_classes = benchmark_dict.get("therapy_classes_mentioned", [])
    if therapy_classes:
        lines.append("Therapy Classes Explicitly Mentioned:")
        for cls in therapy_classes:
            lines.append(f"- {cls}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CORE: BUILD A FRESH BENCHMARK FOR ONE COUNTRY
# ══════════════════════════════════════════════════════════════════════════════

def build_country_benchmark(country: str, blob_names: list[str]) -> str:
    """Extract and persist fresh DOCX + JSON benchmarks for *country*.

    Steps:
      1. Read all SoC PDFs from GCS.
      2. Call Gemini to extract the dual-section benchmark.
      3. Split into plain text and structured JSON.
      4. Save DOCX (human-readable) to GCS.
      5. Save JSON (used for LOT classification) to GCS.
      6. Return the JSON-derived overlay text (or DOCX text as fallback).

    Args:
        country:    Country name matching the GCS subfolder.
        blob_names: GCS blob paths for this country's SoC PDFs.

    Returns:
        Plain-text representation of the benchmark suitable for the overlay
        analysis prompt.
    """
    logger.info(
        "[COUNTRY_LOT] Building fresh benchmark for '%s' from %d PDF(s)...",
        country, len(blob_names),
    )
    pdf_text = extract_country_pdfs_text_gcs(blob_names)

    logger.info("[COUNTRY_LOT] Running Gemini extraction for '%s'...", country)
    raw_output = run_soc_extraction(pdf_text, country)

    plain_text, benchmark_dict = split_soc_extraction_output(raw_output)

    # ── Save DOCX ────────────────────────────────────────────────────────────
    logger.info("[COUNTRY_LOT] Saving DOCX for '%s'...", country)
    docx_bytes = build_soc_benchmark_docx_bytes(country, plain_text)
    save_soc_benchmark_docx(country, docx_bytes)

    # ── Save JSON and return JSON-derived text ────────────────────────────────
    if benchmark_dict is not None:
        logger.info("[COUNTRY_LOT] Saving JSON for '%s'...", country)
        save_soc_benchmark_json(country, benchmark_dict)
        return soc_benchmark_json_to_text(benchmark_dict)

    # ── Fallback: JSON parse failed ───────────────────────────────────────────
    logger.warning(
        "[COUNTRY_LOT] JSON extraction failed for '%s' — using DOCX text as fallback",
        country,
    )
    return extract_text_from_docx_bytes(docx_bytes)


# ══════════════════════════════════════════════════════════════════════════════
# GET-OR-BUILD: used by lot_scoring.py for every country on each scoring run
# ══════════════════════════════════════════════════════════════════════════════

def get_or_build_soc_benchmark(country: str, blob_names: list[str]) -> str:
    """Return overlay-ready benchmark text for *country*.

    Uses the cached JSON when it already exists; otherwise calls
    ``build_country_benchmark`` to generate it from source PDFs.

    This is the single function imported by ``lot_scoring.py`` — the cache
    check and fresh-build path live here so there is no duplication.
    """
    existing_json = load_soc_benchmark_json_if_exists(country)
    if existing_json is not None:
        return soc_benchmark_json_to_text(existing_json)
    return build_country_benchmark(country, blob_names)


# ══════════════════════════════════════════════════════════════════════════════
# RERUN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def rerun_countries(countries: list[str]) -> None:
    """Delete and rebuild DOCX + JSON benchmarks for each country in *countries*.

    Args:
        countries: List of country names to reprocess.  Names must match the
                   subfolder names under ``GCS_SOC_BASE_PATH`` in GCS.
    """
    if not countries:
        logger.info("[COUNTRY_LOT] COUNTRIES_TO_RERUN is empty — nothing to do.")
        return

    logger.info(
        "[COUNTRY_LOT] Will rerun %d country/countries: %s",
        len(countries), ", ".join(countries),
    )

    for country in countries:
        logger.info("[COUNTRY_LOT] ── Processing '%s' ──────────────────────", country)
        try:
            # Step 1: delete existing cached artefacts
            delete_country_benchmarks(country)

            # Step 2: discover source PDFs
            blob_names = discover_country_pdfs(country)
            logger.info(
                "[COUNTRY_LOT] Found %d PDF(s) for '%s': %s",
                len(blob_names), country, blob_names,
            )

            # Step 3: extract and persist fresh benchmarks
            build_country_benchmark(country, blob_names)

            logger.info("[COUNTRY_LOT] '%s' completed successfully.", country)

        except FileNotFoundError as exc:
            logger.error("[COUNTRY_LOT] Skipping '%s' — %s", country, exc)
        except Exception:
            logger.exception("[COUNTRY_LOT] Unexpected error processing '%s' — skipping", country)

    logger.info("[COUNTRY_LOT] All done.")