"""SoC → Lines-of-Therapy (LOT) scoring — core logic.

Reads country-level Standard-of-Care (SoC) PDFs from GCS (one subfolder per
country under ``GCS_SOC_BASE_PATH``), looks up each drug's Mechanism of
Action (MOA) from the BigQuery MOA lookup table, asks Gemini to (1) extract a
per-country SoC LOT benchmark and (2) classify each supplied drug against
that benchmark, then pushes one row per drug/country combination to
BigQuery:

    drug_name, country, lot_score, lot_type, rationale, confidence, final_lot_score

final_lot_score is a per-drug aggregate across all countries, NOT a per-row
value: (US lot_score x US_WEIGHT) + OTHER_COUNTRY_WEIGHT x sum(lot_score for
every other country). It is repeated on every row for that drug.

SoC benchmark extraction (the PDF-read + Gemini-extraction step) is cached as
both a Word document and a structured JSON file per country under
``gs://GCS_BUCKET/{LOT_BENCHMARK_PATH}/``:

    docs/{country_slug}.docx   — human-readable benchmark
    json/{country_slug}.json   — machine-readable benchmark (used for classification)

All DOCX/JSON creation, caching, and deletion logic lives in
``generate_country_lot.py``.  Use ``COUNTRIES_TO_RERUN`` in that module to
force a fresh re-extraction for specific countries without touching the rest.

The public entry point (``line_of_treatment_assessment()``) lives in
``line_of_treatment.py``, which imports everything it needs from this module.

NEW CONFIG REQUIRED
--------------------
Add the following to ``medical_potential/config.py`` (they do not exist yet):

    GCS_SOC_BASE_PATH           # e.g. "SOC"
    GEMINI_FLASH_PREVIEW_MODEL  # e.g. "gemini-2.5-flash"
    LOT_TABLE                   # e.g. "soc_lot_scores"
    LOT_BENCHMARK_PATH          # e.g. "lot_benchmarks"
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass

import fitz  # PyMuPDF  (still used for PDF text extraction here)
from google.cloud import bigquery

from medical_potential.config import (
    GCS_BUCKET,
    GCS_SOC_BASE_PATH,
    GEMINI_FLASH_PREVIEW_MODEL,
    DRUG_DETAILS_FULL_TABLE_ID,
    PROJECT_ID,
)
from medical_potential.gcp_utils import get_bq_client, get_gcs_client
from medical_potential.gemini_utils import gemini_api_call
from medical_potential.line_of_treatment.generate_country_lot import (
    get_or_build_soc_benchmark,
    # Re-export helpers still referenced by line_of_treatment.py
    extract_country_pdfs_text_gcs,
    build_soc_benchmark_docx_bytes,
    save_soc_benchmark_docx,
    save_soc_benchmark_json,
    load_soc_benchmark_json_if_exists,
    soc_benchmark_json_to_text,
    split_soc_extraction_output,
    get_soc_benchmark_docx_blob_name,
    get_soc_benchmark_json_blob_name,
    extract_text_from_docx_bytes,
    load_soc_benchmark_from_docx,
)

logger = logging.getLogger(__name__)

US_WEIGHT = 0.58
OTHER_COUNTRY_WEIGHT = 0.14

# ==============================
# CONFIG / CONSTANTS
# ==============================
# LOT type -> numeric score
LOT_SCORE_MAP: dict[str, int] = {
    "first-line standard of care": 5,
    "strong first-line alternative / dominant second-line": 4,
    "second-line option": 3,
    "third-line or restricted niche use": 2,
    "salvage / last-resort use": 1,
}

# Fallback keyword matching, checked in order, for LOT strings that don't
# exactly match LOT_SCORE_MAP (e.g. minor wording drift from the LLM).
LOT_KEYWORD_FALLBACK: list[tuple[str, int]] = [
    ("salvage", 1),
    ("last-resort", 1),
    ("third-line", 2),
    ("restricted niche", 2),
    ("strong first-line alternative", 4),
    ("dominant second-line", 4),
    ("second-line", 3),
    ("first-line", 5),
]

US_ALIASES = {"us", "usa", "u.s.", "u.s.a.", "united states", "united_states", "united states of america"}


# ==============================
# CLI ARGUMENTS
# ==============================
def parse_drugs() -> list[str]:
    """Reads drug names from CLI args, falling back to an interactive prompt."""
    if len(sys.argv) > 1:
        drugs = sys.argv[1:]
    else:
        user_input = input("No drugs supplied. Enter drug names separated by spaces: ").strip()
        if not user_input:
            raise SystemExit("No drug names provided. Exiting.")
        drugs = user_input.split()
    logger.info("[SOC_LOT] Drugs to analyse: %s", ", ".join(drugs))
    return drugs


# ==============================
# BIGQUERY: MOA LOOKUP
# ==============================
def lookup_moa(drug_names: list[str]) -> dict[str, list[dict[str, str]]]:
    """Maps each drug name to its unique MOA records via BigQuery."""
    bq_client = get_bq_client()
    table_ref = f"`{DRUG_DETAILS_FULL_TABLE_ID}`"

    placeholders = ", ".join(f"@drug_{i}" for i in range(len(drug_names)))
    query = f"""
        SELECT DISTINCT
            Cleaned_Generic_Name,
            Mechanism_of_Action,
            Mechanism_of_Action_Detailed
        FROM {table_ref}
        WHERE LOWER(Cleaned_Generic_Name) IN ({placeholders})
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(f"drug_{i}", "STRING", name.lower())
            for i, name in enumerate(drug_names)
        ]
    )
    results = bq_client.query(query, job_config=job_config).result()

    moa_map_lower: dict[str, list[dict[str, str]]] = {}
    for row in results:
        moa_record = {
            "moa": row.Mechanism_of_Action or "Unknown",
            "moa_detailed": row.Mechanism_of_Action_Detailed or "",
        }
        drug_records = moa_map_lower.setdefault(row.Cleaned_Generic_Name.lower(), [])
        if moa_record not in drug_records:
            drug_records.append(moa_record)

    drug_moa: dict[str, list[dict[str, str]]] = {}
    for drug in drug_names:
        match = moa_map_lower.get(drug.lower())
        if match:
            drug_moa[drug] = match
        else:
            logger.warning("[SOC_LOT] '%s' not found in BigQuery table. Using 'Unknown' as MOA.", drug)
            drug_moa[drug] = [{"moa": "Unknown", "moa_detailed": ""}]

    return drug_moa


# ==============================
# GCS: DISCOVER + READ SoC PDFs
# ==============================
def discover_countries_gcs(base_path: str = GCS_SOC_BASE_PATH) -> dict[str, list[str]]:
    """Lists SoC PDFs in GCS, grouped by country.

    Expects blobs laid out as: {base_path}/{Country}/{...}.pdf
    Returns {country_name: [blob_name, ...]}.
    """
    gcs_client = get_gcs_client()
    bucket = gcs_client.bucket(GCS_BUCKET)
    prefix = f"{base_path.rstrip('/')}/"

    countries: dict[str, list[str]] = {}
    for blob in gcs_client.list_blobs(bucket, prefix=prefix):
        if not blob.name.lower().endswith(".pdf"):
            continue
        relative = blob.name[len(prefix):]
        parts = relative.split("/")
        if len(parts) < 2:
            # No country subfolder — skip stray files directly under base_path
            continue
        country = parts[0]
        countries.setdefault(country, []).append(blob.name)

    if not countries:
        raise FileNotFoundError(f"No country PDFs found under gs://{GCS_BUCKET}/{prefix}")

    for country in countries:
        countries[country].sort()

    return countries


# PDF text extraction (used locally for discover_countries_gcs fallback reads)
def _extract_pdf_text_from_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return "".join(page.get_text("text") for page in doc)
    finally:
        doc.close()


# ==============================
# OVERLAY PROMPT (LOT classification + confidence)
# ==============================
def build_overlay_prompt(drugs: list[str], drug_moa: dict[str, list[dict[str, str]]]) -> str:
    molecule_lines = []
    for i, drug in enumerate(drugs):
        moa_variants = "; ".join(
            f"MOA: {record['moa']}"
            + (f" (Detailed: {record['moa_detailed']})" if record["moa_detailed"] else "")
            for record in drug_moa[drug]
        )
        molecule_lines.append(f"{i + 1}. {drug} - {moa_variants}")
    molecule_list = "\n".join(molecule_lines)

    output_blocks = "\n\n".join(
        f"{drug}:\nMolecule: {drug}\n"
        f"MOA variants: "
        + "; ".join(
            f"{record['moa']}"
            + (f" (Detailed: {record['moa_detailed']})" if record["moa_detailed"] else "")
            for record in drug_moa[drug]
        )
        + "\n"
        f"Final LoT Category:\nRationale:\nConfidence:"
        for drug in drugs
    )

    return f"""You are a Senior Market Access Analyst.

Task:
Determine the most appropriate Line of Treatment (LOT) classification for each molecule below based ONLY on the supplied MOA and the SoC definitions provided above.

Molecules and their MOA data:
{molecule_list}

MOA DATA HANDLING:
- A molecule may have multiple unique MOA records from the source data.
- Treat all listed MOA and detailed-MOA variants as evidence for that molecule.
- Duplicate rows with identical MOA values have already been removed and must not be counted as additional evidence.
- If the variants conflict, evaluate the complete set and explain the relevant mapping in the rationale.

LOT ASSIGNMENT RULES:
- Identify the pharmacologic class, therapeutic modality, or treatment approach from the MOA.
- Match the MOA to the therapy classes, mechanisms, modalities, or treatment approaches explicitly described in the SoC.
- Determine the earliest treatment line in which the matching therapy class, mechanism, modality, or treatment approach appears.
- If the MOA aligns with multiple treatment lines, assign the earliest applicable line.
- Do NOT use historical treatment sequencing, current prescribing patterns, external guidelines, prior knowledge, or assumptions outside the supplied SoC.
- Evaluate the MOA against all SoC pathways and patient segments described in the benchmark.
- When a therapy class, mechanism, modality, or treatment approach appears in multiple SoC pathways, prioritize the earliest treatment line in which it is explicitly recommended.
- Only assign Second-Line when the MOA aligns primarily with therapies described as add-on, substitute, escalation, replacement, or post-failure options relative to First-Line treatment.
- Only assign Third-Line when the MOA aligns primarily with therapies described as later-line escalation, refractory-disease management, restricted-use therapies, or options used after failure of earlier treatment lines.
- Only assign Salvage when the MOA aligns primarily with rescue therapies, transplantation, last-resort interventions, or therapies explicitly described in the Salvage section of the SoC.
- When uncertainty exists, assign the earliest treatment line supported by the MOA-to-SoC mapping.

CLASSIFICATION OPTIONS (use this exact wording for Final LoT Category):
- First-line standard of care
- Strong first-line alternative / dominant second-line
- Second-line option
- Third-line or restricted niche use
- Salvage / last-resort use

CONFIDENCE:
- For each molecule, also output a Confidence score from 0 to 100 reflecting how directly the MOA maps to explicit language in the SoC.
- Use a high confidence (80-100) only when the SoC explicitly names the therapy class/mechanism at a specific line.
- Use a medium confidence (50-79) when the mapping is inferred from a closely related class or modality.
- Use a low confidence (0-49) when the SoC does not clearly address this MOA and the line was inferred indirectly.
- Output Confidence as a bare integer (e.g. "82"), with no % sign or extra words.

OUTPUT FORMAT (repeat exactly for every molecule, keep field labels verbatim):
{output_blocks}

FORMATTING RULES:
- Output plain business English.
- Do NOT use markdown.
- Do NOT use LaTeX.
- Use Unicode symbols directly.
- Produce plain text."""


# ==============================
# TEXT CLEANUP
# ==============================
def clean_text(text: str) -> str:
    replacements = {
        "$\\ge$": "≥",
        "$\\le$": "≤",
        "\\ge": "≥",
        "\\le": "≤",
        "**": "",
        "*": "",
        "#": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\$+", "", text)
    return text.strip()


# ==============================
# GEMINI CALLS
# ==============================
def run_overlay_analysis(soc_text: str, overlay_prompt: str) -> str:
    prompt = (
        "Below is the extracted SoC benchmark. "
        "Use it as the sole reference for LOT assignment.\n\n"
        f"{soc_text}\n\n{overlay_prompt}"
    )
    return gemini_api_call(
        prompt=prompt,
        model=GEMINI_FLASH_PREVIEW_MODEL,
        temperature=0.0,
    ) or "No response received."


# ==============================
# PARSE OVERLAY OUTPUT -> STRUCTURED FIELDS
# ==============================
def split_drug_blocks(text: str, drugs: list[str]) -> dict[str, str]:
    """Splits the overlay LLM output into one text block per drug heading.

    Tries multiple patterns to locate each drug's output block, because
    Gemini's output format isn't always consistent:
      - "Semaglutide:"  (on its own line)
      - "Semaglutide:\n"  (with trailing content on the same line)
      - case-insensitive match
    When only one drug is requested and no heading is found, the entire
    output is treated as that drug's block (the most common single-drug
    failure mode — the LLM just omits the heading).
    """
    positions: list[tuple[int, str]] = []
    for drug in drugs:
        # Pattern 1: drug name on its own line (strict — original)
        pattern_strict = re.compile(rf"(?im)^{re.escape(drug)}\s*:\s*$")
        m = pattern_strict.search(text)
        if m:
            positions.append((m.start(), drug))
            continue

        # Pattern 2: drug name at start of a line followed by colon (lenient)
        pattern_lenient = re.compile(rf"(?im)^{re.escape(drug)}\s*:")
        m = pattern_lenient.search(text)
        if m:
            positions.append((m.start(), drug))
            continue

    positions.sort(key=lambda p: p[0])
    blocks: dict[str, str] = {}
    for i, (start, drug) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        blocks[drug] = text[start:end]

    # Fallback: if only 1 drug was requested and no block heading found,
    # treat the entire output as that drug's block.
    if len(drugs) == 1 and not blocks:
        logger.info(
            "[SOC_LOT] No heading found for '%s' — treating entire overlay output as its block",
            drugs[0],
        )
        blocks[drugs[0]] = text

    return blocks


def extract_field(block: str, label: str) -> str:
    pattern = re.compile(
        rf"(?im)^{re.escape(label)}\s*:\s*(.*?)(?=\n[A-Za-z][A-Za-z /\\-]*:\s*|\Z)",
        re.DOTALL,
    )
    m = pattern.search(block)
    return clean_text(m.group(1)) if m else ""


def parse_overlay_output(overlay_text: str, drugs: list[str]) -> dict[str, dict]:
    """Returns {drug: {"lot_type": ..., "rationale": ..., "confidence": int|None}}."""
    blocks = split_drug_blocks(overlay_text, drugs)
    parsed: dict[str, dict] = {}
    for drug in drugs:
        block = blocks.get(drug, "")
        if not block:
            logger.warning("[SOC_LOT] Could not locate output block for drug '%s'", drug)
        lot_type = extract_field(block, "Final LoT Category")
        rationale = extract_field(block, "Rationale")
        confidence_raw = extract_field(block, "Confidence")

        confidence: int | None = None
        conf_match = re.search(r"\d{1,3}", confidence_raw)
        if conf_match:
            confidence = max(0, min(100, int(conf_match.group(0))))

        parsed[drug] = {"lot_type": lot_type, "rationale": rationale, "confidence": confidence}
    return parsed


# ==============================
# LOT SCORE MAPPING
# ==============================
def map_lot_type_to_score(lot_type: str) -> int | None:
    if not lot_type:
        return None
    normalized = lot_type.strip().lower()
    if normalized in LOT_SCORE_MAP:
        return LOT_SCORE_MAP[normalized]
    for keyword, score in LOT_KEYWORD_FALLBACK:
        if keyword in normalized:
            return score
    logger.warning("[SOC_LOT] Unrecognized LOT type '%s' — no score assigned", lot_type)
    return None


def is_us(country: str) -> bool:
    normalized = re.sub(r"[_\-]+", " ", country).strip().lower()
    return normalized in US_ALIASES


def compute_final_lot_score(rows: list["LotRow"]) -> float | None:
    """Compute one aggregate LOT score for a single molecule.

    final_lot_score = (US lot_score x US_WEIGHT) + OTHER_COUNTRY_WEIGHT x sum(
    lot_score for every other country). Countries with no resolvable
    lot_score are skipped. Returns ``None`` when no country has a resolvable
    score.

    Raises:
        ValueError: If rows contain more than one molecule.
    """
    drug_names = {row.drug_name for row in rows}
    if len(drug_names) > 1:
        raise ValueError("compute_final_lot_score accepts rows for one molecule only")

    us_scores: list[int] = []
    other_scores: list[int] = []
    for r in rows:
        if r.lot_score is None:
            continue
        (us_scores if is_us(r.country) else other_scores).append(r.lot_score)

    if not us_scores and not other_scores:
        return None

    us_score = sum(us_scores) * US_WEIGHT  # normally 0 or 1 US row
    other_score = sum(other_scores) * OTHER_COUNTRY_WEIGHT
    return round(us_score + other_score, 4)


# ==============================
# PER-COUNTRY PROCESSING
# ==============================
@dataclass
class LotRow:
    drug_name: str
    country: str
    lot_score: int | None
    lot_type: str
    rationale: str
    confidence: int | None
    final_lot_score: float | None


def process_country(country: str, blob_names: list[str], drugs: list[str], overlay_prompt: str) -> list[LotRow]:
    n_pdfs = len(blob_names)
    logger.info("[SOC_LOT] Processing: %s (%d PDF%s)", country, n_pdfs, "s" if n_pdfs != 1 else "")

    soc_output = get_or_build_soc_benchmark(country, blob_names)

    logger.info("[SOC_LOT] Running overlay analysis for '%s'...", country)
    overlay_output = run_overlay_analysis(soc_output, overlay_prompt)

    logger.info("[SOC_LOT] Parsing structured LOT results for '%s'...", country)
    parsed = parse_overlay_output(overlay_output, drugs)

    rows: list[LotRow] = []
    for drug in drugs:
        info = parsed.get(drug, {"lot_type": "", "rationale": "", "confidence": None})
        lot_score = map_lot_type_to_score(info["lot_type"])
        rows.append(
            LotRow(
                drug_name=drug,
                country=country,
                lot_score=lot_score,
                lot_type=info["lot_type"],
                rationale=info["rationale"],
                confidence=info["confidence"],
                final_lot_score=None,  # filled in once, after all countries are processed
            )
        )
    return rows