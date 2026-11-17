"""
generate_lot_report.py
───────────────────────
Reads scored data for one molecule from BigQuery `LOT_TABLE` (populated by
``line_of_treatment.py`` / ``lot_scoring.py``) and generates a business-facing
PDF report (target ≤ 3 pages) using Gemini for narrative generation.

Part of the LOT (Line of Treatment) pipeline — imported and called by
``line_of_treatment.py`` as one step of the full assessment (see
``generate_lot_report()`` at the bottom of this file). Not run standalone.

Data sources:
  - BigQuery LOT_TABLE   — per-country lot_score, lot_type, rationale,
                           final_lot_score (confidence intentionally excluded
                           from the query and the report)
  - GCS JSON benchmarks  — per-country structured SoC benchmark written by
                           generate_country_lot.py at
                           {LOT_BENCHMARK_PATH}/json/{country_slug}.json
                           These provide the 1L/2L/3L/Salvage detail used to
                           build the per-country breakdown and to justify
                           the score.

Report structure (≤ 3 pages):
  - Title block  (drug name · date · country count)
  - Score grid: one box per country + a FINAL box, plus the weighting
    formula used to compute it (e.g. "Final Score = US (5) × 0.58 +
    UK (3) × 0.14 + France (4) × 0.14 = 4.32 / 5")
  - Key Line-of-Treatment Findings
      Summary bullets · Geographic Variation · Treatment-Line Positioning
  - Country wise LOT Breakdown
      One block per country (no numeric score — that's in the grid above):
        <Country> — <LOT Type>
          1L: <LLM-summarized SoC text, 2 sentences>
          2L: <LLM-summarized SoC text, 2 sentences>
          3L: <LLM-summarized SoC text, 2 sentences>
          Salvage: <LLM-summarized SoC text, 2 sentences>
          Rationale: <LLM-summarized placement rationale, 2 sentences>
  - Insights and Implications
      Market Access Impact · Geographic Strategy · Competitive Positioning
  - Line of Treatment Profile Summary
      Overall assessment · Score Justification · Cross-country consistency
  - Line of Treatment Scoring Reference  (Score | Classification | Definition)

Confidence and Strategic Recommendation are not surfaced anywhere in this report.

Per-country summarization (1L/2L/3L/Salvage/Rationale, each to 2 sentences)
and the business narrative are both produced by a single Gemini call in
``generate_lot_narrative()`` — see that function for the prompt.
"""

import re
import json
import logging
from io import BytesIO
from datetime import datetime

from google.cloud import bigquery

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)

from medical_potential.config import (
    PROJECT_ID,
    BQ_DATASET_ID,
    LOT_TABLE,
    GCS_BUCKET,
    LOT_BENCHMARK_PATH,
    GEMINI_FLASH_PREVIEW_MODEL,
)
from medical_potential.gcp_utils import get_bq_client, get_gcs_client
from medical_potential.gemini_utils import gemini_api_call

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GEMINI_MODEL = GEMINI_FLASH_PREVIEW_MODEL

# ── Aggregate score weights — MUST match lot_scoring.py's US_WEIGHT / OTHER_COUNTRY_WEIGHT ──
US_WEIGHT = 0.58
OTHER_COUNTRY_WEIGHT = 0.14
US_ALIASES = {"us", "usa", "u.s.", "u.s.a.", "united states", "united_states", "united states of america"}

# ── Colors ────────────────────────────────────────────────────────────────────
DARK_BLUE     = colors.HexColor("#1F3864")
BLUE          = colors.HexColor("#2E75B6")
LIGHT_BLUE_BG = colors.HexColor("#E8EDF3")
WHITE         = colors.white
LIGHT_GRAY    = colors.HexColor("#666666")
DARK_TEXT     = colors.HexColor("#1A1A2E")
DIVIDER_COLOR = colors.HexColor("#D0D7E3")
GREEN         = colors.HexColor("#2E7D32")
RED           = colors.HexColor("#C62828")
AMBER         = colors.HexColor("#B8860B")

LOT_SCORE_LABEL = {
    5: "First-line standard of care",
    4: "Strong first-line alternative / dominant second-line",
    3: "Second-line option",
    2: "Third-line or restricted niche use",
    1: "Salvage / last-resort use",
}

LOT_SCORE_DEFINITION = {
    5: "Drug is recommended as the primary, default treatment option for the majority of eligible patients in this country's guidelines.",
    4: "Drug is a preferred alternative at first line or the dominant choice when escalating to second line; broadly accessible and guideline-endorsed.",
    3: "Drug is used after first-line therapy fails or is insufficient; typically added on or substituted when initial treatment goals are not met.",
    2: "Drug is reserved for specific sub-populations, refractory cases, or situations where earlier lines are contraindicated or exhausted.",
    1: "Drug is used as a last resort when all other treatment options have failed; may include compassionate use or highly restricted access.",
}


# ── Gemini helpers ────────────────────────────────────────────────────────────

def call_gemini(prompt: str) -> str:
    return gemini_api_call(
        prompt=prompt,
        model=GEMINI_MODEL,
        temperature=0.3,
    ).strip()


def _extract_json(text: str):
    text = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ── Country helpers (US weighting, short labels, consistent ordering) ────────

def _is_us(country: str) -> bool:
    normalized = re.sub(r"[_\-]+", " ", (country or "")).strip().lower()
    return normalized in US_ALIASES


_COUNTRY_ABBR = {
    "united states": "US", "usa": "US", "u.s.": "US", "u.s.a.": "US",
    "united states of america": "US",
    "united kingdom": "UK", "u.k.": "UK", "great britain": "UK",
}


def _short_country_label(country: str) -> str:
    key = (country or "").strip().lower()
    if key in _COUNTRY_ABBR:
        return _COUNTRY_ABBR[key]
    return country if len(country) <= 12 else country[:11] + "…"


def _order_countries(countries: list[dict]) -> list[dict]:
    """US first, then the rest alphabetically — used consistently for the
    score grid, the formula text, and the country breakdown so the report
    reads coherently top to bottom.
    """
    return sorted(countries, key=lambda c: (0 if _is_us(c["country"]) else 1, c["country"]))


def _score_color(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return LIGHT_GRAY
    if v >= 4:
        return GREEN
    if v <= 2:
        return RED
    return AMBER


# ── Text normalization (no truncation — never cut off mid-sentence) ──────────

def _clean_display_text(text: str) -> str:
    """Collapse whitespace only. No truncation, no ellipsis, no forced line
    count: the text is handed to a ReportLab Paragraph, which wraps it to
    however many lines it actually needs within the column width. Column
    widths and font sizes elsewhere are tuned so most entries land around
    3 lines, but nothing is ever cut off mid-sentence — a longer entry
    simply takes a few more lines instead of losing content.
    """
    text = " ".join((text or "").split())
    return text or "Not specified"


def _summarize_to_sentences(text: str, max_sentences: int = 3) -> str:
    """Keep at most ``max_sentences`` complete sentences from ``text``.

    Cuts only at sentence-ending punctuation (. ! ?) followed by
    whitespace — never mid-sentence or mid-word. If the text already has
    ``max_sentences`` or fewer, it's returned unchanged. This is what keeps
    the per-country rationale short enough for the report to fit its page
    budget, without losing a sentence partway through.
    """
    text = _clean_display_text(text)
    if text == "Not specified":
        return text
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) <= max_sentences:
        return text
    return " ".join(sentences[:max_sentences])



# ── GCS: load per-country SoC benchmark JSON ──────────────────────────────────

def _slugify_country(country: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", country.strip().lower())
    return slug.strip("_") or "unknown_country"


def load_country_benchmark_json(country: str) -> dict | None:
    """Load the structured SoC benchmark JSON for one country from GCS.

    Returns the parsed dict, or None if the blob does not exist or cannot
    be parsed. This is the same JSON file written by
    ``generate_country_lot.py`` (and consumed by ``lot_scoring.py`` for
    LOT classification) — the report reads it directly rather than
    re-deriving anything from the DOCX.
    """
    base = LOT_BENCHMARK_PATH.strip("/")
    blob_name = f"{base}/json/{_slugify_country(country)}.json"
    try:
        gcs = get_gcs_client()
        bucket = gcs.bucket(GCS_BUCKET)
        blob = bucket.blob(blob_name)
        if not blob.exists():
            logger.warning("[LOT_REPORT] No benchmark JSON found for '%s' at %s", country, blob_name)
            return None
        raw = blob.download_as_text(encoding="utf-8")
        return json.loads(raw)
    except Exception:
        logger.exception("[LOT_REPORT] Failed to load benchmark JSON for '%s'", country)
        return None


def _prompt_snippet(text: str, cap: int = 200) -> str:
    """Collapse whitespace and truncate for inclusion in the Gemini prompt only
    (display wrapping in the PDF is handled separately, at render time).
    """
    text = " ".join((text or "").split())
    if len(text) > cap:
        text = text[: cap - 1].rsplit(" ", 1)[0] + "…"
    return text or "Not specified"


def _lot_line_text(benchmark: dict | None, key: str) -> str:
    """Return the raw (whitespace-collapsed, untruncated) SoC summary text
    for one line of therapy, combining recommended therapies with the
    patient segment when available.
    """
    entry = ((benchmark or {}).get("lines_of_therapy", {}) or {}).get(key) or {}
    therapies = (entry.get("recommended_therapies") or "").strip()
    segment = (entry.get("patient_segment") or "").strip()
    if therapies and segment:
        combined = f"{therapies} — {segment}"
    elif therapies:
        combined = therapies
    elif segment:
        combined = segment
    else:
        combined = ""
    return " ".join(combined.split()) if combined else "Not specified in SoC"


def _lot_line_summaries(benchmark: dict | None) -> dict[str, str]:
    """Return {"1L": "<raw text>", "2L": ..., "3L": ..., "Salvage": ...}
    built from a country's structured SoC benchmark JSON. Text is NOT
    truncated here — it is wrapped to a fixed number of lines at render
    time, against the actual column width/font used in the PDF.
    """
    return {key: _lot_line_text(benchmark, key) for key in ("1L", "2L", "3L", "Salvage")}


# ── BigQuery: load LOT results ─────────────────────────────────────────────────

def load_from_bigquery(molecule_name: str) -> list[dict]:
    """Load the LATEST row per (drug_name, country) from LOT_TABLE."""
    table_ref = f"`{PROJECT_ID}.{BQ_DATASET_ID}.{LOT_TABLE}`"
    query = f"""
        WITH ranked AS (
            SELECT
                drug_name, country, lot_score, lot_type, rationale,
                final_lot_score, created_at,
                ROW_NUMBER() OVER (
                    PARTITION BY drug_name, country ORDER BY created_at DESC
                ) AS _rn
            FROM {table_ref}
            WHERE drug_name = @molecule_name
        )
        SELECT * EXCEPT(_rn) FROM ranked WHERE _rn = 1
        ORDER BY drug_name, country
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("molecule_name", "STRING", molecule_name)]
    )
    logger.info("[LOT_REPORT] Loading latest LOT data from %s...", LOT_TABLE)
    bq_client = get_bq_client()
    rows = [dict(row) for row in bq_client.query(query, job_config=job_config).result()]
    logger.info("[LOT_REPORT] Loaded %d row(s) for '%s'.", len(rows), molecule_name)
    return rows


# ── Build stats dict ───────────────────────────────────────────────────────────

def extract_drug_stats(drug_name: str, rows: list[dict]) -> dict:
    """Build the enriched stats dict, pulling the benchmark JSON for every
    country the drug was scored against.
    """
    def safe(val, fallback="N/A"):
        return str(val).strip() if val is not None and str(val).strip() else fallback

    countries = []
    for row in sorted(rows, key=lambda r: r.get("country") or ""):
        score_int = None
        try:
            score_int = int(float(row["lot_score"]))
        except (ValueError, TypeError, KeyError):
            pass

        country_name = safe(row.get("country"))
        benchmark = load_country_benchmark_json(country_name)

        countries.append({
            "country":      country_name,
            "lot_score":    score_int,
            "lot_type":     safe(row.get("lot_type")),
            "rationale":    safe(row.get("rationale"), ""),
            "lot_lines":    _lot_line_summaries(benchmark),
            "benchmark":    benchmark,   # kept for narrative prompt context
        })

    final_score = None
    try:
        latest = max(rows, key=lambda r: r.get("created_at") or "")
        final_score = round(float(latest["final_lot_score"]), 2)
    except (ValueError, TypeError, KeyError):
        pass

    return {
        "drug_name":       drug_name,
        "countries":       countries,
        "final_lot_score": final_score,
    }


# ── Gemini narrative ───────────────────────────────────────────────────────────

def generate_lot_narrative(stats: dict) -> dict:
    """Generate the full business narrative AND per-country summaries via Gemini.

    A single Gemini call produces:
        key_findings: {
            summary_bullets, geographic_variation_detail, therapy_line_detail
        }
        insights_implications: {
            market_access_impact, geographic_strategy, positioning_implications
        }
        profile_summary: {
            overall_assessment, score_justification, cross_country_consistency
        }
        country_summaries: {
            "<country>": {
                "1L": "<2-sentence summary>",
                "2L": "<2-sentence summary>",
                "3L": "<2-sentence summary>",
                "Salvage": "<2-sentence summary>",
                "rationale": "<2-sentence summary>"
            },
            ...
        }
    The LLM summarises every LOT line and rationale into exactly 2 concise
    sentences, preserving the key clinical and commercial meaning.
    """
    # ── Build full raw data for the prompt ───────────────────────────────────
    country_sections = []
    for c in stats["countries"]:
        lines = c["lot_lines"]
        section = (
            f"Country: {c['country']}\n"
            f"LOT Classification: {c['lot_type'] or 'N/A'} (Score {c['lot_score']}/5)\n"
            f"Rationale: {c['rationale'] or 'Not available'}\n"
            f"1L: {_clean_display_text(lines.get('1L', ''))}\n"
            f"2L: {_clean_display_text(lines.get('2L', ''))}\n"
            f"3L: {_clean_display_text(lines.get('3L', ''))}\n"
            f"Salvage: {_clean_display_text(lines.get('Salvage', ''))}"
        )
        country_sections.append(section)

    # Build the country_summaries schema for the prompt
    country_keys = [c["country"] for c in stats["countries"]]
    country_schema_example = {
        country: {
            "1L": "...", "2L": "...", "3L": "...", "Salvage": "...", "rationale": "..."
        }
        for country in country_keys
    }

    prompt = f"""You are a business-focused market access analyst.

Goal: Produce a concise, business-facing Line of Treatment (LOT) report for
{stats['drug_name']}, for a Market Access / Medical Affairs audience.

Source: Use only the provided data. Do not introduce external assumptions.

=== PROVIDED DATA ===
Drug: {stats['drug_name']}
Aggregate Final LOT Score: {stats['final_lot_score']} / 5

{chr(10).join(f"--- {s}" for s in country_sections)}

=== INSTRUCTIONS ===

PART A — Country-level summarization (country_summaries):
For EACH country listed above, produce a 2-sentence summary of each field.
The 2 sentences must capture the essential clinical and commercial meaning.
Do NOT truncate mid-sentence. Do NOT use bullet points or lists inside the strings.

  - "1L": Summarise the 1st-line SoC description in exactly 2 clear sentences.
  - "2L": Summarise the 2nd-line SoC description in exactly 2 clear sentences.
  - "3L": Summarise the 3rd-line SoC description in exactly 2 clear sentences.
  - "Salvage": Summarise the salvage SoC description in exactly 2 clear sentences.
  - "rationale": Summarise WHY {stats['drug_name']} is placed at that LOT in this country
    in exactly 2 clear sentences, based on the rationale and SoC context provided.

PART B — Business narrative:

1. "key_findings":
   - summary_bullets: 3 to 4 short bullet points. Include the aggregate score
     ("scored X out of 5") in the first bullet; call out US positioning in another.
     Do NOT explain how the score was calculated or weighted.
   - geographic_variation_detail: 2-3 sentences on how treatment-line placement
     varies (or doesn't) across the countries analysed.
   - therapy_line_detail: 2-3 sentences on where the drug typically sits in the
     treatment pathway and what that means for patients reaching it.

2. "insights_implications":
   - market_access_impact: 2-3 sentences on what positioning means for reimbursement.
   - geographic_strategy: 2-3 sentences on how geographic differences shape strategy.
   - positioning_implications: 2-3 sentences on competitive positioning.

3. "profile_summary":
   - overall_assessment: 2-3 sentences on the drug's overall treatment-line profile.
   - score_justification: 3-5 sentences justifying WHY the drug received its aggregate
     LOT score of {stats['final_lot_score']}/5, citing specific countries and their SoC
     placement. Do NOT describe weights or scoring formula.
   - cross_country_consistency: 1-2 sentences on whether positioning is consistent.

Language rules for ALL fields:
- Plain, clear, business-friendly language.
- No internal jargon, no scoring methodology terms, no mention of confidence.
- No markdown. Flowing prose only.

Respond ONLY with a valid JSON object (no markdown fences, no extra text):
{{
  "country_summaries": {json.dumps(country_schema_example, indent=2)},
  "key_findings": {{
    "summary_bullets": ["...", "...", "..."],
    "geographic_variation_detail": "...",
    "therapy_line_detail": "..."
  }},
  "insights_implications": {{
    "market_access_impact": "...",
    "geographic_strategy": "...",
    "positioning_implications": "..."
  }},
  "profile_summary": {{
    "overall_assessment": "...",
    "score_justification": "...",
    "cross_country_consistency": "..."
  }}
}}"""

    text = call_gemini(prompt)
    result = _extract_json(text)
    if result:
        return result

    # Fallback: mechanical 2-sentence trim if Gemini fails
    country_summaries_fallback = {}
    for c in stats["countries"]:
        lines = c["lot_lines"]
        country_summaries_fallback[c["country"]] = {
            k: _summarize_to_sentences(_clean_display_text(lines.get(k, "")), max_sentences=2)
            for k in ("1L", "2L", "3L", "Salvage")
        }
        country_summaries_fallback[c["country"]]["rationale"] = _summarize_to_sentences(
            (c.get("rationale") or "").strip(), max_sentences=2
        )

    return {
        "country_summaries": country_summaries_fallback,
        "key_findings": {
            "summary_bullets": [
                f"{stats['drug_name']} received an aggregate final LOT score of "
                f"{stats['final_lot_score']} out of 5.",
                f"Countries analysed: {', '.join(c['country'] for c in stats['countries']) or 'N/A'}.",
            ],
            "geographic_variation_detail": "See country-by-country breakdown below.",
            "therapy_line_detail": "See country-by-country breakdown below.",
        },
        "insights_implications": {
            "market_access_impact": "See country-by-country breakdown below.",
            "geographic_strategy": "See country-by-country breakdown below.",
            "positioning_implications": "See country-by-country breakdown below.",
        },
        "profile_summary": {
            "overall_assessment": (
                f"{stats['drug_name']} received an aggregate final LOT score of "
                f"{stats['final_lot_score']} out of 5 based on its classification across "
                f"{len(stats['countries'])} countries."
            ),
            "score_justification": "See the country-by-country breakdown for the underlying SoC positioning.",
            "cross_country_consistency": "Consistency is based on available country-level classifications.",
        },
    }


# ── Styles ─────────────────────────────────────────────────────────────────────

def build_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title", parent=base["Normal"],
            fontSize=18, leading=22, textColor=DARK_BLUE,
            alignment=TA_LEFT, fontName="Helvetica-Bold", spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"],
            fontSize=9.5, leading=12, textColor=LIGHT_GRAY,
            alignment=TA_LEFT, fontName="Helvetica", spaceAfter=2,
        ),
        # Full-width colored bar used for every major section heading
        "section_header": ParagraphStyle(
            "SectionHeader", parent=base["Normal"],
            fontSize=11, leading=13, textColor=WHITE,
            fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=5,
            backColor=DARK_BLUE, leftIndent=6, borderPadding=(4, 4, 4, 4),
        ),
        # Smaller bold sub-heading used inside a section (no bar)
        "subheading": ParagraphStyle(
            "Subheading", parent=base["Normal"],
            fontSize=9.5, leading=12, textColor=DARK_BLUE,
            fontName="Helvetica-Bold", spaceBefore=5, spaceAfter=2,
        ),
        "country_heading": ParagraphStyle(
            "CountryHeading", parent=base["Normal"],
            fontSize=9.5, leading=13, textColor=WHITE,
            fontName="Helvetica-Bold", spaceAfter=0,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["Normal"],
            fontSize=8.8, leading=12, textColor=DARK_TEXT,
            fontName="Helvetica", spaceAfter=4, alignment=TA_JUSTIFY,
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["Normal"],
            fontSize=8.8, leading=12, textColor=DARK_TEXT,
            fontName="Helvetica", spaceAfter=2, leftIndent=16, bulletIndent=5,
        ),
        "footer": ParagraphStyle(
            "Footer", parent=base["Normal"],
            fontSize=6.5, leading=9, textColor=colors.HexColor("#999999"),
            fontName="Helvetica", alignment=TA_CENTER, spaceBefore=8,
        ),
        "lot_label": ParagraphStyle(
            "LotLabel", parent=base["Normal"],
            fontSize=8, leading=11, textColor=DARK_BLUE,
            fontName="Helvetica-Bold", alignment=TA_LEFT,
        ),
        "lot_value": ParagraphStyle(
            "LotValue", parent=base["Normal"],
            fontSize=8, leading=11, textColor=DARK_TEXT,
            fontName="Helvetica", alignment=TA_LEFT,
        ),
        "why_label": ParagraphStyle(
            "WhyLabel", parent=base["Normal"],
            fontSize=8, leading=11, textColor=AMBER,
            fontName="Helvetica-Bold", alignment=TA_LEFT,
        ),
        "why_value": ParagraphStyle(
            "WhyValue", parent=base["Normal"],
            fontSize=8, leading=11, textColor=DARK_TEXT,
            fontName="Helvetica-Oblique", alignment=TA_LEFT,
        ),
        "cell": ParagraphStyle(
            "Cell", parent=base["Normal"],
            fontSize=8, leading=11, textColor=DARK_TEXT,
            fontName="Helvetica", alignment=TA_LEFT,
        ),
        "cell_center": ParagraphStyle(
            "CellCenter", parent=base["Normal"],
            fontSize=8, leading=11, textColor=DARK_TEXT,
            fontName="Helvetica", alignment=TA_CENTER,
        ),
        "cell_header": ParagraphStyle(
            "CellHeader", parent=base["Normal"],
            fontSize=8.5, leading=11, textColor=WHITE,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        ),
        "cell_header_left": ParagraphStyle(
            "CellHeaderLeft", parent=base["Normal"],
            fontSize=8.5, leading=11, textColor=WHITE,
            fontName="Helvetica-Bold", alignment=TA_LEFT,
        ),
        "score_big": ParagraphStyle(
            "ScoreBig", parent=base["Normal"],
            fontSize=15, leading=18, fontName="Helvetica-Bold",
            alignment=TA_CENTER,
        ),
        "score_label": ParagraphStyle(
            "ScoreLabel", parent=base["Normal"],
            fontSize=7.5, leading=9, fontName="Helvetica",
            textColor=LIGHT_GRAY, alignment=TA_CENTER,
        ),
        "formula": ParagraphStyle(
            "Formula", parent=base["Normal"],
            fontSize=8, leading=11, textColor=LIGHT_GRAY,
            fontName="Helvetica-Oblique", alignment=TA_CENTER, spaceBefore=4, spaceAfter=2,
        ),
    }


def _render_bullets(items: list, styles: dict, story: list):
    for item in items:
        if item and str(item).strip():
            story.append(Paragraph(f"&#8226; {item}", styles["bullet"]))


# ── Score grid (per-country scores + final, with the weighting formula) ───────

def _build_score_grid(countries: list[dict], final_score, styles: dict) -> Table:
    """One box per country (ordered US-first, then alphabetical) plus a
    distinct FINAL box, each colored green/amber/red by score strength.
    """
    ordered = _order_countries(countries)
    cells = []
    for c in ordered:
        val = c["lot_score"]
        val_display = f"{val}/5" if val is not None else "N/A"
        color = _score_color(val)
        score_style = ParagraphStyle("cs", parent=styles["score_big"], textColor=color)
        cells.append([
            Paragraph(val_display, score_style),
            Paragraph(_short_country_label(c["country"]), styles["score_label"]),
        ])

    final_display = f"{final_score:g}" if final_score is not None else "N/A"
    final_style = ParagraphStyle("fs", parent=styles["score_big"], textColor=DARK_BLUE, fontSize=17)
    final_label_style = ParagraphStyle(
        "fl", parent=styles["score_label"], fontName="Helvetica-Bold", textColor=DARK_BLUE,
    )
    cells.append([
        Paragraph(final_display, final_style),
        Paragraph("FINAL", final_label_style),
    ])

    n = len(cells)
    col_width = min(1.15 * inch, 6.5 * inch / n)
    table = Table([cells], colWidths=[col_width] * n, rowHeights=[0.68 * inch])
    table.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("BOX",           (0, 0), (-2, -1), 0.75, colors.HexColor("#CCCCCC")),
        ("BOX",           (n - 1, 0), (n - 1, 0), 1.5, DARK_BLUE),
        ("INNERGRID",     (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("BACKGROUND",    (0, 0), (n - 2, 0), WHITE),
        ("BACKGROUND",    (n - 1, 0), (n - 1, 0), LIGHT_BLUE_BG),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _build_score_formula_text(countries: list[dict], final_score) -> str:
    """e.g. 'Final Score = US (5) × 0.58 + UK (3) × 0.14 + France (4) × 0.14 = 4.32 / 5'"""
    ordered = _order_countries(countries)
    terms = []
    for c in ordered:
        label = _short_country_label(c["country"])
        score_display = c["lot_score"] if c["lot_score"] is not None else "N/A"
        weight = US_WEIGHT if _is_us(c["country"]) else OTHER_COUNTRY_WEIGHT
        terms.append(f"{label} ({score_display}) &times; {weight:g}")
    formula = " + ".join(terms)
    final_display = f"{final_score:g}" if final_score is not None else "N/A"
    return f"Final Score = {formula} = {final_display} / 5"


# ── Country wise LOT Breakdown (nested per-country blocks) ────────────────────

# Column sizing for the SoC-summary value column. Font size and column width
# are tuned so a typical entry (~150-220 chars) lands around 3 lines, but
# this is a sizing preference only — ReportLab wraps the actual text to
# however many lines it needs, so nothing is ever truncated.
_LOT_VALUE_COL_WIDTH_IN = 6.05
_LOT_VALUE_FONT_SIZE = 7.8


def _country_block(country: dict, summaries: dict, styles: dict) -> Table:
    """Render one country as a self-contained block using LLM-summarized text.

        <Country Name>  ·  <LOT Type / Classification>    [dark header bar]
        1L          <LLM-summarized SoC text>
        2L          <LLM-summarized SoC text>
        3L          <LLM-summarized SoC text>
        Salvage     <LLM-summarized SoC text>
        Rationale   <LLM-summarized rationale>             [amber accent]

    ``summaries`` is the ``country_summaries[country_name]`` dict produced by
    ``generate_lot_narrative`` — all fields are already 2 LLM-written sentences.
    Falls back to the raw field values if the LLM summary is missing.
    """
    lot_value_style = ParagraphStyle(
        "LotValueSized", parent=styles["lot_value"],
        fontSize=_LOT_VALUE_FONT_SIZE, leading=_LOT_VALUE_FONT_SIZE + 2.5,
    )
    why_value_style = ParagraphStyle(
        "WhyValueSized", parent=styles["why_value"],
        fontSize=_LOT_VALUE_FONT_SIZE, leading=_LOT_VALUE_FONT_SIZE + 2.5,
    )

    header_text = f"{country['country']}  ·  {country['lot_type']}"
    rows = [[Paragraph(header_text, styles["country_heading"]), ""]]

    raw_lines = country["lot_lines"]
    for label in ("1L", "2L", "3L", "Salvage"):
        # Use the LLM summary; fall back to the mechanical 2-sentence trim
        text = (summaries.get(label) or "").strip()
        if not text:
            text = _summarize_to_sentences(_clean_display_text(raw_lines.get(label, "")), max_sentences=2)
        rows.append([
            Paragraph(label, styles["lot_label"]),
            Paragraph(text, lot_value_style),
        ])

    # ── Rationale row ─────────────────────────────────────────────────────────
    rationale_text = (summaries.get("rationale") or "").strip()
    if not rationale_text:
        raw = (country.get("rationale") or "").strip()
        rationale_text = _summarize_to_sentences(raw, max_sentences=2) if raw else ""
    has_rationale = bool(rationale_text) and rationale_text.upper() != "N/A"
    if has_rationale:
        rows.append([
            Paragraph("Rationale", styles["why_label"]),
            Paragraph(rationale_text, why_value_style),
        ])

    col_widths = [0.75 * inch, _LOT_VALUE_COL_WIDTH_IN * inch]
    tbl = Table(rows, colWidths=col_widths)
    n_lot_rows = 4  # 1L, 2L, 3L, Salvage
    row_bg = [
        ("BACKGROUND", (0, i), (-1, i), LIGHT_BLUE_BG if i % 2 == 1 else WHITE)
        for i in range(1, n_lot_rows + 1)
    ]
    table_style = [
        ("SPAN",          (0, 0), (-1, 0)),
        ("BACKGROUND",    (0, 0), (-1, 0), DARK_BLUE),
        ("TOPPADDING",    (0, 0), (-1, 0), 4),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ("LEFTPADDING",   (0, 0), (-1, 0), 8),
        ("GRID",          (0, 1), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 1), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
        ("LEFTPADDING",   (0, 1), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 1), (-1, -1), 7),
        *row_bg,
    ]
    if has_rationale:
        why_row_idx = n_lot_rows + 1
        table_style.append(("BACKGROUND", (0, why_row_idx), (-1, why_row_idx), colors.HexColor("#FFF4E0")))
    tbl.setStyle(TableStyle(table_style))
    return tbl


def _country_breakdown_section(countries: list[dict], country_summaries: dict, styles: dict) -> list:
    """Return a list of flowables: one KeepTogether block per country
    (ordered US-first, then alphabetically — matching the score grid).
    ``country_summaries`` is the dict keyed by country name from the narrative.
    """
    flowables = []
    for c in _order_countries(countries):
        summaries = country_summaries.get(c["country"], {})
        flowables.append(KeepTogether([
            _country_block(c, summaries, styles),
            Spacer(1, 5),
        ]))
    return flowables


# ── LOT scoring reference table ────────────────────────────────────────────────

def _scoring_reference_table(styles: dict) -> Table:
    """Table: Score | Classification | Definition."""
    header = [
        Paragraph("Score",          styles["cell_header"]),
        Paragraph("Classification", styles["cell_header_left"]),
        Paragraph("Definition",     styles["cell_header_left"]),
    ]
    rows = [header]
    for score in (5, 4, 3, 2, 1):
        rows.append([
            Paragraph(str(score),                  styles["cell_center"]),
            Paragraph(LOT_SCORE_LABEL[score],      styles["cell"]),
            Paragraph(LOT_SCORE_DEFINITION[score], styles["cell"]),
        ])

    col_widths = [0.55 * inch, 2.05 * inch, 4.4 * inch]
    tbl = Table(rows, colWidths=col_widths)
    row_styles = [
        ("BACKGROUND", (0, i), (-1, i), LIGHT_BLUE_BG if i % 2 == 1 else WHITE)
        for i in range(1, len(rows))
    ]
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), DARK_BLUE),
        ("GRID",          (0, 0), (-1, -1), 0.35, colors.HexColor("#CCCCCC")),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        *row_styles,
    ]))
    return tbl


# ── PDF builder ────────────────────────────────────────────────────────────────

def build_report(stats: dict, narrative: dict) -> bytes:
    """Assemble the ≤3-page PDF and return its bytes."""
    styles = build_styles()
    story  = []
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.55 * inch,
        bottomMargin=0.5 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        title=f"{stats['drug_name']} Line of Treatment Report",
        author="LOT Scorer",
    )

    # ── Title block ────────────────────────────────────────────────────────
    story.append(Paragraph(f"{stats['drug_name']} — Line of Treatment Report", styles["title"]))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%B %d, %Y')}  ·  "
        f"{len(stats['countries'])} countr{'y' if len(stats['countries']) == 1 else 'ies'} analysed",
        styles["subtitle"],
    ))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=DARK_BLUE, spaceAfter=6))

    # ── Score grid + calculation formula ────────────────────────────────────
    if stats["countries"]:
        story.append(_build_score_grid(stats["countries"], stats["final_lot_score"], styles))
        story.append(Paragraph(
            _build_score_formula_text(stats["countries"], stats["final_lot_score"]),
            styles["formula"],
        ))
    story.append(Spacer(1, 6))

    # ── Key Line of Treatment Findings ──────────────────────────────────────
    key_findings = narrative.get("key_findings", {})
    story.append(Paragraph(f"Key Line of Treatment Findings for {stats['drug_name']}", styles["section_header"]))
    if isinstance(key_findings, dict):
        bullets = key_findings.get("summary_bullets", [])
        if isinstance(bullets, list):
            _render_bullets(bullets, styles, story)
        story.append(Spacer(1, 2))

        geo_detail = key_findings.get("geographic_variation_detail", "")
        if geo_detail:
            story.append(Paragraph("Geographic Variation", styles["subheading"]))
            story.append(Paragraph(geo_detail, styles["body"]))

        therapy_detail = key_findings.get("therapy_line_detail", "")
        if therapy_detail:
            story.append(Paragraph("Treatment-Line Positioning", styles["subheading"]))
            story.append(Paragraph(therapy_detail, styles["body"]))
    elif isinstance(key_findings, str):
        story.append(Paragraph(key_findings, styles["body"]))
    story.append(Spacer(1, 2))

    # ── Country-wise LOT Breakdown ──────────────────────────────────────────
    story.append(Paragraph("Country wise LOT Breakdown", styles["section_header"]))
    country_summaries = narrative.get("country_summaries", {})
    if stats["countries"]:
        story.extend(_country_breakdown_section(stats["countries"], country_summaries, styles))
    else:
        story.append(Paragraph("No country-level data available.", styles["body"]))

    # ── Insights and Implications ──────────────────────────────────────────
    insights = narrative.get("insights_implications", {})
    insight_block = [Paragraph("Insights and Implications", styles["section_header"])]
    if isinstance(insights, dict):
        market_access = insights.get("market_access_impact", "")
        if market_access:
            insight_block.append(Paragraph(f"<b>Market Access Impact:</b> {market_access}", styles["body"]))
        geo_strategy = insights.get("geographic_strategy", "")
        if geo_strategy:
            insight_block.append(Paragraph(f"<b>Geographic Strategy:</b> {geo_strategy}", styles["body"]))
        positioning = insights.get("positioning_implications", "")
        if positioning:
            insight_block.append(Paragraph(f"<b>Competitive Positioning:</b> {positioning}", styles["body"]))
    elif isinstance(insights, str):
        insight_block.append(Paragraph(insights, styles["body"]))
    story.append(KeepTogether(insight_block))
    story.append(Spacer(1, 4))

    # ── Line of Treatment Profile Summary (includes Score Justification) ────
    profile = narrative.get("profile_summary", {})
    profile_block = [Paragraph("Line of Treatment Profile Summary", styles["section_header"])]
    if isinstance(profile, dict):
        overall = profile.get("overall_assessment", "")
        if overall:
            profile_block.append(Paragraph(overall, styles["body"]))

        score_just = profile.get("score_justification", "")
        if score_just:
            profile_block.append(Paragraph("Score Justification", styles["subheading"]))
            profile_block.append(Paragraph(score_just, styles["body"]))

        consistency = profile.get("cross_country_consistency", "")
        if consistency:
            profile_block.append(Paragraph(consistency, styles["body"]))
    elif isinstance(profile, str):
        profile_block.append(Paragraph(profile, styles["body"]))
    story.append(KeepTogether(profile_block))
    story.append(Spacer(1, 6))

    # ── LOT Scoring Reference ──────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph("Line of Treatment Scoring Reference", styles["section_header"]),
        _scoring_reference_table(styles),
    ]))

    # ── Footer ─────────────────────────────────────────────────────────────
    story.append(HRFlowable(
        width="100%", thickness=0.4,
        color=colors.HexColor("#CCCCCC"), spaceBefore=10,
    ))
    story.append(Paragraph(
        "Auto-generated from Line of Treatment analysis output. For internal use only.",
        styles["footer"],
    ))

    doc.build(story)
    return buffer.getvalue()


# ── Public entry point ─────────────────────────────────────────────────────────

def generate_lot_report(molecule_name: str) -> bytes:
    """Generate one in-memory PDF report (≤ 3 pages) for one molecule.

    Args:
        molecule_name: Exact molecule name to load from LOT_TABLE.

    Returns:
        Generated PDF bytes.

    Raises:
        ValueError: If molecule_name is empty or no LOT data exists.
        RuntimeError: If the Gemini API key is not configured.
    """
    if not molecule_name or not molecule_name.strip():
        raise ValueError("molecule_name must be provided")
    molecule_name = molecule_name.strip()

    rows = load_from_bigquery(molecule_name)
    if not rows:
        raise ValueError(f"No LOT data found for molecule: {molecule_name}")

    logger.info("[LOT_REPORT] Building stats and loading benchmark JSONs for '%s'...", molecule_name)
    stats = extract_drug_stats(molecule_name, rows)

    logger.info("[LOT_REPORT] Generating narrative for '%s'...", molecule_name)
    narrative = generate_lot_narrative(stats)

    logger.info("[LOT_REPORT] Building PDF for '%s'...", molecule_name)
    report_bytes = build_report(stats, narrative)
    logger.info("[LOT_REPORT] Report generated for '%s'.", molecule_name)
    return report_bytes