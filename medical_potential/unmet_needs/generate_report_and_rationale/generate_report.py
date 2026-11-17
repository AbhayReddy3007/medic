"""Prompt-based unmet-needs PDF report generator."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from medical_potential.config import (
    BQ_DATASET_ID,
    PROJECT_ID,
    TRIAL_MAPPING_TABLE,
    UNMET_NEED_SCORING_TABLE,
    GEMINI_FLASH_PREVIEW_MODEL
)
from medical_potential.gcp_utils import get_bq_client
from medical_potential.gemini_utils import gemini_api_call
from ..compute_unmet_need_score_from_trial_data import aggregate_single_score

logger = logging.getLogger(__name__)

NAVY = colors.HexColor("#1F3864")
BLUE = colors.HexColor("#2E75B6")
GREY = colors.HexColor("#666666")
DARK_TEXT = colors.HexColor("#1A1A2E")

SECTION_ORDER = [
    "Executive Summary",
    "Unmet Needs Addressed with Clinical Evidence",
]

SECTION_TITLES = {
    "Executive Summary": "Executive Summary",
    "Unmet Needs Addressed with Clinical Evidence": "Unmet Needs Addressed with Clinical Evidence",
}


def escape_html(text: str | None) -> str:
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        fontSize=18,
        leading=22,
        textColor=NAVY,
        fontName="Helvetica-Bold",
        spaceAfter=2,
        alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        fontSize=9.5,
        leading=12,
        textColor=GREY,
        fontName="Helvetica",
        spaceAfter=1,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeader",
        fontSize=11.5,
        leading=14,
        textColor=colors.white,
        fontName="Helvetica-Bold",
        spaceBefore=10,
        spaceAfter=0,
        backColor=NAVY,
        alignment=TA_LEFT,
        leftIndent=0,
        rightIndent=0,
        firstLineIndent=0,
        borderPadding=(6, 8, 6, 8),
    ))
    styles.add(ParagraphStyle(
        name="BodyProse",
        fontSize=9.5,
        leading=13,
        textColor=DARK_TEXT,
        fontName="Helvetica",
        spaceAfter=5,
        alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        name="BulletProse",
        fontSize=9.5,
        leading=13,
        textColor=DARK_TEXT,
        fontName="Helvetica",
        spaceAfter=4,
        leftIndent=12,
        alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        name="SnapshotLabel",
        fontSize=8,
        leading=10,
        textColor=GREY,
        fontName="Helvetica-Bold",
        spaceAfter=0,
    ))
    styles.add(ParagraphStyle(
        name="SnapshotValue",
        fontSize=10,
        leading=11,
        textColor=DARK_TEXT,
        fontName="Helvetica-Bold",
        spaceAfter=0,
    ))
    styles.add(ParagraphStyle(
        name="FooterText",
        fontSize=7,
        leading=9,
        textColor=GREY,
        fontName="Helvetica",
    ))
    return styles


def _safe_json_parse(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    payload = text.strip()
    if "```json" in payload:
        payload = payload.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in payload:
        parts = payload.split("```")
        if len(parts) >= 3:
            payload = parts[1].strip()

    try:
        parsed = json.loads(payload)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        # Try to recover JSON object embedded in extra prose.
        start = payload.find("{")
        end = payload.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = payload[start : end + 1]
            try:
                parsed = json.loads(candidate)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
        return None


def _get_section_value(parsed: dict[str, Any], key: str) -> str:
    """Return section text by exact key first, then case/format-insensitive match."""
    value = parsed.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()

    normalized_target = "".join(ch for ch in key.lower() if ch.isalnum())
    for parsed_key, parsed_value in parsed.items():
        if not isinstance(parsed_key, str) or not isinstance(parsed_value, str):
            continue
        normalized_key = "".join(ch for ch in parsed_key.lower() if ch.isalnum())
        if normalized_key == normalized_target and parsed_value.strip():
            return parsed_value.strip()

    return ""


def _strip_markdown_emphasis(text: str) -> str:
    if not text:
        return ""
    return str(text).replace("**", "").replace("__", "")


def _build_summary_table(styles, score_text: str, total_trials_text: str, total_unmet_needs_text: str) -> Table:
    snap_cells = [[
        [
            Paragraph(score_text, ParagraphStyle("score-value", parent=styles["SnapshotValue"], textColor=NAVY)),
            Paragraph("Score", styles["SnapshotLabel"]),
        ],
        [
            Paragraph(total_trials_text, ParagraphStyle("trial-value", parent=styles["SnapshotValue"], textColor=NAVY)),
            Paragraph("Total Trials", styles["SnapshotLabel"]),
        ],
        [
            Paragraph(total_unmet_needs_text, ParagraphStyle("needs-value", parent=styles["SnapshotValue"], textColor=NAVY)),
            Paragraph("Relevant Unmet Needs", styles["SnapshotLabel"]),
        ],
    ]]

    table = Table(snap_cells, colWidths=[2.17 * inch, 2.17 * inch, 2.17 * inch], rowHeights=[0.4 * inch])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#E0E0E0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _build_section_flowables(section_title: str, section_body: str, styles) -> list:
    flowables = [Paragraph(escape_html(section_title.upper()), styles["SectionHeader"]), Spacer(1, 8)]

    for paragraph in (section_body or "").split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if paragraph.startswith("•") and ":" in paragraph:
            bullet_and_name, description = paragraph.split(":", 1)
            bold_part = escape_html(bullet_and_name + ":")
            regular_part = escape_html(description.strip())
            flowables.append(Paragraph(f"<b>{bold_part}</b> {regular_part}", styles["BulletProse"]))
        else:
            flowables.append(Paragraph(escape_html(paragraph), styles["BodyProse"]))

    flowables.append(Spacer(1, 8))
    return flowables


def _fetch_rows_for_molecule(
    table_name: str,
    molecule_name: str,
    use_timestamp_filter: bool = False,
) -> list[dict[str, Any]]:
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{table_name}"
    sql = f"""
        SELECT *
        FROM `{table_id}`
        WHERE molecule_name = @molecule_name
    """

    if use_timestamp_filter:
        sql += f"""
        AND created_at_utc = (
            SELECT MAX(created_at_utc)
            FROM `{table_id}`
            WHERE molecule_name = @molecule_name
        )
        """

    client = get_bq_client()
    from google.cloud import bigquery as _bq

    job_config = _bq.QueryJobConfig(
        query_parameters=[
            _bq.ScalarQueryParameter("molecule_name", "STRING", molecule_name),
        ]
    )
    query_job = client.query(sql, job_config=job_config)
    return [dict(row) for row in query_job.result()]


def build_unmet_needs_report_payload(
    molecule_name: str, 
    aggregated_score: float | None, 
    clinical_trials_assessed: bool = True
) -> dict[str, Any]:
    """Build report input payload from BigQuery rows + aggregated score.
    
    If aggregated_score is None or clinical_trials_assessed is False,
    calculates the aggregated score from the scoring rows using aggregate_single_score.
    """
    molecule = (molecule_name or "").strip()
    if not molecule:
        raise ValueError("molecule_name is required")

    mapping_rows = _fetch_rows_for_molecule(TRIAL_MAPPING_TABLE, molecule)
    scoring_rows = _fetch_rows_for_molecule(UNMET_NEED_SCORING_TABLE, molecule, use_timestamp_filter=True)

    # Calculate aggregated score if null or if clinical trials weren't assessed
    if aggregated_score is None or not clinical_trials_assessed:
        if scoring_rows:
            aggregated_score = aggregate_single_score(scoring_rows)

    total_trials = len({str(r.get("trial_id", "")).strip() for r in mapping_rows if str(r.get("trial_id", "")).strip()})
    total_relevant_unmet_needs = len(
        {
            (
                str(r.get("unmet_need", "")).strip(),
                str(r.get("category", "")).strip(),
                str(r.get("country", "")).strip(),
            )
            for r in scoring_rows
            if str(r.get("unmet_need", "")).strip()
        }
    )

    return {
        "molecule_name": molecule,
        "summary_table": {
            "score": aggregated_score,
            "total_trials": total_trials,
            "total_relevant_unmet_needs": total_relevant_unmet_needs,
        },
        "trial_to_unmet_need_mapping_rows": mapping_rows,
        "molecule_level_unmet_need_scoring_rows": scoring_rows,
        "aggregated_score": aggregated_score,
    }


def _prepare_prompt_payload(data: dict[str, Any]) -> dict[str, Any]:
    mapping_rows = data.get("trial_to_unmet_need_mapping_rows", []) or []
    scoring_rows = data.get("molecule_level_unmet_need_scoring_rows", []) or []

    trial_evidence_snapshot: list[dict[str, Any]] = []
    for row in mapping_rows[:40]:
        trial_evidence_snapshot.append(
            {
                "trial_id": row.get("trial_id"),
                "unmet_need": row.get("unmet_need"),
                "category": row.get("category"),
                "country": row.get("country"),
                "evidence_strength": row.get("evidence_strength"),
                "relevant_endpoint": row.get("relevant_endpoint"),
                "explanation": row.get("explanation"),
                "phase": row.get("phase"),
                "trial_size": row.get("trial_size"),
            }
        )

    scoring_snapshot: list[dict[str, Any]] = []
    for row in scoring_rows[:40]:
        scoring_snapshot.append(
            {
                "unmet_need": row.get("unmet_need"),
                "category": row.get("category"),
                "country": row.get("country"),
                "score_dimension": row.get("score_dimension"),
                "unmet_need_score": row.get("unmet_need_score"),
                "evidence_impact_score": row.get("evidence_impact_score"),
                "weighted_unmet_need_score": row.get("weighted_unmet_need_score"),
                "confidence": row.get("confidence"),
            }
        )

    return {
        "molecule_name": data.get("molecule_name", "Unknown"),
        "summary_table": data.get("summary_table", {}),
        "aggregated_score": data.get("aggregated_score"),
        "trial_mapping_snapshot": trial_evidence_snapshot,
        "scoring_snapshot": scoring_snapshot,
    }


def _fallback_sections(payload: dict[str, Any]) -> dict[str, str]:
    summary = payload.get("summary_table", {})
    score = summary.get("score")
    trials = summary.get("total_trials", 0)
    needs = summary.get("total_relevant_unmet_needs", 0)

    return {
        "Executive Summary": (
            f"This report summarizes unmet-needs evidence for {payload.get('molecule_name', 'Unknown')}. "
            f"Current aggregate unmet-needs score is {score if score is not None else 'N/A'} based on mapped trial evidence. "
            f"Evidence addresses {needs} specific unmet needs across {trials} clinical trials."
        ),
        "Unmet Needs Addressed with Clinical Evidence": (
            f"The molecule demonstrates meaningful clinical benefit with evidence linking {trials} trials to {needs} distinct unmet needs. "
            f"The following bullet points detail the specific clinical evidence and evidence-based conclusions:\n"
            f"• Evidence relevance: The molecule shows evidence relevant to {needs} distinct unmet needs across {trials} trials, with trial-level mappings linking specific endpoints to unmet needs.\n"
            f"• Score composition: The aggregate score reflects combined unmet-need and evidence-impact signals from mapped outcomes.\n"
            f"• Endpoint mapping: Quantitative outcomes are linked to specific unmet needs through documented trial evidence.\n"
            f"• Data considerations: Results should be interpreted with attention to remaining data limitations where evidence is sparse or incomplete."
        ),
    }


async def _extract_sections_via_prompt(payload: dict[str, Any], model_name: str) -> dict[str, str]:

    prompt = f"""
You are a business-focused medical insights analyst. 
Goal: Create a 2-page report for a given molecule focusing on Unmet needs for the indication T2DM and the unmet needs the molecule addresses, highlighting key findings and insights derived from the provided Json input. 
The report is intended for a Medical Affairs business audience. 


Input data (JSON):
{json.dumps(payload, indent=2, default=str)}

Return strictly the below sections as valid JSON with exactly these keys:
- Executive Summary
    - A high-level narrative that first describes the unmet need landscape, including key gaps or limitations in current treatment options. 
    - It should then briefly explain how the drug addresses some of these unmet needs, supported by directional references to clinical evidence (without going into detailed trial-level descriptions). This section should provide a concise "big-picture" view of problem → solution → evidence signal.
- Unmet Needs Addressed with Clinical Evidence
    - Start with a 2-line summary of the clinical evidence landscape.
    - Then provide a bulleted summary with bullet points (•) for each key insight that:
        1. Identifies the specific unmet medical needs
        2. Explains how the drug addresses each need
        3. Supports these claims with explicit clinical trial evidence, including endpoints, results, and numerical data where available
    - Format: 2-line intro paragraph, then bullet points formatted as \"• Section Name: description\" where Section Name will be bolded in the PDF.

Writing Guidelines:

1.Evidence Grounding
    - Write only from the information provided in the input JSON.
    - Do not use external knowledge or assumptions.
    - Do not invent trial results, endpoints, or numerical values.

2. Use of Data
    - Incorporate quantitative evidence (e.g., percentages, sample sizes) wherever available.
    - If evidence is sparse or missing, explicitly acknowledge the limitation.

3. Clarity and Tone
    - Maintain a professional, factual, and concise tone.
    - The audience is non-technical and not familiar with internal analytical frameworks, scoring methodologies, or internal jargon
    - Do not use markdown formatting such as **bold**, __bold__, *italic*, or headings.
    - Do NOT use internal jargon, scoring framework names, or technical modeling terms.    

4. Language and Style Guidelines
    - Avoid methodological explanations (e.g., how score was derived).   
    - Use clear, simple, business-friendly language.   
    - Translate clinical findings into plain-language impact (e.g., what side effects mean for patients and treatment continuation).

5. Interpretation and Narrative Quality
    •	Headline: Transform raw data into a purposeful narrative by focusing on clinical impact and eliminating filler or technical jargon.
    •	Key Insights:
        o	Value-Driven Content: Focus strictly on the unmet medical needs the drug addresses and how the clinical trials directly support these solutions. Restating obvious facts without offering deep interpretation fails to provide value.
        o	Actionable Implications: Ensure every single statement in the narrative adds a fresh insight, logical conclusion, or forward-looking implication.
        o	Clean Narrative Structure: Avoid referencing any internal field names, indices, database structures, or JSON keys within the final text.

6. Formatting Constraints
    - Executive Summary must be a single paragraph.
    - Unmet Needs Addressed with Clinical Evidence must start with a 2-line summary paragraph, then be followed by bullet points.
    - For bullet points, format as \"• Section Name: description\" where the section name (before colon) will be rendered in bold in the PDF.
    - Separate bullets using newline characters (\\n) within the JSON string value.
    - Do not include markdown, headings, or additional keys beyond the required two.

Output Example (Just look at the format not the explanation):

{{
  "Executive Summary": "The current treatment landscape is characterized by limited efficacy in patients with prior treatment exposure and insufficient durability of response, highlighting a need for more effective and sustained therapeutic options. The available clinical evidence suggests that the drug addresses some of these gaps by improving key efficacy outcomes and delaying disease progression. While the evidence indicates a positive therapeutic signal, variations across studies and limited long-term data suggest that not all unmet needs are fully resolved.",

    "Unmet Needs Addressed with Clinical Evidence": "The drug demonstrates meaningful clinical benefit across multiple trial settings, with evidence consistently supporting its role in addressing key unmet therapeutic needs. The following bullet points detail the specific clinical evidence supporting these conclusions:\n• Efficacy in treatment-resistant patients: Clinical trials demonstrate higher response rates in the treatment arms compared to comparators, directly addressing the unmet need for more effective options in patients with prior treatment exposure.\n• Durability of response: Evidence of delayed disease progression and improved progression-free survival indicates the drug addresses the need for more sustained therapeutic benefit.\n• Clinical impact across populations: Multiple trials across different geographies and patient subgroups show consistent evidence signals, though variations in study design and population heterogeneity suggest benefits may differ by subgroup.\n• Data limitations: Limited long-term follow-up duration and underrepresentation of certain populations highlight gaps where additional evidence would strengthen the clinical case."
}}
    
    """.strip()

    try:
        response_text = await asyncio.to_thread(
            gemini_api_call,
            prompt=prompt,
            model=model_name,
        )
        parsed = _safe_json_parse(response_text.strip())
        if isinstance(parsed, dict):
            sections: dict[str, str] = {}
            for key in SECTION_ORDER:
                value = _get_section_value(parsed, key)
                sections[key] = _strip_markdown_emphasis(value)
            if all(sections.get(k) for k in SECTION_ORDER):
                return sections
            logger.warning("[UNMET_NEEDS][REPORT] Prompt output missing required section keys; using fallback.")
        else:
            logger.warning("[UNMET_NEEDS][REPORT] Prompt output was not parseable JSON; using fallback.")
    except Exception:
        logger.exception("[UNMET_NEEDS][REPORT] Prompt generation failed; using fallback.")

    return _fallback_sections(payload)


async def generate_prompt_unmet_needs_report_bytes(
    data: dict[str, Any],
    model_name: str = GEMINI_FLASH_PREVIEW_MODEL
) -> bytes | tuple[bytes, dict[str, str]]:
    payload = _prepare_prompt_payload(data)
    sections = await _extract_sections_via_prompt(payload, model_name=model_name)

    summary = payload.get("summary_table", {})
    score = summary.get("score")
    total_trials = summary.get("total_trials", 0)
    total_unmet_needs = summary.get("total_relevant_unmet_needs", 0)

    score_text = "N/A" if score is None else f"{score}/5"

    molecule = payload.get("molecule_name", "Unknown")
    dimension = "Unmet Needs Assessment"

    buffer = BytesIO()
    styles = build_styles()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        title=f"{molecule} - {dimension}",
    )

    story = [
        Paragraph(escape_html(dimension), styles["ReportTitle"]),
        Paragraph(
            f"Molecule: <b>{escape_html(molecule)}</b>&nbsp;&nbsp;|&nbsp;&nbsp;{escape_html(date.today().isoformat())}",
            styles["ReportSubtitle"],
        ),
        Spacer(1, 4),
        HRFlowable(width="100%", thickness=1, color=NAVY),
        Spacer(1, 6),
    ]

    story.append(_build_summary_table(styles, score_text=score_text, total_trials_text=str(total_trials), total_unmet_needs_text=str(total_unmet_needs)))
    story.append(Spacer(1, 8))

    for key in SECTION_ORDER:
        story.extend(_build_section_flowables(SECTION_TITLES[key], sections.get(key, ""), styles))

    story.extend([
        Spacer(1, 6),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")),
        Paragraph(
            f"Report generated {escape_html(date.today().isoformat())}  |  Analytical narrative generated by Gemini",
            styles["FooterText"],
        ),
    ])

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    return pdf_bytes, sections
