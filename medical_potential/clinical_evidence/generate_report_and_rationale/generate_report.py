#!/usr/bin/env python3
"""
Drug Evidence Report Generator (Business-Facing)
==================================================
Generates a concise, insight-driven PDF report (max 2 pages) for senior
business stakeholders from a clinical evidence JSON payload.

The report focuses on critical insights, business implications, and
decision-relevant takeaways — not scoring methodology or technical detail.

Gemini 2.5 Flash generates the analytical narrative. Falls back to a
structured summary if the API is unavailable.
"""

import json
import os
import re
import sys
from datetime import datetime
from io import BytesIO

from dotenv import load_dotenv

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable
)

load_dotenv()

GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def safe_text(val) -> str:
    if val is None:
        return "N/A"
    s = str(val).strip()
    return s if s else "N/A"


def escape_html(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def lookup_innovator(molecule_name: str, sponsor_hint: str = "") -> str | None:
    import urllib.request
    import urllib.error

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    prompt = (
        f"Identify the original innovator company that first developed or "
        f"discovered the pharmaceutical molecule \"{molecule_name}\". "
        f"The clinical-trial sponsor on record is \"{sponsor_hint or 'Unknown'}\". "
        f"Return ONLY the company name on a single line — no explanation, "
        f"no punctuation beyond what is in the name itself. "
        f"If you are unsure, return the sponsor name as-is."
    )

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 64},
    }).encode("utf-8")

    url = f"{GEMINI_API_URL}?key={api_key}"
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        print(f"  [AI]  Looking up innovator for {molecule_name}...")
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        text = (
            result.get("candidates", [{}])[0]
                  .get("content", {})
                  .get("parts", [{}])[0]
                  .get("text", "")
        ).strip()
        if text:
            print(f"  [AI]  Innovator resolved: {text}")
            return text
        return None
    except Exception as e:
        print(f"  [WARN] Innovator lookup failed ({e}), using sponsor as fallback.",
              file=sys.stderr)
        return None


# ---------------------------------------------------------------------
# Gemini narrative — business-facing analytical report
# ---------------------------------------------------------------------

def build_gemini_prompt(molecule_name: str, json_data: dict) -> str:
    """
    Build a prompt that instructs Gemini to produce the business-facing
    analytical report following the strict formatting guidelines.
    """
    sc = json_data.get("scoring", {})
    reasoning = sc.get("reasoning", {})
    assess = json_data.get("evidence_assessment", {})
    mol = json_data.get("molecule", {})
    consol = json_data.get("evidence_consolidation", {})
    gaps = json_data.get("evidence_gaps", [])
    rwe = json_data.get("rwe_synthesis", "")

    # Collect trial summary info
    trials = json_data.get("clinical_trials", [])
    trial_summary_lines = []
    for t in trials[:20]:
        trial_summary_lines.append(
            f"  {t.get('trial_id','?')} | Phase {t.get('phase','?')} | "
            f"{t.get('study_design','?')} | n={t.get('trial_size','?')} | "
            f"Location: {t.get('trial_location','?')} | "
            f"Status: {t.get('status','?')} | Results: {'Yes' if t.get('results_available') else 'No'}"
        )

    prompt = f"""You are a senior business analyst preparing a detailed analytical report 
on the clinical evidence dimension of the pharmaceutical molecule "{molecule_name}" 
for senior business decision-makers.

YOUR OUTPUT MUST FOLLOW THIS EXACT STRUCTURE (use these exact headings):

## HEADLINE
Write ONE impactful sentence summarizing the overall business implication of the 
clinical evidence for this molecule. This should be the single most important 
takeaway a decision-maker needs.

## EVIDENCE LANDSCAPE
Write 3-5 sentences providing the quantitative context a decision-maker needs. 
Cover: how many trials exist and at what stages, how many patients have been studied, 
what types of study designs dominate the evidence base (e.g., randomized controlled 
trials vs. observational), where trials were conducted geographically and what that 
means for regulatory pathways, and whether FDA or other regulatory approvals exist. 
Cite specific numbers from the data. This section sets the stage — it should tell 
the reader the size and shape of the evidence base before diving into insights.

## KEY INSIGHTS

Provide 4-6 key insights. For EACH insight, write EXACTLY two lines using this format:
Line 1: "Insight: " followed by a short, specific finding (ONE sentence, max 20 words).
         This is the bold headline of the insight.
Line 2: "Why it matters: " followed by the business implication (2-3 sentences, 
         ~40-60 words). This is the explanatory body text.

IMPORTANT: You MUST use exactly the labels "Insight: " and "Why it matters: " — 
these labels are required for formatting.

Example format:
Insight: 95 RCTs across 827 studies create a formidable competitive barrier with gold-standard evidence.
Why it matters: New entrants would need years and billions to replicate this evidence depth, reinforcing market leadership and prescriber trust. The quality and volume of trials provides unrivaled leverage in payer negotiations and formulary positioning.

Insight: FDA approval plus Tier 1 geographic coverage accelerates multi-market entry strategies.
Why it matters: Trials in US, EU, and key global markets streamline parallel regulatory filings and de-risk international commercial strategy. This breadth reduces time-to-market in secondary geographies and lowers regulatory risk for licensing partners.

Be specific. Reference trial counts, patient numbers, geographic reach, study phases, 
or real-world evidence outcomes. Do NOT write generic statements like "the evidence 
is strong" without concrete data.

Prioritize insights that address:
1. Efficacy evidence strength and what it enables (label expansion, payer leverage)
2. Competitive differentiation based on evidence depth or breadth
3. Regulatory positioning (approval status, geographic coverage for multi-market strategy)
4. Real-world evidence alignment or divergence from trial results
5. Evidence gaps that create risk or delay
6. Consistency/replication signals across independent studies

## EVIDENCE GAPS & RISKS
Write 3-5 bullet points (each starting with "- ") identifying the most material 
evidence gaps and the business risk each creates. Focus ONLY on clinical and commercial 
gaps — for example, missing head-to-head comparisons against specific competitors, 
limited long-term safety data, narrow geographic trial coverage, absence of real-world 
evidence in key populations, or missing pediatric/elderly subgroup data.
CRITICAL: Do NOT mention peer-reviewed journals, publications, published literature, 
academic publishing, or the need for more published studies. These are NOT relevant 
evidence gaps for this report. Every gap must be about missing DATA, not missing 
PUBLICATIONS.

## SCORE REFERENCE
Write 2-3 sentences stating the overall score and briefly contextualizing it 
(e.g., "This dimension is rated strong with a composite score of 4.5/5, reflecting 
mature Phase 3 evidence, gold-standard trial designs, and consistent findings across 
global geographies"). Do NOT explain the scoring formula. Do NOT break down sub-metrics 
individually.

## SCORING METHODOLOGY
Write 3-5 sentences. Explain that the composite score integrates five evaluation 
dimensions: Phase maturity (how far the drug has advanced in clinical development), 
Study Design quality (whether evidence comes from randomized controlled trials or 
weaker designs), Geographic breadth (whether trials were conducted in Tier 1 regulatory 
markets like the US and EU), Evidence support (depth of post-launch real-world data), 
and Consistency (whether findings replicate across independent studies). The final score 
averages development maturity with the dominant quality signal and adjusts for 
cross-study consistency. Scores range from 1 to 5, where 5 represents the strongest 
possible evidence base.

## BOTTOM LINE
Write 3-4 sentences stating what a decision-maker should infer from this dimension. 
Be direct and actionable — state whether the evidence base supports investment, 
partnership, or market entry decisions, and flag any conditions or watchpoints.

STRICT RULES:
- Total length: 800-1100 words (the report should comfortably fill ~2 pages)
- NO technical jargon (no "PDGEC", "mode(D,G,E)", "LLM-assessed", "rule-based", 
  "BigQuery", "ClinicalTrials.gov API")
- Do NOT mention peer-reviewed journals, publications, or academic publishing anywhere
- You MUST use "Insight: " and "Why it matters: " labels exactly in KEY INSIGHTS
- Every statement must add insight or implication — no restating obvious facts
- Use clear, natural business language that a non-scientific executive can follow
- Do not use markdown bold (**text**) — use plain text only
- Reference specific numbers, trial counts, and outcomes wherever possible
- Keep paragraphs short (2-4 sentences max)

DATA FOR YOUR ANALYSIS:
======================

Molecule: {molecule_name}
Brand Names: {', '.join(mol.get('brand_names', [])) or 'N/A'}
Innovator: {mol.get('sponsor', 'Unknown')}
Indications: {', '.join(mol.get('indications', [])[:10]) or 'N/A'}

Scores: P(Phase)={sc.get('P')} D(Design)={sc.get('D')} G(Geography)={sc.get('G')} E(Evidence)={sc.get('E')} C(Consistency)={sc.get('C')} Final={sc.get('final')}/5
Evidence Level: {assess.get('evidence_level', 'N/A')} — {assess.get('evidence_detail', 'N/A')}

Phase reasoning: {reasoning.get('P', 'N/A')}
Design reasoning: {reasoning.get('D', 'N/A')}
Geography reasoning: {reasoning.get('G', 'N/A')}
Evidence reasoning: {reasoning.get('E', 'N/A')}
Consistency reasoning: {reasoning.get('C', 'N/A')}

Trial Stats:
  Total trials: {assess.get('total_trials', 0)}
  Trials with results: {assess.get('trials_with_results', 0)}
  Total patients enrolled: {assess.get('total_patients', 0):,}
  Largest trial: {assess.get('largest_trial', 0):,}
  RCTs: {assess.get('rct_count', 0)}
  FDA approved: {'Yes' if assess.get('has_regulatory_approval') else 'No'}
  Phase distribution: {json.dumps(assess.get('phase_distribution', dict()))}
  Design distribution: {json.dumps(assess.get('design_distribution', dict()))}

Consistency: {consol.get('findings_consistency', 'N/A')}
Replication: {consol.get('replication', 'N/A')}

Evidence Gaps: {'; '.join(gaps) if gaps else 'None identified'}

Sample Trials (first 20):
{chr(10).join(trial_summary_lines) if trial_summary_lines else 'No trial details available'}

RWE Synthesis Summary:
{rwe[:3000] if rwe else 'Not available'}

Now write the report. Remember: business language, specific numbers, no jargon, 800-1100 words. 
CRITICAL: In KEY INSIGHTS, every insight headline MUST start with "Insight: " and every body line MUST start with "Why it matters: "."""

    return prompt


def generate_gemini_narrative(molecule_name: str, json_data: dict) -> str | None:
    import urllib.request
    import urllib.error

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("  [WARN] GEMINI_API_KEY not set — skipping AI narrative.")
        return None

    prompt = build_gemini_prompt(molecule_name, json_data)

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.25,
            "maxOutputTokens": 16384,
        },
    }).encode("utf-8")

    url = f"{GEMINI_API_URL}?key={api_key}"
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        print(f"  [AI]  Requesting business narrative for {molecule_name}...")
        with urllib.request.urlopen(req, timeout=90) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        text = (
            result.get("candidates", [{}])[0]
                  .get("content", {})
                  .get("parts", [{}])[0]
                  .get("text", "")
        )
        if text.strip():
            print(f"  [AI]  Narrative received ({len(text)} chars).")
            return text.strip()
        else:
            print("  [WARN] Gemini returned empty response.", file=sys.stderr)
            return None
    except Exception as e:
        print(f"  [ERR] Gemini API call failed: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------
# PDF building — compact, insight-driven, max ~2 pages
# ---------------------------------------------------------------------

NAVY = colors.HexColor("#1F3864")
BLUE = colors.HexColor("#2E75B6")
LIGHT_BLUE = colors.HexColor("#D9E1F2")
GREEN = colors.HexColor("#2E7D32")
RED = colors.HexColor("#C62828")
AMBER = colors.HexColor("#B8860B")
GREY = colors.HexColor("#666666")
DARK_TEXT = colors.HexColor("#1A1A2E")


def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle", fontSize=18, leading=22, textColor=NAVY,
        fontName="Helvetica-Bold", spaceAfter=2, alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle", fontSize=9.5, leading=12, textColor=GREY,
        fontName="Helvetica", spaceAfter=1,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeader", fontSize=11.5, leading=14, textColor=colors.white,
        fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=0,
        backColor=NAVY, leftIndent=6, borderPadding=(5, 5, 5, 5),
    ))
    styles.add(ParagraphStyle(
        name="Headline", fontSize=11, leading=14, textColor=NAVY,
        fontName="Helvetica-Bold", spaceAfter=6, spaceBefore=6,
        alignment=TA_LEFT,
    ))
    # ── KEY INSIGHTS: headline line ──────────────────────────────────
    # Bold blue, larger font — mirrors the screenshot exactly.
    # "Insight: " label rendered inline as bold blue, rest of text same style.
    styles.add(ParagraphStyle(
        name="InsightHeadline",
        fontSize=10.5,
        leading=14,
        textColor=BLUE,
        fontName="Helvetica-Bold",
        spaceBefore=10,
        spaceAfter=2,
        leftIndent=0,
        alignment=TA_LEFT,
    ))
    # ── KEY INSIGHTS: body line ──────────────────────────────────────
    # Regular (non-italic), dark text, justified.
    styles.add(ParagraphStyle(
        name="InsightBody",
        fontSize=9,
        leading=13,
        textColor=DARK_TEXT,
        fontName="Helvetica",
        spaceBefore=0,
        spaceAfter=6,
        leftIndent=0,
        alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        name="BulletText", fontSize=9, leading=13, textColor=DARK_TEXT,
        fontName="Helvetica", spaceAfter=1, spaceBefore=6,
        leftIndent=12, bulletIndent=0, alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        name="BodyProse", fontSize=9, leading=13, textColor=DARK_TEXT,
        fontName="Helvetica", spaceAfter=3, alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        name="BottomLine", fontSize=9, leading=13, textColor=DARK_TEXT,
        fontName="Helvetica", spaceAfter=3, spaceBefore=2,
    ))
    styles.add(ParagraphStyle(
        name="SnapshotLabel", fontSize=8, leading=10, textColor=GREY,
        fontName="Helvetica-Bold", spaceAfter=0,
    ))
    styles.add(ParagraphStyle(
        name="SnapshotValue", fontSize=9, leading=11, textColor=DARK_TEXT,
        fontName="Helvetica-Bold", spaceAfter=0,
    ))
    styles.add(ParagraphStyle(
        name="FooterText", fontSize=7, leading=9, textColor=GREY,
        fontName="Helvetica",
    ))
    styles.add(ParagraphStyle(
        name="ScoreBig", fontSize=20, leading=22, fontName="Helvetica-Bold",
        textColor=NAVY, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name="ScoreLabel", fontSize=7.5, leading=9, fontName="Helvetica",
        textColor=GREY, alignment=TA_CENTER,
    ))
    return styles


def score_color(value, is_consistency=False):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return colors.black
    if is_consistency:
        return GREEN if v >= 0 else RED
    return GREEN if v >= 4 else (RED if v <= 2 else AMBER)


def build_score_grid(scores: dict, styles):
    """Compact 6-box score grid."""
    labels = ["P\nPhase", "D\nDesign", "G\nGeo", "E\nEvidence", "C\nConsist.", "FINAL"]
    keys = ["P", "D", "G", "E", "C", "Final Score"]

    cells = []
    for label, key in zip(labels, keys):
        val = scores.get(key, "N/A")
        is_c = (key == "C")
        is_final = (key == "Final Score")
        try:
            val_display = f"{float(val):g}"
        except (TypeError, ValueError):
            val_display = safe_text(val)
        color = NAVY if is_final else score_color(val, is_consistency=is_c)
        score_style = ParagraphStyle(
            "cellscore", parent=styles["ScoreBig"], textColor=color,
            fontSize=16 if is_final else 14,
        )
        label_style = ParagraphStyle(
            "celllabel", parent=styles["ScoreLabel"],
            fontName="Helvetica-Bold" if is_final else "Helvetica",
        )
        cell_content = [
            Paragraph(val_display, score_style),
            Paragraph(label.replace("\n", "<br/>"), label_style),
        ]
        cells.append(cell_content)

    table = Table([cells], colWidths=[1.1 * inch] * 6, rowHeights=[0.65 * inch])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOX", (0, 0), (-2, -1), 0.75, colors.HexColor("#CCCCCC")),
        ("BOX", (5, 0), (5, 0), 1.25, NAVY),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("BACKGROUND", (0, 0), (4, 0), colors.white),
        ("BACKGROUND", (5, 0), (5, 0), LIGHT_BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def parse_narrative_to_flowables(narrative: str, styles) -> list:
    """
    Parse the Gemini narrative (markdown-ish) into ReportLab flowables.

    KEY INSIGHTS section:
      Lines starting with "Insight: "      → InsightHeadline style (bold blue)
      Lines starting with "Why it matters:"→ InsightBody style (italic dark)

    All other sections map headings → SectionHeader, body → BodyProse.
    """
    flow = []
    current_section = None

    for line in narrative.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # Strip markdown bold markers
        stripped = stripped.replace("**", "")

        # ── Section heading detection ────────────────────────────────
        if stripped.startswith("## HEADLINE"):
            current_section = "headline"
            continue
        elif stripped.startswith("## EVIDENCE LANDSCAPE"):
            current_section = "landscape"
            flow.append(Spacer(1, 4))
            flow.append(Paragraph("EVIDENCE LANDSCAPE", styles["SectionHeader"]))
            flow.append(Spacer(1, 4))
            continue
        elif stripped.startswith("## KEY INSIGHTS"):
            current_section = "insights"
            flow.append(Spacer(1, 4))
            flow.append(Paragraph("KEY INSIGHTS", styles["SectionHeader"]))
            flow.append(Spacer(1, 2))
            continue
        elif stripped.startswith("## EVIDENCE GAPS"):
            current_section = "gaps"
            flow.append(Spacer(1, 4))
            flow.append(Paragraph("EVIDENCE GAPS &amp; RISKS", styles["SectionHeader"]))
            flow.append(Spacer(1, 4))
            continue
        elif stripped.startswith("## SCORE REFERENCE"):
            current_section = "score_ref"
            flow.append(Spacer(1, 4))
            flow.append(Paragraph("SCORE REFERENCE", styles["SectionHeader"]))
            flow.append(Spacer(1, 3))
            continue
        elif stripped.startswith("## SCORING METHODOLOGY"):
            current_section = "methodology"
            flow.append(Spacer(1, 4))
            flow.append(Paragraph("SCORING METHODOLOGY", styles["SectionHeader"]))
            flow.append(Spacer(1, 3))
            continue
        elif stripped.startswith("## BOTTOM LINE"):
            current_section = "bottom"
            flow.append(Spacer(1, 4))
            flow.append(Paragraph("BOTTOM LINE", styles["SectionHeader"]))
            flow.append(Spacer(1, 3))
            continue
        elif stripped.startswith("## "):
            heading = stripped[3:].strip()
            current_section = "other"
            flow.append(Spacer(1, 4))
            flow.append(Paragraph(escape_html(heading), styles["SectionHeader"]))
            flow.append(Spacer(1, 3))
            continue

        # ── Content rendering by section ────────────────────────────
        if current_section == "headline":
            flow.append(Paragraph(escape_html(stripped), styles["Headline"]))

        elif current_section == "landscape":
            flow.append(Paragraph(escape_html(stripped), styles["BodyProse"]))

        elif current_section == "insights":
            # Strip leading list markers if Gemini adds them
            if stripped.startswith("- ") or stripped.startswith("* "):
                stripped = stripped[2:].strip()

            # ── "Insight: <headline text>" ───────────────────────────
            insight_match = re.match(r'^[Ii]nsight\s*\d*\s*[:.\-]\s*(.*)', stripped)
            if insight_match:
                headline_text = insight_match.group(1).strip()
                # Render headline text only — no "Insight: " label
                html = f'<font color="#2E75B6"><b>{escape_html(headline_text)}</b></font>'
                flow.append(Paragraph(html, styles["InsightHeadline"]))
                continue

            # ── "Why it matters: <body text>" ───────────────────────
            why_match = re.match(
                r'^[Ww]hy\s+[Ii]t\s+[Mm]atters\s*[:.\-]\s*(.*)', stripped
            )
            if why_match:
                body_text = why_match.group(1).strip()
                # Render body text only — no label, no italic
                flow.append(Paragraph(escape_html(body_text), styles["InsightBody"]))
                continue

            # Fallback: any other line in insights section → plain bullet
            if stripped:
                flow.append(Paragraph(
                    f"&bull;&nbsp;&nbsp;{escape_html(stripped)}",
                    styles["BulletText"]
                ))

        elif current_section == "gaps":
            if stripped.startswith("- ") or stripped.startswith("* "):
                text = stripped[2:].strip()
            else:
                text = stripped
            # Skip lines about peer-reviewed journals / publications
            lower = text.lower()
            if any(kw in lower for kw in ["peer-review", "peer review", "journal",
                                           "publication", "published", "publishing"]):
                continue
            if text:
                flow.append(Paragraph(
                    f"&bull;&nbsp;&nbsp;{escape_html(text)}",
                    styles["BulletText"]
                ))

        elif current_section in ("score_ref", "methodology"):
            flow.append(Paragraph(escape_html(stripped), styles["BodyProse"]))

        elif current_section == "bottom":
            flow.append(Paragraph(escape_html(stripped), styles["BottomLine"]))

        else:
            flow.append(Paragraph(escape_html(stripped), styles["BodyProse"]))

    return flow


def generate_pdf_report(json_data: dict, molecule_name: str | None = None) -> bytes:
    """Generate a compact, business-facing PDF and return the PDF bytes."""
    if not isinstance(json_data, dict):
        raise TypeError("json_data must be a dict")

    resolved_name = (
        molecule_name
        or json_data.get("molecule", {}).get("generic_name")
        or "unknown_molecule"
    )

    # Keep external inputs minimal: narrative generation and innovator lookup
    # are derived internally from json_data.
    gemini_narrative = generate_gemini_narrative(resolved_name, json_data)
    sponsor_hint = json_data.get("molecule", {}).get("sponsor", "")
    innovator_name = lookup_innovator(resolved_name, sponsor_hint) or sponsor_hint or None

    buffer = BytesIO()

    styles = build_styles()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        title=f"Clinical Evidence Report - {resolved_name}",
    )
    story = []

    mol = json_data.get("molecule", {})
    sc = json_data.get("scoring", {})
    assess = json_data.get("evidence_assessment", {})

    generic_name = mol.get("generic_name", resolved_name)
    sponsor = innovator_name or mol.get("sponsor", "Unknown")

    # ---------- Title ----------
    story.append(Paragraph("Clinical Evidence Report", styles["ReportTitle"]))
    story.append(Paragraph(
        f"Molecule: <b>{escape_html(generic_name)}</b>"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;Innovator: {escape_html(sponsor)}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;{datetime.now().strftime('%B %d, %Y')}",
        styles["ReportSubtitle"]
    ))
    brands = mol.get("brand_names", [])
    indications = mol.get("indications", [])
    if brands or indications:
        meta_parts = []
        if brands:
            meta_parts.append(f"Brands: {', '.join(brands[:5])}")
        if indications:
            meta_parts.append(f"Indications: {', '.join(indications[:5])}")
        story.append(Paragraph("&nbsp;&nbsp;|&nbsp;&nbsp;".join(meta_parts),
                               styles["ReportSubtitle"]))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY))
    story.append(Spacer(1, 6))

    # ---------- Score grid (compact) ----------
    scores = {
        "P": sc.get("P"), "D": sc.get("D"), "G": sc.get("G"),
        "E": sc.get("E"), "C": sc.get("C"), "Final Score": sc.get("final"),
    }
    story.append(build_score_grid(scores, styles))
    story.append(Spacer(1, 6))

    # ---------- Evidence Snapshot (compact stats bar) ----------
    total_trials = assess.get("total_trials", 0)
    rct_count = assess.get("rct_count", 0)
    total_patients = assess.get("total_patients", 0)
    fda_approved = assess.get("has_regulatory_approval", False)
    evidence_level = assess.get("evidence_level", "N/A")
    trials_with_results = assess.get("trials_with_results", 0)

    snap_cells = []
    snap_data = [
        ("Total Trials", str(total_trials)),
        ("With Results", str(trials_with_results)),
        ("RCTs", str(rct_count)),
        ("Patients", f"{total_patients:,}"),
        ("FDA Approved", "Yes" if fda_approved else "No"),
        ("Evidence Level", str(evidence_level)),
    ]
    for label, val in snap_data:
        val_color = GREEN if (label == "FDA Approved" and val == "Yes") else (
            RED if (label == "FDA Approved" and val == "No") else NAVY
        )
        snap_cells.append([
            Paragraph(val, ParagraphStyle("sv", parent=styles["SnapshotValue"], textColor=val_color)),
            Paragraph(label, styles["SnapshotLabel"]),
        ])

    snap_table = Table([snap_cells], colWidths=[1.1 * inch] * 6, rowHeights=[0.4 * inch])
    snap_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#E0E0E0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(snap_table)
    story.append(Spacer(1, 8))

    # ---------- Gemini narrative (the core of the report) ----------
    if gemini_narrative:
        flow = parse_narrative_to_flowables(gemini_narrative, styles)
        story.extend(flow)
    else:
        # Fallback: structured summary from data
        story.append(Paragraph("EXECUTIVE SUMMARY", styles["SectionHeader"]))
        story.append(Spacer(1, 4))

        level = assess.get("evidence_level", "N/A")
        total = assess.get("total_trials", 0)
        with_results = assess.get("trials_with_results", 0)
        rcts = assess.get("rct_count", 0)
        patients = assess.get("total_patients", 0)
        fda = assess.get("has_regulatory_approval", False)

        summary = (
            f"{generic_name} has {total} registered clinical trials, of which "
            f"{with_results} have reported results. {rcts} are randomized controlled trials "
            f"enrolling a total of {patients:,} patients. "
            f"{'The molecule has FDA approval.' if fda else 'No FDA approval is on record.'} "
            f"The overall evidence level is assessed as {level}."
        )
        story.append(Paragraph(escape_html(summary), styles["BodyProse"]))

        gaps = json_data.get("evidence_gaps", [])
        if gaps:
            story.append(Spacer(1, 4))
            story.append(Paragraph("EVIDENCE GAPS", styles["SectionHeader"]))
            story.append(Spacer(1, 3))
            for g in gaps:
                story.append(Paragraph(f"&bull; {escape_html(g)}", styles["BodyProse"]))

        story.append(Spacer(1, 6))
        final_val = sc.get("final", "N/A")
        story.append(Paragraph(
            f"Composite evidence score: {final_val}/5. "
            f"AI narrative was not generated (GEMINI_API_KEY missing or API error).",
            styles["BodyProse"]
        ))

    # ---------- Footer ----------
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")))
    footer_parts = [f"Report generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
    if gemini_narrative:
        footer_parts.append("Analytical narrative generated by Gemini 2.5 Flash")
    story.append(Paragraph("  |  ".join(footer_parts), styles["FooterText"]))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes