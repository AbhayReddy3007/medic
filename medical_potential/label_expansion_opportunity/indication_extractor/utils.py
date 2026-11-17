"""Label Expansion Opportunity — shared utilities.

Holds everything ``trial_analyser.py`` (Module 1) and ``web_analyser.py``
(Module 2) both need but that isn't specific to either one:

- ``extract_json()`` to reliably parse JSON out of a model response
- ``SECONDARY_INDICATION_CRITERIA``, the shared prompt text defining what
  counts as a genuine label-expansion indication
- the scorable-indication safety-net filter

``gemini_generate()`` / ``gemini_generate_with_timeout()`` now live in the
top-level ``label_expansion_opportunity/utils.py`` - imported here only for
this module's own use (the scorable-indication classifier below). Other
modules import those two functions directly from ``..utils``, not from here.
"""

from __future__ import annotations

import json
import logging
import re

from ..utils import gemini_generate

logger = logging.getLogger(__name__)

# ==============================
# PIPELINE MODE TOGGLE
# ==============================
# False (default): run the full pipeline from scratch - trial_analyser
# fetches trials and extracts indications via Gemini + Google Search per
# trial, fda_fetcher fetches FDA brands and label text and extracts
# disease names, then both classify what they found. This is the
# expensive path (one search-grounded Gemini call per trial/brand).
#
# True: skip re-discovery and re-extraction entirely. Instead,
# trial_analyser, fda_fetcher, and web_analyser pull the indications
# already sitting in LE_TABLE for the drug and only re-run classification
# (indication_type / therapy_area / ot_disease_name) on them - one (or a
# few) batched Gemini call(s) over the unique indications, not one call
# per trial/brand or one open-ended web-research pass. Use this after a
# classification prompt/logic change (e.g. fixing how Primary/Secondary
# is decided) when you want existing rows re-classified with the fixed
# logic without paying to re-discover indications that are already
# correctly identified.
PROCESS_INDICATIONS = False

# ==============================
# BATCHING CONSTANTS
# ==============================
TRIALS_PER_CALL = 6
INDICATIONS_PER_CALL = 10


def extract_json(text: str) -> dict | list:
    """Extracts and parses the first JSON object or array from *text*."""
    if not text:
        raise ValueError("Cannot extract JSON from empty text")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    stripped = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    stripped = re.sub(r"\s*```\s*$", "", stripped, flags=re.MULTILINE).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = stripped.find(open_ch)
        if start == -1:
            continue
        depth, end, in_str = 0, start, False
        while end < len(stripped):
            ch = stripped[end]
            if ch == '"' and (end == 0 or stripped[end - 1] != "\\"):
                in_str = not in_str
            elif not in_str:
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(stripped[start : end + 1])
                        except json.JSONDecodeError:
                            break
            end += 1

    raise ValueError(f"No valid JSON found in response (first 200 chars): {text[:200]}")


# ==============================
# SHARED PROMPT CONSTANTS
# ==============================
SECONDARY_INDICATION_CRITERIA = """
A secondary indication qualifies ONLY if ALL of the following are true:

- The indication represents a true expansion - i.e., it is not part of the
  primary indication (for clinical assets) or currently approved label
  (for commercial assets)
- The indication is described at a clear disease-level definition, avoiding
  vague, overlapping, or synonymous representations
- The source must describe observed or measured outcomes in that specific
  indication (e.g., trial results, endpoint readouts, biomarker response),
  not just planned evaluation or exploratory intent

The following must NOT be considered secondary indications:
* Indications mentioned only as hypothesis, targets, or exploratory possibilities
* Pipeline indications without any data or outcomes
* Mechanism based assumptions without clinical or empirical validation
* Early discovery or preclinical signals without human data
* Any indication lacking traceable, verifiable evidence of results
"""


# ==============================
# SCORABLE-INDICATION FILTER (shared safety net)
# ==============================
# Used by the main orchestrator, after merging trial_analyser/fda_fetcher/
# web_analyser output, to drop rows whose "indication" is actually a trial
# endpoint, biomarker, PK parameter, or procedure rather than a genuine
# disease/condition. These can never become an approved label indication,
# so they shouldn't be scored as label-expansion opportunities. This is a
# safety net for cases the extraction prompts themselves didn't already
# exclude.
SCORABLE_BATCH_SIZE = 20


def _classify_scorable_batch(indications: list[str]) -> dict[str, bool]:
    """One Gemini call: classifies each indication as scorable (a genuine
    disease/condition) or not (an endpoint/biomarker/PK parameter/
    procedure). Returns ``{indication_lower: is_scorable}``."""
    if not indications:
        return {}

    ind_list = "\n".join(f"{i + 1}. {ind}" for i, ind in enumerate(indications))
    prompt = f"""You are a pharmaceutical analyst.

Below is a numbered list of strings extracted from clinical trial and FDA
label data. For EACH one, decide whether it is a disease or medical
condition that could plausibly appear as an FDA-approved drug indication
(scorable), or whether it is something else that should NOT be scored as
an indication - a trial endpoint or outcome measure, a biomarker or lab
value, a pharmacokinetic parameter, a physiological measurement, or a
procedure/intervention (not scorable).

Examples of NOT scorable: "Exercise Capacity", "Waist Circumference",
"Postprandial Glucose", "Pharmacokinetics", "Lipid Profile",
"Hepatocyte Ballooning", "Bariatric Surgery".
Examples of scorable: "Type 2 Diabetes", "Obesity", "Heart Failure".

Items:
{ind_list}

Return ONLY a JSON array, one object per item, in the same order:
[
  {{"item": "<exact item text>", "scorable": true or false}}
]
"""
    try:
        text = gemini_generate(
            prompt,
            system_instruction=(
                "Classify each item as a scorable disease/condition or not. "
                "Return ONLY valid JSON."
            ),
            use_search=False,
        )
        parsed = extract_json(text)
        entries = parsed if isinstance(parsed, list) else parsed.get("items", []) if isinstance(parsed, dict) else []
        result: dict[str, bool] = {}
        for e in entries:
            if not isinstance(e, dict):
                continue
            item = (e.get("item") or "").strip().lower()
            if item:
                result[item] = bool(e.get("scorable", True))
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("[UTILS] Scorable classification failed for batch %s: %s", indications, exc)
        # Fail open: on error, keep everything rather than silently
        # dropping potentially-valid indications.
        return {ind.lower(): True for ind in indications}


def filter_scorable_indications(rows: list[dict]) -> list[dict]:
    """Drops rows whose ``indication`` is a trial endpoint, biomarker,
    pharmacokinetic parameter, or procedure rather than an actual
    disease/condition.

    Runs one batched Gemini call per ``SCORABLE_BATCH_SIZE`` unique
    indications (not per row) to keep cost down, then filters ``rows``
    using the result. On any classification failure, the affected
    indications are kept (fail open) rather than silently dropped.
    """
    if not rows:
        return rows

    unique_indications = sorted({
        (r.get("indication") or "").strip()
        for r in rows
        if (r.get("indication") or "").strip()
    })
    if not unique_indications:
        return rows

    batches = [
        unique_indications[i : i + SCORABLE_BATCH_SIZE]
        for i in range(0, len(unique_indications), SCORABLE_BATCH_SIZE)
    ]

    scorable_map: dict[str, bool] = {}
    for batch in batches:
        scorable_map.update(_classify_scorable_batch(batch))

    kept: list[dict] = []
    dropped: list[str] = []
    for r in rows:
        ind = (r.get("indication") or "").strip().lower()
        if not ind or scorable_map.get(ind, True):
            kept.append(r)
        else:
            dropped.append(r.get("indication"))

    if dropped:
        logger.info(
            "[UTILS] Filtered out %d row(s) with non-scorable indication(s): %s",
            len(dropped), sorted(set(dropped)),
        )
    return kept