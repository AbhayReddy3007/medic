"""Reusable unmet-need scoring functions.

This module scores unique unmet needs from mapped trial evidence using Gemini,
then aggregates to a single unmet-need score.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from medical_potential.config import (
    BQ_DATASET_ID,
    PROJECT_ID,
    TRIAL_MAPPING_TABLE,
    UNMET_NEED_SCORING_TABLE,
)
from medical_potential.gcp_utils import get_bq_client
from medical_potential.gemini_utils import gemini_api_call

from .bq_utils import (
    TRIAL_MAPPING_SCHEMA,
    UNMET_NEED_SCORING_SCHEMA,
    export_to_gbq,
    import_from_gbq,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_PARALLEL_SCORING_TASKS = 3
UNMET_NEED_BATCH_SIZE = 3

WEIGHTS = {
    "C": 0.40,
    "S": 0.35,
    "R": 0.25,
    "X": 0.00,
}

# Country Weights
US_WEIGHT_WITH_EU = 0.6
EU_WEIGHT_WITH_US = 0.4
US_WEIGHT_ONLY = 0.6
EU_WEIGHT_ONLY = 0.4
SECONDARY_COUNTRIES_WEIGHT = 0

EU_COUNTRIES = ["France", "UK", "Germany"]
SECONDARY_COUNTRIES = ["China", "India", "Brazil", "Australia", "Russia", "Japan", "Korea"]

COUNTRY_WEIGHTS = {
    "US_WEIGHT_WITH_EU": US_WEIGHT_WITH_EU,
    "EU_WEIGHT_WITH_US": EU_WEIGHT_WITH_US,
    "US_WEIGHT_ONLY": US_WEIGHT_ONLY,
    "EU_WEIGHT_ONLY": EU_WEIGHT_ONLY,
    "SECONDARY_COUNTRIES_WEIGHT": SECONDARY_COUNTRIES_WEIGHT,
}

CATEGORY_TO_DIMENSION = {
    "Treatment Burden & Adherence Complexity": "C",
    "Achievement of Clinical Targets": "C",
    "Durability of Disease Control": "C",
    "Complication Prevention & Management": "C",
    "Safety & Treatment-Related Risk": "S",
    "Diagnostic Accuracy & Early Detection": "X",
    "Access, Affordability & Care Delivery": "X",
    "Quality of Life & Psychosocial Burden": "R",
    "Population-Specific Gaps & Inequities": "X",
}


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------


def build_unique_unmet_need_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate unmet needs and collect supporting evidence per unmet need."""
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}

    for row in records:
        unmet_need = str(row.get("unmet_need", "")).strip()
        category = str(row.get("category", "")).strip()
        country = str(row.get("country", "")).strip()
        if not unmet_need:
            continue

        key = (unmet_need, category, country)
        if key not in grouped:
            grouped[key] = {
                "unmet_need": unmet_need,
                "category": category,
                "country": country,
                "evidence_rows": [],
            }

        grouped[key]["evidence_rows"].append(
            {
                "trial_id": row.get("trial_id"),
                "molecule_name": row.get("molecule_name"),
                "phase": row.get("phase"),
                "phase_status": row.get("phase_status"),
                "evidence_strength": row.get("evidence_strength"),
                "relevant_endpoint": row.get("relevant_endpoint"),
                "explanation": row.get("explanation"),
                "country": row.get("country"),
                "search_sources_used": row.get("search_sources_used"),
            }
        )

    unique_groups = list(grouped.values())
    logger.info("Collapsed to %d unique unmet needs", len(unique_groups))
    return unique_groups


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------


def build_prompt(unmet_need_batch: list[dict[str, Any]]) -> str:
    """Build Gemini prompt for scoring a batch of unique unmet needs."""
    blocks: list[str] = []

    for idx, unmet_need_group in enumerate(unmet_need_batch, 1):
        category = unmet_need_group["category"]
        score_dimension = CATEGORY_TO_DIMENSION.get(category, "C")
        category_guidance = {
            "C": "Category belongs to Clinical Importance (C): assess clinically meaningful benefit vs current SoC and active comparators.",
            "S": "Category belongs to Safety/Diagnostic (S): assess reduction in safety risks or diagnostic limitations vs current SoC.",
            "R": "Category belongs to Real-World Adjustment (R): assess measurable improvements in adherence, access, QoL, treatment uptake, or equity.",
            "X": "Category belongs to Excluded-from-rollup (X): still score strictly from evidence, but this item has zero weight in final aggregate calculation.",
        }[score_dimension]

        evidence_lines: list[str] = []
        for i, ev in enumerate(unmet_need_group["evidence_rows"], 1):
            evidence_lines.append(
                f"Evidence {i}:\n"
                f"- Trial ID: {ev.get('trial_id')}\n"
                f"- Molecule: {ev.get('molecule_name')}\n"
                f"- Phase: {ev.get('phase')} ({ev.get('phase_status')})\n"
                f"- Evidence strength: {ev.get('evidence_strength')}\n"
                f"- Relevant endpoint: {ev.get('relevant_endpoint')}\n"
                f"- Explanation: {ev.get('explanation')}"
            )

        evidence_block = "\n\n".join(evidence_lines)
        blocks.append(
            f"""Unmet Need Item #{idx}:
- unmet_need: {unmet_need_group['unmet_need']}
- category: {unmet_need_group['category']}
- country: {unmet_need_group['country']}
- score_dimension: {score_dimension}
- category_guidance: {category_guidance}

Supporting evidence from mapped trials:
{evidence_block}
"""
        )

    items_block = "\n\n".join(blocks)

    return f"""You are an expert evidence-based pharmaceutical scoring analyst.

Score all unmet-need items below.

{items_block}

Scoring instructions for each item:
1. Output ONE unmet_need_score (integer 1-5) using that item's score_dimension.
   unmet_need_score must clearly indicate how the provided clinical-trial evidence helps solve the unmet need.
   Use this rubric:
       - 5: Demonstrates substantial improvement in clinically meaningful outcomes and is likely to change treatment practice, treatment guidelines, or standard clinical care.
       - 4: Demonstrates clear and clinically meaningful improvement versus current SoC.
       - 3: Demonstrates measurable improvement with moderate clinical benefit.
       - 2: Demonstrates limited improvement versus current SoC.
       - 1: Demonstrates minimal or no meaningful improvement versus current SoC.
2. Output evidence_impact_score (P score, integer 1-5) from evidence maturity:
         - 5: Multiple Phase 3/approved/post-marketing real-world evidence
         - 4: Strong Phase 3 evidence
         - 3: Positive Phase 2 evidence
         - 2: Early clinical evidence
         - 1: Preclinical/theoretical/unsupported
3. Score only the unmet need and its directly relevant evidence, not adjacent endpoints.
4. Be conservative if evidence is indirect, weak, or not comparator-grounded.
5. Provide concise rationale and explicit evidence references from the input.
6. Output country exactly as provided in input (verbatim, no rewriting).
7. Do not mention or reference any index fields (for example batch_item_index) inside rationale text or any summary text.

Return ONLY valid JSON as a LIST of objects (one object per unmet-need item). Each object must follow this schema:
[{{
    "batch_item_index": <integer starting at 1 matching Unmet Need Item #>,
    "unmet_need": "string",
    "category": "string",
    "country": "string",
    "score_dimension": "C|S|R|X",
    "unmet_need_score": 1|2|3|4|5,
    "evidence_impact_score": 1|2|3|4|5,
    "unmet_need_score_rationale": "rationale on giving a particular score",
    "evidence_score_rationale": "rationale on evidence impact score",
    "confidence": "High|Medium|Low"
}}]
"""


# ---------------------------------------------------------------------------
# LLM Invocation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Response Parsing
# ---------------------------------------------------------------------------


def parse_json_list(raw: str) -> list[dict[str, Any]]:
    """Parse model output into a JSON list of objects."""
    if not raw:
        return []

    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()

    start_list = text.find("[")
    end_list = text.rfind("]")
    start_obj = text.find("{")
    end_obj = text.rfind("}")

    if start_list != -1 and end_list != -1 and end_list > start_list:
        text = text[start_list:end_list + 1]
    elif start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        text = text[start_obj:end_obj + 1]
    else:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
        if isinstance(parsed, dict):
            return [parsed]
        return []
    except json.JSONDecodeError:
        pass

    try:
        from json_repair import repair_json  # type: ignore

        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, list):
            return [x for x in repaired if isinstance(x, dict)]
        if isinstance(repaired, dict):
            return [repaired]
        return []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Score Computation Helpers
# ---------------------------------------------------------------------------


def build_score_fallback_record(group: dict[str, Any], reason: str) -> dict[str, Any]:
    """Create a safe default scoring record when parsing or matching fails."""
    return {
        "unmet_need": group["unmet_need"],
        "category": group["category"],
        "country": group["country"],
        "score_dimension": None,
        "unmet_need_score": None,
        "evidence_impact_score": None,
        "unmet_need_score_rationale": reason,
        "evidence_score_rationale": None,
        "confidence": "Low",
    }


def select_parsed_item_for_group(
    parsed_list: list[dict[str, Any]],
    group: dict[str, Any],
    batch_pos: int,
) -> dict[str, Any] | None:
    """Pick the best parsed LLM item for one unmet-need group."""
    for item in parsed_list:
        if item.get("batch_item_index") == batch_pos:
            return item

    for item in parsed_list:
        if (
            item.get("unmet_need") == group.get("unmet_need")
            and item.get("category") == group.get("category")
            and item.get("country") == group.get("country")
        ):
            return item

    if len(parsed_list) == 1:
        return parsed_list[0]
    if len(parsed_list) >= batch_pos:
        return parsed_list[batch_pos - 1]

    return None


def finalize_scoring_record(parsed: dict[str, Any], group: dict[str, Any], molecule_name: str) -> dict[str, Any]:
    """Normalize, validate, and enrich a parsed LLM scoring record."""
    result = dict(parsed)

    # Step 1: Enforce deterministic identity fields from the source group.
    expected_dimension = CATEGORY_TO_DIMENSION.get(group["category"], "C")
    result["molecule_name"] = molecule_name
    result["unmet_need"] = group["unmet_need"]
    result["category"] = group["category"]
    result["country"] = group["country"]
    result["score_dimension"] = expected_dimension

    # Step 2: Validate bounded integer scores coming from LLM output.
    score_val = result.get("unmet_need_score")
    if not isinstance(score_val, int) or not (1 <= score_val <= 5):
        result["unmet_need_score"] = None
    evidence_score_val = result.get("evidence_impact_score")
    if not isinstance(evidence_score_val, int) or not (1 <= evidence_score_val <= 5):
        result["evidence_impact_score"] = None

    # Step 3: Compute per-unmet-need weighted score.
    # 1) evidence_impact_weight = evidence_impact_score / 5
    # 2) weighted_unmet_need_score = unmet_need_score * evidence_impact_weight
    result["num_of_trials_assessed"] = len(group["evidence_rows"])

    if isinstance(result.get("unmet_need_score"), int) and isinstance(result.get("evidence_impact_score"), int):
        evidence_impact_weight = float(result["evidence_impact_score"]) / 5.0
        weighted_unmet_need_score = float(result["unmet_need_score"]) * evidence_impact_weight
        result["evidence_impact_weight"] = round(evidence_impact_weight, 4)
        result["weighted_unmet_need_score"] = round(weighted_unmet_need_score, 4)
        # Keep this key for compatibility with existing downstream aggregation/output.
        result["final_unmet_need_score"] = round(weighted_unmet_need_score, 4)
    else:
        result["evidence_impact_weight"] = None
        result["weighted_unmet_need_score"] = None
        result["final_unmet_need_score"] = None

    return result


# ---------------------------------------------------------------------------
# Batch Scoring Helpers
# ---------------------------------------------------------------------------


def create_unmet_need_batches(unmet_need_groups: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split unmet-need groups into fixed-size batches for prompt packing."""
    return [
        unmet_need_groups[i : i + UNMET_NEED_BATCH_SIZE]
        for i in range(0, len(unmet_need_groups), UNMET_NEED_BATCH_SIZE)
    ]


async def score_unmet_need_batch(
    model: str,
    batch_idx: int,
    total_batches: int,
    batch_groups: list[dict[str, Any]],
    molecule_name: str,
    semaphore: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    """Score one unmet-need batch and return finalized rows for each item."""
    async with semaphore:
        start = ((batch_idx - 1) * UNMET_NEED_BATCH_SIZE) + 1
        end = start + len(batch_groups) - 1
        logger.info(
            "Scoring unmet-need batch %d/%d (items %d-%d)",
            batch_idx,
            total_batches,
            start,
            end,
        )

        try:
            prompt = build_prompt(batch_groups)
            raw = await asyncio.to_thread(
                gemini_api_call,
                prompt=prompt,
                model=model,
            )
            parsed_list = parse_json_list(raw)
        except Exception as e:
            logger.warning("Scoring call failed for batch %d: %s", batch_idx, e)
            parsed_list = []

        if not parsed_list:
            logger.warning("Could not parse score response for batch %d", batch_idx)

        finalized: list[dict[str, Any]] = []
        for batch_pos, group in enumerate(batch_groups, 1):
            parsed_item = select_parsed_item_for_group(parsed_list, group, batch_pos)
            if not parsed_item:
                parsed_item = build_score_fallback_record(group, "Failed to parse Gemini response")
            finalized.append(finalize_scoring_record(parsed_item, group, molecule_name))

        return finalized


# ---------------------------------------------------------------------------
# Main Scoring Orchestration
# ---------------------------------------------------------------------------


async def score_all_unique_unmet_needs(
    model: str,
    unmet_need_groups: list[dict[str, Any]],
    molecule_name: str,
) -> list[dict[str, Any]]:
    """Run Gemini scoring in small batches of unmet needs with semaphore-limited parallelism."""
    # Step 1: Short-circuit when there is nothing to score.
    total = len(unmet_need_groups)
    if total == 0:
        return []

    # Step 2: Prepare concurrency guard and prompt batches.
    semaphore = asyncio.Semaphore(MAX_PARALLEL_SCORING_TASKS)
    batches = create_unmet_need_batches(unmet_need_groups)

    # Step 3: Launch one async task per batch.
    tasks = [
        asyncio.create_task(
            score_unmet_need_batch(
                model=model,
                batch_idx=batch_idx,
                total_batches=len(batches),
                batch_groups=batch_groups,
                molecule_name=molecule_name,
                semaphore=semaphore,
            )
        )
        for batch_idx, batch_groups in enumerate(batches, 1)
    ]

    # Step 4: Wait for all batches and flatten to one row per unmet need.
    batch_outputs = await asyncio.gather(*tasks)

    flattened: list[dict[str, Any]] = []
    for out in batch_outputs:
        flattened.extend(out)
    return flattened


# ---------------------------------------------------------------------------
# Output Aggregation and Persistence
# ---------------------------------------------------------------------------


def _fetch_rows_for_molecule(
    table_name: str,
    molecule_name: str,
    use_timestamp_filter: bool = False,
) -> list[dict[str, Any]]:
    """Fetch BigQuery rows for one molecule, with optional latest timestamp filter."""
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

def aggregate_single_score(scored_rows: list[dict[str, Any]]) -> float | None:
    """Aggregate one final score from row-level weighted unmet-need scores.

    Process:
    1. Compute country-level averages within each scored dimension (C, S, R).
    2. Compute the dimension score using the following country coverage rule:
            - If both US and at least one EU country are present, allocate US_WEIGHT_WITH_EU to US and EU_WEIGHT_WITH_US across the EU countries.
            - If only US is present, allocate US_WEIGHT_ONLY to US.
            - If only EU countries are present, allocate EU_WEIGHT_ONLY across the EU countries.
      - If neither US nor EU is present, the dimension score is 0.
            - Secondary countries contribute only when US or EU coverage is present, using SECONDARY_COUNTRIES_WEIGHT split across the available secondary countries.
      - Excluded dimension X is ignored (zero weight).
    3. Compute final weighted score as:
         round((WEIGHTS["C"] * C) + (WEIGHTS["S"] * S) + (WEIGHTS["R"] * R), 4)
    """

    def normalize_country(country: Any) -> str:
        return str(country or "").strip().upper()

    # Step 1: Group row-level weighted scores by dimension and country.
    dimension_country_scores: dict[str, dict[str, list[float]]] = {"C": {}, "S": {}, "R": {}}
    for row in scored_rows:
        dimension = row.get("score_dimension")
        # BQ table persists weighted_unmet_need_score; in-memory scoring rows also carry
        # final_unmet_need_score for compatibility. Support both for robust aggregation.
        final_score = row.get("final_unmet_need_score")
        if not isinstance(final_score, (int, float)):
            final_score = row.get("weighted_unmet_need_score")
        if dimension in dimension_country_scores and isinstance(final_score, (int, float)):
            country = normalize_country(row.get("country"))
            if not country:
                continue
            if country not in dimension_country_scores[dimension]:
                dimension_country_scores[dimension][country] = []
            dimension_country_scores[dimension][country].append(float(final_score))

    # Step 2: Compute weighted dimension score from country-level averages.
    dimension_averages: dict[str, float] = {}
    eu_normalized = {c.upper() for c in EU_COUNTRIES}
    secondary_normalized = {c.upper() for c in SECONDARY_COUNTRIES}

    for dim, country_scores in dimension_country_scores.items():
        if not country_scores:
            continue

        country_averages = {
            country: (sum(scores) / len(scores))
            for country, scores in country_scores.items()
            if scores
        }
        if not country_averages:
            continue

        weighted_dim_score = 0.0

        # Apply the US/EU coverage rule for the dimension score.
        # If both are present, each contributes its configured shared weight.
        # If only one is present, it receives its configured solo weight.
        # If neither is present, the dimension contributes 0.
        us_present = "US" in country_averages
        eu_present = [c for c in country_averages if c in eu_normalized]

        if us_present and eu_present:
            weighted_dim_score += US_WEIGHT_WITH_EU * country_averages["US"]
            eu_country_weight = EU_WEIGHT_WITH_US / len(eu_present)
            for country in eu_present:
                weighted_dim_score += eu_country_weight * country_averages[country]
        elif us_present:
            weighted_dim_score += US_WEIGHT_ONLY * country_averages["US"]
        elif eu_present:
            eu_country_weight = EU_WEIGHT_ONLY / len(eu_present)
            for country in eu_present:
                weighted_dim_score += eu_country_weight * country_averages[country]
        else:
            dimension_averages[dim] = 0.0
            continue

        # Keep secondary-country handling consistent with the existing model when US/EU are present.
        secondary_present = [c for c in country_averages if c in secondary_normalized]
        if secondary_present and (us_present or eu_present):
            secondary_country_weight = SECONDARY_COUNTRIES_WEIGHT / len(secondary_present)
            for country in secondary_present:
                weighted_dim_score += secondary_country_weight * country_averages[country]

        dimension_averages[dim] = weighted_dim_score

    # Step 3: Return None if there are no usable dimension scores.
    if not dimension_averages:
        return None

    # Step 4: Apply final weighted rollup across C/S/R dimensions.
    final_score = 0.0
    if "C" in dimension_averages:
        final_score += WEIGHTS["C"] * dimension_averages["C"]
    if "S" in dimension_averages:
        final_score += WEIGHTS["S"] * dimension_averages["S"]
    if "R" in dimension_averages:
        final_score += WEIGHTS["R"] * dimension_averages["R"]

    return round(final_score, 2)


async def score_unmet_needs(
    molecule_name: str,
    model: str = "gemini-3-flash-preview",
) -> dict[str, Any]:
    """Score unmet needs from BigQuery mappings, export scores, and return payload."""
    # Step 1: Validate required runtime configuration.
    if not molecule_name or not molecule_name.strip():
        raise ValueError("molecule_name must be provided for unmet-need scoring.")

    molecule = molecule_name.strip()
    logger.info(
        "Running score_unmet_needs in full scoring mode (clinical trials assessed)."
    )

    # Step 2: Import mapped trial-to-unmet-need rows from BigQuery.
    all_mapping_rows = import_from_gbq(
        table_name=TRIAL_MAPPING_TABLE,
        columns=[field.name for field in TRIAL_MAPPING_SCHEMA],
    )

    # Step 3: Keep only rows for the requested molecule.
    mapping_rows = [
        row for row in all_mapping_rows
        if str(row.get("molecule_name", "")).strip() == molecule
    ]

    # Step 4: Build unique unmet-need groups from imported mappings.
    unique_groups = build_unique_unmet_need_groups(mapping_rows)

    # Step 5: Score unmet-need groups in batches using Gemini.
    scored_rows = await score_all_unique_unmet_needs(
        model=model,
        unmet_need_groups=unique_groups,
        molecule_name=molecule,
    )

    # Step 6: Aggregate row-level scores to one molecule-level unmet-need score.
    aggregate_score = aggregate_single_score(scored_rows)

    # Step 7: Export scored rows to BigQuery.
    logger.info("Step 2 complete - exporting %d scored rows to BigQuery", len(scored_rows))
    export_to_gbq(
        rows=scored_rows,
        table_name=UNMET_NEED_SCORING_TABLE,
        schema=UNMET_NEED_SCORING_SCHEMA,
    )

    # Step 8: Build and return output payload.
    payload = {
        "molecule_name": molecule,
        "clinical_trial_evidence_to_unmet_needs_mapping_row_count": len(mapping_rows),
        "clinical_trial_evidence_to_unmet_needs_mapping_rows": mapping_rows,
        "molecule_level_unmet_needs_count": len(scored_rows),
        "molecule_level_unmet_needs_rows": scored_rows,
        "weights_used": {
            "category_weights": WEIGHTS,
            "country_weights": COUNTRY_WEIGHTS,
        },
        "unmet_need_score": aggregate_score,
    }
    return payload
