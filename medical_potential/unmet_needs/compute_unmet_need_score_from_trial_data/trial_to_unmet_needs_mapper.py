"""Core mapping logic for linking unmet needs to clinical trial evidence.

This module contains the detailed implementation for:
1. Loading unmet needs and trial datasets
2. Calling Gemini with web grounding in batches
3. Parsing and normalizing mappings
4. Saving mapping outputs

Use `map_clinical_trial_data_with_unmet_needs` as the main entry point.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

import pandas as pd
from medical_potential.config import (
    BQ_DATASET_ID,
    CLINICAL_EFFICACY_TABLE,
    PROJECT_ID,
    TRIAL_LLM_CONFIDENCE_THRESHOLD,
    TRIAL_MAPPING_TABLE,
    UNMET_NEEDS_TABLE,
)
from medical_potential.gcp_utils import get_bq_client
from medical_potential.gemini_utils import gemini_api_call

from .bq_utils import (
    TRIAL_MAPPING_SCHEMA,
    export_to_gbq,
)

from google.cloud import bigquery as _bq

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRIAL_BATCH_SIZE = 6
MAX_PARALLEL_BATCHES = 3

_url_cache_lock = threading.Lock()
_resolved_url_cache: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


EXCLUDED_CATEGORIES = {
    # "Diagnostic Accuracy & Early Detection",
    # "Access, Affordability & Care Delivery",
    # "Population-Specific Gaps & Inequities",
}




def load_unmet_needs() -> pd.DataFrame:
    """Load unmet needs data from BigQuery table, excluding certain categories.

    Returns:
        DataFrame with columns: unmet_need, category, evidence, country, source, created_at_utc
    """
    # Step 1: Load unmet-needs rows from the BigQuery source table.
    logger.info("Loading unmet needs from BigQuery table: %s.%s.%s", PROJECT_ID, BQ_DATASET_ID, UNMET_NEEDS_TABLE)
    client = get_bq_client()

    sql = f"""
        SELECT unmet_need, category, evidence, country, source, created_at_utc
        FROM `{PROJECT_ID}.{BQ_DATASET_ID}.{UNMET_NEEDS_TABLE}`
        WHERE inclusion_flag = 'Y'
    """
    query_job = client.query(sql)
    df = query_job.result().to_dataframe()

    # Step 2: Apply business-rule exclusions and keep a clean index.
    before = len(df)
    df = df[~df["category"].isin(EXCLUDED_CATEGORIES)].reset_index(drop=True)
    excluded_count = before - len(df)

    logger.info(
        "  Loaded %d unmet needs across %d categories (excluded %d rows from %d categories)",
        len(df),
        df["category"].nunique(),
        excluded_count,
        len(EXCLUDED_CATEGORIES),
    )
    return df


def _get_latest_mapping_timestamp(client, molecule_name: str):
    """Return the latest created_at_utc from the mapping table for the given molecule, or None."""
    mapping_table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{TRIAL_MAPPING_TABLE}"
    sql = f"""
        SELECT MAX(created_at_utc) AS latest_ts
        FROM `{mapping_table_id}`
        WHERE molecule_name = @molecule_name
    """
    job_config = _bq.QueryJobConfig(
        query_parameters=[
            _bq.ScalarQueryParameter("molecule_name", "STRING", molecule_name)
        ]
    )
    result = client.query(sql, job_config=job_config).result()
    row = next(iter(result), None)
    if row is None or row.latest_ts is None:
        return None
    return row.latest_ts


def delete_existing_mappings_for_trial_ids(client, trial_ids: list[str]) -> int:
    """Delete existing mapping rows for provided trial IDs and return affected row count."""
    normalized_ids = [str(tid).strip() for tid in trial_ids if str(tid).strip()]
    if not normalized_ids:
        return 0

    mapping_table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{TRIAL_MAPPING_TABLE}"
    sql = f"""
        DELETE FROM `{mapping_table_id}`
        WHERE trial_id IN UNNEST(@trial_ids)
    """
    job_config = _bq.QueryJobConfig(
        query_parameters=[
            _bq.ArrayQueryParameter("trial_ids", "STRING", normalized_ids),
        ]
    )
    delete_job = client.query(sql, job_config=job_config)
    delete_job.result()

    affected_rows = int(getattr(delete_job, "num_dml_affected_rows", 0) or 0)
    logger.info(
        "Deleted %d existing row(s) for %d trial_id(s) from %s",
        affected_rows,
        len(normalized_ids),
        mapping_table_id,
    )
    return affected_rows


def load_clinical_trials(molecule_name: str, incremental: bool = True) -> pd.DataFrame:
    """Load clinical trials data from BigQuery for a single molecule.

    Args:
        molecule_name: Name of the molecule to load trials for.
        incremental: When True (default), only loads trials created or updated
            after the latest ``created_at_utc`` already present in the mapping
            table for this molecule. Set to False to reload all trials.

    Returns:
        DataFrame with trial metadata including trial_id, molecule_name, phase, etc.
    """
    if not molecule_name or not molecule_name.strip():
        raise ValueError("molecule_name must be provided to load clinical trials")

    stripped_molecule_name = molecule_name.strip()
    client = get_bq_client()

    # Step 1: Determine incremental cutoff timestamp (if applicable).
    cutoff_ts: str | None = None
    if incremental:
        cutoff_ts = _get_latest_mapping_timestamp(client, stripped_molecule_name)
        if cutoff_ts:
            logger.info(
                "Incremental mode: loading trials for '%s' with created_at or updated_at > %s",
                stripped_molecule_name,
                cutoff_ts,
            )
        else:
            logger.info(
                "Incremental mode: no prior mapping found for '%s' — loading all trials",
                stripped_molecule_name,
            )
    else:
        logger.info("Full-load mode: loading all clinical trials for molecule: %s", stripped_molecule_name)

    # Step 2: Fetch candidate trial rows, applying incremental filter when a cutoff exists.
    clinical_efficacy_table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{CLINICAL_EFFICACY_TABLE}"
    if incremental and cutoff_ts is not None:
        sql = f"""
            SELECT *
            FROM `{clinical_efficacy_table_id}`
            WHERE molecule_name = @molecule_name
              AND (
                CAST(created_at AS TIMESTAMP) > @cutoff_ts
                OR CAST(updated_at AS TIMESTAMP) > @cutoff_ts
              )
              AND llm_confidence > {TRIAL_LLM_CONFIDENCE_THRESHOLD}
            ORDER BY trial_id
        """
        job_config = _bq.QueryJobConfig(
            query_parameters=[
                _bq.ScalarQueryParameter("molecule_name", "STRING", stripped_molecule_name),
                _bq.ScalarQueryParameter("cutoff_ts", "TIMESTAMP", cutoff_ts),
            ]
        )
    else:
        sql = f"""
            SELECT *
            FROM `{clinical_efficacy_table_id}`
            WHERE 
                molecule_name = @molecule_name
                AND llm_confidence > {TRIAL_LLM_CONFIDENCE_THRESHOLD}
            ORDER BY trial_id
        """
        job_config = _bq.QueryJobConfig(
            query_parameters=[
                _bq.ScalarQueryParameter("molecule_name", "STRING", stripped_molecule_name),
            ]
        )

    query_job = client.query(sql, job_config=job_config)
    rows = query_job.result()
    df = pd.DataFrame([dict(row) for row in rows])

    if df.empty:
        logger.warning("  No clinical trials found for molecule: %s", molecule_name)
        return df

    # Step 2: Prepare deterministic helper keys for filtering/deduplication.
    # Keep row order for deterministic tie-breaking when phases are equal.
    df = df.reset_index(drop=True)
    df["_row_order"] = df.index

    def _clean_trial_id(value) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip()

    def _extract_phase_rank(value) -> int | None:
        if pd.isna(value):
            return None
        digits = [int(d) for d in re.findall(r"[1-4]", str(value))]
        if not digits:
            return None
        return max(digits)

    df["_trial_id_key"] = df["trial_id"].apply(_clean_trial_id)
    df["_phase_rank"] = df["phase"].apply(_extract_phase_rank)

    # Step 3: Remove rows with unusable trial identifiers or phase values.
    before_filter = len(df)
    df = df[(df["_trial_id_key"] != "") & (df["_phase_rank"].notna())].copy()
    dropped_invalid = before_filter - len(df)

    if df.empty:
        logger.warning(
            "  No valid clinical trials after filtering invalid trial_id/phase rows for molecule: %s",
            molecule_name,
        )
        return df.drop(columns=["_row_order", "_trial_id_key", "_phase_rank"], errors="ignore")

    duplicated_trial_ids = int(df["_trial_id_key"].duplicated(keep=False).sum())

    # Step 4: Deduplicate by normalized trial_id key, preferring highest phase.
    # One row per trial_id: highest phase wins; ties keep first loaded row.
    df = df.sort_values(
        by=["_trial_id_key", "_phase_rank", "_row_order"],
        ascending=[True, False, True],
    )
    df = df.drop_duplicates(subset=["_trial_id_key"], keep="first")
    df = df.sort_values(by=["_trial_id_key", "_row_order"], ascending=[True, True]).reset_index(drop=True)

    df = df.drop(columns=["_row_order", "_trial_id_key", "_phase_rank"], errors="ignore")

    if dropped_invalid:
        logger.info("  Dropped %d rows with missing trial_id or non-parseable phase (1-4)", dropped_invalid)
    if duplicated_trial_ids:
        logger.info("  Found %d rows belonging to duplicate trial_id groups before dedup", duplicated_trial_ids)

    # Step 5: Return stable, deduplicated trial metadata for downstream LLM batching.
    logger.info("  Loaded %d clinical trials covering %d molecules", len(df), df["molecule_name"].nunique())
    return df


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(
    trial_batch: list[dict],
    unmet_needs: list[dict],
) -> str:
    """Build the LLM prompt for a batch of trials against all unmet needs.

    Args:
        trial_batch: List of trial metadata dicts (up to TRIAL_BATCH_SIZE entries).
        unmet_needs: Full list of unmet need dicts with keys: unmet_need, category, evidence, country.

    Returns:
        Formatted prompt string.
    """
    # Step 1: Render trial and unmet-need context blocks for the model.
    trials_text = ""
    for i, trial in enumerate(trial_batch, 1):
        trials_text += f"""
Trial {i}:
    - Batch Trial Index: {i}
  - Trial ID: {trial.get('trial_id', 'N/A')}
  - Molecule: {trial.get('molecule_name', 'N/A')}
  - Dosage: {trial.get('dosage', 'N/A')}
  - Phase: {trial.get('phase', 'N/A')} ({trial.get('phase_status', 'N/A')})
  - Trial Size: {trial.get('trial_size', 'N/A')} participants
  - Location: {trial.get('trial_location', 'N/A')}
  - Duration: {trial.get('trial_start_date', 'N/A')} to {trial.get('trial_completion_date', 'N/A')}
  - HbA1c Change: {trial.get('hba1c_change_pct', 'N/A')}% over {trial.get('hba1c_duration', 'N/A')}
  - Weight Change: {trial.get('weight_change_pct', 'N/A')}% over {trial.get('weight_duration', 'N/A')}
  - ALT Reduction: {trial.get('alt_reduction_pct', 'N/A')}% over {trial.get('alt_duration', 'N/A')}
  - MASH Resolution: {trial.get('mash_resolution_pct', 'N/A')}% over {trial.get('mash_duration', 'N/A')}
  - Source URL: {trial.get('source_url', 'N/A')}
""".strip() + "\n"

    needs_text = ""
    for j, need in enumerate(unmet_needs, 1):
        needs_text += (
            f"Unmet Need #{j} [{need.get('category', 'N/A')}]:\n"
            f"  - unmet_need_index: {j}\n"
            f"  - Need: {need.get('unmet_need', 'N/A')}\n"
            f"  - Evidence: {need.get('evidence', 'N/A')}\n"
            f"  - Country: {need.get('country', 'N/A')}\n"
        )

    # Step 2: Compose strict instructions and output schema constraints.
    prompt = f"""You are a pharmaceutical research analyst specializing in Type 2 Diabetes Mellitus (T2DM).

You are given {len(trial_batch)} clinical trial(s) and a list of {len(unmet_needs)} documented unmet needs in T2DM care.

Your task is to use provided trial data and web search to determine whether each clinical trial contains results, endpoints, or evidence that RESOLVES or IMPROVES any of the listed unmet needs.

## CLINICAL TRIALS TO ANALYZE:

{trials_text}

## UNMET NEEDS IN T2DM:

{needs_text}

## INSTRUCTIONS:

For each trial, search for the trial using its Trial ID on ClinicalTrials.gov and relevant published literature. Then assess which unmet needs (if any) are addressed by that trial's design, endpoints, or reported outcomes.

For each trial-unmet_need pair where there IS a connection using both provided trial data and search results data:
- Explain specifically how the trial's data addresses the unmet need
- When explaining the connection, ALWAYS cite the specific quantitative evidence (e.g., "HbA1c reduction of 1.5%", "weight loss of 8.2 kg")
- Quote the trial endpoints and outcomes directly with their numerical values
- Reference the source of the evidence (e.g., the trial result, published paper, or ClinicalTrials.gov data) when referring to web-search findings.
- For web-search trial results, use only evidence from the highest completed phase and any ongoing phase updates for that trial.
- For example, instead of saying "The trial shows weight loss", say "According to [source], the trial showed a mean weight reduction of 4.5 kg over 12 weeks".
- Indicate the strength of evidence (Strong / Moderate / Weak)
- Note the specific trial endpoint or outcome that is relevant
- Use trial_id and molecule_name exactly as provided in the trial input (no rewriting).
- Include trial_batch_index exactly matching the Batch Trial Index from input.
- Pick unmet_need_index strictly from the numbered "Unmet Need #<index>" list above.
- Use the exact index for the unmet need item you are mapping; do not invent or shift indices.
- Do not mention or reference any index fields (for example trial_batch_index or unmet_need_index) inside explanation/rationale text or any summary text.
- Never shorten, normalize, strip, or alter trial_id (for example, keep text inside parentheses exactly as-is).
- Output country exactly as provided in the unmet-need input (verbatim, no rewriting).

Return your analysis as a JSON array. Each object in the array must follow this exact schema:

{{
    "trial_batch_index": <integer Batch Trial Index from the trial input>,
  "trial_id": "<trial ID>",
  "molecule_name": "<molecule name>",
  "unmet_need_index": <integer 1-based index matching the unmet need list above>,
  "unmet_need": "<exact unmet need text>",
  "category": "<unmet need category>",
    "country": "<exact unmet need country from input>",
  "addresses_unmet_need": true | false,
  "evidence_strength": "Strong" | "Moderate" | "Weak" | null,
  "relevant_endpoint": "<specific endpoint or outcome relevant to the unmet need, or null>",
  "explanation": "<concise explanation of how the trial addresses this need, including quoted quantitative evidence and source references, or null if it does not>",
  "search_sources_used": ["<url or reference>"]
}}

Only include pairs where "addresses_unmet_need" is true. Do NOT include pairs where the trial has no relevant data for the unmet need.

Return ONLY valid JSON - no markdown, no preamble, no explanation outside the JSON array.
"""
    return prompt


# ---------------------------------------------------------------------------
# LLM call with search grounding
# ---------------------------------------------------------------------------


async def call_gemini_with_search(prompt: str, model: str) -> str:
    """Call the shared Gemini transport with Google Search grounding.

    Retry and response handling are centralized in ``gemini_api_call`` so
    concurrent batches do not apply a second, overlapping retry policy.
    """
    return await asyncio.to_thread(
        gemini_api_call,
        prompt=prompt,
        model=model,
        google_search=True,
    )


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def parse_json_response(raw: str, context: str = "") -> list[dict]:
    """Robustly extract a JSON array from a raw LLM response."""
    if not raw:
        return []

    # Step 1: Remove markdown fencing and isolate likely JSON payload.
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()

    start_bracket = text.find("[")
    start_brace = text.find("{")

    if start_bracket == -1 and start_brace == -1:
        logger.warning("No JSON found in response%s", f" for {context}" if context else "")
        return []

    if start_bracket != -1 and (start_brace == -1 or start_bracket < start_brace):
        text = text[start_bracket:]
    else:
        text = text[start_brace:]
        if not text.startswith("["):
            text = f"[{text}]"

    # Step 2: Parse directly via stdlib JSON first.
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else [result]
    except json.JSONDecodeError:
        pass

    # Step 3: Fallback to json_repair for malformed-but-recoverable responses.
    try:
        from json_repair import repair_json  # type: ignore

        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, list):
            return repaired
        if isinstance(repaired, dict):
            return [repaired]
    except ImportError:
        pass
    except Exception as e:
        logger.warning("json_repair failed%s: %s", f" for {context}" if context else "", e)

    return []


def normalize_results(items: list, context: str = "") -> list[dict]:
    """Normalize mixed parsed items into a clean list of dicts."""
    # Step 1: Accept dict/list/JSON-string variants and coerce to list[dict].
    normalized: list[dict] = []

    for item in items:
        if isinstance(item, dict):
            normalized.append(item)
            continue

        if isinstance(item, str):
            candidate = item.strip()
            if not candidate:
                continue

            try:
                reparsed = json.loads(candidate)
            except Exception:
                reparsed = None

            if isinstance(reparsed, dict):
                normalized.append(reparsed)
            elif isinstance(reparsed, list):
                normalized.extend(x for x in reparsed if isinstance(x, dict))
            else:
                logger.warning(
                    "Dropping non-dict string result%s: %s",
                    f" for {context}" if context else "",
                    candidate[:140],
                )
            continue

        if isinstance(item, list):
            normalized.extend(x for x in item if isinstance(x, dict))
            continue

        logger.warning(
            "Dropping unsupported result type%s: %s",
            f" for {context}" if context else "",
            type(item).__name__,
        )

    return normalized


# ---------------------------------------------------------------------------
# Source URL Resolution (Vertex Endpoint Links)
# ---------------------------------------------------------------------------


def _is_vertex_grounding_redirect(url: str) -> bool:
    """Check if URL is a Vertex AI grounding redirect endpoint."""
    parsed = urlparse(url)
    return (
        parsed.netloc in {"vertexaisearch.cloud.google.com", "vertexaisearch.google.com"}
        and parsed.path.startswith("/grounding-api-redirect/")
    )


def _resolve_source_url(url: str, timeout: float = 10.0) -> str:
    """Resolve a URL, following redirects, and cache the final destination."""
    candidate = (url or "").strip()
    if not candidate:
        return ""

    if not _is_vertex_grounding_redirect(candidate):
        return candidate

    with _url_cache_lock:
        cached = _resolved_url_cache.get(candidate)
    if cached:
        return cached

    resolved = candidate
    try:
        request = Request(candidate, headers={"User-Agent": "Mozilla/5.0"})
        opener = build_opener(HTTPRedirectHandler())
        with opener.open(request, timeout=timeout) as response:
            final_url = response.geturl()
            if final_url:
                resolved = final_url
    except Exception as e:
        logger.debug("Could not resolve grounding redirect URL %s: %s", candidate, e)

    with _url_cache_lock:
        _resolved_url_cache[candidate] = resolved

    return resolved


def _normalize_search_sources(value) -> list[str] | str:
    """Resolve and deduplicate source URLs while preserving order."""
    if isinstance(value, str):
        return _resolve_source_url(value)

    if not isinstance(value, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for source in value:
        if not isinstance(source, str):
            continue
        resolved = _resolve_source_url(source)
        if resolved and resolved not in seen:
            seen.add(resolved)
            normalized.append(resolved)
    return normalized


# ---------------------------------------------------------------------------
# Unmet-Need Metadata Attachment
# ---------------------------------------------------------------------------


def _attach_unmet_need_metadata(results: list[dict], unmet_needs: list[dict]) -> list[dict]:
    """Attach canonical unmet-need metadata (including country) from input list.

    Priority:
    1) unmet_need_index (deterministic)
    2) exact (unmet_need, category) match
    """
    needs_by_index: dict[int, dict] = {idx: need for idx, need in enumerate(unmet_needs, 1)}
    needs_by_identity: dict[tuple[str, str], dict] = {}
    for need in unmet_needs:
        identity = (str(need.get("unmet_need", "")), str(need.get("category", "")))
        if identity not in needs_by_identity:
            needs_by_identity[identity] = need

    enriched: list[dict] = []
    for record in results:
        source_need = None

        idx = record.get("unmet_need_index")
        if isinstance(idx, int):
            source_need = needs_by_index.get(idx)

        if source_need is None:
            identity = (str(record.get("unmet_need", "")), str(record.get("category", "")))
            source_need = needs_by_identity.get(identity)

        if source_need:
            record["unmet_need"] = source_need.get("unmet_need")
            record["category"] = source_need.get("category")
            record["country"] = source_need.get("country")

        enriched.append(record)

    return enriched


# ---------------------------------------------------------------------------
# Trial Metadata Attachment
# ---------------------------------------------------------------------------


def _attach_trial_metadata(results: list[dict], trial_batch: list[dict]) -> list[dict]:
    """Attach source trial metadata onto each mapping record.

    This enriches records with molecule_name and phase-related fields while
    enforcing canonical trial_id from the input trial batch.
    """
    trial_lookup: dict[object, dict] = {}
    trial_index_lookup: dict[int, dict] = {}

    def _trial_id_core(value: object) -> str:
        """Conservative matching key for IDs that may lose trailing descriptors."""
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        return text.split("(", 1)[0].strip().lower()

    trial_core_lookup: dict[str, dict] = {}

    # Step 1: Build lookup maps from the source trial batch for fast reconciliation.
    # - exact trial_id lookup
    # - batch index lookup (deterministic primary key)
    # - conservative core-id lookup (fallback when LLM truncates IDs)
    for trial in trial_batch:
        key = trial.get("trial_id")
        if key is not None:
            trial_lookup[key] = trial
            core = _trial_id_core(key)
            if core and core not in trial_core_lookup:
                trial_core_lookup[core] = trial

    for idx, trial in enumerate(trial_batch, 1):
        trial_index_lookup[idx] = trial

    enriched: list[dict] = []
    for record in results:
        # Step 2: Resolve each LLM record back to a source trial using priority:
        # batch index -> exact trial_id -> core trial_id fallback.
        source = None

        batch_idx = record.get("trial_batch_index")
        if isinstance(batch_idx, int):
            source = trial_index_lookup.get(batch_idx)

        if source is None:
            source = trial_lookup.get(record.get("trial_id"))

        if source is None:
            source = trial_core_lookup.get(_trial_id_core(record.get("trial_id")))

        if source:
            # Step 3: Enforce canonical identity and attach deterministic metadata
            # from source data (overrides any LLM drift in identifiers/fields).
            record["trial_id"] = source.get("trial_id")
            record["molecule_name"] = source.get("molecule_name")
            record["phase"] = source.get("phase")
            record["phase_status"] = source.get("phase_status")
            record["trial_size"] = source.get("trial_size")
            record["trial_location"] = source.get("trial_location")

        enriched.append(record)

    return enriched


# ---------------------------------------------------------------------------
# Batch processing pipeline
# ---------------------------------------------------------------------------


async def process_batch(
    batch_index: int,
    trial_batch: list[dict],
    unmet_needs: list[dict],
    model: str,
) -> list[dict]:
    """Run a single LLM call for a batch of trials and return parsed results."""
    # Step 1: Build one prompt for this trial batch and call the LLM.
    trial_ids = [t.get("trial_id", "?") for t in trial_batch]
    logger.info("Batch %d - processing trials: %s", batch_index, trial_ids)

    prompt = build_prompt(trial_batch, unmet_needs)
    raw_response = await call_gemini_with_search(prompt, model=model)

    # Step 2: Parse/normalize model output and reconcile to source unmet-need metadata.
    parsed_results = parse_json_response(raw_response, context=f"batch {batch_index} ({trial_ids})")
    results = normalize_results(parsed_results, context=f"batch {batch_index} ({trial_ids})")

    # Step 3: Enforce canonical unmet-need fields including country from input list.
    results = _attach_unmet_need_metadata(results, unmet_needs)

    # Step 4: Reconcile to source trial metadata.
    results = _attach_trial_metadata(results, trial_batch)

    logger.info("Batch %d - found %d trial-unmet_need mappings", batch_index, len(results))
    return results


async def run_pipeline(
    trials_df: pd.DataFrame,
    unmet_needs_df: pd.DataFrame,
    model: str,
) -> list[dict]:
    """Run the full batched mapping pipeline with semaphore-limited parallelism."""
    # Step 1: Convert inputs to records and create fixed-size trial batches.
    trials = trials_df.to_dict(orient="records")
    unmet_needs = unmet_needs_df.to_dict(orient="records")

    batches = [
        trials[i : i + TRIAL_BATCH_SIZE]
        for i in range(0, len(trials), TRIAL_BATCH_SIZE)
    ]

    logger.info(
        "Starting pipeline: %d trials -> %d batches (batch size=%d, max_parallel=%d)",
        len(trials),
        len(batches),
        TRIAL_BATCH_SIZE,
        MAX_PARALLEL_BATCHES,
    )

    semaphore = asyncio.Semaphore(MAX_PARALLEL_BATCHES)

    async def _run_batch(batch_index: int, trial_batch: list[dict]) -> list[dict]:
        """Execute one batch under semaphore control and isolate failures per batch."""
        async with semaphore:
            try:
                return await process_batch(batch_index, trial_batch, unmet_needs, model=model)
            except Exception as e:
                trial_ids = [t.get("trial_id", "?") for t in trial_batch]
                logger.error("Batch %d failed for trials %s: %s", batch_index, trial_ids, e)
                return []

    # Step 2: Run batches concurrently with per-batch fault isolation.
    tasks = [
        asyncio.create_task(_run_batch(batch_index, trial_batch))
        for batch_index, trial_batch in enumerate(batches, 1)
    ]

    batch_results_list = await asyncio.gather(*tasks)

    # Step 3: Flatten successful batch outputs into one result list.
    all_results: list[dict] = []
    for batch_results in batch_results_list:
        all_results.extend(batch_results)

    return all_results


# ---------------------------------------------------------------------------
# Output normalization
# ---------------------------------------------------------------------------


def prepare_mapping_rows(results: list[dict]) -> list[dict]:
    """Normalize mapping rows for downstream scoring and BigQuery export."""
    # Step 1: Normalize records and clean/resolve source links.
    clean_results = normalize_results(results, context="prepare_mapping_rows")

    transformed_records = 0
    for record in clean_results:
        if "search_sources_used" not in record:
            continue
        original = record.get("search_sources_used")
        normalized = _normalize_search_sources(original)
        if normalized != original:
            transformed_records += 1
        record["search_sources_used"] = normalized

    if transformed_records:
        logger.info("Resolved/cleaned source links in %d records", transformed_records)

    return clean_results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def map_clinical_trial_data_with_unmet_needs(model: str, molecule_name: str, incremental: bool = True) -> dict:
    """Run end-to-end mapping and return artifacts for downstream steps.

    Args:
        model: Gemini model name to use for mapping.
        molecule_name: Name of the molecule to process.
        incremental: When True (default), only processes trials newer than the
            latest mapping already stored in BigQuery. Set to False to reprocess all trials.
    """
    # Step 1: Load source datasets from BigQuery.
    unmet_needs_df = load_unmet_needs()
    trials_df = load_clinical_trials(molecule_name=molecule_name, incremental=incremental)

    # Step 2: Short-circuit when there are no valid trial IDs to process.
    assessed_trial_ids = {
        str(tid).strip()
        for tid in trials_df.get("trial_id", pd.Series(dtype="object")).tolist()
        if str(tid).strip()
    }
    if not assessed_trial_ids:
        logger.info(
            "No clinical trials with valid trial_id to assess for molecule '%s'. Skipping mapping and export.",
            molecule_name,
        )
        return {
            "mapping_rows": [],
            "clinical_trials_assessed": False,
            "clinical_trials_assessed_count": 0,
        }

    # Step 3: Execute batched trial-to-unmet-need mapping.
    results = await run_pipeline(trials_df, unmet_needs_df, model=model)

    logger.info("Pipeline complete - total mappings found: %d", len(results))
    # Step 4: Normalize rows for downstream scoring and BQ export.
    clean_results = prepare_mapping_rows(results)

    # Step 5: Delete existing rows for any trial_id that appears in this run,
    # so re-processed trials don't leave stale duplicates in the table.
    trial_ids_in_run = list({row["trial_id"] for row in clean_results if row.get("trial_id")})
    if trial_ids_in_run and clean_results:
        bq_client = get_bq_client()
        delete_existing_mappings_for_trial_ids(bq_client, trial_ids_in_run)

    # Step 6: Export mapping rows to BigQuery (clinical_trial_evidence_to_unmet_need_mapping).
    write_disposition = (
        _bq.WriteDisposition.WRITE_APPEND
        if incremental
        else _bq.WriteDisposition.WRITE_TRUNCATE
    )
    logger.info(
        "Exporting %d rows to BigQuery clinical evidence to unmet needs mapping table with (write_disposition=%s)",
        len(clean_results),
        write_disposition,
    )
    export_to_gbq(
        rows=clean_results,
        table_name=TRIAL_MAPPING_TABLE,
        schema=TRIAL_MAPPING_SCHEMA,
        write_disposition=write_disposition,
    )

    return {
        "mapping_rows": clean_results,
        "clinical_trials_assessed": True,
        "clinical_trials_assessed_count": len(assessed_trial_ids),
    }
