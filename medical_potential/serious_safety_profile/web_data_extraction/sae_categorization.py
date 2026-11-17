"""SAE categorization helpers: summarization and classification."""

import asyncio
import json
import logging
from typing import Optional
from datetime import datetime, timezone

import pandas as pd

from ..llm_runtime import gemini_call_with_search, safe_json_parse
from .prompts import SAE_EXPECTEDNESS_AND_CATEGORY_PROMPT_TEMPLATE
from .utils import load_serious_safety_trials_data_from_bq, _normalize_trial_for_scoring
from medical_potential.gcp_utils import get_bq_client
from google.cloud import bigquery as _bq
from medical_potential.config import PROJECT_ID, BQ_DATASET_ID, SERIOUS_SAFETY_PROFILE_SAE_SUMMARY

logger = logging.getLogger(__name__)


def summarize_sae_across_trials(trials: list[dict]):
    """Summarize SAEs across the supplied `trials` list.

    This function expects the `trials` list returned by `extract_serious_safety_data()`
    (i.e., `extraction.get("trials", [])`). 

    Returns a list of dicts: {sae_name, number_of_studies, list_of_studies}.
    """
    if not trials:
        return []

    sae_map = {}
    for idx, trial in enumerate(trials):
        trial_id = trial.get("Trial ID") or trial.get("NCT_ID") or f"Trial {idx+1}"
        events = trial.get("Serious Events Reported", [])
        # If events is a dict, convert to list of names
        if isinstance(events, dict):
            events = list(events.keys())
        for sae in events:
            if isinstance(sae, dict):
                sae_name = sae.get("name") or sae.get("event_name") or str(sae)
            else:
                sae_name = str(sae)
            if not sae_name:
                continue
            if sae_name not in sae_map:
                sae_map[sae_name] = {
                    "sae_name": sae_name,
                    "number_of_studies": 0,
                    "list_of_studies": [],
                }
            if trial_id not in sae_map[sae_name]["list_of_studies"]:
                sae_map[sae_name]["number_of_studies"] += 1
                sae_map[sae_name]["list_of_studies"].append(trial_id)
    return list(sae_map.values())


def update_existing_sae_events_in_table(molecule_name: str, sae_summary: list[dict]):
    """Update existing SAE rows in the SERIOUS_SAFETY_PROFILE_SAE_SUMMARY table.

    For each SAE in `sae_summary`, check if it exists for `molecule_name`.
    - If exists: increment `number_of_studies`, append new `list_of_studies` (deduplicated),
      and set `updated_at` = CURRENT_TIMESTAMP().
    - If not exists: leave for insertion by caller.

    Returns a tuple: (existing_sae_names, new_sae_names)
    """
    client = get_bq_client()
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{SERIOUS_SAFETY_PROFILE_SAE_SUMMARY}"

    # Fetch existing SAEs for molecule
    sql = f"""
        SELECT sae_name, number_of_studies, list_of_studies
        FROM `{table_id}`
        WHERE LOWER(molecule_name) = @molecule
    """
    job_config = _bq.QueryJobConfig(
        query_parameters=[_bq.ScalarQueryParameter("molecule", "STRING", molecule_name.strip().lower())]
    )
    rows = [dict(r.items()) for r in client.query(sql, job_config=job_config).result()]

    existing_map = {str(r.get("sae_name", "")).strip().lower(): r for r in rows}

    existing = []
    new = []

    for item in sae_summary:
        sae_name = str(item.get("sae_name", "")).strip()
        if not sae_name:
            continue
        key = sae_name.lower()
        incoming_count = int(item.get("number_of_studies", 0) or 0)
        incoming_list = list(item.get("list_of_studies") or [])

        if key in existing_map:
            row = existing_map[key]
            current_count = int(row.get("number_of_studies", 0) or 0)
            current_list = list(row.get("list_of_studies") or [])

            # Merge current and incoming study lists, deduplicate while preserving order
            merged_list = list(dict.fromkeys(current_list + incoming_list))
            # Compute updated count from the unique merged list (handles overlaps)
            updated_count = len(merged_list)

            # Use explicit UTC timestamp via CURRENT_TIMESTAMP() which is in UTC in BigQuery
            update_sql = f"""
                UPDATE `{table_id}`
                SET number_of_studies = @num, list_of_studies = @list, updated_at = CURRENT_TIMESTAMP()
                WHERE LOWER(molecule_name) = @molecule AND LOWER(sae_name) = @sae_name
            """
            update_job_config = _bq.QueryJobConfig(
                query_parameters=[
                    _bq.ScalarQueryParameter("num", "INT64", int(updated_count)),
                    _bq.ArrayQueryParameter("list", "STRING", merged_list),
                    _bq.ScalarQueryParameter("molecule", "STRING", molecule_name.strip().lower()),
                    _bq.ScalarQueryParameter("sae_name", "STRING", sae_name.lower()),
                ]
            )
            client.query(update_sql, job_config=update_job_config).result()
            existing.append(item)
        else:
            new.append(item)

    return existing, new


def update_new_sae_events_to_summary_table(molecule_name: str, classified_sae_events: list[dict]):
    """Insert newly classified SAE events into the SAE summary table.

    Args:
        molecule_name: molecule name string
        classified_sae_events: list of dicts returned by classify_sae_expectedness_and_category()

    Returns:
        inserted_items: list of dicts that were inserted (same shape as input)
    """
    if not classified_sae_events:
        return []

    client = get_bq_client()
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{SERIOUS_SAFETY_PROFILE_SAE_SUMMARY}"

    rows_to_insert = []
    for item in classified_sae_events:
        sae_name = str(item.get("sae_name", "")).strip()
        if not sae_name:
            continue
        num = int(item.get("number_of_studies", 0) or 0)
        lst = list(item.get("list_of_studies") or [])
        exp_cls = item.get("expectedness_classification")
        exp_rat = item.get("expectedness_rationale")
        sev_cat = item.get("severity_category")
        sev_rat = item.get("severity_rationale")
        sev_orig = item.get("severity_category_original")
        post_proc = item.get("severity_category_post_processed")
        # Ensure boolean value for schema
        post_proc_bool = bool(post_proc) if post_proc is not None else False
        post_note = item.get("severity_category_post_process_note")

        rows_to_insert.append(
            {
                "molecule_name": molecule_name,
                "sae_name": sae_name,
                "number_of_studies": num,
                "list_of_studies": lst,
                "expectedness_classification": exp_cls,
                "expectedness_rationale": exp_rat,
                "severity_category": sev_cat,
                "severity_rationale": sev_rat,
                "severity_category_original": sev_orig,
                "severity_category_post_processed": post_proc_bool,
                "severity_category_post_process_note": post_note,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
        )

    if not rows_to_insert:
        return []

    try:
        df = pd.DataFrame(rows_to_insert)

        load_job_config = _bq.LoadJobConfig(write_disposition=_bq.WriteDisposition.WRITE_APPEND)

        job = client.load_table_from_dataframe(df, table_id, job_config=load_job_config)
        job.result()

        # Return input items that correspond to inserted rows (preserve original structure)
        return classified_sae_events
    except Exception:
        logger.exception("[SAE_INSERT] Failed to bulk insert SAE events for %s", molecule_name)
        return []


def _normalize_expectedness(value: str) -> str:
    v = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    prompt_categories = {
        "unrelated": "unrelated",
        "expected": "expected",
        "unexpected": "unexpected",
        "expected_but_rare": "expected_but_rare",
        "others": "others",
    }
    if v in prompt_categories:
        return prompt_categories[v]

    if "rare" in v:
        return "expected_but_rare"
    if "unexpected" in v or "not_expected" in v:
        return "unexpected"
    if "other" in v:
        return "others"
    return "expected"


def _normalize_severity_category(value: str) -> str:
    v = str(value or "").strip().lower()
    if (
        "catastrophic" in v
        or "death" in v
        or "permanent" in v
        or "irreversible" in v
    ):
        return "Catastrophic"
    if (
        "severe" in v
        or "life-threatening" in v
        or "stroke" in v
        or "heart attack" in v
        or "respiratory failure" in v
        or "allergic reaction" in v
    ):
        return "Severe / Life-Threatening"
    if "organ" in v:
        return "Organ-Specific"
    if "manageable" in v or "non-specific" in v:
        return "Manageable / Non-specific"
    if "other" in v:
        return "Others"
    return "Others"


def _post_process_severity_category(sae_name: str, severity_category: str) -> tuple[str, bool, str, str]:
    """Apply deterministic category corrections for known LLM failure modes.

    Current hardcoded rule:
    - If severity is Catastrophic but SAE name indicates cancer/malignancy and does not
      explicitly indicate death/fatal/permanent irreversible outcome, downgrade to
      Severe / Life-Threatening.
    """
    original = str(severity_category or "").strip()
    name = str(sae_name or "").strip().lower()

    if original != "Catastrophic":
        return original, False, "", original

    cancer_terms = (
        "cancer",
        "carcinoma",
        "malignan",
        "neoplasm",
        "metastatic",
        "tumor",
        "tumour",
        "adenocarcinoma",
    )
    explicit_catastrophic_terms = (
        "death",
        "fatal",
        "died",
        "mortality",
        "permanent disability",
        "irreversible",
        "end-stage",
        "end stage",
        "organ failure",
    )

    is_cancer_like = any(term in name for term in cancer_terms)
    has_explicit_catastrophic_signal = any(term in name for term in explicit_catastrophic_terms)

    if is_cancer_like and not has_explicit_catastrophic_signal:
        corrected = "Severe / Life-Threatening"
        note = (
            "Post-processing override applied: remapped Catastrophic to "
            "Severe / Life-Threatening for cancer-like SAE without explicit "
            "death/fatal/permanent irreversible outcome."
        )
        return corrected, True, note, original

    return original, False, "", original


def _classification_error(sae_item: dict, error_details: str, raw_response: str = "") -> dict:
    sae_name = str(sae_item.get("sae_name", "")).strip() or "Unknown SAE"
    message = f"Grounded model response could not be used; {error_details}; manual review required."
    return {
        "sae_name": sae_name,
        "number_of_studies": sae_item.get("number_of_studies", 0),
        "list_of_studies": sae_item.get("list_of_studies", []),
        "expectedness_classification": None,
        "expectedness_rationale": message,
        "severity_category": None,
        "severity_rationale": message,
        "severity_category_original": None,
        "severity_category_post_processed": False,
        "severity_category_post_process_note": "",
        "gemini_error": error_details,
        "parse_error": True,
        "raw_response": raw_response,
    }


async def _classify_sae_event_batch(
    sae_items: list[dict],
    molecule_class: str,
    semaphore: asyncio.Semaphore,
    model_name: str,
) -> list[dict]:
    sae_events = [
        {
            "sae_name": sae_name,
            "number_of_studies": item.get("number_of_studies", 0),
            "list_of_studies": list(item.get("list_of_studies") or []),
        }
        for item in sae_items
        for sae_name in [str(item.get("sae_name", "")).strip()]
        if sae_name
    ]
    sae_names = ", ".join(event["sae_name"] for event in sae_events)

    prompt = SAE_EXPECTEDNESS_AND_CATEGORY_PROMPT_TEMPLATE.format(
        molecule_class=molecule_class,
        sae_events_json=json.dumps(sae_events, ensure_ascii=True),
    )

    try:
        async with semaphore:
            raw_response = await gemini_call_with_search(prompt=prompt, model=model_name)
    except Exception as exc:
        error_msg = f"SAE classification failed for batch [{sae_names}] ({type(exc).__name__}): {exc}"
        logger.error("[SAE_CLASSIFY] %s", error_msg)
        raise RuntimeError(error_msg) from exc

    parsed = safe_json_parse(raw_response, context=f"sae_classification_batch:{sae_names}")
    if not isinstance(parsed, list) or len(parsed) != len(sae_items) or not all(isinstance(item, dict) for item in parsed):
        parse_details = (
            f"Expected JSON list with {len(sae_items)} objects, got {type(parsed).__name__} "
            f"with length {len(parsed) if isinstance(parsed, list) else 'n/a'}; "
            f"raw_response_preview={str(raw_response)[:300]}"
        )
        return [_classification_error(item, parse_details, raw_response) for item in sae_items]

    results = []
    for source_item, parsed_item in zip(sae_items, parsed):
        sae_name = str(source_item.get("sae_name", "")).strip() or "Unknown SAE"
        normalized_severity = _normalize_severity_category(parsed_item.get("severity_category", ""))
        final_severity, post_processed, post_process_note, original_severity = _post_process_severity_category(
            sae_name=sae_name,
            severity_category=normalized_severity,
        )
        results.append({
            "sae_name": sae_name,
            "number_of_studies": source_item.get("number_of_studies", 0),
            "list_of_studies": source_item.get("list_of_studies", []),
            "expectedness_classification": _normalize_expectedness(parsed_item.get("expectedness_classification", "")),
            "expectedness_rationale": str(parsed_item.get("expectedness_rationale", "")).strip(),
            "severity_category": final_severity,
            "severity_rationale": str(parsed_item.get("severity_rationale", "")).strip(),
            "severity_category_original": original_severity,
            "severity_category_post_processed": post_processed,
            "severity_category_post_process_note": post_process_note,
            "parse_error": False,
            "raw_response": raw_response,
        })
    return results


async def classify_sae_expectedness_and_category(
    sae_study_summary: list[dict],
    model_name: str,
    molecule_class: str = "GLP-1",
    max_concurrency: int = 1,
    batch_size: int = 2,
) -> list[dict]:
    """Classify SAE expectedness and severity category using grounded Gemini calls.

    Runs one grounded async call per batch in parallel with a semaphore cap.
    """
    if not sae_study_summary:
        logger.info("[SAE_CLASSIFY] No SAE events provided. Skipping classification.")
        return []

    sae_study_summary = [
        item for item in sae_study_summary
        if str(item.get("sae_name", "")).strip()
    ]
    if not sae_study_summary:
        logger.info("[SAE_CLASSIFY] No named SAE events provided. Skipping classification.")
        return []

    concurrency = max(1, int(max_concurrency))
    batch_size = max(1, int(batch_size))
    sae_batches = [
        sae_study_summary[index:index + batch_size]
        for index in range(0, len(sae_study_summary), batch_size)
    ]
    total_items = len(sae_study_summary)
    logger.info(
        "[SAE_CLASSIFY] Starting expectedness/severity classification: total_events=%d, batches=%d, batch_size=%d, molecule_class=%s, max_concurrency=%d",
        total_items,
        len(sae_batches),
        batch_size,
        molecule_class,
        concurrency,
    )

    semaphore = asyncio.Semaphore(concurrency)

    async def _classify_with_context(batch: list[dict]):
        try:
            result = await _classify_sae_event_batch(
                batch,
                molecule_class=molecule_class,
                semaphore=semaphore,
                model_name=model_name,
            )
            return batch, result, None
        except Exception as exc:
            return batch, None, exc

    tasks = [
        asyncio.create_task(_classify_with_context(batch))
        for batch in sae_batches
    ]

    results = []
    completed = 0
    success_count = 0
    failure_count = 0
    parse_error_count = 0

    for future in asyncio.as_completed(tasks):
        batch, batch_results, task_error = await future
        completed += len(batch)

        if task_error is not None:
            error_details = f"{type(task_error).__name__}: {task_error}"
            failure_count += len(batch)
            results.extend(_classification_error(item, f"Grounded call failed with error: {error_details}") for item in batch)
        else:
            success_count += len(batch)
            parse_error_count += sum(1 for result in batch_results if result.get("parse_error"))
            results.extend(batch_results)

        logger.info(
            "[SAE_CLASSIFY] Progress: %d/%d completed (%.1f%%)",
            completed,
            total_items,
            (completed / total_items) * 100,
        )

    logger.info(
        "[SAE_CLASSIFY] Completed classification: total=%d, success=%d, failures=%d, parse_errors=%d",
        total_items,
        success_count,
        failure_count,
        parse_error_count,
    )

    return results
