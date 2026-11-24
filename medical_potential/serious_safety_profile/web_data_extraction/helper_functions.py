"""Serious Safety Profile – Dimension 5: Core Functions."""

import ast
import asyncio
import logging
import re
from typing import Optional

from ..llm_runtime import gemini_call_with_search, safe_json_parse
from .clinical_trial_extractor import extract_clinical_trial_data

from .prompts import (
    SERIOUS_SAFETY_EXTRACTION_PROMPT,
    SERIOUS_SAFETY_JSON_FIELDS,
    REGULATORY_IMPACT_PROMPT_TEMPLATE,
    APPROVAL_MARKET_STATUS_PROMPT_TEMPLATE,
    POST_MARKETING_SAFETY_PROMPT_TEMPLATE,
)

from .utils import (
                load_serious_safety_trials_data_from_bq, 
                _normalize_trial_for_scoring
                )

logger = logging.getLogger(__name__)

from medical_potential.gcp_utils import get_bq_client
from google.cloud import bigquery as _bq
from medical_potential.config import (
    PROJECT_ID,
    BQ_DATASET_ID,
    CLINICAL_EFFICACY_TABLE,
    CLINICAL_TRIALS_SERIOUS_SAFETY_DATA,
    TRIAL_LLM_CONFIDENCE_THRESHOLD,
)



# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _parse_int(value) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.upper() in {"N/A", "NONE", "NULL", "UNKNOWN"}:
        return None
    try:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        cleaned = re.sub(r"[^\d]", "", s)
        return int(cleaned) if cleaned else None
    except (TypeError, ValueError):
        return None


def _parse_float(value) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.upper() in {"N/A", "NONE", "NULL", "UNKNOWN"}:
        return None
    try:
        if isinstance(value, (int, float)):
            return float(value)
        cleaned = re.sub(r"[^\d.]", "", s)
        return float(cleaned) if cleaned else None
    except (TypeError, ValueError):
        return None


def _calculate_sae_rates(trial: dict) -> None:
    """Compute SAE rates and delta SAE from counts when rates are missing."""
    drug_rate = _parse_float(trial.get("SAE Rate Drug (%)"))
    control_rate = _parse_float(trial.get("SAE Rate Control (%)"))

    drug_n = _parse_int(trial.get("SAE Drug (N)"))
    control_n = _parse_int(trial.get("SAE Control (N)"))
    drug_arm_n = _parse_int(trial.get("Drug Arm Size (N)"))
    control_arm_n = _parse_int(trial.get("Control Arm Size (N)"))

    if drug_rate is None and drug_n is not None and drug_arm_n and drug_arm_n > 0:
        drug_rate = round((drug_n / drug_arm_n) * 100, 2)
    if control_rate is None and control_n is not None and control_arm_n and control_arm_n > 0:
        control_rate = round((control_n / control_arm_n) * 100, 2)

    trial["SAE Rate Drug (%)"] = drug_rate
    trial["SAE Rate Control (%)"] = control_rate
    trial["Delta SAE (%)"] = (
        round(drug_rate - control_rate, 2)
        if (drug_rate is not None and control_rate is not None)
        else None
    )


def _get_latest_serious_safety_timestamp(client, molecule_name: str):
    """Return the latest update timestamp for a molecule from the serious safety table, or None."""
    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{CLINICAL_TRIALS_SERIOUS_SAFETY_DATA}"
    sql = f"""
        SELECT MAX(created_at) AS latest_ts
        FROM `{table_id}`
        WHERE LOWER(TRIM(molecule_name)) = @molecule_name
    """
    job_config = _bq.QueryJobConfig(
        query_parameters=[_bq.ScalarQueryParameter("molecule_name", "STRING", molecule_name)]
    )
    result = client.query(sql, job_config=job_config).result()
    row = next(iter(result), None)
    if row is None or getattr(row, "latest_ts", None) is None:
        return None
    return row.latest_ts


def extract_incremental_trials(molecule_name: str, incremental_mode: bool = True) -> list:
    """
    Return clinical trials for a molecule.

    When incremental_mode is True, only trials created or updated after the
    latest serious safety processing timestamp are returned. Otherwise, all
    trials for the molecule are returned.
    """
    if not molecule_name or not molecule_name.strip():
        raise ValueError("molecule_name must be provided")

    try:
        client = get_bq_client()
    except Exception:
        logger.exception("BigQuery auth failed")
        return []

    molecule_safe = molecule_name.strip().lower()
    clinical_table = f"{PROJECT_ID}.{BQ_DATASET_ID}.{CLINICAL_EFFICACY_TABLE}"

    # Full mode: return all trials for molecule
    if not incremental_mode:
        logger.info("[Incremental] Full mode: fetching all trials for %s", molecule_name)
        sql = f"""
            SELECT DISTINCT molecule_name, dosage, trial_id, phase, trial_location, secondary_locations, trial_size
            FROM `{clinical_table}`
            WHERE LOWER(molecule_name) = @molecule
            AND llm_confidence > {TRIAL_LLM_CONFIDENCE_THRESHOLD}
            ORDER BY trial_id
        """
        job_config = _bq.QueryJobConfig(query_parameters=[_bq.ScalarQueryParameter("molecule", "STRING", molecule_safe)])
        try:
            rows = [dict(r.items()) for r in client.query(sql, job_config=job_config).result()]
            return [
                {
                    "molecule_name": r.get("molecule_name"),
                    "dosage": r.get("dosage"),
                    "trial_id": r.get("trial_id"),
                    "phase": r.get("phase"),
                    "trial_location": r.get("trial_location"),
                    "secondary_locations": r.get("secondary_locations"),
                    "trial_size": r.get("trial_size"),
                }
                for r in rows
            ]
        except Exception:
            logger.exception("BigQuery query error (full mode) for %s", molecule_name)
            return []

    # Incremental mode - find last serious safety update timestamp
    try:
        last_update = _get_latest_serious_safety_timestamp(client, molecule_safe)
    except Exception:
        logger.exception("Failed to read last update from serious safety table for %s", molecule_name)
        last_update = None

    # If no prior update, treat as first run and return all trials
    if not last_update:
        logger.info("No prior serious-safety update found for %s - returning full trial set", molecule_name)
        return extract_incremental_trials(molecule_name, incremental_mode=False)

    # Fetch trials created/updated after last_update
    try:
        sql_fetch = f"""
            SELECT DISTINCT 
                molecule_name, 
                dosage, 
                trial_id, 
                phase, 
                trial_location, 
                secondary_locations, 
                trial_size
            FROM 
                `{clinical_table}`
            WHERE 
                LOWER(molecule_name) = @molecule
                AND llm_confidence > {TRIAL_LLM_CONFIDENCE_THRESHOLD}
                AND (
                    CAST(created_at AS TIMESTAMP) > @cutoff_ts
                    OR CAST(updated_at AS TIMESTAMP) > @cutoff_ts
                )
            ORDER BY 
                trial_id
        """
        job_config_fetch = _bq.QueryJobConfig(query_parameters=[
            _bq.ScalarQueryParameter("molecule", "STRING", molecule_safe),
            _bq.ScalarQueryParameter("cutoff_ts", "TIMESTAMP", last_update),
        ])
        fetched = [dict(r.items()) for r in client.query(sql_fetch, job_config=job_config_fetch).result()]

        if not fetched:
            logger.info("No clinical trials added/updated since last serious-safety update for %s", molecule_name)
            return []

        rows_out = [
            {
                "molecule_name": r.get("molecule_name"),
                "dosage": r.get("dosage"),
                "trial_id": r.get("trial_id"),
                "phase": r.get("phase"),
                "trial_location": r.get("trial_location"),
                "secondary_locations": r.get("secondary_locations"),
                "trial_size": r.get("trial_size"),
            }
            for r in fetched
        ]
        logger.info("Retrieved %d incremental trials for %s", len(rows_out), molecule_name)
        return rows_out
    except Exception:
        logger.exception("BigQuery fetch error for incremental trials for %s", molecule_name)
        return []


def _parse_serious_events_field(raw) -> list[dict]:
    """Parse the 'Serious Events Reported' field into a list of {name, severity_category} dicts.

    Expects Gemini's dict format: {event_name: {Severity Category: ..., ...}}.
    Returns an empty list for any other type.
    """
    if isinstance(raw, str):
        cleaned = raw.strip()
        if not cleaned or cleaned in {"{}", "[]", "null", "None", "N/A"}:
            return []

        if (cleaned.startswith("'") and cleaned.endswith("'")) or (
            cleaned.startswith('"') and cleaned.endswith('"')
        ):
            cleaned = cleaned[1:-1].strip()

        try:
            # Convert pseudo-dict text to a valid Python dict string.
            normalized = re.sub(r"([{,]\s*)([^:{},]+?)\s*:", r'\1"\2":', cleaned)
            normalized = re.sub(
                r":\s*([A-Za-z][A-Za-z\s\-]+)(?=\s*[,}])",
                lambda m: ': "' + m.group(1).strip() + '"',
                normalized,
            )
            parsed_raw = ast.literal_eval(normalized)
            raw = parsed_raw if isinstance(parsed_raw, dict) else {}
        except (SyntaxError, ValueError):
            return []

    if not isinstance(raw, dict):
        return []

    events = []
    for event_name, details in raw.items():
        if isinstance(details, dict):
            sev = (
                details.get("Severity Category")
                or details.get("severity_category")
                or details.get("severity")
                or "Unknown"
            )
        else:
            sev = "Unknown"

        name = str(event_name).strip()
        if not name:
            continue
        events.append({"name": name, "severity_category": str(sev).strip()})
    return events


def _build_trial_summary(trials: list) -> str:
    """Build a compact text summary of trial safety data for prompt injection."""
    lines = []
    for i, trial in enumerate(trials[:10], 1):
        trial_id = trial.get("Trial ID") or trial.get("NCT_ID") or f"Trial {i}"
        phase = trial.get("Phase", "")
        sae_drug = trial.get("SAE Rate Drug (%)")
        sae_ctrl = trial.get("SAE Rate Control (%)")
        delta = trial.get("Delta SAE (%)")
        events = trial.get("Serious Events Reported", "")
        deaths = trial.get("Deaths Reported", "Unknown")
        drug_deaths = trial.get("Drug-Related Deaths", "Unknown")

        lines.append(
            f"  - {trial_id} (Phase {phase}): "
            f"SAE Drug={sae_drug}%, SAE Control={sae_ctrl}%, Delta={delta}%, "
            f"Deaths={deaths}, Drug-Related Deaths={drug_deaths}, "
            f"Events=[{events}]"
        )
    return "\n".join(lines) if lines else "  (no trial data provided)"


# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================

async def extract_serious_safety_data(
    molecule_name: str,
    extraction_model: str,
    max_trials: int = None,
    incremental_trials: list[dict] | None = None,
) -> dict:
    """Extract trial data with serious-safety fields for Dimension 5."""
    result = await extract_clinical_trial_data(
        molecule_name=molecule_name,
        extraction_model=extraction_model,
        max_trials=max_trials,
        extra_fields_prompt=SERIOUS_SAFETY_EXTRACTION_PROMPT,
        extra_fields_json=SERIOUS_SAFETY_JSON_FIELDS,
        base_trials=incremental_trials,
    )

    for trial in result.get("trials", []):
        _calculate_sae_rates(trial)

    return result


async def identify_regulatory_impact(
    molecule_name: str,
    model_name: str,
) -> dict:
    """Step 7: Identify the likely regulatory impact of a molecule's safety profile.

    Uses Google Search grounding to look up approved labels, regulatory assessment
    reports, safety communications and announcements, then synthesises findings into
    a structured regulatory impact assessment with a detailed rationale.

    Args:
        molecule_name: The drug / molecule name (e.g. "semaglutide").
        trials: List of extracted trial dicts from extract_serious_safety_data.

    Returns:
        dict with keys:
            regulatory_consequence  – highest-severity regulatory action found
            consequence_level       – integer 0-6
            score_modifier          – float penalty (0.0 = none, 1.0 = maximum)
            rationale               – detailed explanation linking trial findings to
                                      the regulatory consequence
            key_safety_drivers      – list of the primary safety findings driving concern
            grounding_summary       – brief summary of known public regulatory status
            raw_response            – raw Gemini text (for debugging)
    """
    # Load persisted trials and normalise for the regulator impact prompt
    persisted = load_serious_safety_trials_data_from_bq(molecule_name)
    trials = [_normalize_trial_for_scoring(r) for r in persisted]
    for trial in trials:
        _calculate_sae_rates(trial)

    trial_summary = _build_trial_summary(trials)

    prompt = REGULATORY_IMPACT_PROMPT_TEMPLATE.format(
        molecule_name=molecule_name,
        trial_summary=trial_summary,
    )

    raw_response = await gemini_call_with_search(prompt=prompt, model=model_name, step_name="step9_regulatory_impact")

    parsed = safe_json_parse(raw_response, context="regulatory_impact")

    if not isinstance(parsed, dict):
        print(f"  Could not parse regulatory impact JSON – returning raw response.")
        return {
            "regulatory_consequence": "Unknown",
            "rationale": raw_response[:500] if raw_response else "",
            "key_safety_drivers": [],
            "grounding_summary": "",
            "raw_response": raw_response,
        }

    parsed.setdefault("raw_response", raw_response)
    return parsed


async def determine_approval_market_status(
    molecule_name: str,
    model_name: str,
) -> dict:
    """Step 0: Determine whether a molecule is approved and marketed."""
    response = {
        "checked": True,
        "is_approved": False,
        "is_marketed": False,
        "status_summary": "",
        "major_markets": [],
        "parse_error": False,
        "parse_error_stage": None,
        "raw_status_response": "",
    }

    status_prompt = APPROVAL_MARKET_STATUS_PROMPT_TEMPLATE.format(
        molecule_name=molecule_name,
    )

    status_raw = await gemini_call_with_search(prompt=status_prompt, model=model_name, step_name="step0_approval_market_status")
    response["raw_status_response"] = status_raw
    status = safe_json_parse(status_raw, context="approval_market_status")

    if not isinstance(status, dict):
        response["parse_error"] = True
        response["parse_error_stage"] = "approval_market_status"
        response["status_summary"] = "Unable to determine approval/marketed status due to JSON parse failure"
        return response

    response["is_approved"] = bool(status.get("is_approved", False))
    response["is_marketed"] = bool(status.get("is_marketed", False))
    response["status_summary"] = status.get("status_summary", "")
    response["major_markets"] = status.get("major_markets", [])
    return response


async def identify_post_marketing_safety(
    molecule_name: str,
    model_name: str,
) -> dict:
    """Assess post-marketing serious safety data for marketed drugs."""
    response = {
        "evaluated": False,
        "parse_error": False,
        "parse_error_stage": None,
        "status_summary": "",
        "new_serious_risks": None,
        "new_serious_risks_rationale": None,
        "stronger_warning_added": None,
        "stronger_warning_added_rationale": None,
        "rems_added": None,
        "rems_added_rationale": None,
        "boxed_warning_added": None,
        "boxed_warning_added_rationale": None,
        "major_label_restriction": None,
        "major_label_restriction_rationale": None,
        "withdrawal_from_market": None,
        "withdrawal_from_market_rationale": None,
        "risk_level": "unknown",
        "key_findings": [],
        "grounding_summary": "",
        "sources_summary": [],
        "raw_post_marketing_response": "",
    }

    post_prompt = POST_MARKETING_SAFETY_PROMPT_TEMPLATE.format(
        molecule_name=molecule_name,
    )
    post_raw = await gemini_call_with_search(prompt=post_prompt, model=model_name, step_name="step10_post_marketing_safety")
    response["raw_post_marketing_response"] = post_raw
    post = safe_json_parse(post_raw, context="post_marketing_safety")

    if not isinstance(post, dict):
        response["evaluated"] = True
        response["parse_error"] = True
        response["parse_error_stage"] = "post_marketing_safety"
        response["status_summary"] = "Failed to parse post-marketing safety details JSON"
        return response

    response["evaluated"] = True
    response["new_serious_risks"] = post.get("new_serious_risks")
    response["new_serious_risks_rationale"] = post.get("new_serious_risks_rationale")
    response["stronger_warning_added"] = post.get("stronger_warning_added")
    response["stronger_warning_added_rationale"] = post.get("stronger_warning_added_rationale")
    response["rems_added"] = post.get("rems_added")
    response["rems_added_rationale"] = post.get("rems_added_rationale")
    response["boxed_warning_added"] = post.get("boxed_warning_added")
    response["boxed_warning_added_rationale"] = post.get("boxed_warning_added_rationale")
    response["major_label_restriction"] = post.get("major_label_restriction")
    response["major_label_restriction_rationale"] = post.get("major_label_restriction_rationale")
    response["withdrawal_from_market"] = post.get("withdrawal_from_market")
    response["withdrawal_from_market_rationale"] = post.get("withdrawal_from_market_rationale")
    response["risk_level"] = post.get("risk_level", "unknown")
    response["key_findings"] = post.get("key_findings", [])
    response["grounding_summary"] = post.get("grounding_summary", "")
    response["sources_summary"] = post.get("sources_summary", [])
    return response
