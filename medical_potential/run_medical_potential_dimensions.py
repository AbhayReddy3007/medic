"""Run one or more medical-potential dimensions for a molecule.

Each dimension (e.g. clinical_efficacy, unmet_needs) is a separate scoring pipeline
registered in DIMENSION_FUNCTIONS. This module runs them one after another for a
single molecule, isolates failures, and logs Gemini usage per dimension to BigQuery.

Usage (from async code):

    from medical_potential import run_medical_potential_dimensions

    # All dimensions
    result = await run_medical_potential_dimensions("semaglutide")

    # Selected dimensions: list, tuple, or comma-separated string
    result = await run_medical_potential_dimensions(
        "semaglutide", dimension_names=["clinical_efficacy", "unmet_needs"]
    )

From a synchronous script, wrap the call with asyncio.run(...). For batch runs over
all GLP-1 molecules, use run_medical_potential_scoring_for_glp_molecules.py.
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any, Callable

from .bq_utils import append_gemini_stats_to_bigquery
from .gemini_utils import get_gemini_call_stats, reset_gemini_call_stats
from .clinical_efficacy import clinical_efficacy_assessment
from .unmet_needs.unmet_needs_assessment import unmet_needs_assesment
from .serious_safety_profile import run_serious_safety_assessment
from .clinical_evidence import clinical_evidence_scoring
from .moa_innovation import get_moa_innovation_assessment
from .tolerability_and_patient_burden import tolerability_and_patient_burden_assessment
from .line_of_treatment import line_of_treatment_assessment
from .label_expansion_opportunity import label_expansion
from .generate_pillar_summary import generate_pillar_summary

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Registry of valid dimension keys; add new dimensions here. Insertion order is the default run order.
# Each runner must accept `molecule_name` as a keyword argument and may be sync or async.
DIMENSION_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "clinical_efficacy": clinical_efficacy_assessment,
    "unmet_needs": unmet_needs_assesment,
    "serious_safety_profile": run_serious_safety_assessment,
    "clinical_evidence": clinical_evidence_scoring,
    "moa_innovation": get_moa_innovation_assessment,
    "tolerability_and_patient_burden": tolerability_and_patient_burden_assessment,
    "line_of_treatment": line_of_treatment_assessment,
    "label_expansion": label_expansion,
    # Keep last: it summarizes the other dimensions' PDF reports already uploaded to GCS.
    "pillar_summary": generate_pillar_summary,
}


def _normalize_dimension_names(dimension_names: str | list[str] | tuple[str, ...] | None) -> list[str]:
    # None means "run everything".
    if dimension_names is None:
        return list(DIMENSION_FUNCTIONS.keys())

    if isinstance(dimension_names, str):
        raw_names = [part.strip() for part in dimension_names.split(",")]
    else:
        raw_names = [str(name).strip() for name in dimension_names]

    normalized = [name for name in raw_names if name]
    # An empty selection (e.g. "" or " , ") falls back to all dimensions instead of running nothing.
    if not normalized:
        return list(DIMENSION_FUNCTIONS.keys())
    return normalized


def _load_dimension_function(dimension_name: str):
    if dimension_name not in DIMENSION_FUNCTIONS:
        available = ", ".join(sorted(DIMENSION_FUNCTIONS))
        raise ValueError(f"Unknown dimension '{dimension_name}'. Available dimensions: {available}")

    return DIMENSION_FUNCTIONS[dimension_name]


async def _call_dimension_runner(runner, molecule_name: str, **kwargs: Any) -> Any:
    result = runner(molecule_name=molecule_name, **kwargs)
    # Runners are a mix of sync and async functions; await only when needed.
    if inspect.isawaitable(result):
        return await result
    return result


async def run_medical_potential_dimensions(
    molecule_name: str,
    dimension_names: str | list[str] | tuple[str, ...] | None = None,
    **dimension_kwargs: Any,
) -> dict[str, Any]:
    """Run one or more medical-potential dimensions for a molecule.

    Args:
        molecule_name: Molecule to process.
        dimension_names: Optional single dimension name, comma-separated names, or list of names.
        **dimension_kwargs: Additional keyword arguments passed to each dimension runner.

    Returns:
        Dictionary with keys:
            molecule_name: the stripped molecule name.
            requested_dimensions: dimension keys that were run, in order.
            results: {dimension_key: runner return value} for successful dimensions.
            failures: {dimension_key: error message} for failed dimensions.

    Raises:
        ValueError: if molecule_name is empty. Dimension errors are not raised;
            they are reported in `failures`.
    """
    if not molecule_name or not molecule_name.strip():
        raise ValueError("molecule_name must be provided")

    molecule = molecule_name.strip()
    selected_dimensions = _normalize_dimension_names(dimension_names)

    results: dict[str, Any] = {
        "molecule_name": molecule,
        "requested_dimensions": selected_dimensions,
        "results": {},
        "failures": {},
    }

    for dimension_name in selected_dimensions:
        logger.info("Running dimension '%s' for molecule '%s'", dimension_name, molecule)
        # Stats are global per process, so reset to attribute usage to this dimension only.
        reset_gemini_call_stats()
        run_status = "success"
        error_message = None
        start_time = time.perf_counter()
        try:
            runner = _load_dimension_function(dimension_name)
            results["results"][dimension_name] = await _call_dimension_runner(
                runner,
                molecule_name=molecule,
                **dimension_kwargs,
            )
        except Exception as exc:
            # A failing dimension (including an unknown key) must not stop the remaining ones.
            logger.exception("Dimension '%s' failed for molecule '%s'", dimension_name, molecule)
            results["failures"][dimension_name] = str(exc)
            run_status = "failed"
            error_message = str(exc)
        finally:
            execution_time_seconds = round(time.perf_counter() - start_time, 3)
            logger.info(
                "Dimension '%s' for molecule '%s' completed in %.3f seconds (%s)",
                dimension_name,
                molecule,
                execution_time_seconds,
                run_status,
            )
            # Runs for failures too, so cost and duration are recorded for every attempt.
            try:
                append_gemini_stats_to_bigquery(
                    molecule_name=molecule,
                    dimension_name=dimension_name,
                    stats_by_model=get_gemini_call_stats(),
                    run_status=run_status,
                    error_message=error_message,
                    execution_time_seconds=execution_time_seconds,
                )
            except Exception:
                # Stats logging is best-effort and must not mask the dimension outcome.
                logger.exception(
                    "Failed to persist Gemini stats for dimension '%s' and molecule '%s'",
                    dimension_name,
                    molecule,
                )

    return results
