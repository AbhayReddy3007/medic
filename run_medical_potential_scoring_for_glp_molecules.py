"""Run medical potential scoring for one or more GLP molecules."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from medical_potential import run_medical_potential_dimensions
from medical_potential.gcp_utils import get_bq_client
from medical_potential.config import DRUG_DETAILS_FULL_TABLE_ID

GLP1_PRODUCT_LIST_QUERY = f"""
SELECT DISTINCT cleaned_generic_name
FROM `{DRUG_DETAILS_FULL_TABLE_ID}`
WHERE
(
    UPPER(cleaned_Target) LIKE '%GLUCAGON LIKE PEPTIDE 1%'
    OR UPPER(cleaned_Target) LIKE '%GLP-1%'
    OR UPPER(cleaned_Target) LIKE '%GLUCAGON LIKE PEPTIDE-1%'
    OR (
        data_source = 'IPD'
        AND Mechanism_of_Action = 'Glucagon-like peptide-1 (GLP-1) agonist'
    )
)
AND Mechanism_of_Action IS NOT NULL
AND LOWER(Mechanism_of_Action) NOT LIKE '%antagonist%'
"""


def _is_test_mode(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() == "yes"


def _parse_list_value(raw_value: str | None) -> list[str]:
    if not raw_value or not raw_value.strip():
        return []

    return [item.strip() for item in raw_value.replace(";", ",").split(",") if item.strip()]


def _parse_int_env(value: str | None, default: int | None = None) -> int | None:
    if value is None or not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _split_molecules_for_task(molecules: list[str], task_index: int, task_count: int) -> list[str]:
    if task_count <= 1:
        return molecules

    total = len(molecules)
    if total == 0:
        return []

    if task_index < 0 or task_index >= task_count:
        return []

    # Evenly distribute molecules across tasks.
    # Example: total=10, task_count=3 -> base_size=3, remainder=1
    # Task 0 gets 4 items, tasks 1 and 2 get 3 items each.
    base_size = total // task_count
    remainder = total % task_count

    # First `remainder` tasks get one extra molecule so distribution stays balanced.
    if task_index < remainder:
        start = task_index * (base_size + 1)
        end = start + base_size + 1
    else:
        # Remaining tasks get exactly `base_size` molecules.
        # Offset start index by all molecules assigned to the larger leading buckets.
        start = remainder * (base_size + 1) + (task_index - remainder) * base_size
        end = start + base_size

    return molecules[start:end]


def _fetch_glp1_molecules() -> list[str]:
    client = get_bq_client()
    query_job = client.query(GLP1_PRODUCT_LIST_QUERY)
    results = query_job.result()
    molecules = [row.cleaned_generic_name for row in results if row.cleaned_generic_name]
    return molecules


async def run_medical_potential_scoring_for_glp_molecules(
    molecules: list[str],
    dimensions: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not molecules:
        raise ValueError("At least one molecule is required")

    results: list[dict[str, Any]] = []
    for molecule_name in molecules:
        result = await run_medical_potential_dimensions(
            molecule_name=molecule_name,
            dimension_names=dimensions,
        )
        results.append(result)
    return results


def main() -> None:
    test_mode = _is_test_mode(os.getenv("TEST_MODE"))
    if test_mode:
        test_molecules_value = os.getenv("TEST_MOLECULES")
        if test_molecules_value and test_molecules_value.strip().lower() == "all":
            molecules = _fetch_glp1_molecules()
            if not molecules:
                raise ValueError("No GLP-1 molecules found from the query")
        else:
            molecules = _parse_list_value(test_molecules_value)
            if not molecules:
                raise ValueError("TEST_MODE=yes requires TEST_MOLECULES or TEST_MOLECULES=all")

        dimensions_from_env = _parse_list_value(os.getenv("TEST_DIMENSIONS"))
        dimensions = dimensions_from_env or None
    else:
        molecules = _fetch_glp1_molecules()
        if not molecules:
            raise ValueError("No GLP-1 molecules found from the query")

        dimensions = None

    task_index = _parse_int_env(os.getenv("CLOUD_RUN_TASK_INDEX"), default=0) or 0
    task_count = _parse_int_env(os.getenv("CLOUD_RUN_TASK_COUNT"), default=1) or 1
    task_molecules = _split_molecules_for_task(molecules, task_index=task_index, task_count=task_count)

    if not task_molecules:
        print(f"No molecules assigned to task index {task_index} of {task_count}; exiting without work.")
        return

    payload = asyncio.run(
        run_medical_potential_scoring_for_glp_molecules(
            molecules=task_molecules,
            dimensions=dimensions,
        )
    )
    # print(payload)


if __name__ == "__main__":
    main()

