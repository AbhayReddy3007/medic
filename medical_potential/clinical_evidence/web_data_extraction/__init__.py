"""Web data extraction for the clinical evidence dimension.

Collects molecule identity, trial registry data, innovator/FDA entries and
Google-Search-grounded real-world evidence.
"""

from .innovator import innovator_deep_dive
from .molecule_normalization import normalize_molecule
from .real_world_evidence import run_rwe_summarizer
from .trial_sources import (
    enrich_bq_trials,
    global_trial_scan,
    merge_trial_lists,
    search_eu_ctr,
    search_who_ictrp,
)

__all__ = [
    "enrich_bq_trials",
    "global_trial_scan",
    "innovator_deep_dive",
    "merge_trial_lists",
    "normalize_molecule",
    "run_rwe_summarizer",
    "search_eu_ctr",
    "search_who_ictrp",
]
