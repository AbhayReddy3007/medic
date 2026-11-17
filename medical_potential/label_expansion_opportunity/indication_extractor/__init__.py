"""Indication extractor sub-package.

Holds the three research modules that find candidate label-expansion
indications for a drug, plus their shared utilities:

- ``utils``           — ``extract_json()``, the ``SECONDARY_INDICATION_CRITERIA``
  prompt text, and the scorable-indication safety-net filter. The shared
  Gemini call wrappers (``gemini_generate()`` / ``gemini_generate_with_timeout()``)
  live one level up, in ``label_expansion_opportunity/utils.py``.
- ``trial_analyser``  — Module 1: mines registered clinical trials
  (read from BigQuery) for indications and their trial phase.
- ``web_analyser``    — Module 2: uses Gemini + Google Search to find
  label-expansion signals directly from public web sources.
- ``fda_fetcher``     — Module 3: fetches FDA-approved indications via
  the openFDA API + Gemini extraction.
"""

from ..utils import gemini_generate
from .trial_analyser import analyse as analyse_trials, fetch_trial_rows
from .web_analyser import analyse as analyse_web
from .fda_fetcher import analyse as analyse_fda
from .utils import (
    SECONDARY_INDICATION_CRITERIA,
    extract_json,
)

__all__ = [
    "analyse_trials",
    "fetch_trial_rows",
    "analyse_web",
    "analyse_fda",
    "SECONDARY_INDICATION_CRITERIA",
    "extract_json",
    "gemini_generate",
]