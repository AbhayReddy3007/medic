"""Constants used by the Dimension 5 scoring engine."""

DEFAULT_SCORE_NO_DELTA_SAE = 1 # When there are no SAE rates available for control or drug arm
DEFAULT_SCORE_NO_SAE = 1 # When there are no SAE events found in the trials

BAND_TO_SCORE: dict[str, int] = {
    "S1": 5,
    "S2": 4,
    "S3": 3,
    "S4": 2,
}

SEVERITY_LEVEL: dict[str, int] = {
    "manageable": 1,
    "concerning": 2,
    "important": 2,
    "clinically important": 2,
    "organ": 2,
    "organ-specific": 2,
    "severe": 3,
    "life-threatening": 3,
    "life threatening": 3,
    "catastrophic": 4,
    "critical": 4,
    "death": 4,
    "fatal": 4
}

