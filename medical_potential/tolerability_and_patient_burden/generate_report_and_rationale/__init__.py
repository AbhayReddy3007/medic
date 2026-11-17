"""Package exposing report generation helpers for tolerability.

This package exposes the main `generate_tolerability_report` entrypoint so
other modules can import it via the package name rather than the module file.
"""
from .generate_tolerability_report import generate_tolerability_report
from .generate_rationale import generate_rationale_for_score

__all__ = ["generate_tolerability_report",
           "generate_rationale_for_score"]
