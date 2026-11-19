"""Scoring logic package for the clinical evidence dimension (PDGEC)."""

from .deterministic_scores import compute_final_score, score_design, score_geography, score_phase
from .evidence_assessment import assess_all_evidence, consolidation_summary, identify_evidence_gaps
from .llm_scores import score_consistency, score_evidence

__all__ = [
    "assess_all_evidence",
    "compute_final_score",
    "consolidation_summary",
    "identify_evidence_gaps",
    "score_consistency",
    "score_design",
    "score_evidence",
    "score_geography",
    "score_phase",
]
