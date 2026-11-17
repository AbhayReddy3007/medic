"""Centralized Market Potential runtime configuration."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# GCP project and authentication Config
# ---------------------------------------------------------------------------
PROJECT_ID = os.getenv("PROJECT_ID", "cognito-dev-380506")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# ---------------------------------------------------------------------------
# Cloud Storage Config
# ---------------------------------------------------------------------------
GCS_BUCKET = os.getenv("GCS_BUCKET", "cognito-gcs")
GCS_REPORT_BASE_PATH = os.getenv("GCS_REPORT_BASE_PATH", "Cognito_new/reports")
GCS_PIPELINE_CACHE_BASE_PATH = os.getenv("GCS_PIPELINE_CACHE_BASE_PATH", "Cognito_new/pipeline_cache")
GCS_MEDICAL_POTENTIAL_SUBFOLDER = os.getenv("GCS_MEDICAL_POTENTIAL_SUBFOLDER", "Medical Potential")
PILLAR_NAME = os.getenv("PILLAR_NAME", "Medical Potential")

# Line of treatment storage config
GCS_SOC_BASE_PATH = os.getenv("GCS_SOC_BASE_PATH", "Cognito_new/SOC_documents")
LOT_BENCHMARK_PATH = os.getenv("LOT_BENCHMARK_PATH", "Cognito_new/SOC_documents")

# ---------------------------------------------------------------------------
# Medical Potential Dimension Names
# ---------------------------------------------------------------------------
CLINICAL_EVIDENCE_DIMENSION_NAME = os.getenv("CLINICAL_EVIDENCE_DIMENSION_NAME", "Clinical Evidence")
CLINICAL_EFFICACY_DIMENSION_NAME = os.getenv("CLINICAL_EFFICACY_DIMENSION_NAME", "Clinical Efficacy")
MOA_INNOVATION_DIMENSION_NAME = os.getenv("MOA_INNOVATION_DIMENSION_NAME", "MoA Innovation")
SERIOUS_SAFETY_PROFILE_DIMENSION_NAME = os.getenv("SERIOUS_SAFETY_PROFILE_DIMENSION_NAME", "Serious Safety Profile")
TOLERABILITY_AND_PATIENT_BURDEN_DIMENSION_NAME = os.getenv(
    "TOLERABILITY_AND_PATIENT_BURDEN_DIMENSION_NAME",
    "Tolerability & Patient Burden",
)
UNMET_NEEDS_DIMENSION_NAME = os.getenv("UNMET_NEEDS_DIMENSION_NAME", "Unmet Need Targeted")
LINE_OF_TREATMENT_DIMENSION_NAME = os.getenv("LINE_OF_TREATMENT_DIMENSION_NAME", "Line of Treatment")
LABEL_EXPANSION_OPPORTUNITY_DIMENSION_NAME = os.getenv("LABEL_EXPANSION_OPPORTUNITY_DIMENSION_NAME", "Label Expansion Opportunity")

MEDICAL_POTENTIAL_DIMENSION_NAMES = (
    CLINICAL_EVIDENCE_DIMENSION_NAME,
    CLINICAL_EFFICACY_DIMENSION_NAME,
    MOA_INNOVATION_DIMENSION_NAME,
    SERIOUS_SAFETY_PROFILE_DIMENSION_NAME,
    TOLERABILITY_AND_PATIENT_BURDEN_DIMENSION_NAME,
    UNMET_NEEDS_DIMENSION_NAME,
    LINE_OF_TREATMENT_DIMENSION_NAME,
    LABEL_EXPANSION_OPPORTUNITY_DIMENSION_NAME
)

# ---------------------------------------------------------------------------
# LLM Model Config
# ---------------------------------------------------------------------------
GEMINI_FLASH_PREVIEW_MODEL = os.getenv("GEMINI_FLASH_PREVIEW_MODEL", "gemini-3-flash-preview")
GEMINI_PRO_MODEL = os.getenv("GEMINI_PRO_MODEL", "gemini-3.1-pro-preview")
GEMINI_SERVICE_TIER = os.getenv("GEMINI_SERVICE_TIER", None)

# ---------------------------------------------------------------------------
# Big Query Config
# ---------------------------------------------------------------------------

BQ_DATASET_ID = os.getenv("BQ_DATASET_ID", "data_mart")

# Drug Details Full table_id
DRUG_DETAILS_FULL_TABLE_ID = os.getenv("DRUG_DETAILS_FULL_TABLE_ID", 'cognito-prod-394707.cognito_prod_datamart.vw_drug_details_full')

# Dimension scores Table
DIM_SCORES_TABLE = os.getenv("DIM_SCORES_TABLE", "dim_scores")

# Gemini model usage statistics table
GEMINI_STATS_TABLE = os.getenv("GEMINI_STATS_TABLE", "gemini_model_api_call_logs")

# Pillar Summary Table
PILLAR_SUMMARY_TABLE = os.getenv("PILLAR_SUMMARY_TABLE","dim_pillar_summary")

# Clinical Efficacy Tables
CLINICAL_EFFICACY_TABLE = os.getenv("CLINICAL_EFFICACY_TABLE", "clinical_efficacy_glp1")
GD_CLINICAL_TRIALS_FULL_TABLE_ID = os.getenv(
    "GD_CLINICAL_TRIALS_FULL_TABLE_ID",
    "cognito-prod-394707.cognito_prod_datamart.vw_clinical_trials_GD",
) 

# Clinical Evidence Tables
CLINICAL_EVIDENCE_SCORING_TABLE = os.getenv(
    "CLINICAL_EVIDENCE_SCORING_TABLE", "molecule_evidence_scoring"
)
CLINICAL_EVIDENCE_TRIAL_DETAILS_TABLE = os.getenv(
    "CLINICAL_EVIDENCE_TRIAL_DETAILS_TABLE", "molecule_trial_details"
)
CLINICAL_EVIDENCE_PUBLICATION_DETAILS_TABLE = os.getenv(
    "CLINICAL_EVIDENCE_PUBLICATION_DETAILS_TABLE", "molecule_publication_details"
)


# Unmet Needs Tables
UNMET_NEEDS_TABLE = os.getenv("UNMET_NEEDS_TABLE", "unmet_needs_all_countries")
TRIAL_MAPPING_TABLE = os.getenv("TRIAL_MAPPING_TABLE", "clinical_trial_evidence_to_unmet_need_mapping")
UNMET_NEED_SCORING_TABLE = os.getenv("UNMET_NEED_SCORING_TABLE", "molecule_level_unmet_needs_scoring")

# MoA innovation Tables
MOA_INNOVATION_TABLE = os.getenv("MOA_INNOVATION_TABLE", "moa_innovation_output_data")
MOA_INNOVATION_EVAL_TABLE = os.getenv("MOA_INNOVATION_EVAL_TABLE", "moa_innovation_eval_results")

# Tolerability table name
CLINICAL_TRIALS_TOLERABILITY_DATA_TABLE = os.getenv("CLINICAL_TRIALS_TOLERABILITY_DATA_TABLE", "clinical_trials_tolerability_data")
TOLERABILITY_TABLE = os.getenv("TOLERABILITY_TABLE", "tolerability_and_patient_burden_assessment")

# Serious Safety Profile Table name
CLINICAL_TRIALS_SERIOUS_SAFETY_DATA = os.getenv("CLINICAL_TRIALS_SERIOUS_SAFETY_DATA", "clinical_trials_serious_safety_data")
SERIOUS_SAFETY_PROFILE_SAE_SUMMARY = os.getenv("SERIOUS_SAFETY_PROFILE_SAE_SUMMARY","serious_safety_profile_sae_summary")

# Line of Treatment tables and storage paths
LOT_TABLE = os.getenv("LOT_TABLE", "glp1_lot_output_data")
BENCHMARK_RERUN = os.getenv("BENCHMARK_RERUN", False)
COUNTRIES_TO_RERUN = ["US"]

#LABEL EXPANSION
DRUG_DETAILS_TABLE_ID = 'drug_details'
LABEL_EXPANSION_OPPORTUNITY_TABLE = 'label_expansion_opportunity_table'
LE_TABLE = "glp_indications_table"
LE_SCORE_CALCULATION_TABLE = "label_expansion_score_calculation"
OT_MOA_TABLE = "ot_moa_mapping"
OT_DISEASE_TABLE = "ot_disease_mapping"
DATA_FETCHER_TABLE = "data_fetched_le"
START_FROM = "discovery" # "discovery" | "moa_mapping" | "indication_mapping" | "scoring"

# ---------------------------------------------------------------------------
# LLM Confidence Thresholds
# ---------------------------------------------------------------------------

# Minimum llm_confidence required for a clinical trial row to be considered high-confidence
TRIAL_LLM_CONFIDENCE_THRESHOLD = float(os.getenv("TRIAL_LLM_CONFIDENCE_THRESHOLD", "0.6"))

