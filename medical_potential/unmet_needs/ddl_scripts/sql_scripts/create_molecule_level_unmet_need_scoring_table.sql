CREATE OR REPLACE TABLE `cognito-dev-380506.data_mart.molecule_level_unmet_needs_scoring` (
  molecule_name STRING OPTIONS (description = "Name of the molecule being scored"),
  unmet_need STRING OPTIONS (description = "Description of the unmet need addressed by trial evidence"),
  category STRING OPTIONS (description = "Unmet need category (e.g. Clinical Importance, Safety, Real-World)"),
  country STRING OPTIONS (description = "Country associated with the scored unmet need"),
  score_dimension STRING OPTIONS (description = "Scoring dimension: C (Clinical), S (Safety), or R (Real-World)"),
  unmet_need_score FLOAT64 OPTIONS (description = "LLM-assigned unmet need score (1-5)"),
  evidence_impact_score FLOAT64 OPTIONS (description = "Evidence impact / maturity score (1-5)"),
  unmet_need_score_rationale STRING OPTIONS (description = "Rationale for the unmet need score"),
  evidence_score_rationale STRING OPTIONS (description = "Rationale for the evidence impact score"),
  confidence STRING OPTIONS (description = "Confidence level of the scoring: High, Medium, or Low"),
  num_of_trials_assessed INT64 OPTIONS (description = "Number of clinical trials assessed for this unmet need"),
  evidence_impact_weight FLOAT64 OPTIONS (description = "Computed weight based on evidence impact score"),
  weighted_unmet_need_score FLOAT64 OPTIONS (description = "Final weighted unmet need score after applying evidence impact weight"),
  created_at_utc TIMESTAMP OPTIONS (description = "UTC timestamp when the record was created")
)
OPTIONS (
  description = "Molecule-level unmet needs scoring table storing LLM-scored unmet needs with evidence impact weights"
);
