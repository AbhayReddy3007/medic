CREATE OR REPLACE TABLE `cognito-dev-380506.data_mart.clinical_trial_evidence_to_unmet_need_mapping` (
  trial_id STRING,
  molecule_name STRING,
  unmet_need STRING,
  category STRING,
  country STRING,
  addresses_unmet_need STRING,
  evidence_strength STRING,
  relevant_endpoint STRING,
  explanation STRING,
  search_sources_used STRING,
  phase STRING,
  phase_status STRING,
  trial_size FLOAT64,
  trial_location STRING,
  created_at_utc TIMESTAMP OPTIONS (description = "UTC timestamp when the record was created")
)
OPTIONS (
  description = "Clinical trial evidence to unmet needs mapping table"
);
