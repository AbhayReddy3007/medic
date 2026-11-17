CREATE OR REPLACE TABLE `cognito-dev-380506.data_mart.serious_safety_profile_sae_summary` (
    molecule_name STRING,
    sae_name STRING,
    number_of_studies INT64,
    list_of_studies ARRAY<STRING>,
    expectedness_classification STRING,
    expectedness_rationale STRING,
    severity_category STRING,
    severity_rationale STRING,
    severity_category_original STRING,
    severity_category_post_processed BOOLEAN,
    severity_category_post_process_note STRING,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);