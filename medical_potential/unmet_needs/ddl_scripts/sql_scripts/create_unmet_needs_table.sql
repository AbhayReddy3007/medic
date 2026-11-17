CREATE OR REPLACE TABLE `cognito-dev-380506.data_mart.unmet_needs_all_countries` (
  unmet_need STRING OPTIONS (description = "Description of the unmet need in T2DM care"),
  category STRING OPTIONS (description = "Category of the unmet need"),
  evidence STRING OPTIONS (description = "Supporting evidence for the unmet need"),
  country STRING OPTIONS (description = "Country associated with the unmet need"),
  source STRING OPTIONS (description = "Data source for the unmet need record"),
  inclusion_flag STRING OPTIONS (description = "'Y' indicates to include the unmet need for processing in scoring/ report"),
  created_at_utc TIMESTAMP OPTIONS (description = "Record creation timestamp in UTC")
)
OPTIONS (
  description = "Unmet needs reference table for T2DM across multiple countries and sources"
);
