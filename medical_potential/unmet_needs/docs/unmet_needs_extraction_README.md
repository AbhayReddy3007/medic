# Unmet Needs Extraction

## Purpose

Build and refresh a single, standardized unmet-needs baseline across countries.
This baseline is the source-of-truth input for molecule-level assessment.

Main entry point:
- `unmet_needs_extraction/run_unmet_needs_extraction.py`

Supporting modules:
- `unmet_needs_extraction/unmet_needs_extraction_from_soc.py` (SoC PDF extraction)
- `unmet_needs_extraction/unmet_needs_extraction_from_web.py` (web-grounded extraction)
- `unmet_needs_extraction/utils.py` (JSON normalization, token helpers)
- `unmet_needs_extraction/prompts.py` (prompt templates)

## How This Pipeline Works

1. Initialize execution context
- Load credentials, model settings, and country configuration.
- Build country-to-input mapping for SoC PDFs and web countries.

2. Extract unmet needs from SoC documents
- For each configured SoC country, read all available PDF guidance docs.
- Send content to Gemini and request unmet-need extraction in structured form.
- Normalize each response into a consistent row schema.

3. Extract unmet needs from web-grounded research
- For each configured web country, run a grounded multi-turn Gemini workflow.
- Capture generated unmet needs plus source context.
- Normalize into the same schema as SoC extraction.

4. Merge and standardize output rows
- Concatenate SoC and web results.
- Add `created_at_utc` so every row has clear load lineage.

5. Load baseline into BigQuery
- Write to `cognito-prod-394707.cognito_prod_datamart.unmet_needs_all_countries`.
- Use explicit schema and append mode.

6. Publish run-level telemetry
- Print combined token usage to support run monitoring and cost governance.

## BigQuery Data Flow

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}, 'flowchart': {'nodeSpacing': 55, 'rankSpacing': 80, 'curve': 'basis'}} }%%
flowchart TB
    SOC_INPUT[SoC PDFs by country] --> SOC_EXTRACT[SoC extraction pipeline]
    WEB_INPUT[Web grounded inputs by country] --> WEB_EXTRACT[Web extraction pipeline]

    SOC_EXTRACT --> NORMALIZE[Normalize to shared schema]
    WEB_EXTRACT --> NORMALIZE

    NORMALIZE --> STAMP[Add created_at_utc]
    STAMP --> BQ_UNMET[(BQ: unmet_needs_all_countries)]
```

## Function Input-to-Output Flow

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}, 'flowchart': {'nodeSpacing': 55, 'rankSpacing': 85, 'curve': 'basis'}} }%%
flowchart TB
    IN1[Input: SoC PDF paths by country] --> R[run_extraction]
    IN2[Input: web country list] --> R
    IN3[Input: model config + credentials] --> R

    R --> SOC[extract_unmet_needs_from_soc]
    SOC --> SOC_ASYNC[_extract_pdf_with_gemini]
    SOC_ASYNC --> SOC_SYNC[_sync_gemini_pdf_call]
    SOC_SYNC --> NORM[_normalize_unmet_need_output]

    R --> WEB[extract_unmet_needs_from_web_multi_turn]
    WEB --> WEB_ASYNC[_extract_web_multi_turn]
    WEB_ASYNC --> WEB_SYNC[_sync_multi_turn_grounded_gemini_call]
    WEB_SYNC --> NORM

    NORM --> MERGE[run_extraction merges country outputs]
    MERGE --> CLIENT[get_bq_client]
    CLIENT --> LOAD[load_table_from_dataframe]
    LOAD --> OUT[(Output: BQ unmet_needs_all_countries)]
```

## Inputs and Outputs

Inputs:
- SoC PDF directories by country
- Web country list
- Gemini model + credentials

Output table:
- `cognito-prod-394707.cognito_prod_datamart.unmet_needs_all_countries`

Key output columns:
- `unmet_need`, `category`, `evidence`, `country`, `source`, `created_at_utc`

## Recommended Practices

- Validate input availability before extraction (missing directory/file checks).
- Keep prompt templates versioned and auditable.
- Use explicit BQ schema (avoid schema autodetect in production).
- Add post-load data checks:
  - row count thresholds,
  - required field non-null checks,
  - duplicate pattern detection.
- Define a refresh policy (append + dedup, or periodic replace) and document it.
