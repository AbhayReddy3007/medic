# Medical Potential Pillar Summary

The `generate_pillar_summary` module creates one concise executive summary for a molecule after the Medical Potential dimension reports have been generated. It combines product metadata from BigQuery with the first page of each available dimension PDF in Google Cloud Storage, asks Vertex AI Gemini to synthesize the evidence, and appends the resulting summary to BigQuery.

This is a summary workflow, not an independently scored dimension. It does not calculate a numeric score, generate a PDF, or upload a payload cache.

## Module Structure

```text
generate_pillar_summary/
|-- __init__.py                         Public package export
|-- README.md                           Implementation documentation
|-- generate_pillar_summary.py          Public orchestrator and Gemini call
|-- dimension_report_extraction.py      GCS report loading and first-page extraction
|-- bq_utils.py                          Product metadata and persistence helpers
`-- prompts.py                           Clinical and marketed prompt templates
```

The public package export is:

```python
from medical_potential.generate_pillar_summary import generate_pillar_summary
```

## Pipeline Position

The module is registered as `pillar_summary`, after all eight Medical Potential dimensions, in `medical_potential/run_medical_potential_dimensions.py`:

```text
clinical_efficacy
        |
unmet_needs
        |
serious_safety_profile
        |
clinical_evidence
        |
moa_innovation
        |
tolerability_and_patient_burden
        |
line_of_treatment
        |
label_expansion
        |
pillar_summary  <-- this module
```

When `dimension_names` is omitted from `run_medical_potential_dimensions()`, the orchestrator runs every registered entry, including `pillar_summary`, in registry order. The summary therefore expects the current dimension PDFs to already exist in GCS. If only selected dimensions are run, `pillar_summary` is included only when it is explicitly selected.

## Workflow Overview

```text
Step 1  Load product details and eligible dosage strengths from BigQuery
        |
Step 2  Select the clinical-stage or marketed/pre-registration prompt
        |
Step 3  Format product metadata into the selected prompt
        |
Step 4  Download each configured dimension PDF from GCS
        |
        +--> Keep only the first page of each readable PDF
        +--> Record missing or unreadable reports and continue
        |
Step 5  Call Gemini with the prompt and PDF attachments at temperature 0
        |
Step 6  Append the generated summary to the pillar-summary BigQuery table
```

The function returns the raw model-generated summary after the BigQuery write succeeds.

## Public API

```python
def generate_pillar_summary(
    molecule_name: str,
    model_name: str = GEMINI_FLASH_PREVIEW_MODEL,
) -> str
```

### Arguments

| Argument | Description |
| --- | --- |
| `molecule_name` | Molecule name used in the BigQuery filters and GCS path. Leading and trailing whitespace is removed. |
| `model_name` | Gemini model passed to the shared `gemini_api_call()` helper. It defaults to `GEMINI_FLASH_PREVIEW_MODEL`. |

An empty or whitespace-only molecule name raises `ValueError`. The function is synchronous and must be called after the required dimension reports have been uploaded, unless a metadata-only summary is acceptable.

## Product Metadata Inputs

`get_drug_detials()` reads two BigQuery sources. The helper name contains the existing `detials` spelling and is kept for compatibility.

### Product details query

The source table is `DRUG_DETAILS_FULL_TABLE_ID`. Rows are selected when:

- `Cleaned_generic_name` matches the requested molecule case-insensitively.
- `Mechanism_of_Action` is not null.

The query reads `Cleaned_generic_name`, `Highest_Development_Stage`, `Cleaned_Indication`, and `Mechanism_of_Action`. The returned prompt metadata is normalized to this shape:

```json
{
  "drug_name": "molecule name",
  "Highest Development Stage": "PHASE III",
  "Primary indication": ["Type 2 Diabetes", "Obesity"],
  "Secondary Indication": ["..."],
  "Dosage Strength details": ["..."]
}
```

The implementation always uses `Type 2 Diabetes` and `Obesity` as the primary-indication list. Secondary indications are created by splitting `Cleaned_Indication` on commas, removing duplicates, and excluding indications matching Type 2 Diabetes or obesity. The mechanism-of-action column is used as a source-row eligibility condition but is not included in the returned metadata dictionary; the attached reports provide the evidence used for mechanism and differentiation language.

### Dosage query

Dosages come from `CLINICAL_EFFICACY_TABLE` and must satisfy all of these conditions:

- Molecule matches case-insensitively.
- `llm_confidence` is greater than `TRIAL_LLM_CONFIDENCE_THRESHOLD`.
- `dosage` is non-null and not blank.
- Values equal to `n/a` or `none` are ignored, case-insensitively.
- Remaining values are de-duplicated case-insensitively and sorted by the query.

If no product rows or dosage values are available, the function still proceeds. The prompt instructs Gemini to omit unavailable dosage strengths and secondary indications.

## Prompt Selection

The highest development stage is selected from all matching product rows using this priority order:

| Rank | Stage |
| ---: | --- |
| 4 | `MARKETED` |
| 3 | `PRE-REGISTRATION` |
| 2 | `PHASE III` |
| 1 | `PHASE II` |
| 0 | `PHASE I` |

Stage comparison is case-insensitive and ignores repeated whitespace. The marketed prompt is selected when the normalized stage starts with `MARKETED`, `PRE-REGISTRATION`, or `PRE-REGISTERED`. Every other stage, including a missing or unrecognized stage, uses the clinical-molecule prompt.

### Clinical-stage prompt

The clinical prompt asks for a one-paragraph summary covering:

1. Product profile: molecule class, dosage, highest phase, indications, and secondary indications.
2. Differentiation: mechanism, efficacy signals, unmet needs, and clinical differentiation.
3. Opportunity: evidence maturity, label expansion, addressable market, and adoption potential.
4. Risk and outlook: the most important limitation or competitive threat and the overall outlook.

### Marketed/pre-registration prompt

The marketed prompt asks for a one-paragraph summary covering:

1. Product profile: molecule class, marketed status, dosage, indications, and secondary indications.
2. Positioning: treatment-line role, efficacy advantages, unmet needs, and competitive differentiation.
3. Franchise potential: evidence maturity, label expansion, sustained adoption, and market relevance.
4. Risk and outlook: the most important safety, tolerability, commercial, or durability risk and the overall outlook.

## Gemini Generation Contract

Both prompt templates require Gemini to:

- Return exactly one paragraph.
- Stay within a hard limit of 100 words.
- Avoid headings and bullet points.
- Avoid naming Cognito dimensions.
- Use the supplied product metadata to anchor stage, dosage, and indication context.
- Use only the attached reports and supplied metadata.
- Avoid invented information or unsupported external knowledge.
- Omit unavailable dosage strengths and secondary indications.
- Use an objective analyst tone and clinically appropriate terminology.
- Synthesize supporting and contradictory evidence before assessing the outlook.

The call uses the selected model, `temperature=0`, and the first-page PDF attachments. It does not enable Google Search grounding. The shared Gemini transport retries transient failures and records model usage for the surrounding dimension orchestrator.

## Dimension Report Inputs

`get_dimension_reports_from_gcs()` uses the configured dimension names from `MEDICAL_POTENTIAL_DIMENSION_NAMES`. By default, these are:

- `Clinical Evidence`
- `Clinical Efficacy`
- `MoA Innovation`
- `Serious Safety Profile`
- `Tolerability & Patient Burden`
- `Unmet Need Targeted`
- `Line of Treatment`
- `Label Expansion Opportunity`

For each name, the module reads this current-report path:

```text
gs://{GCS_BUCKET}/{GCS_REPORT_BASE_PATH}/{molecule}/{PILLAR_NAME}/{dimension_name}.pdf
```

For example:

```text
gs://cognito-gcs/Cognito_new/reports/semaglutide/Medical Potential/Clinical Efficacy.pdf
```

Only the first page is attached to Gemini. `extract_first_page_pdf()` creates a new in-memory PDF, so the complete source report is not sent to the model. This keeps the summary input focused on the report's executive content and limits attachment size.

### Missing and unreadable reports

- A missing GCS object is recorded in `missing_reports` and skipped.
- A PDF with no pages or a PDF that cannot be parsed is recorded as missing/unreadable and skipped.
- If some reports are available, generation continues with those reports.
- If no reports are available, generation continues from product metadata only and logs a warning.
- GCS download errors other than `NotFound` are not converted into missing-report entries and may fail the call.

The public `generate_pillar_summary()` function does not return the `missing_reports` list. Use `get_dimension_reports_from_gcs()` directly when a caller needs to inspect the paths that were skipped.

## BigQuery Output

`push_pillar_summary_to_bq()` appends one row to:

```text
{PROJECT_ID}.{BQ_DATASET_ID}.{PILLAR_SUMMARY_TABLE}
```

The default table is `data_mart.dim_pillar_summary` in the configured project. The table is created if needed with this schema:

| Field | Type | Mode | Value |
| --- | --- | --- | --- |
| `molecule_name` | `STRING` | `NULLABLE` | Requested, trimmed molecule name |
| `Pillar` | `STRING` | `NULLABLE` | `PILLAR_NAME`, normally `Medical Potential` |
| `Summary` | `STRING` | `NULLABLE` | Raw Gemini response |
| `created_at` | `TIMESTAMP` | `REQUIRED` | Current UTC timestamp |

Writes use `WRITE_APPEND`. There is no update or de-duplication step, so rerunning the same molecule creates another timestamped row. Downstream consumers should select the desired latest row explicitly.

## Configuration

Configuration is loaded from environment variables and `.env` by `medical_potential/config.py`.

| Variable | Default | Used for |
| --- | --- | --- |
| `PROJECT_ID` | `cognito-dev-380506` | BigQuery and Vertex AI project |
| `BQ_DATASET_ID` | `data_mart` | Dataset for the output table and clinical efficacy table |
| `DRUG_DETAILS_FULL_TABLE_ID` | `cognito-prod-394707.cognito_prod_datamart.vw_drug_details_full` | Product metadata source |
| `CLINICAL_EFFICACY_TABLE` | `clinical_efficacy_glp1` | Dosage source |
| `TRIAL_LLM_CONFIDENCE_THRESHOLD` | `0.6` | Minimum confidence for dosage rows; the query requires a value greater than this threshold |
| `PILLAR_SUMMARY_TABLE` | `dim_pillar_summary` | Summary output table name |
| `GCS_BUCKET` | `cognito-gcs` | Bucket containing dimension PDFs |
| `GCS_REPORT_BASE_PATH` | `Cognito_new/reports` | Root prefix for current dimension reports |
| `PILLAR_NAME` | `Medical Potential` | GCS pillar folder and output `Pillar` value |
| `GEMINI_FLASH_PREVIEW_MODEL` | `gemini-3-flash-preview` | Default summary model |
| `GOOGLE_APPLICATION_CREDENTIALS` | empty | Optional service-account JSON path; otherwise Google Application Default Credentials are used |

The dimension-report producers use `GCS_MEDICAL_POTENTIAL_SUBFOLDER` when uploading reports, while this module uses `PILLAR_NAME` when reading them. Keep these values aligned, normally both `Medical Potential`, or the summary reader will not find the files produced by the scoring pipeline.

## Usage

### Generate a summary directly

```python
from medical_potential.generate_pillar_summary import generate_pillar_summary

summary = generate_pillar_summary("semaglutide")
print(summary)
```

To select a model explicitly:

```python
summary = generate_pillar_summary(
    molecule_name="semaglutide",
    model_name="gemini-3-flash-preview",
)
```

The direct call requires the product tables, GCS reports, credentials, and Vertex AI access described above. It writes the output row before returning the string.

### Run through the dimension orchestrator

```python
import asyncio

from medical_potential.run_medical_potential_dimensions import run_medical_potential_dimensions

result = asyncio.run(
    run_medical_potential_dimensions(
        molecule_name="semaglutide",
        dimension_names=["pillar_summary"],
    )
)
```

This form assumes the dimension PDFs already exist. To run all dimensions and then generate the summary, omit `dimension_names`:

```python
result = asyncio.run(
    run_medical_potential_dimensions("semaglutide")
)
```

The orchestrator records the summary result under `result["results"]["pillar_summary"]`. If the summary fails, the orchestrator records the exception under `result["failures"]["pillar_summary"]` and continues its normal failure-isolation behavior. Gemini usage statistics are also persisted by the orchestrator for the `pillar_summary` run.

## Operational Behavior and Troubleshooting

### Product metadata is incomplete

The module does not require a matching product row or dosage row to continue. The summary can still be generated from attached reports, but stage, dosage, or indication context may be incomplete. Check the source-table filters and molecule spelling when metadata is unexpectedly absent.

### Reports are missing

Check the exact molecule folder, pillar folder, dimension label, and report prefix. Dimension names are case-sensitive in the GCS object path. Also verify that `PILLAR_NAME` matches the folder used by the report uploaders.

### The output contains multiple rows for one molecule

This is expected after repeated runs because the output table is append-only. Query by `molecule_name` and order by `created_at` to select the latest result.

### The summary exceeds the requested format

The prompt requests one paragraph and a maximum of 100 words, but the model response is returned without local validation or truncation. Downstream consumers should validate the returned text if strict enforcement is required.

