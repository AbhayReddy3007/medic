# DRL Cognito AI

DRL Cognito AI calculates medical-potential scores for GLP-1 molecules. The pipeline combines BigQuery data, Google Cloud Storage assets, Vertex AI Gemini analysis, clinical-trial and public-web evidence, and deterministic scoring logic. Results and generated reports are written to the configured BigQuery tables and GCS paths.

The main batch entry point is `run_medical_potential_scoring_for_glp_molecules.py`. It discovers GLP-1 molecules from BigQuery in production mode and can process a supplied list of molecules and dimensions in test mode.

## Repository layout

```text
run_medical_potential_scoring_for_glp_molecules.py  # Batch entry point
medical_potential/
  config.py                                        # Environment-backed configuration
  gcp_utils.py                                     # BigQuery and GCS helpers
  gemini_utils.py                                  # Vertex AI Gemini transport
  run_medical_potential_dimensions.py              # Dimension orchestration
  clinical_efficacy/                               # Clinical efficacy pipeline
  clinical_evidence/                               # Clinical evidence pipeline
  moa_innovation/                                  # MoA innovation pipeline
  serious_safety_profile/                          # Serious safety pipeline
  tolerability_and_patient_burden/                 # Tolerability pipeline
  unmet_needs/                                     # Unmet-needs pipeline
  line_of_treatment/                               # Treatment-line pipeline
  label_expansion_opportunity/                     # Label expansion pipeline
  generate_pillar_summary/                         # Pillar summary pipeline
Dockerfile                                         # Cloud Run Jobs image
requirements.txt                                   # Python dependencies
```

## Medical-potential dimensions and summary module

The orchestration layer supports these medical-potential dimension keys:

| Key | Purpose |
| --- | --- |
| `clinical_efficacy` | Clinical efficacy and trial outcome assessment |
| `clinical_evidence` | Strength and quality of clinical evidence |
| `moa_innovation` | Mechanism-of-action innovation assessment |
| `serious_safety_profile` | Serious adverse-event and safety assessment |
| `tolerability_and_patient_burden` | Tolerability and patient-burden assessment |
| `unmet_needs` | Unmet-need targeting assessment |
| `line_of_treatment` | Treatment-pathway positioning |
| `label_expansion` | Label-expansion opportunity assessment |

The `generate_pillar_summary` module is not a dimension. It runs after the dimensions and creates a single medical-potential pillar summary for the molecule using the available dimension reports and product metadata.

When no dimensions are specified, all dimensions above are requested and the pillar-summary module runs afterward. A failure in one dimension is recorded in that molecule's result while the remaining dimensions continue to run.


## Configuration

Configuration is loaded by `medical_potential/config.py` from environment variables and `.env`. The most important settings are:

| Variable | Default | Description |
| --- | --- | --- |
| `PROJECT_ID` | `cognito-dev-380506` | Google Cloud project used by BigQuery and Vertex AI |
| `BQ_DATASET_ID` | `data_mart` | BigQuery dataset for output tables |
| `DRUG_DETAILS_FULL_TABLE_ID` | `cognito-prod-394707.cognito_prod_datamart.vw_drug_details_full` | Fully-qualified source table used to discover GLP-1 molecules |
| `GCS_BUCKET` | `cognito-gcs` | GCS bucket for reports, caches, and source files |
| `GCS_REPORT_BASE_PATH` | `Cognito_new/reports` | Root report path in the bucket |
| `GCS_PIPELINE_CACHE_BASE_PATH` | `Cognito_new/pipeline_cache` | Root pipeline-cache path |
| `GCS_MEDICAL_POTENTIAL_SUBFOLDER` | `Medical Potential` | Report/cache subfolder |
| `GEMINI_FLASH_PREVIEW_MODEL` | `gemini-3-flash-preview` | Gemini Flash model |
| `GEMINI_PRO_MODEL` | `gemini-3.1-pro-preview` | Gemini Pro model |
| `GEMINI_SERVICE_TIER` | unset | Set to `flex` to enable flex mode for gemini api |
| `GOOGLE_APPLICATION_CREDENTIALS` | required | Required Google Cloud authentication; set this to a service-account JSON path or configure Application Default Credentials |

The remaining table names and dimension labels are also configurable in `medical_potential/config.py`.

## Run the scoring pipeline locally

### Production mode

Production mode discovers all eligible GLP-1 molecules from `DRUG_DETAILS_FULL_TABLE_ID` and runs all dimensions:

```powershell
python run_medical_potential_scoring_for_glp_molecules.py
```

### Test mode for selected molecules

Use comma- or semicolon-separated values. Test mode is useful for validating a small change without processing the full molecule list:

```powershell
$env:TEST_MODE="yes"
$env:TEST_MOLECULES="semaglutide,tirzepatide"
python run_medical_potential_scoring_for_glp_molecules.py
```

Restrict the run to selected dimension keys:

```powershell
$env:TEST_DIMENSIONS="clinical_efficacy,clinical_evidence"
python run_medical_potential_scoring_for_glp_molecules.py
```

To run every molecule returned by the discovery query while retaining test-mode behavior, set `TEST_MOLECULES=all`:

```powershell
$env:TEST_MODE="yes"
$env:TEST_MOLECULES="all"
python run_medical_potential_scoring_for_glp_molecules.py
```

Unset test variables before a normal run:

```powershell
Remove-Item Env:TEST_MODE, Env:TEST_MOLECULES, Env:TEST_DIMENSIONS -ErrorAction SilentlyContinue
```

The script also honors `CLOUD_RUN_TASK_INDEX` and `CLOUD_RUN_TASK_COUNT`. When these variables are present, the molecule list is evenly split across Cloud Run tasks, and each task processes only its assigned slice.

## Outputs and operational notes

- Dimension scores, rationales, and model-usage statistics are persisted to configured BigQuery tables.
- PDF reports and JSON payload caches are uploaded to GCS under the configured report and cache prefixes.
- A molecule can have successful and failed dimensions in the same run; inspect the execution logs and output tables rather than relying only on the process exit status.
- Gemini calls may use web search grounding for selected dimensions. Review Vertex AI quotas and costs before increasing task count.
