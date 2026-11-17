# Tolerability and Patient Burden Pipeline

The `tolerability_and_patient_burden` pipeline evaluates how well a molecule is tolerated in clinical trials and how much treatment burden patients may experience. It uses clinical-trial discontinuations due to adverse events, placebo and Standard of Care (SoC) comparisons, common adverse events, adverse-event severity, persistence, and management requirements.

The pipeline produces a deterministic score from 1 to 5, a detailed Gemini-generated justification, a concise dimension rationale, a BigQuery payload, a PDF report, and a JSON payload cache in GCS.

## Module Structure

```text
tolerability_and_patient_burden/
|-- __init__.py                                      Public package export
|-- README.md                                        Implementation documentation
|-- tolerability_and_patient_burden.py               Main orchestrator
|-- tolerability_tools.py                            Trial selection, extraction, scoring, and rationale
|-- tolerability_constants.py                        Extraction fields, weights, thresholds, and fallbacks
|-- gemini_extractor.py                              Grounded batched clinical-trial extraction
|-- bq_utils.py                                      BigQuery schemas and persistence helpers
|-- ddl_scripts/
|   `-- clinical_trials_tolerability_data_ddl.sql    Trial-table DDL
`-- generate_report_and_rationale/
    |-- __init__.py                                  Report helper exports
    |-- generate_tolerability_report.py               Enrichment, narrative, and PDF generation
    `-- generate_rationale.py                        Concise score rationale
```

## Pipeline Overview

```text
Step 0   Select new, updated, or all eligible clinical-efficacy trials
         |
Step 1   Extract tolerability fields from trial records with grounded Gemini
         |
Step 2   Replace refreshed trial rows and persist them to BigQuery
         |
Step 3   Load persisted trials, weight evidence, calculate rates, and score
         |
         +--> Grounded Gemini lookup of the Standard of Care benchmark
         +--> Gemini justification for the computed score (not search grounded)
         |
Step 4   Append the complete tolerability result to BigQuery
         |
Step 5   Generate the concise dimension-score rationale (not search grounded)
         |
Step 6   Append the dimension score and rationale to the shared score table
         |
Step 7   Generate the PDF report, optionally enriching thin data with grounded Gemini
         |
Step 8   Upload the complete payload cache to GCS
```

The public entry point is `tolerability_and_patient_burden_assessment()` in `tolerability_and_patient_burden.py`. It persists artifacts during execution and currently returns `None`.

## Step 0 - Trial Selection and Incremental Mode

`extract_incremental_trials()` reads eligible trial metadata from the configured `CLINICAL_EFFICACY_TABLE`.

- Full mode returns all distinct rows for the molecule where `llm_confidence` is above `TRIAL_LLM_CONFIDENCE_THRESHOLD`.
- Incremental mode reads the latest `created_at` from `CLINICAL_TRIALS_TOLERABILITY_DATA_TABLE`.
- The first incremental run has no prior timestamp and therefore behaves like a full run.
- Later incremental runs select trials that were created or updated after the stored timestamp, as well as clinical-efficacy trials not yet present in the tolerability trial table.
- Empty selection skips extraction, scoring, and BigQuery writes. A skip payload is still uploaded to the GCS dimension cache.

The selected metadata includes molecule, trial ID, dosage, phase, primary and secondary locations, and trial size.

## Step 1 - Grounded Trial Tolerability Extraction

`extract_tolerability_data()` delegates to the shared two-step clinical-trial extractor with tolerability-specific fields. Trial IDs are processed in batches of six with up to four concurrent workers. The extractor uses Google Search-grounded Gemini to search ClinicalTrials.gov, linked publications, and international registries.

The extracted fields include:

- Drug and placebo arm sizes
- Raw discontinuation counts due to adverse events for drug and placebo arms
- Active comparator name, arm size, and discontinuation count for head-to-head trials
- Top common non-serious adverse events with frequencies
- Overall AE severity: Mild, Moderate, Severe, Mixed, or unavailable
- AE persistence: Transient, Persistent, Mixed, Unknown, or unavailable
- Management required: None, Occasional, Regular, or unavailable

The extractor preserves the trial metadata supplied by BigQuery as authoritative. It uses `json_repair` and additional recovery logic for malformed, truncated, or wrapped Gemini JSON. ClinicalTrials.gov API v2 is used to override trial start and primary-completion dates when those fields are present in the shared extractor output.

Discontinuation rates are calculated locally from raw counts:

```text
Discontinuation Rate (%) = Discontinued Due to AE Count / Arm Size * 100
```

Drug-arm and comparator rates require a valid arm size. If the drug arm size is unavailable, the implementation estimates it as 65% of total trial size. Missing placebo or comparator arm data produces no rate for that arm.

## Step 2 - Trial Persistence

Refreshed trial rows are written to `CLINICAL_TRIALS_TOLERABILITY_DATA`.

- In incremental mode, matching molecule and trial-ID rows are deleted before refreshed rows are appended.
- In full mode, the trial-table load uses `WRITE_TRUNCATE`.
- Numeric values are normalized before loading into the explicit BigQuery schema.
- The persisted table contains trial metadata, arm sizes, raw discontinuation counts, comparator data, AE summaries, and derived discontinuation rates.

## Step 3 - Weighting, Rate Aggregation, and Scoring

`compute_tolerability_score()` loads all persisted trial rows for the molecule, not only the newly extracted rows. It normalizes them into the extractor's canonical field names, applies evidence weighting, calculates weighted discontinuation rates, fetches the SoC benchmark, and applies the scoring rules.

### Trial weighting

Each trial receives a weight based on phase, geography, dosage relevance, and sample size:

```text
Trial Weight = Phase Weight * Geography Weight * Dosage Weight * Sample Size
```

Phase weights:

| Phase | Weight |
|---|---:|
| Phase 3/4 | 1.00 |
| Phase 2 | 0.75 |
| Phase 1/2 | 0.60 |
| Phase 1 or unknown | 0.40 |

Geography weights:

| Tier | Regions | Weight |
|---|---|---:|
| Tier 1 | US, EU, UK | 1.00 |
| Tier 2 | Canada, Japan, Australia, Switzerland | 0.85 |
| Tier 3 | Other or unknown regions | 0.65 |

For approved molecules, dosage weights are approved `1.00`, off-label `0.70`, and exploratory `0.40`. For clinical-stage molecules, dosage weights are target `1.00`, alternative `0.75`, and exploratory `0.50`. The current orchestrator calls the scorer without passing approval status, so the default clinical-stage dosage weighting is used unless the scorer is called directly with `is_approved=True`.

Trials are prioritized by phase and evidence tier. Within a phase group, lower drug discontinuation is preferred, followed by higher calculated weight. Weighted rates use:

```text
Weighted Rate = Sum(Trial Weight * Trial Rate) / Sum(Trial Weight)
```

The drug rate counts trials with valid drug-arm data. The placebo rate is calculated independently from trials with valid placebo-arm data.

### Standard of Care benchmark

`get_soc_benchmark_dynamic()` sends the molecule class, optional indication, active comparators, and head-to-head trial context to Gemini with Google Search grounding enabled. The search prioritizes head-to-head trials, recent Phase 3 evidence, pooled analyses, real-world evidence, and then FDA labels.

A successful response returns a SoC drug, discontinuation rate, source, and `dynamic: true`. If the call fails, the response is malformed, or no numeric rate is returned, the implementation falls back to the hardcoded benchmark for the drug class, such as Liraglutide at 10% for `GLP-1`, with `dynamic: false`.

### Base score

The base score compares the weighted drug discontinuation rate with placebo:

| Condition | Base score |
|---|---:|
| Drug rate is at or below placebo rate | 5 |
| Drug rate is above placebo and below 5% | 4 |
| Drug rate is at least 5% and below 10% | 3 |
| Drug rate is at least 10% and below 20% | 2 |
| Drug rate is at least 20% | 1 |
| Drug rate is unavailable | 3 |

### SoC adjustment

The weighted drug rate is compared with the dynamic or fallback SoC rate:

| Comparison | Adjustment |
|---|---:|
| More than 2 percentage points better than SoC | +1 |
| Within 2 percentage points of SoC | 0 |
| More than 2 percentage points worse than SoC | -1 |

### Patient-burden adjustment

The first five prioritized trials are checked for severity, persistence, and management requirements:

| Evidence | Burden adjustment |
|---|---:|
| Severe AEs or regular management required | -2 |
| Persistent or moderate AEs, when the severe/regular condition is not met | -1 |
| Otherwise, mild/transient burden | 0 |

The final score is calculated and clamped to the range 1-5:

```text
Final Score = clamp(Base Score + SoC Adjustment + Burden Adjustment, 1, 5)
```

### Guardrail

The guardrail fails when both conditions are true:

- Drug discontinuation is greater than or equal to the SoC discontinuation rate.
- At least one of the first five prioritized trials shows persistent AEs or regular management requirements.

A failed guardrail records the reason that the drug is not competitive on tolerability. The guardrail does not independently change the numeric score.

### Result shape

The score result includes:

- `tolerability_score` and `score_numeric`
- Weighted drug and SoC rates, SoC source, and difference versus SoC
- Key AEs, AE frequencies, and severity summary
- Comparisons versus placebo and SoC
- Guardrail status and reason
- Detailed justification
- Base score, SoC adjustment, burden adjustment, and trial counts in `_scoring_breakdown`

The detailed justification is generated from the extracted values, weighted-rate calculations, adjustments, guardrail status, and the top prioritized trials. If Gemini fails, a structured local justification is returned.

## Steps 4-6 - BigQuery Score and Dimension Persistence

### Step 4 - Tolerability payload

`append_tolerability_to_bq()` appends one row to `TOLERABILITY_TABLE` with the final score, discontinuation rates, SoC source, AE summary, comparisons, guardrail data, justification, score breakdown, and timestamps. The table is append-only.

### Step 5 - Concise rationale

`generate_rationale_for_score()` reduces the result to the score, supporting discontinuation data, AE summary, trial count, and justification. Gemini generates a short plain-text rationale for the Medical Affairs audience. The prompt requires the rationale to use only supplied data and avoid external facts or specific numeric values. If generation fails, the fallback is `Rationale has not been generated.`

### Step 6 - Shared dimension score

`append_dimension_score_to_bigquery()` appends the numeric score and generated rationale to `DIM_SCORES_TABLE` using `TOLERABILITY_AND_PATIENT_BURDEN_DIMENSION_NAME`.

## Step 7 - PDF Report

`generate_tolerability_report()` loads the latest row for the molecule from `TOLERABILITY_TABLE` and builds a business-facing PDF with ReportLab. The report contains:

1. Key tolerability findings
2. Discontinuation-rate details
3. Adverse-event details
4. Comparator analysis
5. Insights and implications
6. Tolerability profile summary
7. Tolerability scoring reference

Before narrative generation, the report checks whether BigQuery data has a sufficiently long justification, key AEs, and a drug discontinuation rate. If the data is thin, it calls Gemini with Google Search grounding to enrich the molecule's tolerability information. If the data is sufficient, enrichment is skipped.

The final narrative call uses Gemini without Google Search grounding and is instructed to use only the loaded or enriched report data. If the narrative response is not valid JSON, a local fallback narrative is rendered. The orchestrator converts the report to PDF bytes and uploads the base and archive report artifacts through `upload_dimension_report_pdf_to_gcs()`.

## Step 8 - GCS Payload Cache

The orchestrator uploads the complete `score_result` to the dimension cache through `upload_dimension_payload_cache_to_gcs()`. The cache includes the score, scoring breakdown, trial counts, comparison data, guardrail, detailed justification, generated rationale and prompt payload, and report metadata when report generation succeeds.

When no incremental trials are available, the cache contains a skip payload with `status: "skipped_no_incremental_trials"`; extraction, scoring, and BigQuery writes are not performed.

## Step Input / Output Reference

| Step | Input | Output |
|---|---|---|
| 0. Trial selection | Molecule, clinical-efficacy table, incremental mode | Eligible trial metadata or skip result |
| 1. Tolerability extraction | Selected trial metadata | Trial-level arm, discontinuation, comparator, and AE fields |
| 2. Trial persistence | Extracted trial records | Rows in `CLINICAL_TRIALS_TOLERABILITY_DATA` |
| 3. Scoring | Persisted trial rows, drug class, indication | Weighted rates, SoC comparison, score, guardrail, justification |
| 4. Tolerability persistence | Score result | Row appended to `TOLERABILITY_TABLE` |
| 5. Concise rationale | Reduced score payload | Plain-text rationale and prompt payload |
| 6. Dimension score | Numeric score and rationale | Row appended to `DIM_SCORES_TABLE` |
| 7. PDF report | Latest tolerability row | PDF bytes and base/archive GCS URIs |
| 8. Payload cache | Complete score result | JSON cache uploaded to GCS |

## Gemini Usage

| Step | Task | Gemini capability | Google Search grounded? |
|---|---|---|---|
| 1 | Extract clinical-trial tolerability fields | Grounded structured extraction of arm sizes, AE discontinuation counts, comparators, common AEs, severity, persistence, and management burden | **Yes** |
| 3a | Identify the Standard of Care benchmark | Grounded search for the SoC drug and its AE discontinuation rate, prioritizing head-to-head and recent evidence | **Yes** |
| 3b | Generate the detailed score justification | Explain extracted values, weighted discontinuation rates, score adjustments, patient burden, and guardrail status | **No** |
| 5 | Generate the concise dimension rationale | Plain-text qualitative rationale based only on the computed score result | **No** |
| 7a | Enrich thin report data | Grounded tolerability research when stored BigQuery justification, AE, or discontinuation data is insufficient | **Yes, conditional** |
| 7b | Write the PDF report narrative | Structured business-facing narrative generated from loaded or enriched report data | **No** |

Step 1 and Step 3a explicitly pass `google_search=True`. Step 7a passes `use_search=True` only when the latest BigQuery row does not contain sufficient report data. Steps 3b, 5, and 7b use Gemini without Google Search grounding and rely on data already supplied to their prompts.

## Programmatic Usage

Run from the repository root:

```python
import asyncio

from medical_potential.tolerability_and_patient_burden import (
    tolerability_and_patient_burden_assessment,
)


async def main():
    await tolerability_and_patient_burden_assessment(
        molecule_name="semaglutide",
        drug_class="GLP-1",
        indication="obesity",
        incremental_mode=True,
    )


asyncio.run(main())
```

The public function persists results and returns `None`. Inspect `TOLERABILITY_TABLE`, `DIM_SCORES_TABLE`, the generated report URI, or the GCS payload cache for artifacts.

## Configuration and Dependencies

Configuration is read from `medical_potential.config` and the repository environment:

- `PROJECT_ID`
- `BQ_DATASET_ID`
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GEMINI_API_KEY`
- `GEMINI_FLASH_PREVIEW_MODEL`
- `CLINICAL_EFFICACY_TABLE`
- `CLINICAL_TRIALS_TOLERABILITY_DATA_TABLE`
- `TOLERABILITY_TABLE`
- `TOLERABILITY_AND_PATIENT_BURDEN_DIMENSION_NAME`
- `DIM_SCORES_TABLE`
- `TRIAL_LLM_CONFIDENCE_THRESHOLD`
- `GCS_BUCKET`
- `GCS_REPORT_BASE_PATH`
- `GCS_PIPELINE_CACHE_BASE_PATH`
- `GCS_MEDICAL_POTENTIAL_SUBFOLDER`

Install dependencies from `requirements.txt`. The module uses Google Cloud BigQuery and Storage, Gemini access, `json_repair`, `pandas`, `aiohttp`, and ReportLab.

