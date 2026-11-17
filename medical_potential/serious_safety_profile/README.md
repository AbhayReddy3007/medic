# Serious Safety Profile Pipeline

The `serious_safety_profile` pipeline evaluates the serious safety risk of a molecule using clinical-trial evidence, serious adverse event (SAE) classification, regulatory signals, and post-marketing findings. It produces an explainable score from 1 to 5, a concise rationale, structured pipeline output, and an optional PDF report.

## Module Structure

```text
serious_safety_profile/
|-- serious_safety_profile.py              Main 15-step orchestrator
|-- run_serious_safety_parallel.py         Batch runner
|-- bq_utils.py                            Trial and SAE BigQuery I/O
|-- llm_runtime.py                         Gemini calls and JSON parsing
|-- web_data_extraction/
|   |-- clinical_trial_extractor.py        Trial extraction
|   |-- helper_functions.py                Selection and grounded assessments
|   |-- sae_categorization.py              SAE aggregation and classification
|   |-- trial_level_weight_calculator.py   Weighting and rate aggregation
|   |-- prompts.py                         Grounded extraction prompts
|   `-- utils.py                           BigQuery loading and normalization
|-- scoring_logic/
|   |-- serious_safety_profile_scoring.py  Deterministic score
|   |-- trial_banding.py                   S1-S4 bands and weight tiers
|   |-- adjustments.py                     Regulatory and post-market caps
|   |-- parsing.py                         Rate, severity, and presence parsing
|   `-- constants.py                       Score and severity mappings
|-- generate_report_and_rationale/
|   |-- generate_report.py                 PDF generation
|   `-- generate_rationale.py              Score rationale generation
`-- business_logic_for_dimension/          Reference business-logic documents
```

## Pipeline Overview

```text
Step 0   Approval and market-status assessment
         |
Step 1   Incremental or full clinical-trial selection from BigQuery
         |
Step 2   Google Search Grounded Serious Adverse events (SAE) extraction from trial records
         |
Step 3   Persist refreshed trial-level safety data
         |
Step 4   Summarize unique SAEs across trials
         |
Step 5   Merge SAE occurrences into the classification cache
         |
Step 6   Classify new SAEs by expectedness and severity
         |
Step 7   Persist new SAE classifications
         |
Step 8   Calculate trial weights and aggregate SAE rates
         |
Step 9   Assess regulatory impact
         |
Step 10  Assess post-marketing safety for marketed molecules
         |
Step 11  Calculate the deterministic 1-5 score
         |
Step 12  Generate and upload the PDF report (optional)
         |
Step 13  Generate the score rationale
         |
Step 14  Append the score and rationale to BigQuery
         |
Step 15  Cache the complete payload in GCS
```

The public entry point is `run_serious_safety_assessment()` in `serious_safety_profile.py`. Incremental mode is enabled by default. When no new or updated clinical-efficacy trials are found, extraction and trial persistence are skipped and existing SAE classifications are loaded from BigQuery.

## Step 0 - Approval and Market Status

- Uses Gemini with search grounding to determine whether the molecule is approved and marketed.
- Returns `is_approved`, `is_marketed`, `status_summary`, and `major_markets`.
- The approval result controls dosage weighting.
- The marketed result controls whether post-marketing assessment runs.

## Step 1 - Trial Selection

The source is the configured `CLINICAL_EFFICACY_TABLE` in BigQuery.

- Filters to the molecule and rows above `TRIAL_LLM_CONFIDENCE_THRESHOLD`.
- Full mode returns all eligible trials.
- Incremental mode compares clinical-efficacy `created_at` and `updated_at` values with the latest `created_at` timestamp in `CLINICAL_TRIALS_SERIOUS_SAFETY_DATA`.
- The first run has no prior timestamp and therefore behaves like a full run.
- Updated trial IDs in `CLINICAL_EFFICACY_TABLE` are deleted from the serious-safety table before replacement, preventing duplicate rows.

## Step 2 - Clinical-Trial SAE Extraction

`extract_serious_safety_data()` processes the selected trial records for the serious-safety assessment.

Trial identifier, phase, dosage, geography, and total size come from the clinical-trial metadata selected in Step 1. Grounded Gemini extraction is used for the serious-safety fields:

- Drug-arm and control-arm sample sizes
- SAE counts and rates for both arms
- Serious events reported
- Deaths reported and drug-related deaths
- Rationale for drug-related death assessment

When rates are absent, they are calculated as:

```text
SAE Rate (%) = SAE Count / Arm Size * 100
Delta SAE (%) = SAE Rate Drug (%) - SAE Rate Control (%)
```

The extracted SAE fields are added to the selected trial records. The resulting records are persisted for later aggregation and scoring.

## Step 3 - Trial-Level Safety Data Persistence

Extracted records are appended to `CLINICAL_TRIALS_SERIOUS_SAFETY_DATA` in the configured project and dataset. In incremental mode, only refreshed trial IDs are replaced. The table contains molecule and trial metadata, arm sizes, SAE counts and rates, reported events, death fields, delta SAE, and `created_at`.

## Step 4 - SAE Summarization

`summarize_sae_across_trials()` converts trial-level `Serious Events Reported` values into one record per SAE. The same trial is counted once per SAE, even if the event appears more than once in that trial.

The intermediate shape is:

```json
{
  "sae_name": "example event",
  "number_of_studies": 2,
  "list_of_studies": ["trial-1", "trial-2"]
}
```

## Step 5 - Existing SAE Cache Update

The pipeline checks each summarized SAE against existing rows for the molecule in `SERIOUS_SAFETY_PROFILE_SAE_SUMMARY`. Existing rows are updated with merged, de-duplicated study IDs. New events are passed to Step 6 for classification.

## Step 6 - SAE Expectedness and Severity Classification

New SAE events are sent to grounded Gemini in batches. The orchestrator uses the `GLP-1` molecule class and a batch size of 4.

Each event is classified as:

- Expected, expected but rare, unexpected, or another normalized category
- Manageable/non-specific, organ-specific, severe/life-threatening, or catastrophic

The implementation normalizes model labels and applies a deterministic correction for cancer-like SAE names that are labeled catastrophic without an explicit death, fatality, permanent disability, irreversible outcome, or organ-failure signal. Failed model calls and malformed JSON are retained as parse-error records for review.

## Step 7 - New SAE Classification Persistence

Newly classified events are appended to `SERIOUS_SAFETY_PROFILE_SAE_SUMMARY`.

Cached classification fields include:

- `sae_name`
- `number_of_studies`
- `list_of_studies`
- `expectedness_classification`
- `expectedness_rationale`
- `severity_category`
- `severity_rationale`
- Deterministic post-processing fields and parse-error metadata

## Step 8 - Trial Weighting and SAE Aggregation

Persisted trials are normalized and weighted using four factors:

```text
Trial Weight = Phase Weight * Geography Weight * Dosage Weight * Trial Size
```

### Phase weights

| Phase | Weight |
|---|---:|
| 3/4 | 1.00 |
| 2 | 0.75 |
| 1/2 | 0.60 |
| 1 | 0.40 |

Missing or unrecognized phases use the conservative 0.40 weight.

### Geography weights

| Regulatory tier | Regions | Weight |
|---|---|---:|
| Tier 1 | US, EU, UK | 1.00 |
| Tier 2 | Canada, Japan, Australia, Switzerland | 0.85 |
| Tier 3 | Other regions or unknown | 0.65 |

For multi-region values, the highest recognized regional weight is used.

### Dosage weights

For approved molecules: approved dose `1.00`, off-label dose `0.70`, exploratory dose `0.40`.

For clinical-stage molecules: target dose `1.00`, alternative dose `0.75`, exploratory dose `0.50`.

The pipeline normalizes each trial weight by the total weight and calculates arm-specific weighted rates:

```text
Normalized Weight = Trial Weight / Sum of Trial Weights
Weighted Rate = Sum(Trial Weight * Rate) / Sum(Trial Weight)
Weighted Delta SAE = Weighted Drug Rate - Weighted Control Rate
```

The output includes the weighted rates, delta rate, number of trials used, and the full trial list with raw and normalized weights.

## Step 9 - Regulatory Impact

Gemini with search grounding reviews public regulatory information and the persisted trial safety summary. It returns a structured regulatory assessment, including the highest relevant regulatory consequence, rationale, safety drivers, grounding summary, and raw response.

The scoring engine recognizes these cap categories:

| Regulatory signal | Maximum score |
|---|---:|
| Market-threatening action, withdrawal, or rejection | 1 |
| Established serious human risk | 2 |
| Monitoring or REMS | 3 |
| Informational or precautionary action | No cap |
| No action or unrecognized signal | No cap |

## Step 10 - Post-Marketing Safety

Post-marketing review runs only when Step 0 identifies the molecule as marketed. Grounded Gemini evaluates:

- New serious risks
- Stronger warnings
- REMS additions
- Boxed-warning additions
- Major label restrictions
- Withdrawal from the market

New serious risks reduce the score by 1. A boxed warning or withdrawal caps the score at 2; a REMS or major label restriction caps it at 3. Non-marketed molecules receive an explicit `evaluated: false` result with a skip reason.

## Step 11 - Score Calculation

Implemented in `scoring_logic/serious_safety_profile_scoring.py`, the score is deterministic after the upstream evidence has been collected.

### 11.1 Trial banding and base score

Trial banding converts each trial's SAE burden into an S1-S4 risk band. The input is `Delta SAE (%)`, which is calculated as the drug-arm SAE rate minus the control-arm SAE rate. If an explicit delta is unavailable, the scorer derives it from the two arm rates. A trial with missing rates has no usable delta and is excluded from band counts.

Each usable trial is assigned a band:

| Band | Delta SAE | Base score |
|---|---:|---:|
| S1 | `<= 0` | 5 |
| S2 | `> 0` and `<= 0.5` | 4 |
| S3 | `> 0.5` and `<= 2.0` | 3 |
| S4 | `> 2.0` | 2 |

The bands are interpreted as follows:

- **S1:** The drug does not have a higher SAE rate than control. This is the most favorable trial-level signal.
- **S2:** The drug has a small positive excess SAE rate, up to and including 0.5 percentage points.
- **S3:** The excess is greater than 0.5 and up to and including 2.0 percentage points.
- **S4:** The excess is greater than 2.0 percentage points and represents the strongest trial-level concern.

#### Weight tiers

Band counts are not treated as a simple unweighted vote. Before the bands are compared, each persisted trial already has a `normalized_weight` from Step 8. The scorer sorts trials from highest to lowest normalized weight and assigns:

1. **High tier:** trials making up the leading cumulative weight, through the 50% threshold. If the next trial has the same weight as the boundary trial, it is kept in the same tier.
2. **Medium tier:** subsequent trials through approximately the next 30% of cumulative weight, again keeping equal-weight boundary trials together.
3. **Low tier:** all remaining trials.

Trials with missing or negative normalized weights are treated as having weight 0 for tier assignment. The scorer records both `band` and `weight_tier` on every trial and separately counts all bands and bands within the high tier.

#### Selecting the base score

The base score is selected in this order:

1. If the combined normalized weight of S4 trials is greater than 0.20, S4 is present and the base score is forced to 2, regardless of the other band counts.
2. If S4 trials make up no more than 0.20 of normalized weight, every S4 trial is reclassified as S3 before counts are finalized. This prevents a small amount of low-weight evidence from dominating the base score.
3. If no S4 remains, the scorer compares the number of high-tier trials in S1, S2, and S3. The band with the largest high-tier count becomes dominant and maps to its table score.
4. If high-tier counts tie, the band with the lower score is selected. In practice, this makes the tie-break conservative: S3 is preferred over S2, and S2 over S1.

For example, if the high-tier counts are S1=2, S2=1, and S3=0, the dominant band is S1 and the base score is 5. If the counts are S1=1, S2=1, and S3=1, S3 wins the tie and the base score is 3. If S4 has 25% of the normalized weight, the base score is 2 even if S1 has more high-tier trials.

When every trial has an unusable delta, or all available drug and control SAE rates are zero or missing, the scorer uses the no-delta default base score of 1. A separate no-SAE-events check also applies the default score of 1 when the SAE classification cache contains no events. Pattern, expectedness, regulatory, and post-marketing adjustments are applied after this base-score stage.

### 11.2 Pattern and severity adjustments

- At least two trials with positive delta SAE and drug-related deaths associated with severity-4 events force the score to 1.
- Life-threatening events in the presence of an S4 trial reduce the score by 1.
- Repeated organ-specific events reduce the score by 1.
- Repeated life-threatening events reduce the score by 1.
- For marketed molecules, repetition is based on 5% of trials; for non-marketed molecules, it requires at least two studies.
- A molecule with no classified SAE events receives the no-SAE default score of 1.

### 11.3 Expectedness adjustments

An event is evaluated using severity and presence rate across studies:

| Severity | Clinically meaningful presence |
|---|---|
| Life-threatening or catastrophic | At least 3%; 1-3% only when unexpected |
| Organ-specific | At least 10%; 3-10% only when unexpected |
| Manageable | At least 10% and unexpected |

Each clinically meaningful unexpected event reduces the score by 1. Each clinically meaningful expected-but-rare event also reduces the score by 1.

### 11.4 Final score

Regulatory and post-marketing caps are applied after clinical-trial adjustments. The result is finally clamped to the range 1-5:

```text
Final Score = max(1, min(5, adjusted score))
```

The score result preserves the step-by-step reasoning, key drivers, adjustments applied, trial banding, and SAE categorization.

## Step 12 - Report and Rationale Generation

### PDF report

When `generate_pdf_report=True`, Gemini generates structured report content and the module renders it to PDF. The report covers:

1. Serious safety profile landscape
2. Clinical-trial safety
3. Serious adverse events found
4. Regulatory assessment
5. Post-marketing safety

The PDF is uploaded through `upload_dimension_report_pdf_to_gcs()` using the configured report base path, Medical Potential subfolder, dimension name, molecule, and timestamp.

### Rationale

`generate_serious_safety_rationale()` produces the concise business rationale used with the score. The score and rationale are appended to the configured `DIM_SCORES_TABLE` with dimension name `Serious Safety Profile`.

### Pipeline cache

The complete output payload is uploaded through `upload_dimension_payload_cache_to_gcs()`. The cache includes trial data, SAE classifications, regulatory and post-marketing assessments, score reasoning, report content, rationale, and report/rationale prompt payloads.

## Step Input / Output Reference

| Step | Input | Output |
|---|---|---|
| 0. Market status | Molecule name | Approval/marketed flags, status summary, major markets |
| 1. Trial selection | Molecule name, clinical-efficacy table, incremental mode | Eligible full or newly changed trial metadata |
| 2. SAE extraction | Selected trials, extraction model | Trial-level SAE fields and derived rates |
| 3. Trial persistence | Extracted trial records | Rows in `CLINICAL_TRIALS_SERIOUS_SAFETY_DATA` |
| 4. SAE summarization | Extracted trial records | Unique SAE names and study occurrence lists |
| 5. Existing SAE merge | SAE summary plus cached rows | Updated occurrence counts for known events; list of new events |
| 6. SAE classification | New SAE events, molecule class | Expectedness, severity, rationales, parse metadata |
| 7. SAE persistence | Classified new events | Rows in `SERIOUS_SAFETY_PROFILE_SAE_SUMMARY` |
| 8. Weighted aggregation | Persisted trial rows, approval status | Weighted drug/control rates, delta, normalized trial weights |
| 9. Regulatory assessment | Molecule name and trial summary | Regulatory consequence, cap inputs, rationale, sources |
| 10. Post-marketing assessment | Marketed molecule name | Post-marketing risks, adjustments, cap inputs, sources |
| 11. Score calculation | Aggregation, SAE cache, regulatory and post-marketing results | Score, reasoning, key drivers, applied adjustments |
| 12. PDF report | Report input payload | PDF uploaded to GCS and structured report content |
| 13. Rationale | Report input payload | Concise score rationale |
| 14. Score persistence | Molecule, dimension, score, rationale | Row appended to `DIM_SCORES_TABLE` |
| 15. Payload cache | Complete output payload | JSON cache uploaded to GCS |

## Gemini Usage

| Step | Task | Gemini capability |
|---|---|---|
| 0 | Determine approval and marketed status | Grounded search, extraction, and classification |
| 2 | Extract clinical-trial SAE fields | Grounded structured extraction |
| 6 | Classify SAE expectedness and severity | Batched structured classification |
| 9 | Assess regulatory impact | Grounded regulatory research and synthesis |
| 10 | Assess post-marketing safety | Grounded post-market research and synthesis |
| 12 | Write the PDF report | Structured narrative generation |
| 13 | Write the score rationale | Concise business-language generation |

All Gemini calls use the configured `GEMINI_FLASH_PREVIEW_MODEL` unless a caller supplies a different model to an internal function. JSON responses are parsed defensively using the shared runtime helpers.

## Programmatic Usage

Run from the repository root:

```python
import asyncio

from medical_potential.serious_safety_profile import run_serious_safety_assessment


async def main():
    result = await run_serious_safety_assessment(
        molecule_name="semaglutide",
        max_trials=10,
        generate_pdf_report=True,
        incremental_mode=True,
    )
    print(result["safety_score"]["score"])


asyncio.run(main())
```

The main result contains `molecule_name`, `approval_market_status`, `total_trials`, `sae_aggregation`, `sae_event_categorization`, `regulatory_impact`, `post_marketing_safety`, `safety_score`, report content, rationale, and GCS URIs when those artifacts are created.

## Configuration and Dependencies

Configuration is read from `medical_potential.config` and the repository `.env` file.

Required or commonly used settings:

- `PROJECT_ID`
- `BQ_DATASET_ID`
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GEMINI_API_KEY` or the configured Gemini authentication mechanism
- `GCS_BUCKET`
- `GCS_REPORT_BASE_PATH`
- `GCS_PIPELINE_CACHE_BASE_PATH`
- `GCS_MEDICAL_POTENTIAL_SUBFOLDER`
- `CLINICAL_EFFICACY_TABLE`
- `CLINICAL_TRIALS_SERIOUS_SAFETY_DATA`
- `SERIOUS_SAFETY_PROFILE_SAE_SUMMARY`
- `DIM_SCORES_TABLE`
- `TRIAL_LLM_CONFIDENCE_THRESHOLD`

Install the repository dependencies from `requirements.txt`.

