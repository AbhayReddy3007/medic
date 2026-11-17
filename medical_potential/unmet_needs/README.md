# Unmet Needs Pipeline

The `unmet_needs` module evaluates whether a molecule addresses documented unmet medical needs using clinical-trial evidence. It has two related workflows:

1. **Unmet-needs baseline extraction:** periodically builds a multi-country repository of unmet needs from Standard of Care (SoC) PDFs and web research.
2. **Molecule-level assessment:** translates a molecule's clinical-trial evidence into a country-specific view of how meaningfully it addresses existing unmet needs, helping assess its potential to deliver differentiated medical value across markets.

The molecule-level assessment produces a weighted score, row-level scoring evidence, a business-facing PDF, a concise rationale, BigQuery rows, and a complete JSON payload cache in GCS.

## Module Structure

```text
unmet_needs/
|-- __init__.py                                      Package marker
|-- README.md                                        Implementation documentation
|-- unmet_needs_assessment.py                       Six-stage molecule orchestrator
|-- compute_unmet_need_score_from_trial_data/
|   |-- __init__.py                                 Mapping/scoring exports
|   |-- trial_to_unmet_needs_mapper.py              Trial loading, grounded mapping, persistence
|   |-- unmet_need_scorer.py                        Row scoring and country/dimension aggregation
|   `-- bq_utils.py                                 BigQuery schemas and import/export helpers
|-- generate_report_and_rationale/
|   |-- __init__.py                                 Report helper exports
|   |-- generate_report.py                           Payload assembly and PDF generation
|   `-- generate_rationale.py                       Concise rationale generation
|-- unmet_needs_extraction/
|   |-- run_unmet_needs_extraction.py               Baseline refresh runner
|   |-- unmet_needs_extraction_from_soc.py          PDF-based SoC extraction
|   |-- unmet_needs_extraction_from_web.py          Three-turn grounded web extraction
|   |-- prompts.py                                   Baseline extraction prompts
|   `-- utils.py                                     Parsing, normalization, and token helpers
|-- business_logic_documents/                        Reference business logic documents
|-- soc_documents/                                   Country-specific SoC PDFs
`-- outputs/                                         Local JSON/Excel extraction outputs
```

## Pipeline Overview

### Baseline extraction

```text
Step 1   Extract unmet needs from country-specific SoC PDFs
         |
Step 2   Research additional countries with a three-turn grounded web workflow
         |
Step 3   Normalize and combine the extracted records
         |
Step 4   Replace refreshed country rows in the baseline BigQuery table
```

### Molecule-level assessment

```text
Step 1   Load the unmet-needs baseline and eligible clinical trials
         |
Step 2   Map trial evidence to unmet needs with Google Search-grounded Gemini
         |
Step 3   Persist refreshed trial-to-unmet-need mappings
         |
Step 4   Score unique unmet needs from persisted mappings
         |
Step 5   Aggregate row-level scores into one molecule-level score
         |
Step 6   Persist scoring rows and build the report payload
         |
Step 7   Generate and upload the PDF report
         |
Step 8   Generate the concise rationale and append the shared dimension score
         |
Step 9   Cache the complete assessment payload in GCS
```

The public molecule-level orchestrator is `unmet_needs_assesment()` in `unmet_needs_assessment.py`. It persists artifacts during execution and currently returns `None`.

The command-line runner can process one molecule, an explicit list, or molecules returned by a BigQuery query. It can also run molecule assessments in parallel worker processes.

## Baseline Extraction Workflow

The baseline extraction workflow is implemented under `unmet_needs_extraction/` and is run by `run_unmet_needs_extraction.py`.

### Step 1 - SoC document extraction

Country-specific PDF files are loaded from the configured local directories for the US, UK, Germany, and France. All PDFs found for a country are sent together to Gemini with a structured JSON response requirement.

This call uses the PDF documents supplied in the request and does **not** enable the Google Search tool. The extracted records are normalized into patient-focused unmet needs with supporting evidence, country, and `source: "soc_document"`.

### Step 2 - Web extraction

The default web countries are China, India, Brazil, Australia, Russia, Japan, and Korea. Each country is processed concurrently through a three-turn Gemini workflow:

1. Broad epidemiological and public-health landscape research using Google Search grounding.
2. Focused clinical, therapeutic, financial-barrier, and care-delivery research using Google Search grounding and the conversation history.
3. Structured JSON synthesis from the conversation history with no Google Search tool.

Grounding sources from the first two turns are collected and de-duplicated. Extracted records receive `source: "web_search"` and their country label.

### Step 3 - Normalize and combine

Successful SoC and web results are converted to DataFrames and concatenated. The extraction runner adds:

- `created_at_utc` with the current UTC timestamp
- `inclusion_flag: "N"` for the newly generated rows

The baseline table is intentionally refreshed by country rather than globally. Countries represented in the current run have their existing rows deleted before the new rows are appended.

### Step 4 - Baseline persistence

The baseline is written to the configured `UNMET_NEEDS_TABLE`. The expected fields are:

- `unmet_need`
- `category`
- `evidence`
- `country`
- `source`
- `inclusion_flag`
- `created_at_utc`

The molecule-level mapper reads only rows with `inclusion_flag = 'Y'`. The extraction runner currently writes new rows with `N`, so governance or a subsequent promotion process must mark approved baseline records as included before assessments can use them.

## Molecule-Level Assessment Workflow

## Step 1 - Assessment Trial and Baseline Loading

`map_clinical_trial_data_with_unmet_needs()` loads two sources:

- The baseline unmet-needs table, filtered to `inclusion_flag = 'Y'`.
- The configured `CLINICAL_EFFICACY_TABLE`, filtered to `llm_confidence > TRIAL_LLM_CONFIDENCE_THRESHOLD`.

In incremental mode, the mapper reads the latest mapping timestamp for the molecule and selects clinical-efficacy trials whose `created_at` or `updated_at` is newer. With no previous mapping timestamp, incremental mode behaves like a full load. Full mode loads all eligible trials for the molecule.

Before mapping, the mapper removes rows with missing trial IDs or unparseable phases and deduplicates by normalized trial ID. When duplicate trial IDs exist, the highest phase is retained; ties preserve the first loaded row.

If no eligible trials remain, the mapper returns `clinical_trials_assessed: false`. The orchestrator skips scoring and BigQuery score writes, uploads a skip payload to GCS, and returns.

## Step 2 - Trial-to-Unmet-Need Mapping

Eligible trials are processed in batches of six with at most three mapping batches running concurrently. Each prompt contains trial metadata, clinical endpoints, the full included unmet-needs baseline, and country/category context.

The grounded Gemini mapping call searches for each trial ID on ClinicalTrials.gov and in relevant published literature. It determines whether trial design, endpoints, or reported outcomes resolve or improve each listed unmet need. A mapping record may include:

- `addresses_unmet_need`
- `evidence_strength`: Strong, Moderate, or Weak
- `relevant_endpoint`
- Quantitative explanation of the relationship between trial evidence and unmet need
- `search_sources_used`
- Trial metadata such as phase, status, size, and location

Only trial-need relationships supported by the provided trial data and search results should be retained. Source URLs returned as Vertex grounding redirects are resolved where possible and cached to avoid repeating redirect requests.

Malformed or truncated model output is handled through JSON cleanup, standard JSON parsing, `json_repair`, and field-level recovery. A failed batch is isolated and returns no mappings so other batches can complete.

## Step 3 - Mapping Persistence

The mapper normalizes the results and refreshes the `TRIAL_MAPPING_TABLE`.

- Incremental mode deletes existing mapping rows for the trial IDs being reprocessed, then appends the refreshed mappings.
- Full mode loads the mapping table with `WRITE_TRUNCATE`.
- Tables are created when needed using `TRIAL_MAPPING_SCHEMA`.
- Every exported row receives a shared UTC `created_at_utc` timestamp.

The mapping schema contains:

| Field | Purpose |
|---|---|
| `trial_id` | Clinical trial identifier |
| `molecule_name` | Requested molecule |
| `unmet_need` | Baseline unmet-need statement |
| `category` | Baseline category |
| `country` | Evidence geography |
| `addresses_unmet_need` | Mapping decision from Gemini |
| `evidence_strength` | Strong, Moderate, or Weak |
| `relevant_endpoint` | Trial endpoint linked to the need |
| `explanation` | Evidence-based explanation |
| `search_sources_used` | Serialized source references |
| `phase` | Trial phase |
| `phase_status` | Trial status |
| `trial_size` | Trial enrollment |
| `trial_location` | Trial location |
| `created_at_utc` | Mapping load timestamp |

## Step 4 - Row-Level Unmet-Need Scoring

`score_unmet_needs()` imports all mapping rows for the requested molecule from `TRIAL_MAPPING_TABLE`. The scoring stage does not apply the incremental cutoff again; it re-scores the complete set of mapped rows for the molecule.

Mappings are grouped by the tuple `(unmet_need, category, country)`. Each unique group retains its supporting trial evidence. Groups are scored in batches of three with at most three concurrent Gemini scoring tasks.

The scoring prompt asks Gemini to return two independent 1-5 values:

### Unmet-need benefit score

| Score | Meaning |
|---:|---|
| 5 | Substantial clinically meaningful improvement likely to change treatment practice, guidelines, or standard care |
| 4 | Clear and clinically meaningful improvement versus current SoC |
| 3 | Measurable improvement with moderate clinical benefit |
| 2 | Limited improvement versus current SoC |
| 1 | Minimal or no meaningful improvement |

### Evidence-impact score

| Score | Evidence maturity |
|---:|---|
| 5 | Multiple Phase 3, approved, or post-marketing real-world evidence |
| 4 | Strong Phase 3 evidence |
| 3 | Positive Phase 2 evidence |
| 2 | Early clinical evidence |
| 1 | Preclinical, theoretical, or unsupported evidence |

The score is conservative when evidence is indirect, weak, or not comparator-grounded. The model also returns concise rationales and a confidence label of High, Medium, or Low. The implementation enforces source identity fields and accepts only integer scores from 1 through 5. Invalid or unparseable values become null, and a failed batch produces low-confidence fallback rows.

Each valid scored row receives:

```text
Evidence Impact Weight = Evidence Impact Score / 5
Weighted Unmet-Need Score = Unmet-Need Score * Evidence Impact Weight
```

The category is mapped to a rollup dimension:

| Category | Dimension |
|---|---|
| Treatment Burden & Adherence Complexity | C |
| Achievement of Clinical Targets | C |
| Durability of Disease Control | C |
| Complication Prevention & Management | C |
| Safety & Treatment-Related Risk | S |
| Quality of Life & Psychosocial Burden | R |
| Diagnostic Accuracy & Early Detection | X |
| Access, Affordability & Care Delivery | X |
| Population-Specific Gaps & Inequities | X |

`X` rows are still scored and persisted, but receive zero weight in the final molecule score.

## Step 5 - Molecule-Level Aggregation

`aggregate_single_score()` aggregates valid weighted row scores in four stages:

1. Group scores by rollup dimension and country.
2. Average scores within each country for each dimension.
3. Apply US/EU coverage weights to each dimension.
4. Apply the final C/S/R dimension weights.

Country handling:

- When both US and one or more EU countries are present, US receives `0.6` and EU countries share `0.4`.
- With US only, US receives `0.6`.
- With EU only, EU countries share `0.4`.
- Secondary countries have a configured weight of `0` and therefore do not change the score.
- A dimension with neither US nor EU evidence contributes zero.

Final dimension weights are:

| Dimension | Meaning | Weight |
|---|---|---:|
| C | Clinical Importance | 0.40 |
| S | Safety/Diagnostic | 0.35 |
| R | Real-World Adjustment | 0.25 |
| X | Excluded from rollup | 0.00 |

The final score is:

```text
Final Score = round((0.40 * C) + (0.35 * S) + (0.25 * R), 2)
```

If no valid dimension scores remain, aggregation returns `None`. Confidence is retained per scoring row but is not aggregated into a molecule-level confidence value.

## Step 6 - Scoring Persistence and Report Payload

Scored rows are appended to `UNMET_NEED_SCORING_TABLE` using `UNMET_NEED_SCORING_SCHEMA`. The table stores the two raw scores, rationales, confidence, trial count, evidence-impact weight, weighted row score, dimension, country, and timestamp.

The report payload loads:

- All mapping rows for the molecule
- Only the latest timestamped scoring rows for the molecule
- The aggregate score, total unique trials, and total relevant unmet needs

The mapping query is not timestamp-filtered, while scoring rows are limited to the latest scoring timestamp. This means the report can include historical mapping rows alongside the latest score rows.

## Step 7 - PDF Report

`generate_prompt_unmet_needs_report_bytes()` prepares a compact report payload and asks Gemini to write two sections:

1. Executive Summary
2. Unmet Needs Addressed with Clinical Evidence

The prompt requires the narrative to use only the supplied structured data, include quantitative endpoints where available, acknowledge sparse evidence, and avoid internal scoring jargon. If JSON parsing fails, local fallback sections are generated.

The content is rendered with ReportLab into a business-facing PDF. The orchestrator uploads the PDF through `upload_dimension_report_pdf_to_gcs()` and records base and archive GCS URIs in the final payload.

## Step 8 - Rationale and Shared Dimension Score

`generate_unmet_needs_rationale()` receives a reduced scoring payload. It excludes mapping rows, filters out dimension `X`, keeps up to 500 scoring rows, and asks Gemini for approximately 50 words describing the major unmet needs driving the result. The prompt requires plain text and disallows external facts, internal field names, and methodology explanations.

This rationale call does not use Google Search grounding. If generation fails or returns empty text, the fallback is `Rationale has not been generated.`

The numeric aggregate score and rationale are appended to the shared `DIM_SCORES_TABLE` using `UNMET_NEEDS_DIMENSION_NAME`.

## Step 9 - GCS Payload Cache

The complete assessment payload is uploaded through `upload_dimension_payload_cache_to_gcs()`. It includes:

- Molecule and incremental-mode metadata
- Trial-assessment status and count
- Mapping and scoring results
- Aggregate score and configured weights
- Report input data and report metadata
- Generated rationale
- Report and cache GCS URIs when available

When no eligible incremental trials exist, the cache contains `status: "skipped_no_incremental_trials"` and the nested mapping payload. Mapping, scoring, report, and shared score writes are skipped.

## Step Input / Output Reference

| Step | Input | Output |
|---|---|---|
| 1. Trial and baseline loading | Baseline table, clinical-efficacy table, molecule, incremental mode | Included unmet needs and eligible deduplicated trials |
| 2. Trial mapping | Trial batches and baseline unmet needs | Trial-to-unmet-need mappings with evidence and sources |
| 3. Mapping persistence | Normalized mappings | Rows in `TRIAL_MAPPING_TABLE` |
| 4. Row scoring | Mapping rows grouped by unmet need/category/country | Benefit score, evidence-impact score, rationales, weighted row score |
| 5. Aggregation | Valid scored rows | Country-weighted C/S/R aggregate score |
| 6. Scoring/report payload | Mapping rows, latest scoring rows, aggregate score | BigQuery scoring rows and report input payload |
| 7. PDF report | Report input payload | PDF and base/archive GCS URIs |
| 8. Dimension rationale | Reduced scoring payload | Plain-text rationale and shared dimension score |
| 9. Payload cache | Complete assessment output | JSON cache in GCS |

## Gemini Usage

| Step | Task | Gemini capability | Google Search grounded? |
|---|---|---|---|
| Baseline 1 | Extract unmet needs from SoC PDFs | Structured document extraction from supplied PDF content | **No** |
| Baseline 2a | Research country-level epidemiological and public-health gaps | Multi-turn web research | **Yes** |
| Baseline 2b | Research clinical, therapeutic, financial, and care-delivery barriers | Multi-turn web research | **Yes** |
| Baseline 2c | Synthesize web findings into structured unmet-needs JSON | Structured synthesis from grounded conversation history | **No** |
| Assessment 2 | Map clinical-trial evidence to unmet needs | Trial-ID and literature research with structured mapping | **Yes** |
| Assessment 4 | Score benefit and evidence maturity for each unmet need | Rubric-guided structured scoring from persisted mapping evidence | **No** |
| Assessment 7 | Generate the PDF report narrative | Structured narrative generation from the report payload | **No** |
| Assessment 8 | Generate the concise dimension rationale | Plain-text qualitative rationale from supplied scoring data | **No** |

The baseline web workflow uses Google Search only in its first two research turns; its third turn synthesizes the existing conversation without a search tool. The molecule-level trial mapper explicitly passes `google_search=True`. Row-level scoring, PDF narrative generation, and concise rationale generation use Gemini without Google Search grounding.

## Programmatic Usage

Run from the repository root:

```python
import asyncio

from medical_potential.unmet_needs.unmet_needs_assessment import (
    unmet_needs_assesment,
)


async def main():
    await unmet_needs_assesment(
        molecule_name="semaglutide",
        incremental_mode=True,
    )


asyncio.run(main())
```

The assessment function persists its outputs and returns `None`.

## Configuration and Dependencies

Configuration is read from `medical_potential.config` and the repository environment:

- `PROJECT_ID`
- `BQ_DATASET_ID`
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GEMINI_API_KEY`
- `GEMINI_FLASH_PREVIEW_MODEL`
- `CLINICAL_EFFICACY_TABLE`
- `UNMET_NEEDS_TABLE`
- `TRIAL_MAPPING_TABLE`
- `UNMET_NEED_SCORING_TABLE`
- `UNMET_NEEDS_DIMENSION_NAME`
- `DIM_SCORES_TABLE`
- `TRIAL_LLM_CONFIDENCE_THRESHOLD`
- `GCS_BUCKET`
- `GCS_REPORT_BASE_PATH`
- `GCS_PIPELINE_CACHE_BASE_PATH`
- `GCS_MEDICAL_POTENTIAL_SUBFOLDER`

The module uses Google Cloud BigQuery and Storage, Gemini access, `google-genai`, `pandas`, `json_repair`, `aiohttp`/HTTP redirect handling, and ReportLab. The baseline extraction additionally requires local SoC PDF inputs.

