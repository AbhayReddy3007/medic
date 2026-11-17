# MoA Innovation Pipeline

The `moa_innovation` pipeline evaluates how innovative and strategically meaningful a molecule's mechanism of action (MoA) is for an indication. It combines internal drug-detail data, grounded Gemini research, clinical and biological evidence, competitive-landscape analysis, and a five-point innovation rubric.

The pipeline classifies the mechanism into one of these categories:

- First-in-Class
- Best-in-Class
- Me-too/Fast Follower
- Weak/Outdated
- Poor/Invalid

It persists the complete assessment to BigQuery, appends the dimension score, generates a business-facing PDF report, and caches the full payload in GCS.

## Module Structure

```text
moa_innovation/
|-- __init__.py                              Public package export
|-- moa_innovation.py                        Main 13-stage orchestrator
|-- bq_utils.py                              Query and BigQuery persistence helpers
`-- generate_report_and_rationale/
    |-- generate_moa_report.py               Latest-row loading and PDF generation
    |-- generate_rationale.py                 Concise score rationale
    `-- __init__.py                           Report helper exports
```

## Pipeline Overview

```text
Step 1   Resolve MoA and primary indication for the given drug
         |
Step 2   Build the current mechanism landscape
         |
Step 3   Classify the molecule's MoA position
         |
         +------------------------------------------------+
         |                                                |
Step 4   Check biological rationale              Step 5   Check clinical validation
         |                                                |
         +-------------------------+----------------------+
                                   |
                         Step 6   Check mechanism failures
                                   |
Step 7   Check mechanistic improvement
         |
Step 8   Check mechanism currency
         |
Step 9   Generate final classification, score, and evidence summary
         |
Step 10  Append the complete MoA payload to BigQuery
         |
Step 11  Append the dimension score and concise rationale
         |
Step 12  Generate and upload the PDF report
         |
Step 13  Cache the complete payload in GCS
```

Steps 4, 5, and 6 are independent research checks and run concurrently after Step 3. Steps 7 and 8 run after that parallel group, and Step 9 consumes all intermediate results.

The public entry point is `get_moa_innovation_assessment()` in `moa_innovation.py`. It performs the assessment and persists the resulting artifacts. The current function does not return the formatted assessment object to its caller.

## Step 1 - MoA and Primary Indication Resolution

The pipeline queries `MOA_INNOVATION_QUERY` through BigQuery. The query reads distinct drug names, mechanisms of action, and cleaned indications from `DRUG_DETAILS_FULL_TABLE_ID` and is currently filtered to GLP-1-related target or mechanism records.

For the requested molecule:

- Matching rows are selected by exact or substring match on `cleaned_generic_name`.
- The first matching mechanism is used as the molecule's mechanism statement.
- All unique indications are collected.
- A single indication is used directly.
- When multiple indications are available, Gemini selects the primary indication based on first approval or establishment, mechanism validation, and strongest clinical evidence.
- If BigQuery does not provide a mechanism, grounded Gemini search is used to find the mechanism and indication.
- If the indication is still unavailable, the GLP-1 fallback is `Type 2 Diabetes / Obesity`.

The step records the selected data, all available indications, and the source used in `step1_moa_data`.

## Step 2 - Mechanism Landscape

Gemini with Google Search builds the current treatment landscape for the selected indication. The research prompt asks for:

- Currently approved drugs and their mechanisms
- Phase 2/3 pipeline drugs and their mechanisms
- Current standard-of-care treatments
- The role of each mechanism, such as first-line, second-line, or emerging
- Dominant mechanism classes
- Whether the indication is crowded, dominated by specific mechanisms, or open to new mechanisms
- Sources consulted

The structured result is stored as `step2_landscape` and becomes the evidence base for MoA position classification.

## Step 3 - MoA Position Classification

Gemini compares the molecule's mechanism with the landscape and classifies its position using the following categories:

- **First-in-Class:** The first meaningful drug targeting a novel biological mechanism in the indication, with no prior approved or clearly established class.
- **Best-in-Class:** A known mechanism used in a clearly better way through meaningful mechanistic differentiation, such as dual agonism, improved selectivity, or better downstream biology.
- **Me-too/Fast Follower:** A validated mechanism with little mechanism-level innovation, competing mainly on dosing, formulation, convenience, or other non-MoA factors.
- **Weak/Outdated:** An older mechanism whose strategic importance has declined because newer approaches have replaced it.
- **Poor/Invalid:** A mechanism with weak biological rationale or a mechanism-level clinical failure without a convincing reason this molecule should succeed.

The result includes boolean evidence flags, differentiation details, a classification rationale, and sources. It is stored as `step3_classification`.

## Steps 4-6 - Independent Evidence Checks

After the MoA position is classified, the pipeline runs three independent grounded research checks concurrently using `asyncio.gather()`.

### Step 4 - Biological Rationale

The pipeline evaluates whether the target or pathway is biologically linked to the disease and whether modulating it is expected to produce meaningful benefit. It checks:

- Disease pathophysiology and target/pathway linkage
- Scientific evidence that target modulation affects the disease
- Consensus, seminal papers, and definitive reviews
- Translational and mechanistic evidence

For multi-mechanism drugs, each component is evaluated separately. A combination is considered strongly supported when most components have validated biological rationale; novel combinations of individually validated mechanisms are not penalized solely for being combinations.

The result includes target pathways, component-level analysis, disease-biology linkage, key publications, rationale flags, explanation, and sources. It is stored as `step4_rationale`.

### Step 5 - Clinical Validation

The pipeline searches FDA and EMA sources, ClinicalTrials.gov, and peer-reviewed publications to determine whether the mechanism has been validated in humans. It checks:

- Approved drugs using the same mechanism in the same indication
- Successful Phase 2/3 trials
- Published human efficacy evidence
- Overall validation strength: Strong, Moderate, Limited, or None

The result is stored as `step5_validation`.

### Step 6 - Mechanism Failures

The pipeline searches completed, terminated, and withdrawn trials and publications for failures involving the same mechanism in the indication. It explicitly distinguishes:

- **Mechanism-level failure:** the biological approach does not work in humans.
- **Molecule-level failure:** the individual asset failed because of pharmacokinetics, toxicity, dose, trial design, or another molecule-specific issue.

The result records failed drugs, trial IDs, failure reasons and types, whether the mechanism is invalidated, evidence for invalidation, and whether the assessed molecule could overcome earlier failures. It is stored as `step6_failures`.

## Step 7 - Mechanistic Improvement

If Step 3 classifies the molecule as First-in-Class, this step is skipped because no comparison with an existing class is required.

For other classifications, Gemini evaluates whether the molecule has real mechanism-level improvement over the established class. Valid differentiators include:

- Dual, triple, or multi-agonism instead of a single receptor
- Better target selectivity
- Biased signaling
- More relevant downstream biology
- A mechanism designed to overcome known class limitations

Dosing, formulation, device, price, and marketing advantages do not count as mechanistic improvements. The output includes the improvement type, comparison class, scientific basis, limitations addressed, and an assessment of Significant, Moderate, Minimal, or None. It is stored as `step7_improvement`.

## Step 8 - Mechanism Currency

Gemini assesses whether the mechanism remains strategically current. The assessment covers:

- Whether the mechanism remains part of current standard of care
- Whether treatment guidelines still recommend it
- Whether newer mechanisms have displaced it
- Advantages and disadvantages relative to newer approaches
- Strategic relevance: High, Moderate, Low, or Minimal

The output includes the mechanism status, guideline status, competitive position, newer mechanisms, and currency conclusion. It is stored as `step8_currency`.

## Step 9 - Final Classification and Score

The final scoring call receives the complete outputs from Steps 1-8. Gemini applies the following rubric:

| Score | Label | Criteria |
|---:|---|---|
| 5 | Exceptional | True First-in-Class mechanism, strong biology, and meaningful potential to create a new treatment class |
| 4 | Strong | Validated mechanism with clearly superior mechanistic implementation, typically Best-in-Class |
| 3 | Moderate | Validated mechanism with limited mechanism-level differentiation, typically Me-too/Fast Follower |
| 2 | Weak | Older or strategically outdated mechanism that has been displaced by better approaches |
| 1 | Poor | Weak/speculative biology, repeated mechanism-level failures, or clinical invalidation without a rescue hypothesis |

The final prompt also requires:

- A classification decision following the rubric decision tree
- A guardrail status of `PASS` or `FAIL`
- Confidence score from 0.0 to 1.0
- Confidence tier from Tier 1 to Tier 3
- Detailed justification for the score
- A mandatory explanation of why the score was not higher
- Primary, secondary, and tertiary source lists

### Guardrail

The guardrail fails only when the same mechanism has been clinically invalidated in the same indication and there is no compelling scientific reason for the assessed molecule to succeed. One failed molecule alone does not invalidate the mechanism.

### Confidence tiers

- **Tier 1:** Strong peer-reviewed or authoritative evidence.
- **Tier 2:** Some reliance on secondary material or moderate inference.
- **Tier 3:** Limited evidence and inference-heavy conclusion.

Confidence score interpretation:

| Range | Interpretation |
|---:|---|
| 0.90-1.00 | Very high confidence |
| 0.70-0.89 | High confidence |
| 0.50-0.69 | Moderate confidence |
| 0.30-0.49 | Low confidence |
| 0.00-0.29 | Very low confidence |

The formatted result is stored in `step9_final` and includes the final score, classification, guardrail, confidence, justification, sources, and all raw intermediate data.

## Steps 10 and 11 - BigQuery Persistence

### Step 10 - Complete payload persistence

`append_moa_innovation_payload_to_bigquery()` appends one normalized row to `MOA_INNOVATION_TABLE`. The row includes:

- Drug name, indication, and mechanism statement
- MoA classification and score
- Guardrail status
- Confidence score and tier
- Six justification fields
- Unique grounding URLs referenced by Gemini grounding supports
- A `grounding_urls_by_step` JSON object mapping each uniquely named Gemini call to its referenced URLs
- Narrative rationale
- Primary, secondary, and tertiary sources serialized as JSON strings
- Analysis date, processing time, and `created_at`

Grounding URLs are retained only when Gemini grounding metadata contains a support
that references the corresponding grounding chunk at least once. Unreferenced,
invalid, or URL-less chunks are excluded. URLs are grouped by the explicit MoA
call that made the search, including empty arrays for calls without grounded
results. The identifiers are `step1_primary_indication`, `step1_moa_search`,
`step2_mechanism_landscape`, `step3_moa_classification`,
`step4_biological_rationale`, `step5_clinical_validation`,
`step6_mechanism_failures`, `step7_mechanistic_improvement`,
`step8_mechanism_currency`, and `step9_final_scoring`. The former
per-justification URL columns are no longer written for new rows; legacy columns
may remain on existing BigQuery tables.

The write uses append-only semantics and creates the table if needed using `MOA_INNOVATION_SCHEMA`.

### Step 11 - Dimension score persistence

The score is appended to `DIM_SCORES_TABLE` through `append_dimension_score_to_bigquery()` using the configured `MOA_INNOVATION_DIMENSION_NAME`. A separate concise rationale is generated from the final output and stored with the score.

## Step 12 - Report and Rationale Generation

### Concise score rationale

`generate_rationale_for_score()` prepares a reduced payload containing the molecule, mechanism, score, classification, justification, and narrative rationale. Gemini generates a plain-text rationale of approximately 50 words without Google Search. If `GEMINI_API_KEY` is unavailable or generation fails, the fallback is `Rationale has not been generated.`

### PDF report

`generate_moa_report()` loads the latest BigQuery row for the molecule by descending `created_at`. It extracts the stored fields and parses the JSON source lists.

If fewer than three of the five main justification fields contain at least 50 characters, the report generator enriches the data with a grounded Gemini call. Enrichment fills only fields where the new response is longer and merges sources when primary sources are absent.

The report narrative is then generated as structured JSON with these sections:

1. Executive Summary
2. Mechanism of Action Overview
3. Categorization and Rationale
4. Scoring and Implications
5. Sources
6. MoA Innovation scoring reference

The report is rendered to PDF with ReportLab and returned as bytes. The orchestrator uploads it through `upload_dimension_report_pdf_to_gcs()` and stores base and archive GCS URIs in `report_metadata`.

## Step Input / Output Reference

| Step | Input | Output |
|---|---|---|
| 1. MoA resolution | Molecule name, MoA BigQuery query | Mechanism, indications, selected primary indication, source |
| 2. Landscape | Molecule, mechanism, indication | Approved and pipeline landscape, SOC, dominant classes, sources |
| 3. Position classification | Molecule, mechanism, indication, landscape | FIC/BIC/Me-too/Outdated/Poor classification and rationale |
| 4. Biological rationale | Molecule, mechanism, indication | Target/pathway analysis, component validation, publications |
| 5. Clinical validation | Molecule, mechanism, indication | Approved same-mechanism drugs, trials, human evidence, strength |
| 6. Mechanism failures | Molecule, mechanism, indication | Failure records, invalidation assessment, rescue hypothesis |
| 7. Mechanistic improvement | Classification plus molecule context | Differentiation type, scientific basis, class limitations addressed |
| 8. Mechanism currency | Molecule, mechanism, indication | Status, guideline position, newer mechanisms, strategic relevance |
| 9. Final scoring | All outputs from Steps 1-8 | Score, classification, guardrail, confidence, justification, sources |
| 10. Payload persistence | Formatted output | Row appended to `MOA_INNOVATION_TABLE` |
| 11. Dimension score | Score and generated rationale | Row appended to `DIM_SCORES_TABLE` |
| 12. PDF report | Latest MoA BigQuery row | Report PDF and base/archive GCS URIs |
| 13. Payload cache | Complete output payload | JSON cache uploaded to GCS |

## Gemini Usage

| Step | Task | Gemini capability | Google Search grounded? |
|---|---|---|---|
| 1 | Select the primary indication or recover missing MoA data | Structured extraction and selection | **Yes** |
| 2 | Research the indication's treatment and mechanism landscape | Search and structured synthesis | **Yes** |
| 3 | Classify the MoA position | Evidence-based structured classification | **Yes** |
| 4 | Evaluate biological rationale | Scientific literature research and component analysis | **Yes** |
| 5 | Evaluate clinical validation | Regulatory, trial, and publication research | **Yes** |
| 6 | Identify mechanism-level versus molecule-level failures | Trial and publication research with causal distinction | **Yes** |
| 7 | Evaluate mechanistic differentiation | Comparative mechanism analysis | **Yes** |
| 8 | Assess current strategic relevance | Guideline and competitive-landscape research | **Yes** |
| 9 | Produce final score and evidence summary | Rubric-guided structured judgment | **Yes** |
| 12a | Enrich thin report data | Grounded enrichment before report writing, only when stored justification data is insufficient | **Yes, conditional** |
| 12b | Write the PDF report narrative | Structured report generation from loaded/enriched data | **No** |
| 13 | Generate concise score rationale | Plain-text qualitative rationale | **No** |

Steps 1-9 use Gemini with Google Search grounding through `_gemini_search_call()`. Step 12a uses Google Search grounding only when the BigQuery justification fields are insufficient. Step 12b and Step 13 use the configured Gemini model without Google Search grounding. Responses are extracted from JSON or repaired with `json_repair`; raw responses are retained when structured parsing is unavailable.

## Programmatic Usage

Run from the repository root:

```python
import asyncio

from medical_potential.moa_innovation import get_moa_innovation_assessment


async def main():
    await get_moa_innovation_assessment(
        molecule_name="semaglutide",
        indication=None,
    )


asyncio.run(main())
```

The assessment is persisted to BigQuery and GCS during execution. The public function currently returns `None`; inspect the persisted `MOA_INNOVATION_TABLE`, `DIM_SCORES_TABLE`, report URI, or payload cache for the generated artifacts.

## Configuration and Dependencies

Configuration is read from `medical_potential.config` and the repository `.env` file.

Common settings:

- `PROJECT_ID`
- `BQ_DATASET_ID`
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GEMINI_API_KEY`
- `GEMINI_FLASH_PREVIEW_MODEL`
- `DRUG_DETAILS_FULL_TABLE_ID`
- `MOA_INNOVATION_TABLE`
- `MOA_INNOVATION_DIMENSION_NAME`
- `DIM_SCORES_TABLE`
- `GCS_BUCKET`
- `GCS_REPORT_BASE_PATH`
- `GCS_PIPELINE_CACHE_BASE_PATH`
- `GCS_MEDICAL_POTENTIAL_SUBFOLDER`

Install repository dependencies from `requirements.txt`. The module uses Google Cloud BigQuery and Storage, Gemini access, `json_repair`, `pandas`, and ReportLab.

