# Label Expansion Opportunity Pipeline

The `label_expansion()` pipeline identifies potential label-expansion opportunities for a drug. It discovers candidate indications, maps them to Open Targets diseases, scores the opportunities, and generates a business-facing report.

## Pipeline Overview

```text
Step 1   Indication Discovery for the Drug
         ├── Clinical Trial Analysis
         ├── FDA Label Analysis
         └── Web/Public Signal Analysis
         └── Indication Filtering 
                    │
                    ▼
Step 2   Filter, Merge & Push
         De-duplicate and upsert all indications
                    │
                    ▼
Step 3   MOA → Open Targets Target Mapping
                    │
                    ▼
Step 4   Indication → Open Targets Disease Mapping
         Map each secondary indication to a validated Open Targets disease ID
                    │
                    ▼
Step 5   Trial Data Fetching & Trial Selection
         Enrich each indication with trial metadata and select the best trial
                    │
                    ▼
Step 6   Score Calculation
                    │
                    ▼
Step 7   Rationale & PDF Report Generation
```

> **Primary indications** (i.e. the drug's established, approved uses) are discovered and recorded but are not scored. Only **Secondary indications** — emerging or unapproved uses with active clinical or commercial interest — proceed through scoring.

---

## Folder Layout

```text
medical_potential/
└── label_expansion_opportunity/
    ├── label_expansion_opportunity.py        # orchestrator / entry point
    ├── multiple_drugs.py                     # batch runner (excluded from handoff scope)
    ├── bq_utils.py                           # shared BQ schemas + push functions
    ├── __init__.py
    │
    ├── indication_extractor/
    │   ├── trial_analyser.py                 # Step 1: clinical trial extraction
    │   ├── fda_fetcher.py                    # Step 1b: FDA label extraction
    │   ├── web_analyser.py                   # Step 2: web/public signal extraction
    │   ├── utils.py                          # shared Gemini wrapper, PROCESS_INDICATIONS toggle
    │   └── __init__.py
    │
    ├── ot_mapping/
    │   ├── moa_mapping.py                    # Step 4: MOA → OT target
    │   ├── indication_mapping.py             # Step 5: indication → OT disease (Path A/B/C)
    │   ├── ot_utils.py                       # shared OT GraphQL, push_mappings (MERGE upsert)
    │   └── __init__.py
    │
    ├── scoring/
    │   ├── data_fetcher.py                   # Step 6a: enrich rows, association_score
    │   ├── trial_selector.py                 # Step 6b: trial_weight, ta_i, best-per-TA-I
    │   ├── score_calculator.py               # Step 6c: full formula, SCORING_FORMULAS, push
    │   └── __init__.py
    │
    └── generate_report_and_rationale/
        ├── generate_report.py                # PDF report generator + GCS upload helpers
        ├── generate_rationale.py             # short plain-English rationale generator
        └── __init__.py
 
```

---

## Step 1 — Discovery

The pipeline discovers indications from three independent sources and stored in gbq

### 1.1 Clinical Trial Analysis

- Retrieves registered clinical trials for the drug from BigQuery.
- Uses Gemini and Google Search to analyze each trial.
- Extracts the disease or indication being studied.
- Classifies the indication as `Primary` or `Secondary`.
- Generates a standardized Open Targets disease-name hint.
- Supports incremental processing by skipping trials already processed.

### 1.2 FDA Label Analysis

- Queries the openFDA API for the drug's approved brands.
- Retrieves the raw `indications_and_usage` section from FDA labels.
- Uses Gemini to extract FDA-approved indications.
- Classifies the extracted indications.

### 1.3 Web/Public Signal Analysis

- Uses Gemini with Google Search.
- Researches public and pipeline signals such as:
  - Investor materials
  - Press releases
  - Pipeline information
- Identifies potential label-expansion candidates.

### 1.4 Indication Filtering

- Merges outputs from Steps 1.1–1.3 into a unified candidate list.
- Removes non-disease extractions such as:
  - Trial endpoints
  - Biomarkers
  - Pharmacokinetic parameters
  - Procedures
- The filtered list is what enters Step 2 for de-duplication and persistence.

**Artifacts produced:** cleaned indication list, ready for upsert.

## Step 2 — Filter, Merge & Push

- Combines indications discovered from clinical trials, FDA labels, and web research.
- De-duplicates records using:
  - Drug name
  - Indication
  - Trial ID
- Resolves conflicts by preferring trial-sourced information over web-sourced information.
- Upserts the resulting indications into the label-expansion table.

**Artifacts produced:** upserted rows in the `glp_indication_table` BigQuery table — the canonical indication record used by all downstream steps.

## Step 3 — MOA → Open Targets Target Mapping

- Retrieves the drug's Mechanism of Action (MOA) from the drug details table in BigQuery.
- Resolves the MOA to an Open Targets target 
- Retrieves the corresponding Ensembl ID.
- Stores the MOA-to-target mapping.
- The resulting Ensembl IDs are used during disease mapping.

**Artifacts produced:** MOA-to-Ensembl ID mapping row in `ot_moa_mapping` table, used as the biological reference point for all disease association queries.

## Step 4 — Indication → Open Targets Disease Mapping

Secondary indications are resolved to actual Open Targets diseases through a layered mapping process.

### 4.1 Normalization

- Normalizes spelling.
- Handles hyphenation and parentheticals.
- Standardizes indication names.

### 4.2 Semantic Clustering

- Groups synonymous indications under a canonical disease name.
- Reduces duplicate Open Targets queries and improves mapping consistency across sources.
- Example: different names referring to the same disease can be grouped together.

### 4.3 Path A — Target-Disease Matching

- Uses Gemini to match the indication against the drug's Open Targets target-disease relationships.

### 4.4 Path B — Open Targets Text Search

- For indications not resolved by Path A
- Searches Open Targets for matching diseases.
- Uses the indication name as a search hint.
- Ranks candidates and selects the closest match by name

### 4.5 Path C — Gemini + Google Search

- Performs a grounded search when the first two paths cannot resolve the indication.
- Uses Gemini and Google Search to identify the appropriate Open Targets disease.

### 4.6 Validation

- Validates the proposed disease mapping.
- Rejects false positives caused by superficial word overlap.
- Unresolved mappings remain null and can be retried during later runs.

**Artifacts produced:** Open Targets disease ID, disease name, and association score written to the `ot_indication_mapping` BigQuery table for each secondary indication.

## Step 5 — Data Fetching & Trial Selection

Enriches each mapped secondary indication with the trial-level evidence needed for scoring.

### 5.1 Data Fetching

For each secondary indication with a confirmed Open Targets disease mapping:

- Retrieves relevant trial information.
- Enriches trial records with:
  - Region
  - Sample size
  - Dosage
- Uses clinical trials and serious safety profile data to retrieve trial information.
- Retrieves the Open Targets association score between the drug target and disease using API.
- Caches fetched information to avoid repeated searches.

### 5.2 Trial Weight Calculation

A weight is calculated for each trial using:

- Clinical trial phase
- Geography
- Sample size
- Dosage confidence

Approved-phase rows are assigned a trial weight of `1.0`.

### 5.3 Best Trial Selection

- Builds the `ta_i` key from:
  - Therapy area
  - Open Targets disease

- Selects the highest-weighted trial for each `ta_i`.

## Step 6 — Score Calculation

The scoring pipeline calculates a composite Final Score for each label-expansion opportunity.

The calculation proceeds through the following stages:

| # | Component | What it captures |
|---|-----------|-----------------|
| 1 | `prior` | Open Targets association score — the baseline biological plausibility of the drug-disease link |
| 2 | `maturity_weight` | How far along the clinical development path the leading trial has progressed |
| 3 | `effective_indications` | The count of indications with sufficient evidence to contribute meaningfully to the score |
| 4 | `effective_therapy_areas` | The number of distinct therapy areas represented across effective indications |
| 5 | `Q_i` | Combined trial-quality weight for indication *i*, aggregating phase, geography, size, and dosage signals |
| 6 | `e_phase_i` | Phase-specific evidence scaling factor |
| 7 | `e_i = Q_i × e_phase_i` | Overall evidence strength for indication *i* |
| 8 | `link = 1 − (1−prior)(1−e_i)` | Bayesian-style combination of biological prior and clinical evidence |
| 9 | `link_ta` | Maturity-weighted average of `link` values across all indications within a therapy area |
| 10 | `B_ind` | Indication breadth — how many indications contribute meaningfully |
| 11 | `B_ta` | Therapy-area breadth — how many therapy areas contribute meaningfully |
| 12 | `B` | Overall breadth — combined indication and therapy-area spread |
| 13 | `coherence` | How closely related the indications are across therapy areas (a focused cluster scores higher than scattered disease areas) |
| 14 | Coherence adjustment | `C = 0.1 + 0.9 × coherence^1.75` — softens the penalty for moderate coherence while rewarding tightly focused expansion profiles |
| 15 | **Final Score** | `Final Score = 1 + 4 × B × C` — scales to 1–5, where 5 represents the broadest, most coherent, and best-evidenced expansion opportunity |

**Artifacts produced:** Final Score, all intermediate components, and formula breakdown in the `label_expansion_score_calculation` BigQuery table.

## Step 7 — Rationale & PDF Report Generation

Translates the scored opportunity landscape into a business-ready narrative and formal report.

### 7.1 Rationale Generation

- Uses Gemini to generate a concise plain-English rationale.
- Rationale is limited to 50 words.
- Does not include scores or internal field names.

### 7.2 PDF Report Generation

The report contains:

1. Summary
   - Therapy-area count
   - Secondary-indication count
   - Top Final Score

2. Business Narrative
   - Headline
   - Indication Landscape
   - Key Insights
   - Evidence Gaps & Risks
   - Bottom Line

3. Expansion Indications Table
   - Grouped by therapy area

4. Methodology
   - Breadth calculation
   - Coherence calculation
   - Final Score calculation
   - Full scored-indication table

### 7.3 Artifact Storage

The final pipeline stores:

- PDF report
- JSON payload/cache
- Score
- Rationale

in the corresponding GCS and BigQuery locations.


## Step Input / Output Reference

| Step | Input | Output |
|------|-------|--------|
| 1.1 Clinical Trial Analysis | Drug name; registered trial records (BigQuery trials table) | Trial-sourced indication rows: indication name, Primary/Secondary classification, Open Targets disease-name hint |
| 1.2 FDA Label Analysis | Drug name; openFDA brand lookup → raw `indications_and_usage` label text | FDA-sourced indication rows: indication name, Primary/Secondary classification |
| 1.3 Web / Public Signal Analysis | Drug name | Web-sourced indication rows: indication name, Primary/Secondary classification, supporting rationale |
| 1.4 Indication Filtering | Combined raw indication rows from Steps 1.1–1.3 | Cleaned indication list, with trial endpoints / biomarkers / PK parameters / procedures removed |
| 2. Filter, Merge & Push | Cleaned indication list from Step 1.4 | De-duplicated rows upserted into the `glp_indications_table` BigQuery table |
| 3. MOA → Target Mapping | Drug's Mechanism of Action (drug details BigQuery table) | MOA → Open Targets target name + Ensembl gene ID. Stored in ot_moa_mapping |
| 4.1 Normalization | Raw indication text from `glp_indications_table` table | Standardized indication text (spelling, hyphenation, parentheticals, common synonyms collapsed) |
| 4.2 Semantic Clustering | Normalized indications for the drug | Indications grouped under one canonical name per distinct disease |
| 4.3 Direct Match (OT Search) | Canonical indication name  | Confirmed Open Targets disease ID for exact-name matches; unresolved indications pass through |
| 4.4 Path A — Target-Disease Matching | Indications unresolved by Direct Match; Ensembl target ID from Step 3 | Open Targets disease ID for indications matched to the drug's known target-disease associations |
| 4.5 Path B — Open Targets Text Search | Indications unresolved by Path A; OT disease-name hint | Best-scoring Open Targets disease ID from ranked text-search candidates |
| 4.6 Path C — Gemini + Google Search | Indications unresolved by Path A and Path B | Open Targets disease ID identified via grounded web reasoning |
| 4.7 Validation | All candidate mappings from Paths A–C | Confirmed mappings (false-positive word-overlap matches rejected and left `null` for retry). All the confirmed mappings are then pushed to ot_indication_mapping |
| 5.1 Data Fetching | Secondary indications with a confirmed Open Targets disease mapping | Enriched trial records (region, sample size, dosage) + Open Targets association score, cached in BigQuery |
| 5.2 Trial Weight Calculation | Enriched trial records from Step 5.1 | Composite `trial_weight` per trial (phase, geography, sample size, dosage confidence) |
| 5.3 Best Trial Selection | Weighted trial records, keyed by `ta_i` (therapy area + OT disease ID) | One best-trial record per indication. Then TA-I combination is created using that trial. |
| 6. Score Calculation | Best-trial records from Step 5.3 | Final Score (1–5) and full formula breakdown per indication, written to `label_expansion_score_calculation table` |
| 7.1 Rationale Generation | Scored indications for the drug | Plain-English rationale (≤50 words) per opportunity set |
| 7.2 PDF Report Generation | Scored indications, Step 7.1 rationale | Structured PDF report (summary snapshot, business narrative, expansion indications table, methodology) |
| 7.3 Persistence | Generated PDF + JSON payload | PDF + archived PDF (GCS), JSON payload + archived JSON (GCS), dimension score (BigQuery) |

---

## Gemini Usage

Gemini is used at six distinct points in the pipeline. Each call is purpose-built for a specific reasoning or retrieval task that structured data queries alone cannot perform.

| Step | Task | Gemini capability used |
|------|------|----------------------|
| 1.1 Clinical Trial Analysis | Extract the disease indication from a trial's title, conditions, and interventions; classify as Primary or Secondary; generate an Open Targets disease name hint | Text extraction + classification, optionally grounded with Google Search |
| 1.2 FDA Label Analysis | Parse the raw `indications_and_usage` label text and extract normalized, classified indications | Text extraction + classification |
| 1.3 Web / Public Signal Analysis | Scan investor materials, press releases, and pipeline databases for forward-looking expansion signals not yet in the trial registry | Grounded generation with Google Search |
| 4.2 Semantic Clustering | Group indications describing the same disease under one canonical name (e.g. "T2DM" / "Type 2 Diabetes" / "adult-onset diabetes") before any Open Targets matching is attempted | Text clustering / reasoning |
| 4.4 Path A — Target-Disease Matching | Match a normalized indication against the Open Targets target-disease association list for the drug's Ensembl target ID | Structured matching / reasoning |
| 4.6 Path C — Disease Mapping | Resolve an indication to its correct Open Targets disease entry when the direct match, Path A, and Path B all fail | Grounded reasoning with Google Search |
| 4.7 Validation | Reject candidate disease mappings (from Path A/B/C) that are only superficial word-overlap matches rather than genuine clinical matches | Structured verification / reasoning |
| 7.1 / 7.2 Report Generation | Generate the plain-English rationale (50-word cap) and the full structured business narrative (Headline, Indication Landscape, Key Insights, Evidence Gaps, Bottom Line) for the PDF report | Long-form structured generation |

## Pipeline Re-entry

The pipeline can be restarted from specific stages to reuse previously generated data:

```text
discovery
    ↓
moa_mapping
    ↓
indication_mapping
    ↓
scoring
    ↓
report generation
```

This allows later stages to be rerun without repeating earlier discovery and mapping operations.
