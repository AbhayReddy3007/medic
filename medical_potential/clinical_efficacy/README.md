# Clinical Efficacy Assessment Pipeline

The `clinical_efficacy_assessment()` pipeline measures how well a drug works, based on the clinical trials run on it. For one molecule it:

1. finds every known clinical trial across registries and secondary sources,
2. checks that each trial really studies that drug,
3. extracts four efficacy endpoints per trial (weight loss, HbA1c, MASH, ALT) with Gemini and Google Search grounding,
4. scores the molecule on a 1–5 scale using a phase-anchored algorithm,
5. stores everything in BigQuery,
6. writes a narrative rationale and a business-facing PDF report.

---

## Pipeline Overview

```text
Step 1   Fetch Trials                                   (trial_fetcher.py)
         Resolve aliases → search all registries/sources for primary drug + aliases
         → deduplicate → LLM confidence check ("is this trial really about this drug?")
         → normalise → raw trial rows
                    │
                    ▼
Step 2   Enrich + Score                                 (fetcher.py)
         ├── 2.1  Enrichment: one Gemini+Search call per batch of trials extracts
         │         metadata, locations and 4 endpoints (value, duration, rationale, confidence)
         └── 2.2  Scoring: per-endpoint 1–5 score → weighted Clinical Efficacy Score (0–5)
                    │
                    ▼
Step 2b  Rationale Generation                           (generate_rationale.py)
         Gemini writes a 3-sentence rationale from the score breakdown
                    │
                    ▼
Step 3   Persist to BigQuery                            (push_to_bq.py + gcp_utils)
         ├── Incremental insert/update into the clinical_efficacy table
         └── Append "Clinical Efficacy" row to the shared dimension-score table
                    │
                    ▼
Step 4   PDF Report + Output Payload                    (generate_efficacy_report.py)
         Read stored scores + trials from BigQuery → Gemini narrative → PDF → GCS
         Full JSON payload → pipeline cache (GCS)
```

> **`step1_only` mode:** runs only Step 1 (fetch) and Step 3 (push raw rows to BigQuery). No enrichment, scoring, rationale or report. Useful for refreshing the trial universe cheaply.

---

## Folder Layout

```text
clinical_efficacy/
├── __init__.py                        exposes clinical_efficacy_assessment
├── clinical_efficacy.py               pipeline orchestrator (Steps 1–4)
├── trial_fetcher.py                   Step 1: multi-source trial discovery
├── alias_resolver.py                  alias lookup (BigQuery) + Gemini alias cleaning
├── llm_eval.py                        Step 1: drug-match confidence check
├── fetcher.py                         Step 2: enrichment + scoring (live scoring logic)
├── push_to_bq.py                      Step 3: incremental BigQuery writer
├── utils.py                           Gemini client, model names, batch/call-budget config
│
├── generate_report_and_raitonale/     Steps 2b + 4 (folder name must match the import
│   ├── __init__.py                    in clinical_efficacy.py — note the spelling)
│   ├── generate_rationale.py          3-sentence score rationale
│   └── generate_efficacy_report.py    PDF report builder
│
├── ctgov_trials.py, ctgov_port.py     ClinicalTrials.gov
├── ctis_drug_trials.py                EU CTIS
├── eudract_drug_trials.py             EudraCT
├── euct_trials.py                     EU CT register helper
├── ctri_trials.py                     India CTRI
├── pubmed_api.py, who.py              PubMed, WHO ICTRP
├── fetch_trial_ids.py                 international registries (Gemini + Search)
├── nice_trials.py                     NICE appraisals (Gemini + Search)
├── innovator_web.py                   innovator company website scan
├── trade_and_conferences.py           trade press + conference abstracts
│
├── generate_all_reports.py            standalone: PDFs for all molecules, in parallel
├── efficacy_scorer.py                 NOT used by the pipeline (see Known Issues)
└── enrich_outcomes.py.py              stub that raises ImportError by design
```

---

## Running It

The pipeline is an `async` function. Run it from Python:

```python
import asyncio
from medical_potential.clinical_efficacy import clinical_efficacy_assessment

asyncio.run(clinical_efficacy_assessment("CagriSema"))
```

| Argument | Default | Effect |
|----------|---------|--------|
| `molecule_name` | required | Drug to assess |
| `max_records` | `None` | Cap on records per registry per search term |
| `top_n` | `None` | Keep only the top N trials by completeness before enrichment |
| `workers` | `6` | Concurrent enrichment batches |
| `no_score` | `False` | Skip scoring (and therefore rationale and dimension score) |
| `no_report` | `False` | Skip the PDF report **and the output payload upload** |
| `step1_only` | `False` | Fetch + push raw rows only |
| `skip_fetch`, `json_path` | `False`, `None` | Intended to reuse a saved trials JSON (see Known Issues) |



---

## Input / Output Table

| Step | Input | Output |
|------|-------|--------|
| 1 — Fetch | Drug name | List of trial rows (one dict per trial) with registry metadata, `alias_names`, `llm_confidence`, `llm_confidence_rationale`, `trial_source` |
| 2.1 — Enrich | Trial rows with `llm_confidence > 0.6` | Same rows with metadata gaps filled and, per endpoint: value, duration, rationale, confidence, plus `eval_*_confidence` |
| 2.2 — Score | Enriched rows | `weighted_score`, per-endpoint results, `data_coverage`, `score_breakdown` text |
| 2b — Rationale | Score result | 3-sentence narrative string |
| 3 — Persist | Enriched rows + score + rationale | Rows in the clinical-efficacy BigQuery table; row in the shared dimension-score table |
| 4 — Report | Stored rows + score columns from BigQuery | PDF uploaded to GCS (plus archived copy); JSON output payload cached in GCS |

---

## Step 1 — Fetch Trials

**1a. Alias resolution.** `alias_resolver.resolve_aliases()` looks the drug up in the BigQuery alias table, then asks Gemini once to deduplicate and normalise the alias list (hyphen variants collapsed, non-drug entries removed). The result is `[drug, alias_1, alias_2, …]`. If the resolver is unavailable, only the drug name is searched.

**1b. Multi-source search.** Two phases, deduplicated across all terms by `trial_id`:

- **Phase A — primary name through all 11 sources**

  | Source | How it is queried |
  |--------|-------------------|
  | ClinicalTrials.gov, EU CTIS, EudraCT, CTRI, PubMed, WHO ICTRP | Direct registry APIs, no Gemini |
  | GD Clinical Trials | BigQuery table (curated; trials auto-assigned confidence 1.0) |
  | International registries (INTL) | Gemini + Search, registries searched `trial_registries_per_call` at a time |
  | NICE appraisals | Gemini + Search |
  | Trade publications + conferences | Gemini + Search, within a fixed call budget |
  | Innovator website | Gemini discovers the company and URLs, pages are fetched and regex-scanned, then Gemini extracts IDs; fixed call budget |

- **Phase B — aliases.** API sources run once per alias. INTL runs per batch of aliases. NICE, trade/conference and innovator run once with all aliases combined (`alias1 OR alias2 …`).

**1c. Deduplication.** Rows with the same `trial_id` (case-insensitive) collapse to the richest row, with ties broken by source priority (ClinicalTrials.gov first).

**1d. Drug-match confidence (`llm_eval.py`).** Registry searches can return false positives, so each trial gets an `llm_confidence` between 0 and 1.

- Rows from ClinicalTrials.gov, EU CTIS and the GD table are set to 1.0 automatically.
- All other rows are verified by Gemini Pro (grounded with Search) in batches, asking whether the trial genuinely studies the drug or an alias, based on title, indication, sponsor, phase and source.
- Rows with `llm_confidence ≤ 0.6` are carried forward but are **not enriched** and are excluded from the report.

**1e. Normalisation.** Locations, dates (`YYYY-MM-DD`) and phases (`Phase 1/2/3/4`) are cleaned. If `top_n` is set, rows are ranked by completeness (phase, enrolment, dates, acronym, sponsor, completion status) and trimmed.

---

## Step 2 — Enrich + Score

### 2.1 Enrichment

Only rows with `llm_confidence > 0.6` are enriched. Each Gemini + Search call handles a batch of trials and returns, per trial:

- registry metadata and locations (filling blanks only, never overwriting existing values),
- `dosage`,
- four endpoints, each with value, duration, rationale and confidence: `hba1c_change_pct`, `weight_change_pct`, `alt_reduction_pct`, `mash_change_pct`,
- an `eval_*_confidence` (0–1) per endpoint.

The prompt enforces a source hierarchy: **P1** registry results pages and primary papers, then **P2** innovator press releases and pipeline pages, then **P3** trade press, journals and conference abstracts. Rationales must cite the exact source and tier. Values are reported as positive numbers, and `"N/A"` is used when no source has data.

**Every trial is asked about all four endpoints.** Most trials only report one or two, so the rest come back `N/A`. That is expected.

Rows at or below the confidence threshold get `N/A` for all outcome columns.

> **Partial enrichment** (`CLINICAL_TRIALS_PARTIAL_ENRICHMENT=true`, default off): trials already in BigQuery get metadata-only backfill and **no endpoint extraction**. See Known Issues for the scoring consequence.

### 2.2 Scoring

Scoring is pure arithmetic with no Gemini calls. It lives in `fetcher.py` (`compute_clinical_efficacy_score`).

**Endpoints and weights**

| Endpoint | Field | Weight |
|----------|-------|--------|
| Weight loss | `weight_change_pct` | 0.40 |
| HbA1c reduction | `hba1c_change_pct` | 0.40 |
| MASH | `mash_change_pct` | 0.10 |
| ALT reduction | `alt_reduction_pct` | 0.10 |

**Value → score**

| Adjusted value | Score |
|----------------|-------|
| ≥ 22 % | 5 |
| 16 – 21.9 % | 4 |
| 10 – 15.9 % | 3 |
| 5 – 9.9 % | 2 |
| < 5 % | 1 |

**Phase penalty** (applied to the raw value before the table above): Phase 3 and 4 × 1.00, Phase 2 × 0.85, Phase 1 × 0.65.

**Which trial represents each endpoint.** Each endpoint is scored independently over the same set of rows, so weight loss may come from one trial and HbA1c from another:

1. **Eligible rows:** parseable `phase`, parseable value for that endpoint, and `trial_size > 0`. There is no confidence filter at this stage.
2. **Fallback:** if *no* row has a usable `trial_size`, the size requirement is dropped and the same selection runs on rows with just a phase and value.
3. **Highest phase wins:** Phase 4 / 3, then 2, then 1. A lower-phase value is never used while a higher phase has data, even if the lower number is larger.
4. **Within that phase:** rows are deduplicated per `trial_id` (best arm kept), then the single highest value is taken. This is a best-case trial, not an average.
5. The penalty is applied, the table converts the result to 1–5, and the winning `trial_id`, dosage and duration are recorded.

**Final score**

```text
weighted_score = Σ (endpoint score × endpoint weight)   over endpoints that have a score
```

Endpoints with no usable data contribute nothing and are **not** renormalised. A molecule with only weight-loss data can therefore score at most 5 × 0.40 = 2.0. `data_coverage` (e.g. `2/4 endpoints scored (missing: mash, alt)`) records this.

**Outputs:** `weighted_score`, `endpoints`, `data_coverage`, and `score_breakdown`, a line-per-endpoint text such as:

```text
weight_loss  | adj=18.90%  score=4  weight=40%  (Phase 3 data used)
hba1c        | N/A  weight=40%  (No valid data for this endpoint)
```

The score, coverage and breakdown are written onto every row of the molecule.

---

## Step 2b — Rationale Generation

`generate_score_rationale()` sends the score, coverage and the exact trial, phase, dosage and duration behind each endpoint to Gemini, which returns **exactly three sentences**: overall score and performance, strongest endpoint(s) with specifics, then data gaps and what the score reflects. If generation fails, the text is `"Rationale has not been generated."`. If the module cannot be imported, the step is skipped with a warning and the column stays empty.

---

## Step 3 — Persist to BigQuery

`push_to_bq.save_clinical_efficacy_to_bq()` writes to `{PROJECT_ID}.{BQ_DATASET_ID}.{CLINICAL_EFFICACY_TABLE}`, one row per `(molecule_name, trial_id)`. The table and any missing columns are created automatically.

| Situation | Action |
|-----------|--------|
| New `trial_id` for the molecule | **INSERT** the full row, including score, coverage, breakdown and rationale |
| Existing `trial_id` whose phase or status changed, or whose title/study was blank | **UPDATE** only `phase`, `phase_status`, blank `trial_title`/`trial_study`, `updated_at` |
| Existing and unchanged | Skip |

Per-endpoint `*_confidence` columns are stored as FLOAT64 (`High` = 1.0, `Medium` = 0.6, `Low` = 0.4, `N/A` = 0). After the push, `append_dimension_score_to_bigquery()` adds the molecule's score and rationale to the shared dimension-score table under dimension "Clinical Efficacy", pillar "Medical Potential".



---

## Step 4 — PDF Report and Output Payload

`generate_efficacy_report(molecule)` does **not** recompute anything. It:

1. Loads the molecule's rows from BigQuery, keeping only `llm_confidence > TRIAL_LLM_CONFIDENCE_THRESHOLD`.
2. Takes the stored `efficacy_score`, coverage, rationale and breakdown from the **first** row returned (ordered by `phase DESC, created_at DESC`) and regex-parses the breakdown text back into per-endpoint results.
3. Looks up display details (trial ID, dosage, duration) from the trial rows.
4. If the stored rationale is under 80 characters or no endpoint scored, makes one extra Gemini + Search call to gather supporting efficacy data. These display-only fields (`weight_loss_pct`, `hba1c_pct`, …) are never written to BigQuery and never affect the score.
5. Makes one Gemini call for the narrative, producing Key Findings, Insights and Implications, and Profile Summary.
6. Builds the PDF with ReportLab: header, key findings (weight loss, HbA1c, liver endpoints), endpoint summary table (raw value, adjusted value, score, phase, trial and dosage), insights, profile summary, scoring reference.
7. Uploads it to GCS via `upload_dimension_report_pdf_to_gcs` (archived copy included).

The full output payload (score, rationale, report URIs, enriched rows) is then cached in GCS. This happens only when `no_report=False`.

---

## Gemini Usage

| Stage | Calls | Batch size / budget |
|-------|-------|---------------------|
| Alias cleaning | 1 per molecule | Whole alias list in one prompt |
| International registries | 1 per registry batch | `TRIAL_REGISTRIES_PER_CALL` = 3 registries; alias batches of `INTERNATIONAL_ALIAS_BATCH` = 10 |
| NICE | 1 (+1 fallback if nothing found) | — |
| Trade + conferences | up to 8 per search term | `TRADE_CALLS` = 4, `CONFERENCE_CALLS` = 4 |
| Innovator website | up to 3 | `INNOVATOR_WEB_SOURCES_CALLS` = 3 |
| Drug-match check | 1 per batch | `EVAL_TRIAL_BATCH` = 6 trials, Gemini Pro, run sequentially |
| Endpoint enrichment | 1 per batch | `ENDPOINT_EXTRACTION_BATCH` = 6 trials |
| Metadata-only backfill (partial mode) | 1 per batch | `METADATA_BATCH` = 10 trials |
| Scoring | 0 | — |
| Rationale | 1 per molecule | — |
| Report narrative | 1 per molecule, +1 if data is thin | — |

Registry-API sources (ClinicalTrials.gov, CTIS, EudraCT, CTRI, WHO, PubMed) cost no Gemini calls.

---

## Configuration

Set via environment variables (read in `utils.py`) or `medical_potential.config`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | Gemini access (required) |
| `GEMINI_FLASH_PREVIEW_MODEL` | `gemini-3-flash-preview` | Enrichment, search, rationale, report |
| `GEMINI_PRO_MODEL` | `gemini-3.1-pro-preview` | Drug-match confidence check |
| `ENDPOINT_EXTRACTION_BATCH` | 6 | Trials per enrichment call |
| `EVAL_TRIAL_BATCH` | 6 | Trials per confidence-check call |
| `METADATA_BATCH` | 10 | Trials per metadata-only call |
| `TRIAL_REGISTRIES_PER_CALL` | 3 | Registries per international search call |
| `INTERNATIONAL_ALIAS_BATCH` | 10 | Aliases per international search call |
| `INNOVATOR_WEB_SOURCES_CALLS` | 3 | Gemini call budget for the innovator scan |
| `TRADE_CALLS`, `CONFERENCE_CALLS` | 4, 4 | Call budgets for trade and conference scans |
| `CLINICAL_TRIALS_PARTIAL_ENRICHMENT` | `false` | Metadata-only for trials already in BigQuery |
| `BQ_LOCATION` | `asia-south1` | BigQuery location used by the report builder |

---

## Persistence Summary

| Artifact | Location | Contents |
|----------|----------|----------|
| Trial rows + score columns | BigQuery clinical-efficacy table | One row per molecule/trial: metadata, endpoints, confidences, score, coverage, breakdown, rationale |
| Dimension score | Shared BigQuery dimension-score table | Molecule, "Clinical Efficacy", score, rationale |
| PDF report | GCS (via `upload_dimension_report_pdf_to_gcs`) | Business-facing efficacy report, with archived copy |
| Output payload | GCS pipeline cache (via `upload_dimension_payload_cache_to_gcs`) | Score result, rationale, report URIs, enriched rows |

---