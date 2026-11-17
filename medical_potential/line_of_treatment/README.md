# Line of Treatment (LOT) Assessment Pipeline

The `line_of_treatment_assessment()` pipeline determines where a drug sits in the real-world treatment pathway for each country it operates in — first-line standard of care, second-line option, salvage use, and so on. It reads each country's official Standard of Care (SoC) guidance, extracts a structured benchmark of how that country sequences therapies, classifies the drug against that benchmark using its Mechanism of Action, scores the positioning across all countries into a single aggregate Final LOT Score, and produces a business-facing PDF report with a country-by-country breakdown.

---

## Pipeline Overview

```text
Step 1   MOA Lookup
         Read drug's Mechanism of Action from BigQuery → MOA records
                    │
                    ▼
Step 2   LOT classification
         Construct the LOT classification prompt using MOA records 
                    │
                    ▼
Step 3   Discover SoC Documents
         List country SoC PDF folders from GCS 
                    │
                    ▼
Step 4   Per Country SoC Benchmark Creation/Utilisation + LOT Classification
         For each country:
         ├── 4.1  Load or Build SoC Benchmark
         │         ├── Cache hit  → load JSON from GCS
         │         └── Cache miss → extract PDFs → Gemini extraction
         │                          → DOCX + JSON written to GCS
         └── 4.2  Overlay Analysis
                   Run overlay prompt against benchmark → LOT type, rationale, confidence
                    │
                    ▼
Step 5   Final LOT Score Calculation
         Aggregate per-country scores → weighted Final LOT Score (float)
                    │
                    ▼
Step 6   Persist Country-Level Results
         Write one row per drug/country → LOT table (BigQuery)
                    │
                    ▼
Step 7   Summarize Results
         Log per-country row counts
                    │
                    ▼
Step 8   PDF Report Generation
         Load scores + benchmark JSONs from BigQuery + GCS
         → Gemini narrative + country summaries → PDF (GCS)
                    │
                    ▼
Step 9   Rationale Generation & Dimension Score
         Gemini rationale (50 words) → dimension score table (BigQuery)
                    │
                    ▼
Step 10  Upload Output Payload
         Full JSON payload → pipeline cache (GCS)
```

> **BENCHMARK_RERUN mode**: when `BENCHMARK_RERUN = True` in `config.py`, the pipeline stops after rebuilding the SoC benchmark DOCX and JSON for the countries listed in `COUNTRIES_TO_RERUN`. No scoring, report, or BigQuery writes occur. This lets you refresh guidance documents without triggering a full re-score.

---
## Folder Layout

```
line_of_treatment/
├── line_of_treatment.py                    # orchestrator
│
├── lot_scoring.py
│   ├── lookup_moa()                        # BigQuery → MOA records
│   ├── build_overlay_prompt()              # MOA records → prompt string
│   ├── discover_countries_gcs()            # GCS listing → {country: [pdf blobs]}
│   ├── process_country()                   # per-country entry point
│   │   └── generate_country_lot.get_or_build_soc_benchmark()
│   │       ├── cache hit  → load JSON from GCS → plain text
│   │       └── cache miss → read PDFs → Gemini → save DOCX + JSON → plain text
│   ├── run_overlay_analysis()              # SoC text + prompt → Gemini → raw output
│   ├── parse_overlay_output()              # raw output → {lot_type, rationale, confidence}
│   ├── map_lot_type_to_score()             # lot_type string → int 1–5
│   └── compute_final_lot_score()           # per-country scores → weighted float
│
├── bq_utils.py
│   └── push_results_to_bigquery()          # LotRow list → BigQuery LOT_TABLE
│
└── generate_report_and_rationale/
    ├── generate_lot_report.py
    │   ├── load_from_bigquery()            # LOT_TABLE → scored rows
    │   ├── load_country_benchmark_json()   # GCS → benchmark JSON per country
    │   ├── extract_drug_stats()            # rows + JSONs → stats dict
    │   ├── generate_lot_narrative()        # Gemini → narrative + country summaries
    │   └── build_report()                  # stats + narrative → PDF bytes
    │
    └── generate_rationale.py
        ├── fetch_final_lot_data()          # BigQuery → per-country LOT data
        ├── _prepare_prompt_payload()       # reduce data → minimal prompt input
        └── generate_final_lot_rationale()  # Gemini → 50-word rationale string
 ```
---

## Input / Output Table

| Step | Input | Output |
|------|-------|--------|
| 1 — MOA Lookup | Drug name | MOA records from `drug_details` BigQuery table |
| 2 — LOT classification Prompt | MOA records (all variants per drug) | Structured plain-text LOT classification prompt |
| 3 — Discover SoC Documents | GCS bucket path (`GCS_SOC_BASE_PATH`) | 
| 4.1 — Load Benchmark (cache hit) | Uses the existing benchmark|  |
| 4.1 — Build Benchmark (cache miss) | Country SoC PDFs from GCS | Plain-text benchmark DOCX (`lot_benchmarks/docs/{country}.docx`) + structured JSON (`lot_benchmarks/json/{country}.json`) written to GCS |
| 4.2 — Overlay Analysis | SoC benchmark text + LOT classification Prompt | LOT type, rationale, confidence score per drug per country |
| 5 — Final LOT Score | Per-country `lot_score` values | Single weighted `final_lot_score` float |
| 6 — Persist Results | `LotRow` records (drug, country, lot_score, lot_type, rationale, confidence, final_lot_score) | Rows upserted into LOT BigQuery table |
| 7 — Summarize | `all_rows` list | Per-country row count logged |
| 8 — PDF Report | LOT scores from BigQuery + benchmark JSONs from GCS | PDF report uploaded to GCS; archived copy also stored |
| 9 — Rationale + Dimension Score | Per-country LOT data from BigQuery | 50-word plain-text rationale; row in `dim_scores` BigQuery table |
| 10 — Output Payload | All pipeline outputs | Full JSON payload uploaded to GCS pipeline cache; archived copy also stored |

---

## Step 1 — MOA Lookup

Establishes the biological basis for the drug's LOT classification before any SoC documents are read.

- Queries the `drug_details` BigQuery table for all distinct Mechanism of Action records associated with the drug.
- A single drug can have multiple MOA records from different source databases (e.g. "GLP-1 Receptor Agonist" and "Glucagon-like peptide-1 (GLP-1) agonist" for the same molecule). All variants are retained and passed forward.
- If no MOA record is found, the drug is assigned `"Unknown"` as its MOA and a warning is logged. The pipeline continues, though classification confidence will be lower.


---

## Step 2 — Build Overlay Prompt

Constructs the structured classification prompt that will be used in Step 4.2 for every country.

- Formats all MOA variants for the drug into a single prompt that instructs Gemini to act as a Senior Market Access Analyst.
- The prompt includes explicit LOT assignment rules , the five permitted classification labels, and a confidence scoring rubric.
- The prompt is built once and reused across all countries, since the drug's MOA does not change — only the SoC benchmark changes per country.

---

## Step 3 — Discover SoC Documents

Identifies which countries have SoC guidance available for processing.

- Lists all SOC PDFs using the GCS client.
- Expects the folder structure `{base_path}/{Country}/{document}.pdf` — each country occupies its own subfolder.



---

## Step 4 — Per-Country SoC Benchmark + LOT Classification

The core of the pipeline. For each discovered country, the pipeline first resolves the SoC benchmark (from cache or fresh extraction), then classifies the drug against it.

### 4.1 — Load or Build SoC Benchmark

The benchmark encodes each country's official treatment sequencing — which therapies are used at 1L, 2L, 3L, and Salvage — extracted from the country's SoC PDF(s).

**Cache hit (JSON exists in GCS):**
- Checks for an existing structured JSON benchmark at `gs://{GCS_BUCKET}/{LOT_BENCHMARK_PATH}/json/{country_slug}.json`.
- If found, loads and parses it directly.
- No PDFs are read and no Gemini call is made — the cached benchmark is used as-is.

**Cache miss (no JSON in GCS):**
- Downloads all SoC PDFs for the country from GCS using PyMuPDF to extract their text content.
- Concatenates the text of all documents, labelling each by filename, into a single input for Gemini.
- Calls Gemini (see Gemini Usage below) to extract the benchmark.
- Splits the response on the `---JSON---` marker into a plain-text section and a structured JSON section.
- Saves the plain-text section as a human-readable DOCX at `gs://{GCS_BUCKET}/{LOT_BENCHMARK_PATH}/docs/{country_slug}.docx`.
- Saves the structured JSON this becomes the cache for all future runs.


### 4.2 — Overlay Analysis

Classifies the drug into a LOT category for this country by running the overlay prompt against the SoC benchmark.

- Provides the SoC benchmark text as context and the LOT classification prompt (built in Step 2) as the classification task.
- Gemini returns a structured plain-text block per drug containing:
  - **Final LoT Category** — one of five permitted labels
  - **Rationale** — explanation of why the drug's MOA maps to that treatment line
  - **Confidence** — integer 0–100 reflecting how directly the MOA matches explicit SoC language
- The output is parsed to extract these three fields using regex-based field extraction.

- The LOT type label is mapped to a numeric score (1–5) using the `LOT_SCORE_MAP` lookup, with a keyword-based fallback for minor wording variations.


---

## Step 5 — Final LOT Score Calculation

Aggregates the per-country LOT scores into a single weighted score for the drug.

The formula reflects the commercial importance of the US market relative to other markets:

```text
Final LOT Score = (US lot_score × 0.58) + (sum of all other country lot_scores × 0.14)
```

- Countries without a resolvable LOT score (e.g. where the overlay returned an unrecognised classification) are excluded from the calculation.
- Returns `None` if no country produced a scoreable result.
- The `final_lot_score` is attached to every `LotRow` so it appears alongside the country-level detail in BigQuery.



---

## Step 6 — Persist Country-Level Results

Writes the per-country classification results to BigQuery for downstream consumption by the report generator and rationale module.

- Upserts one row per drug/country combination into the LOT BigQuery table.
- Each row carries: `drug_name`, `country`, `lot_score`, `lot_type`, `rationale`, `confidence`, `final_lot_score`, and a `created_at` timestamp.
- The `(drug_name, country)` pair is used as the upsert key — re-running the pipeline for the same drug overwrites the prior result rather than creating duplicates.



---

## Step 7 — Summarize Results

Logs a per-country summary of how many rows were successfully produced, for operational monitoring.

- Iterates over all discovered countries and counts the `LotRow` records for each.
- Logs the count at `[INFO]` level so it appears in the pipeline run log.

No new artifacts are produced at this step.

---

## Step 8 — PDF Report Generation

Produces the business-facing PDF report combining the scored results from BigQuery with the SoC benchmark detail from GCS.

- Loads the latest LOT scores per country from BigQuery (one row per `drug_name/country`).
- Loads the structured SoC benchmark JSON from GCS for each country — these provide the 1L/2L/3L/Salvage detail used to populate the country breakdown section of the report.
- Makes a single Gemini call (see Gemini Usage below) that produces both: per-country LLM summaries of each LOT line and rationale (2 sentences each), and the full business narrative sections (Key Findings, Insights and Implications, Profile Summary).
- Renders the PDF using ReportLab with the following structure:
  1. Score grid — one box per country + FINAL box, with the weighting formula shown
  2. Key Line of Treatment Findings — bullet summary, geographic variation, treatment-line positioning
  3. Country wise LOT Breakdown — one block per country: 1L/2L/3L/Salvage + Rationale row
  4. Insights and Implications — market access impact, geographic strategy, competitive positioning
  5. Line of Treatment Profile Summary — overall assessment, score justification, cross-country consistency
  6. Line of Treatment Scoring Reference — Score | Classification | Definition table
- Uploads the PDF to GCS and stores an archived (timestamped) copy.

**Artifacts produced:**
- PDF report at `gs://{GCS_BUCKET}/reports/{molecule}/Line of Treatment.pdf`
- Archived PDF at `gs://{GCS_BUCKET}/reports/{molecule}/archived/Line of Treatment_{timestamp}.pdf`

---

## Step 9 — Rationale Generation & Dimension Score

Generates a concise qualitative rationale for the drug's Final LOT Score and persists the dimension-level result.

- Fetches all per-country LOT data for the drug from the LOT BigQuery table.
- Reduces the data to the fields needed for a concise explanation (country, lot_score, lot_type, confidence, a short rationale excerpt).
- Calls Gemini (see Gemini Usage below) to produce a plain-English rationale of at most 50 words, written for a Medical Affairs / Market Access audience.
- Appends the Final LOT Score and rationale to the shared `dim_scores` BigQuery table — the single table that aggregates scores across all pipeline dimensions for dashboard consumption.

**Artifacts produced:** row in `dim_scores` BigQuery table containing molecule name, dimension name, Final LOT Score, and rationale.

---

## Step 10 — Upload Output Payload

Packages the complete pipeline output into a structured JSON payload and caches it in GCS.

- Assembles a JSON object containing: molecule name, all per-country `LotRow` results, final LOT score, rationale, rationale prompt payload, and GCS URIs for the PDF report.
- Uploads to the pipeline cache location in GCS.
- Stores an archived (timestamped) copy for version history.

**Artifacts produced:**
- JSON payload at `gs://{GCS_BUCKET}/pipeline_cache/{molecule}/Line of Treatment/output_payload.json`
- Archived JSON at `gs://{GCS_BUCKET}/pipeline_cache/{molecule}/Line of Treatment/archived/output_payload_{timestamp}.json`

---

## Gemini Usage

Gemini is used at three distinct points in the pipeline. Each call is purpose-built for a specific reasoning task that structured data queries cannot perform.

| Step | Task | Input | Output | Gemini capability |
|------|------|-------|--------|-------------------|
| 4.1 — SoC Benchmark Extraction | Read one or more SoC PDFs for a country (potentially in mixed languages) and extract the country's treatment sequencing for T2DM / Obesity as both a plain-text benchmark and a structured JSON | Concatenated text of all country SoC PDFs + system instruction defining the extraction schema | Plain-text benchmark (Section 1) + structured JSON with 1L/2L/3L/Salvage fields per line of therapy (Section 2), separated by `---JSON---` marker | Text extraction + schema-constrained structured output; multilingual (translates non-English documents to English) |
| 4.2 — LOT Classification (Overlay Analysis) | Classify the drug's Mechanism of Action against the country's SoC benchmark and assign it to a treatment line | SoC benchmark plain-text + overlay prompt containing MOA variants, assignment rules, five permitted LOT labels, and confidence rubric | Structured plain-text block per drug: `Final LoT Category`, `Rationale`, `Confidence` (0–100 integer) | Constrained classification with chain-of-thought rationale; temperature 0 for deterministic output |
| 8 — Report Narrative + Country Summaries | Summarize each country's 1L/2L/3L/Salvage and Rationale into 2 concise sentences each, and generate the full business narrative for the PDF report | Per-country LOT scores, types, rationales, and full SoC benchmark text for all countries | `country_summaries` dict (2-sentence summaries per LOT line and rationale per country) + `key_findings`, `insights_implications`, and `profile_summary` narrative sections | Long-form structured generation with JSON output schema; single call produces both the per-country summaries and the full narrative |
| 9 — Final Rationale Generation | Produce a concise qualitative explanation of the drug's Final LOT Score for Medical Affairs / Market Access stakeholders | Reduced per-country data (country, lot_score, lot_type, confidence, rationale excerpt) as JSON | Plain-text rationale, maximum 50 words, explaining the primary drivers behind the Final LOT Score | Constrained generation; temperature 0 |

---

## SoC Benchmark Cache

The SoC benchmark extraction (Step 4.1 cache miss) is the most expensive operation in the pipeline — it reads PDF files and makes a Gemini call per country. The results are cached in GCS so that subsequent pipeline runs reuse them without re-reading PDFs or calling Gemini again.

```text
gs://{GCS_BUCKET}/{LOT_BENCHMARK_PATH}/
  docs/{country_slug}.docx   ← human-readable benchmark (Word document)
  json/{country_slug}.json   ← machine-readable benchmark (used for classification and report)
```

The JSON is the authoritative cache. The DOCX is written alongside it for human inspection but is not used programmatically. To force a fresh extraction for a specific country, either delete its JSON blob from GCS directly, or set `BENCHMARK_RERUN = True` and `COUNTRIES_TO_RERUN = ["Country"]` in `config.py` and run the pipeline — it will delete both artefacts and rebuild them before exiting without running the scoring pipeline.

---

## LOT Score Reference

| Score | Classification | Definition |
|-------|---------------|------------|
| 5 | First-line standard of care | Drug is the primary, default treatment option for the majority of eligible patients per this country's guidelines |
| 4 | Strong first-line alternative / dominant second-line | Drug is a preferred alternative at first line or the dominant choice when escalating to second line; broadly accessible and guideline-endorsed |
| 3 | Second-line option | Drug is used after first-line therapy fails or is insufficient; typically added on or substituted when initial treatment goals are not met |
| 2 | Third-line or restricted niche use | Drug is reserved for specific sub-populations, refractory cases, or situations where earlier lines are contraindicated or exhausted |
| 1 | Salvage / last-resort use | Drug is used as a last resort when all other treatment options have failed; may include compassionate use or highly restricted access |

### Final LOT Score Formula

```text
Final LOT Score = (US lot_score × 0.58) + (sum of all other country lot_scores × 0.14)
```

The US market is weighted at 0.58 to reflect its disproportionate commercial importance. All other markets each contribute 0.14. The score ranges from 1 to 5 and is rounded to 4 decimal places.

---

## Pipeline Re-entry

The pipeline supports re-entry from specific stages, allowing later steps to be rerun without repeating earlier work — for example, to regenerate the report after a formatting change without re-running the Gemini classification across all countries.

```text
benchmark_rebuild   (BENCHMARK_RERUN=True — rebuilds SoC JSON/DOCX only, then exits)
    ↓
scoring             (Steps 1–7 — MOA lookup through BigQuery persist)
    ↓
report_generation   (Step 8 — reads from BigQuery + GCS benchmark JSONs)
    ↓
rationale           (Step 9 — reads from BigQuery LOT table)
    ↓
payload_upload      (Step 10 — assembles and caches the full output)
```

Each re-entry point reads from the BigQuery or GCS artifacts written by the preceding step, so intermediate results are fully preserved and reusable across runs.

---

