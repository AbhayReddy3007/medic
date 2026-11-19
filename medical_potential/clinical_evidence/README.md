# Clinical Evidence Analyzer

Scores the clinical evidence behind a pharmaceutical molecule and turns the result into a short business report.

## Folder structure

```text
clinical_evidence/
  clinical_evidence.py            # Orchestrator: clinical_evidence_scoring (entry point)
  bq_utils.py                     # BigQuery reads (trials) and write-back, flat row builders
  llm_runtime.py                  # Gemini transport: global concurrency cap, plain + grounded calls
  utils.py                        # Buffered logging, HTTP and text helpers
  scoring_logic/
    deterministic_scores.py       # P, D, G and the final-score formula
    llm_scores.py                 # E and C (Gemini, with rule-based fallbacks)
    evidence_assessment.py        # Evidence level, consolidation, gap identification
  web_data_extraction/
    molecule_normalization.py     # Step 0: openFDA + ClinicalTrials.gov
    trial_sources.py              # Step 1: CT.gov, EU CTR, WHO ICTRP, merge
    innovator.py                  # Step 2: FDA product entries, sponsor links
    real_world_evidence.py        # Step 3: Google-Search-grounded RWE synthesis
  generate_report_and_rationale/
    generate_report.py            # PDF report
    generate_rationale.py         # Short score rationale
```

The modules work together:

| Script | What it does | Main output |
|---|---|---|
| `clinical_evidence.py` | Collects trial and real-world evidence for a molecule, scores it on five dimensions (P, D, G, E, C) and produces a final score out of 5 | Score  in GBQ table |
| `generate_report.py` | Reads the GBQ table from the first script and writes an executive-friendly PDF | PDF report |
| `generate_rationale.py` | Reads the GBQ table from the first script and writes an executive-friendly rationale for the generated score | 200 word summary |

---

## Setup

Python 3.10+ is required.

```
pip install requests python-dotenv google-generativeai google-cloud-bigquery beautifulsoup4 openpyxl reportlab
```

Create a config or`.env` file next to the scripts:

```
GEMINI_API_KEY=<your Gemini API key>
BQ_SERVICE_ACCOUNT=<path to BigQuery service-account JSON key>
```


- `BQ_SERVICE_ACCOUNT` is needed for the BigQuery trial source

---


## How `clinical_evidence.py` works

| Step | What happens | Sources |
|---|---|---|
| 0. Molecule normalisation | Resolves the generic name, brand names, sponsor and indications | openFDA, ClinicalTrials.gov |
| 1. Global trial scan | Collects trials from four registries, enriches BigQuery trials with ClinicalTrials.gov detail, then merges and de-duplicates | BigQuery (`clinical_efficacy` table), ClinicalTrials.gov, EU Clinical Trials Register, WHO ICTRP |
| 2. Innovator check | Looks up FDA product entries and builds sponsor-website search links | openFDA |
| 3. Real-world evidence synthesis | Produces post-launch evidence summaries across six data sources and merges them into one synthesis (MIMIC-IV,CMS Medicare Files,ResDAC,CDC Open Data,NIH,Innovator websites). Each source call is grounded with Google Search and its source URLs are kept | Gemini (Google Search grounding) |
| 4. Evidence quality assessment | Counts phases, designs, patients and results, and assigns an evidence level | Rule-based |
| 5. Evidence consolidation | Checks whether HbA1c and weight endpoints agree across trials, and whether phases are replicated | Rule-based |
| 6. Scoring | Computes P, D, G, E, C and the final score; lists evidence gaps | Rules and Gemini |

Evidence level (Step 4):

- **STRONG**: a Phase 3/4 trial exists, at least 2 trials have results, and there are at least 2 RCTs or an FDA approval
- **MODERATE**: at least one RCT with results
- **LIMITED**: only Phase 1/2 studies
- **INSUFFICIENT**: nothing with results

---

## The score

Each dimension is scored on its own, then combined.

| Dimension | Meaning | How it is scored | Method |
|---|---|---|---|
| **P** Phase | How far development has progressed | Phase 3/4 = 5, Phase 2 = 4, Phase 1 = 2, none = 0 | Rules |
| **D** Design | Quality of the best study design | RCT = 5, non-randomised = 3, single-arm = 2, observational = 2, none = 0 | Rules |
| **G** Geography | Where trials were run | US/EU/UK (Tier 1) = 5; Canada, Switzerland, Australia, Japan (Tier 2) = 4; other or unknown = 3 | Rules |
| **E** Evidence support | Strength of post-launch real-world evidence | One of 0, 1, 3, 4, 5 (5 = approved with strong, multi-source real-world evidence; 1 = trials registered but no results) | **Gemini**, rule-based fallback |
| **C** Consistency | Whether findings agree and replicate | One of +0.5, +0.25, -0.5, -1.0 | **Gemini**, rule-based fallback |

```
Final = min(5, max(1, (P + mode(D, G, E)) / 2 + C))
```

If D, G and E are all different (no mode), the median of the three is used. Example: P=5, D=5, G=5, E=4, C=+0.5 gives (5 + 5) / 2 + 0.5 = 5.5, which is capped at 5.0.

---

## Where Gemini is used

All gemini calls 

| # | Script | Purpose | Input to Gemini | What comes back |
|---|---|---|---|---|
| 1 | `clinical_evidence.py` Step 3 | Per-source real-world evidence summary. Six calls: MIMIC-IV, CMS Medicare, ResDAC, CDC/NHANES, NIH All of Us, innovator website | Drug name and data-source name | A 150-200 word summary of HbA1c, weight, cardiovascular/renal and safety findings, or a statement that data is limited |
| 2 | `clinical_evidence.py` Step 3 | Integrated real-world evidence synthesis | The six source summaries | A 300-400 word synthesis covering consistency, strength, alignment with trials and gaps. Stored in the JSON and feeds E and C |
| 3 | `clinical_evidence.py` Step 6 | **E score** (Evidence Support) | RWE synthesis, FDA-approval flag, count of trials with results, endpoint availability | `SCORE` (0, 1, 3, 4 or 5) and a short reasoning |
| 4 | `clinical_evidence.py` Step 6 | **C score** (Consistency) | Trial counts, RCT count, endpoint values, terminated trials, replication summary, RWE synthesis | `SCORE` (+0.5, +0.25, -0.5 or -1.0), `PCT` (estimated share of consistent studies) and reasoning |
| 5 | `generate_rationale.py` | Rationale | GBQ score table | A single company name |
| 6 | `generate_report.py` | Business narrative for the PDF | Scores, evidence statistics, trial list, gaps, RWE synthesis | Headline, evidence landscape, key insights, gaps and risks, score reference, methodology note and bottom line |

Everything else (molecule normalisation, registry searches, phase/design/geography scoring, evidence level, consolidation, gap identification, file writing) is plain code with no Gemini involvement.

**Grounding (calls 1):** each per-source RWE call runs with Google Search grounding, so summaries come from retrieved pages. The prompt tells the model to report only figures found in those pages. The grounding URLs are collected per source and de-duplicated, then returned as `rwe_grounding_urls` (and `rwe_grounding_urls_by_source` in `report_input_payload`) in the output payload cached to GCS. The synthesis (call 2) and the E and C scores are not grounded; they read the grounded summaries. The databases themselves (MIMIC-IV, CMS, ResDAC and so on) are still not queried directly, so figures should be spot-checked against the listed URLs before decisions rely on them.

---

## Output files

`clinical_evidence.py` writes to `output/` (timestamped):

| File | Contents |
|---|---|
| `<molecule>_evidence_data_<ts>.json` | Everything: molecule info, scores and reasoning, trials, consolidation, gaps, RWE synthesis. This is the input to `ce_report.py`. |
| `<molecule>_evidence_report_<ts>.txt` | Full text report |
| `<molecule>_scoring_<ts>.csv` | One row of scores per molecule |
| `<molecule>_trials_<ts>.csv` | One row per trial |
| `<molecule>_publications_<ts>.csv` | Publication rows (publication search is currently disabled, so this is empty) |
| `pdgec_scores_<ts>.xlsx` | Excel summary of P, D, G, E, C, final score, evidence level, trial counts, gaps and consistency commentary for all molecules in the run |

`ce_report.py` writes `<molecule>_evidence_report_<ts>.pdf`.

## The PDF report

Written for senior business readers: no scoring jargon up front, and the scoring explanation is kept to the end. It contains:

1. Title block with molecule, innovator, brands and indications
2. A score strip and an evidence snapshot (trials, results, RCTs, patients, FDA status, evidence level)
3. Headline: one-line overall implication
4. Evidence landscape: the size and shape of the evidence base
5. Key insights, each with its business implication
6. Evidence gaps and risks
7. Score reference: a brief, one-line mention
8. Scoring methodology: a short explanation at the end
9. Bottom line: what a decision-maker should take away


