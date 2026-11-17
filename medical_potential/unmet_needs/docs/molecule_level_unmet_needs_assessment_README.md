# Molecule-Level Unmet Needs Assessment (Biweekly)

## Purpose

Determine how well each molecule addresses known unmet needs and compute a weighted molecule-level score.

Primary orchestrators:
- Core orchestrator (no CLI): `unmet_needs_assessment.py`
  - Function: `unmet_needs_assesment(molecule_name, incremental_mode=True)`
- Call the core orchestrator from application code or a scheduled workflow;

Core implementation modules:
- `compute_unmet_need_score_from_trial_data/trial_to_unmet_needs_mapper.py`
- `compute_unmet_need_score_from_trial_data/unmet_need_scorer.py`
- `compute_unmet_need_score_from_trial_data/bq_utils.py`

## How This Pipeline Works

1. Resolve the execution mode
- Incremental mode: process only trials newer than the last processed timestamp for the molecule.
- Full refresh mode: process all eligible trials and write mapping output using truncate behavior.

2. Load and prepare molecule trial evidence
- Read trial rows from `clinical_efficacy`.
- In incremental mode, use latest `created_at_utc` from mapping table as cutoff.
- Keep one canonical row per `trial_id` using phase-priority and deterministic tie-breaking.

3. Load unmet-needs baseline
- Read unmet-need records from `unmet_needs_all_countries`.
- Apply configured category exclusions so only in-scope needs are assessed.

4. Generate trial-to-unmet-need mappings
- Batch trials and call Gemini with trial + unmet-need context.
- Parse and normalize model outputs.
- Re-attach canonical trial and unmet-need metadata before persistence.

5. Keep mapping table idempotent for processed trials
- Identify `trial_id`s processed in current run.
- Delete existing mapping rows for those IDs, then write refreshed rows.

6. Score unmet needs for the molecule
- Read mapping rows back from BQ.
- Group into unique unmet-need units.
- Score each group with Gemini and compute weighted row-level scores.
- Aggregate to a molecule-level unmet-need score.

7. Persist scoring output and return run payload
- Write scoring rows to `molecule_level_unmet_needs_scoring`.
- Return operational indicators (`clinical_trials_assessed`, counts, scoring status).

## BigQuery Data Flow

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}, 'flowchart': {'nodeSpacing': 55, 'rankSpacing': 85, 'curve': 'basis'}} }%%
flowchart TB
    CE[(BQ: clinical_efficacy)] --> TL[Trial loader + incremental filter]
    UN[(BQ: unmet_needs_all_countries)] --> UL[Unmet-need loader]

    TL --> MAP[Trial-to-unmet-need mapping]
    UL --> MAP

    MAP --> DEL[Delete rows for processed trial_ids]
    DEL --> MAPBQ[(BQ: clinical_trial_evidence_to_unmet_need_mapping)]

    MAPBQ --> SCORE[Unmet-need scorer]
    SCORE --> SCOREBQ[(BQ: molecule_level_unmet_needs_scoring)]
```

## Function Input-to-Output Flow

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}, 'flowchart': {'nodeSpacing': 55, 'rankSpacing': 85, 'curve': 'basis'}} }%%
flowchart TB
    I1[Input: molecule_name] --> ORCH[unmet_needs_assesment]
    I2[Input: incremental_mode] --> ORCH

    ORCH --> MAP[map_clinical_trial_data_with_unmet_needs]
    MAP --> LTRIAL[load_clinical_trials]
    LTRIAL --> CUT[_get_latest_mapping_timestamp]
    MAP --> LUNMET[load_unmet_needs]
    MAP --> PIPE[run_pipeline]
    PIPE --> BATCH[process_batch]
    BATCH --> LLMMAP[call_gemini_with_search]
    BATCH --> PARSE[parse_json_response + normalize_results]
    MAP --> PREP[prepare_mapping_rows]
    MAP --> CLEAN[delete_existing_mappings_for_trial_ids]
    MAP --> EXP_MAP[export_to_gbq -> mapping table]

    ORCH --> SCORE[score_unmet_needs]
    SCORE --> IMP[import_from_gbq]
    SCORE --> GROUP[build_unique_unmet_need_groups]
    SCORE --> SCOREB[score_all_unique_unmet_needs]
    SCOREB --> SCORE1[score_unmet_need_batch]
    SCORE1 --> LLM_SCORE[call_gemini]
    SCORE1 --> FINALIZE[finalize_scoring_record]
    SCORE --> AGG[aggregate_single_score]
    SCORE --> EXP_SCORE[export_to_gbq -> scoring table]

    EXP_SCORE --> O1[Output: scoring rows + aggregate score]
    ORCH --> O2[Output: final payload with run indicators]
```

## Key BigQuery Tables

Input tables:
- `cognito-prod-394707.cognito_prod_datamart.unmet_needs_all_countries`
- `cognito-prod-394707.cognito_prod_datamart.clinical_efficacy`

Output tables:
- `cognito-prod-394707.cognito_prod_datamart.clinical_trial_evidence_to_unmet_need_mapping`
- `cognito-prod-394707.cognito_prod_datamart.molecule_level_unmet_needs_scoring`

## Operating Modes

- Incremental (default): process only newly created/updated trials.
- Full refresh: process all candidate trials and truncate mapping-write target before load.

## Recommended Practices

- Keep all writes schema-driven using shared schema definitions in `compute_unmet_need_score_from_trial_data/bq_utils.py`.
- Use parameterized SQL for every runtime input.
- Keep all timestamps and cutoffs in UTC.
- Preserve idempotency with delete-before-append or full-refresh truncate mode.
- Monitor operational quality:
  - no-trials-assessed frequency,
  - parser fallback/error rates,
  - BQ row counts loaded vs expected.
- Store run metadata (molecule, mode, row counts, start/end timestamps) for audit and troubleshooting.
