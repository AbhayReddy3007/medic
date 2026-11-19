"""BigQuery reads and write-back for the clinical evidence dimension."""

from medical_potential.config import (
    BQ_DATASET_ID,
    CLINICAL_EFFICACY_TABLE,
    CLINICAL_EVIDENCE_PUBLICATION_DETAILS_TABLE,
    CLINICAL_EVIDENCE_SCORING_TABLE,
    CLINICAL_EVIDENCE_TRIAL_DETAILS_TABLE,
    PROJECT_ID,
    TRIAL_LLM_CONFIDENCE_THRESHOLD,
)
from medical_potential.gcp_utils import get_bq_client

from .scoring_logic.deterministic_scores import _get_highest_phase
from .utils import get_innovator, log, safe_str

CLINICAL_EFFICACY_TABLE_ID = f"{PROJECT_ID}.{BQ_DATASET_ID}.{CLINICAL_EFFICACY_TABLE}"
def query_bigquery(molecule: dict) -> list:
    try:
        client = get_bq_client()
    except Exception as e:
        log.err(f"BigQuery auth failed: {e}"); return []

    terms = molecule.get("synonyms", [molecule["generic_name"].lower()])
    where = " OR ".join(f"LOWER(molecule_name) LIKE '%{t}%'" for t in terms)
    sql = f"""
        SELECT trial_id, phase, trial_size, company_name,
               hba1c_change_pct, weight_change_pct, molecule_name
        FROM `{CLINICAL_EFFICACY_TABLE_ID}`
                WHERE ({where})
                    AND llm_confidence > {TRIAL_LLM_CONFIDENCE_THRESHOLD}
        ORDER BY phase, trial_id
    """
    log.info("Executing BigQuery query...")
    try:
        rows = []
        for row in client.query(sql).result():
            hba1c_val = None
            weight_val = None
            try:
                if row.hba1c_change_pct is not None:
                    hba1c_val = float(row.hba1c_change_pct)
            except (ValueError, TypeError):
                pass
            try:
                if row.weight_change_pct is not None:
                    weight_val = float(row.weight_change_pct)
            except (ValueError, TypeError):
                pass
            rows.append({
                "trial_id": row.trial_id, "phase": row.phase,
                "trial_size": row.trial_size, "company_name": row.company_name,
                "hba1c_change_pct": hba1c_val, "weight_change_pct": weight_val,
                "molecule_name": row.molecule_name, "source": "BigQuery",
                "results_available": (hba1c_val is not None or weight_val is not None),
            })
        log.ok(f"Retrieved {len(rows)} trials from BigQuery")
        return rows
    except Exception as e:
        log.err(f"BigQuery query error: {e}"); return []


def fetch_bq_molecules() -> list:
    """
    Returns a sorted list of distinct molecule names from the
    clinical_efficacy BigQuery table.
    Falls back to an empty list if BQ is unavailable.
    """
    try:
        client = get_bq_client()
    except Exception as e:
        log.err(f"BigQuery auth failed: {e}"); return []

    sql = f"""
        SELECT DISTINCT TRIM(molecule_name) AS molecule_name
        FROM `{CLINICAL_EFFICACY_TABLE_ID}`
        WHERE molecule_name IS NOT NULL AND TRIM(molecule_name) != ''
          AND llm_confidence > {TRIAL_LLM_CONFIDENCE_THRESHOLD}
        ORDER BY molecule_name
    """
    try:
        results = [row.molecule_name for row in client.query(sql).result()]
        log.ok(f"Found {len(results)} distinct molecules in BigQuery table.")
        return results
    except Exception as e:
        log.err(f"Failed to fetch molecule list from BigQuery: {e}")
        return []


def build_scoring_row(molecule, trials, publications, assessment,
                      consolidation, P, D, G, E, C, final, gaps, reasoning, ts):
    return {
        "molecule_name":           molecule["generic_name"],
        "brand_names":             "; ".join(molecule.get("brand_names", [])),
        "innovator":               get_innovator(trials, molecule),
        "dev_codes":               "; ".join(molecule.get("dev_codes", [])),
        "total_clinical_studies":  assessment["total_trials"],
        "studies_with_results":    assessment["trials_with_results"],
        "peer_reviewed_publications": assessment["total_publications"],
        "highest_trial_phase":     f"Phase {_get_highest_phase(trials)}" if _get_highest_phase(trials) > 0 else "None",
        "total_patients_enrolled": assessment["total_patients"],
        "largest_trial_size":      assessment["largest_trial"],
        "rct_count":               assessment["rct_count"],
        "has_regulatory_approval": assessment["has_regulatory_approval"],
        "score_phase_P":           P,
        "reasoning_phase_P":       safe_str(reasoning.get("P", "")),
        "score_design_D":          D,
        "reasoning_design_D":      safe_str(reasoning.get("D", "")),
        "score_geography_G":       G,
        "reasoning_geography_G":   safe_str(reasoning.get("G", "")),
        "score_evidence_E":        E,
        "reasoning_evidence_E":    safe_str(reasoning.get("E", "")),
        "score_consistency_C":     C,
        "reasoning_consistency_C": safe_str(reasoning.get("C", "")),
        "final_score":             final,
        "evidence_level":          assessment["evidence_level"],
        "evidence_detail":         assessment["evidence_detail"],
        "findings_consistency":    consolidation["findings_consistency"],
        "replication":             consolidation["replication"],
        "evidence_gaps":           "; ".join(gaps),
        "report_generated_at":     ts,
    }


def build_trial_rows(molecule, trials, ts):
    rows = []
    for t in trials:
        comps = t.get("comparators", [])
        comp_str = "; ".join(
            f"{c['arm_label']} ({c['arm_type']})" for c in comps
            if c.get("arm_label") != "None identified"
        ) or "None"
        rows.append({
            "molecule_name":     molecule["generic_name"],
            "trial_id":          t.get("trial_id", ""),
            "title":             safe_str(t.get("title", "")),
            "phase":             t.get("phase", ""),
            "study_design":      t.get("study_design", ""),
            "trial_size":        t.get("trial_size", ""),
            "trial_location":    t.get("trial_location", ""),
            "comparator_arms":   comp_str,
            "duration":          t.get("duration", ""),
            "status":            t.get("status", ""),
            "innovator":         t.get("company_name", ""),
            "results_available": t.get("results_available", False),
            "hba1c_change_pct":  t.get("hba1c_change_pct", ""),
            "weight_change_pct": t.get("weight_change_pct", ""),
            "source":            t.get("source", ""),
            "report_generated_at": ts,
        })
    return rows


def build_pub_rows(molecule, publications, ts):
    rows = []
    for p in publications:
        rows.append({
            "molecule_name":     molecule["generic_name"],
            "pmid":              p.get("pmid", ""),
            "doi":               p.get("doi", ""),
            "title":             safe_str(p.get("title", "")),
            "journal":           safe_str(p.get("journal", "")),
            "year":              p.get("year", ""),
            "study_phase":       p.get("study_phase", ""),
            "sample_size":       p.get("sample_size", ""),
            "population":        p.get("population", ""),
            "duration":          p.get("duration", ""),
            "key_outcomes":      safe_str(p.get("key_outcomes", "")),
            "safety_findings":   safe_str(p.get("safety_findings", "")),
            "citation_count":    p.get("citation_count", 0),
            "report_generated_at": ts,
        })
    return rows


def write_bq_to_bigquery(scoring_row, trial_rows, pub_rows):
    """Optionally write results back to BigQuery if credentials available."""
    try:
        client = get_bq_client()

        # molecule_scoring table
        table_scoring = f"{PROJECT_ID}.{BQ_DATASET_ID}.{CLINICAL_EVIDENCE_SCORING_TABLE}"
        errors = client.insert_rows_json(table_scoring, [scoring_row])
        if errors:
            log.warn(f"BQ insert errors (scoring): {errors}")
        else:
            log.ok(f"Inserted scoring row into {table_scoring}")

        # trial_details table
        if trial_rows:
            table_trials = f"{PROJECT_ID}.{BQ_DATASET_ID}.{CLINICAL_EVIDENCE_TRIAL_DETAILS_TABLE}"
            errors = client.insert_rows_json(table_trials, trial_rows)
            if errors:
                log.warn(f"BQ insert errors (trials): {errors}")
            else:
                log.ok(f"Inserted {len(trial_rows)} rows into {table_trials}")

        # publication_details table
        if pub_rows:
            table_pubs = f"{PROJECT_ID}.{BQ_DATASET_ID}.{CLINICAL_EVIDENCE_PUBLICATION_DETAILS_TABLE}"
            errors = client.insert_rows_json(table_pubs, pub_rows)
            if errors:
                log.warn(f"BQ insert errors (pubs): {errors}")
            else:
                log.ok(f"Inserted {len(pub_rows)} rows into {table_pubs}")

        return True
    except Exception as e:
        log.warn(f"BQ write-back failed: {e}")
        return False
