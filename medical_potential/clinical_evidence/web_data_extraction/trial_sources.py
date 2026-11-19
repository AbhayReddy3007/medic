"""Step 1: trial registry sources (ClinicalTrials.gov, EU CTR, WHO ICTRP) and merging."""

import re
import time
from collections import OrderedDict

from ..utils import http_get, log
from .molecule_normalization import CTGOV_API, _combo_name_variants
def classify_design(design_info: dict) -> str:
    alloc = design_info.get("allocation", "").upper()
    model = design_info.get("interventionModel", "").upper()
    stype = design_info.get("studyType", "").upper()
    if "RANDOMIZED" in alloc and "NON" not in alloc:
        return "Randomized Controlled Trial (RCT)"
    if "NON_RANDOMIZED" in alloc or "NON-RANDOMIZED" in alloc:
        return "Non-Randomized Controlled Study"
    if "SINGLE_GROUP" in model or "SINGLE GROUP" in model:
        return "Single-arm Study"
    if "OBSERVATIONAL" in stype:
        return "Observational / Real-world Study"
    return f"Other ({alloc or model or stype or 'Unknown'})"


def extract_comparators(arm_groups: list) -> list:
    comps = []
    for arm in arm_groups:
        atype = arm.get("type", "").upper()
        if atype in ("PLACEBO_COMPARATOR", "ACTIVE_COMPARATOR",
                      "SHAM_COMPARATOR", "NO_INTERVENTION"):
            comps.append({
                "arm_label": arm.get("label", "Unknown"),
                "arm_type": arm.get("type", "Unknown"),
                "description": arm.get("description", ""),
            })
    return comps or [{"arm_label": "None identified", "arm_type": "N/A", "description": ""}]


def extract_location_countries(contacts_mod: dict) -> list:
    """Return list of country names."""
    locs = contacts_mod.get("locations", [])
    countries = list(OrderedDict.fromkeys(
        l.get("country", "") for l in locs if l.get("country")
    ))
    return countries[:15]


def parse_ctgov_study(study: dict) -> dict:
    proto = study.get("protocolSection", {})
    id_mod       = proto.get("identificationModule", {})
    status_mod   = proto.get("statusModule", {})
    design_mod   = proto.get("designModule", {})
    sponsor_mod  = proto.get("sponsorCollaboratorsModule", {})
    arms_mod     = proto.get("armsInterventionsModule", {})
    contacts_mod = proto.get("contactsLocationsModule", {})

    phases = design_mod.get("phases", [])
    phase_str = ", ".join(phases) if phases else "N/A"
    design_info = design_mod.get("designInfo", {})
    design_info["studyType"] = design_mod.get("studyType", "")
    enrollment = design_mod.get("enrollmentInfo", {}).get("count", None)
    has_results = study.get("hasResults", False)
    arm_groups = arms_mod.get("armGroups", [])

    start = status_mod.get("startDateStruct", {}).get("date", "")
    end   = status_mod.get("primaryCompletionDateStruct", {}).get("date", "")
    duration = f"{start} to {end}" if start and end else "Not specified"

    countries = extract_location_countries(contacts_mod)

    return {
        "trial_id":          id_mod.get("nctId", ""),
        "title":             id_mod.get("officialTitle", id_mod.get("briefTitle", "")),
        "phase":             phase_str,
        "study_design":      classify_design(design_info),
        "trial_size":        enrollment,
        "trial_location":    ", ".join(countries) if countries else "Not specified",
        "trial_countries":   countries,
        "comparators":       extract_comparators(arm_groups),
        "duration":          duration,
        "status":            status_mod.get("overallStatus", ""),
        "company_name":      sponsor_mod.get("leadSponsor", {}).get("name", "Unknown"),
        "results_available": has_results,
        "hba1c_change_pct":  None,
        "weight_change_pct": None,
        "molecule_name":     "",
        "source":            "ClinicalTrials.gov",
    }


def _ctgov_search(query_intr: str, max_results: int = 20) -> list:
    all_studies = []
    page_token = None
    while len(all_studies) < max_results:
        params = {
            "query.intr": query_intr,
            "pageSize": min(20, max_results - len(all_studies)),
            "format": "json",
        }
        if page_token:
            params["pageToken"] = page_token
        r = http_get(CTGOV_API, params=params)
        if not r or r.status_code != 200:
            break
        data = r.json()
        studies = data.get("studies", [])
        if not studies:
            break
        all_studies.extend(studies)
        page_token = data.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.3)
    return all_studies


def global_trial_scan(molecule: dict, max_per_query: int = 20) -> list:
    generic = molecule["generic_name"]
    sponsor = molecule.get("sponsor", "")
    indications = molecule.get("indications", [])

    queries = [generic, f"{generic} phase 3", f"{generic} phase 2",
               f"{generic} randomized trial"]
    for ind in indications[:4]:
        queries.append(f"{generic} {ind}")
    if sponsor and sponsor != "Unknown":
        queries.append(f"{generic} {sponsor.split('/')[0].strip()}")

    # Combination therapies (e.g. "Cagrilintide+Semaglutide") are rarely
    # registered verbatim with that exact separator. Add the alternate
    # spellings (slash, "and", space-joined, reversed order) as additional
    # searches so trials registered under a different convention are still
    # found. We deliberately do NOT search each component in isolation
    # here -- that would pull in single-ingredient trials unrelated to the
    # combination and pollute the result set.
    if molecule.get("is_combination") and molecule.get("components"):
        for variant in _combo_name_variants(molecule["components"]):
            if variant.lower() != generic.lower():
                queries.append(variant)

    # De-duplicate queries (case-insensitive) while preserving order.
    seen_q = set()
    deduped_queries = []
    for q in queries:
        qk = q.lower().strip()
        if qk and qk not in seen_q:
            seen_q.add(qk)
            deduped_queries.append(q)
    queries = deduped_queries

    seen_nct = set()
    all_trials = []
    for q in queries:
        log.info(f"  Searching: '{q}'")
        raw = _ctgov_search(q, max_results=max_per_query)
        new = 0
        for study in raw:
            nct = (study.get("protocolSection", {})
                   .get("identificationModule", {}).get("nctId", ""))
            if nct and nct not in seen_nct:
                seen_nct.add(nct)
                rec = parse_ctgov_study(study)
                rec["molecule_name"] = generic
                all_trials.append(rec)
                new += 1
        log.info(f"    -> {new} new (total: {len(all_trials)})")
        time.sleep(0.4)
    log.ok(f"Global scan: {len(all_trials)} unique trials")
    return all_trials


def enrich_trial_from_ctgov(nct_id: str) -> dict:
    enrichment = {"study_design": "Unknown", "comparators": [], "title": "",
                  "status": "", "trial_location": "", "trial_countries": [],
                  "duration": ""}
    if not nct_id or not nct_id.upper().startswith("NCT"):
        return enrichment
    try:
        r = http_get(f"{CTGOV_API}/{nct_id}", params={"format": "json"})
        if not r or r.status_code != 200:
            return enrichment
        parsed = parse_ctgov_study(r.json())
        for k in enrichment:
            enrichment[k] = parsed.get(k, enrichment[k])
        enrichment["results_available"] = parsed.get("results_available", False)
    except Exception as e:
        log.warn(f"Enrichment failed for {nct_id}: {e}")
    return enrichment


def enrich_bq_trials(bq_rows: list) -> list:
    log.info(f"Enriching {len(bq_rows)} BQ trials...")
    for i, trial in enumerate(bq_rows):
        nct = trial.get("trial_id", "")
        log.info(f"  [{i+1}/{len(bq_rows)}] {nct}")
        enrichment = enrich_trial_from_ctgov(nct)
        trial.update(enrichment)
        if trial.get("hba1c_change_pct") is not None or trial.get("weight_change_pct") is not None:
            trial["results_available"] = True
        time.sleep(0.3)
    return bq_rows


def merge_trial_lists(*trial_lists):
    """Merge multiple trial lists, de-duplicating by trial_id. First list wins."""
    merged = OrderedDict()
    for tlist in trial_lists:
        for t in tlist:
            tid = t.get("trial_id", "")
            if tid and tid not in merged:
                merged[tid] = t
    return list(merged.values())


EU_CTR_SEARCH = "https://www.clinicaltrialsregister.eu/ctr-search/search"

def search_eu_ctr(molecule: dict, max_results: int = 20) -> list:
    """Search EU Clinical Trials Register via its web search."""
    generic = molecule["generic_name"]
    log.info(f"Searching EU CTR for '{generic}'...")

    trials = []
    r = http_get(EU_CTR_SEARCH, params={"query": generic, "output": "xml"}, timeout=20)
    if not r or r.status_code != 200:
        # Try HTML fallback
        r = http_get(EU_CTR_SEARCH, params={"query": generic}, timeout=20)
        if not r or r.status_code != 200:
            log.warn("EU CTR search failed or unreachable")
            return []

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")

        # Parse EudraCT result entries
        result_blocks = soup.select(".result, tr.result, div.result")
        if not result_blocks:
            # Try alternative selectors
            result_blocks = soup.find_all("table", class_="result")

        for block in result_blocks[:max_results]:
            trial = _parse_eu_ctr_block(block, generic)
            if trial and trial.get("trial_id"):
                trials.append(trial)

        # If structured parsing failed, try link-based extraction
        if not trials:
            for link in soup.find_all("a", href=True):
                href = link.get("href", "")
                if "ctr-search/trial" in href or "EudraCT" in link.get_text():
                    eudract = re.search(r"(\d{4}-\d{6}-\d{2})", href + link.get_text())
                    if eudract:
                        tid = eudract.group(1)
                        if tid and not any(t["trial_id"] == tid for t in trials):
                            title_text = link.get_text(strip=True)[:200]
                            trials.append({
                                "trial_id": tid,
                                "title": title_text,
                                "phase": "N/A",
                                "study_design": "Unknown",
                                "trial_size": None,
                                "trial_location": "Europe",
                                "trial_countries": ["European Union"],
                                "comparators": [{"arm_label": "None identified", "arm_type": "N/A", "description": ""}],
                                "duration": "Not specified",
                                "status": "Unknown",
                                "company_name": molecule.get("sponsor", "Unknown"),
                                "results_available": False,
                                "hba1c_change_pct": None,
                                "weight_change_pct": None,
                                "molecule_name": generic,
                                "source": "EU CTR",
                            })
                            if len(trials) >= max_results:
                                break

    except ImportError:
        log.warn("beautifulsoup4 needed for EU CTR parsing")
    except Exception as e:
        log.warn(f"EU CTR parse error: {e}")

    log.ok(f"EU CTR: {len(trials)} trials found")
    return trials


def _parse_eu_ctr_block(block, generic: str) -> dict:
    """Parse a single EU CTR result block."""
    text = block.get_text(" ", strip=True)
    eudract = re.search(r"(\d{4}-\d{6}-\d{2})", text)
    if not eudract:
        return {}

    # Phase detection
    phase = "N/A"
    for p_pat, p_label in [(r"Phase\s*IV", "PHASE4"), (r"Phase\s*III", "PHASE3"),
                            (r"Phase\s*II", "PHASE2"), (r"Phase\s*I\b", "PHASE1")]:
        if re.search(p_pat, text, re.IGNORECASE):
            phase = p_label; break

    return {
        "trial_id": eudract.group(1),
        "title": text[:200],
        "phase": phase,
        "study_design": "Unknown",
        "trial_size": None,
        "trial_location": "Europe",
        "trial_countries": ["European Union"],
        "comparators": [{"arm_label": "None identified", "arm_type": "N/A", "description": ""}],
        "duration": "Not specified",
        "status": "Unknown",
        "company_name": "",
        "results_available": False,
        "hba1c_change_pct": None,
        "weight_change_pct": None,
        "molecule_name": generic,
        "source": "EU CTR",
    }


# -- 1E: WHO ICTRP (trialsearch.who.int) --
WHO_ICTRP_SEARCH = "https://trialsearch.who.int/Trial2.aspx"
WHO_ICTRP_API = "https://trialsearch.who.int/api/search"

def search_who_ictrp(molecule: dict, max_results: int = 20) -> list:
    """Search WHO International Clinical Trials Registry Platform."""
    generic = molecule["generic_name"]
    log.info(f"Searching WHO ICTRP for '{generic}'...")

    trials = []

    # Try the search page
    r = http_get("https://trialsearch.who.int/Default.aspx", timeout=20)
    search_url = f"https://trialsearch.who.int/Trial2.aspx?ConditionID=&TrialID=&SearchFor={generic.replace(' ','+')}"
    r2 = http_get(search_url, timeout=20)

    if not r2 or r2.status_code != 200:
        # Fallback: try direct URL patterns
        r2 = http_get(
            "https://trialsearch.who.int/default.aspx",
            params={"SearchFor": generic}, timeout=20
        )

    if r2 and r2.status_code == 200:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r2.text, "html.parser")

            # WHO ICTRP returns results in table rows or div blocks
            for row in soup.find_all(["tr", "div"], class_=re.compile(r"result|trial", re.I)):
                text = row.get_text(" ", strip=True)
                trial = _parse_who_result(text, generic)
                if trial:
                    trials.append(trial)
                    if len(trials) >= max_results:
                        break

            # Alternative: look for trial ID patterns in all links
            if not trials:
                for link in soup.find_all("a", href=True):
                    href = link.get("href", "")
                    text = link.get_text(strip=True)
                    # WHO ICTRP links to various registries
                    trial_id = None
                    # ISRCTN pattern
                    m = re.search(r"(ISRCTN\d+)", text + href, re.IGNORECASE)
                    if m: trial_id = m.group(1)
                    # CTRI pattern (India)
                    m = re.search(r"(CTRI/\d{4}/\d+/\d+)", text + href)
                    if m: trial_id = m.group(1)
                    # JPRN pattern (Japan)
                    m = re.search(r"(JPRN-\S+)", text + href)
                    if m: trial_id = m.group(1)
                    # ACTRN pattern (Australia/NZ)
                    m = re.search(r"(ACTRN\d+)", text + href, re.IGNORECASE)
                    if m: trial_id = m.group(1)
                    # ChiCTR pattern (China)
                    m = re.search(r"(ChiCTR\S+)", text + href)
                    if m: trial_id = m.group(1)

                    if trial_id and not any(t["trial_id"] == trial_id for t in trials):
                        trials.append({
                            "trial_id": trial_id,
                            "title": text[:200] or trial_id,
                            "phase": "N/A",
                            "study_design": "Unknown",
                            "trial_size": None,
                            "trial_location": "International",
                            "trial_countries": [],
                            "comparators": [{"arm_label": "None identified", "arm_type": "N/A", "description": ""}],
                            "duration": "Not specified",
                            "status": "Unknown",
                            "company_name": molecule.get("sponsor", "Unknown"),
                            "results_available": False,
                            "hba1c_change_pct": None,
                            "weight_change_pct": None,
                            "molecule_name": generic,
                            "source": "WHO ICTRP",
                        })
                        if len(trials) >= max_results:
                            break

        except ImportError:
            log.warn("beautifulsoup4 needed for WHO ICTRP parsing")
        except Exception as e:
            log.warn(f"WHO ICTRP parse error: {e}")
    else:
        log.warn("WHO ICTRP unreachable")

    log.ok(f"WHO ICTRP: {len(trials)} trials found")
    return trials


def _parse_who_result(text: str, generic: str) -> dict:
    """Parse a WHO ICTRP result text block."""
    if generic.lower() not in text.lower():
        return None

    # Try to find a trial ID
    trial_id = None
    for pattern in [r"(NCT\d+)", r"(ISRCTN\d+)", r"(\d{4}-\d{6}-\d{2})",
                    r"(CTRI/\d{4}/\d+/\d+)", r"(ACTRN\d+)", r"(ChiCTR\S+)",
                    r"(JPRN-\S+)", r"(KCT\d+)", r"(DRKS\d+)"]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            trial_id = m.group(1)
            break

    if not trial_id:
        return None

    phase = "N/A"
    for p_pat, p_label in [(r"Phase\s*4", "PHASE4"), (r"Phase\s*3", "PHASE3"),
                            (r"Phase\s*2", "PHASE2"), (r"Phase\s*1", "PHASE1"),
                            (r"Phase\s*IV", "PHASE4"), (r"Phase\s*III", "PHASE3"),
                            (r"Phase\s*II", "PHASE2"), (r"Phase\s*I\b", "PHASE1")]:
        if re.search(p_pat, text, re.IGNORECASE):
            phase = p_label; break

    return {
        "trial_id": trial_id,
        "title": text[:200],
        "phase": phase,
        "study_design": "Unknown",
        "trial_size": None,
        "trial_location": "International",
        "trial_countries": [],
        "comparators": [{"arm_label": "None identified", "arm_type": "N/A", "description": ""}],
        "duration": "Not specified",
        "status": "Unknown",
        "company_name": "",
        "results_available": False,
        "hba1c_change_pct": None,
        "weight_change_pct": None,
        "molecule_name": generic,
        "source": "WHO ICTRP",
    }
