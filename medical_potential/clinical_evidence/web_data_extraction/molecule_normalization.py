"""Step 0: molecule normalisation from openFDA and ClinicalTrials.gov."""

import re
from collections import OrderedDict

from ..utils import http_get, log

CTGOV_API = "https://clinicaltrials.gov/api/v2/studies"
FDA_API = "https://api.fda.gov/drug/drugsfda.json"
def _fetch_brand_names_from_fda(name: str) -> tuple:
    """Fetch brand names, manufacturer, and dev codes from openFDA."""
    brand_names = set()
    manufacturers = set()
    app_numbers = set()

    for field in ["generic_name", "brand_name", "substance_name"]:
        r = http_get(FDA_API, params={
            "search": f'openfda.{field}:"{name}"', "limit": 20,
        })
        if r and r.status_code == 200:
            try:
                for res in r.json().get("results", []):
                    openfda = res.get("openfda", {})
                    for bn in openfda.get("brand_name", []):
                        brand_names.add(bn)
                    for mfg in openfda.get("manufacturer_name", []):
                        manufacturers.add(mfg)
                    app_no = res.get("application_number", "")
                    if app_no:
                        app_numbers.add(app_no)
            except Exception:
                pass

    return list(brand_names), list(manufacturers), list(app_numbers)


def _fetch_info_from_ctgov(name: str) -> dict:
    """Fetch sponsor, dev codes, indications from ClinicalTrials.gov."""
    info = {"sponsor": "Unknown", "dev_codes": [], "indications": [],
            "other_ids": []}
    try:
        r = http_get(CTGOV_API, params={
            "query.intr": name, "pageSize": 20, "format": "json",
        })
        if not r or r.status_code != 200:
            return info

        studies = r.json().get("studies", [])
        if not studies:
            return info

        # Sponsor from first study
        sp = (studies[0].get("protocolSection", {})
              .get("sponsorCollaboratorsModule", {})
              .get("leadSponsor", {}).get("name", "Unknown"))
        info["sponsor"] = sp

        # Collect indications + other IDs (potential dev codes)
        indications = set()
        other_ids = set()
        for s in studies[:20]:
            proto = s.get("protocolSection", {})
            # Conditions
            conds = proto.get("conditionsModule", {}).get("conditions", [])
            indications.update(c.lower() for c in conds)
            # Secondary IDs (dev codes like NN9535, LY3298176, etc.)
            id_mod = proto.get("identificationModule", {})
            for sec_id in id_mod.get("secondaryIdInfos", []):
                sid = sec_id.get("id", "")
                if sid and not sid.startswith("NCT"):
                    other_ids.add(sid)
            org_id = id_mod.get("orgStudyIdInfo", {}).get("id", "")
            if org_id and not org_id.startswith("NCT"):
                other_ids.add(org_id)

        info["indications"] = list(indications)[:15]
        info["dev_codes"] = list(other_ids)[:10]

    except Exception as e:
        log.warn(f"ClinicalTrials.gov discovery error: {e}")

    return info


_COMBO_SEPARATOR_PATTERN = re.compile(r"\s*[+/]\s*")


def _split_combo_components(name: str) -> list:
    """
    Split a combination-therapy name into its individual active-ingredient
    components. Recognizes '+' and '/' as separators, e.g.:
        "Cagrilintide+Semaglutide" -> ["Cagrilintide", "Semaglutide"]
        "cagrilintide/semaglutide" -> ["cagrilintide", "semaglutide"]
    Returns a single-item list (just the original, trimmed) for
    non-combination names.
    """
    parts = [p.strip() for p in _COMBO_SEPARATOR_PATTERN.split(name) if p.strip()]
    return parts if len(parts) > 1 else [name.strip()]


def _combo_name_variants(components: list) -> list:
    """
    Given ["Cagrilintide", "Semaglutide"], generate the alternate ways a
    combination therapy's name commonly appears across FDA/registry
    sources, so lookups aren't limited to a single '+'-joined spelling.
    """
    variants = set()
    if len(components) < 2:
        return []
    variants.add("+".join(components))
    variants.add("/".join(components))
    variants.add(" and ".join(components))
    variants.add(" ".join(components))
    # Reversed order too -- registries aren't consistent about which
    # active ingredient is listed first.
    rev = list(reversed(components))
    variants.add("+".join(rev))
    variants.add("/".join(rev))
    variants.add(" and ".join(rev))
    variants.add(" ".join(rev))
    return sorted(variants)


def normalize_molecule(name: str) -> dict:
    """
    Step 0: Build a normalized molecule record entirely from web sources.

    Handles combination therapies (e.g. "Cagrilintide+Semaglutide") by
    detecting the '+' / '/' separator, looking up each active ingredient
    individually on top of the combined name, and merging the results.
    Single-ingredient molecules ("Semaglutide", "Orforglipron Calcium")
    go through the same path unchanged -- _split_combo_components just
    returns a one-item list for them, so nothing extra happens.
    """
    key = name.strip().lower()
    components = _split_combo_components(name)
    is_combo = len(components) > 1

    record = {
        "generic_name": name.strip().title(),
        "brand_names": [],
        "dev_codes": [],
        "sponsor": "Unknown",
        "indications": [],
        "synonyms": [key],
        "is_combination": is_combo,
        "components": components if is_combo else [],
    }

    if is_combo:
        log.info(f"'{name}' detected as a combination therapy: "
                  f"{' + '.join(components)}")
        # Seed synonyms with alternate separator spellings so downstream
        # BigQuery LIKE-matching and search queries aren't limited to the
        # exact '+'-joined string the user typed.
        record["synonyms"].extend(v.lower() for v in _combo_name_variants(components))

    # 1) Fetch brand names + manufacturer from openFDA
    log.info(f"Looking up '{name}' on openFDA...")
    brands, manufacturers, app_nos = _fetch_brand_names_from_fda(name)

    # 2) Fetch sponsor, dev codes, indications from ClinicalTrials.gov
    log.info(f"Looking up '{name}' on ClinicalTrials.gov...")
    ctgov_info = _fetch_info_from_ctgov(name)

    if is_combo:
        # The exact combo string often isn't registered verbatim anywhere,
        # so also look up each individual active ingredient and merge in
        # anything the combined-name lookup didn't already find. This is
        # what actually makes combo names like "Cagrilintide+Semaglutide"
        # return useful sponsor/brand/indication data instead of coming
        # back empty.
        combo_brands, combo_mfgs = set(brands), set(manufacturers)
        combo_indications = set(ctgov_info["indications"])
        combo_dev_codes = set(ctgov_info["dev_codes"])
        combo_sponsor = ctgov_info["sponsor"]

        for component in components:
            log.info(f"  Looking up component '{component}' on openFDA...")
            c_brands, c_mfgs, _ = _fetch_brand_names_from_fda(component)
            combo_brands.update(c_brands)
            combo_mfgs.update(c_mfgs)
            record["synonyms"].append(component.lower())

            log.info(f"  Looking up component '{component}' on ClinicalTrials.gov...")
            c_ctgov = _fetch_info_from_ctgov(component)
            combo_indications.update(c_ctgov["indications"])
            combo_dev_codes.update(c_ctgov["dev_codes"])
            record["synonyms"].extend(d.lower() for d in c_ctgov["dev_codes"])
            if combo_sponsor == "Unknown" and c_ctgov["sponsor"] != "Unknown":
                combo_sponsor = c_ctgov["sponsor"]

        brands = list(combo_brands)
        manufacturers = list(combo_mfgs)
        ctgov_info["indications"] = list(combo_indications)[:15]
        ctgov_info["dev_codes"] = list(combo_dev_codes)[:10]
        ctgov_info["sponsor"] = combo_sponsor
        if combo_brands:
            log.ok(f"Combined component brand names: "
                   f"{', '.join(sorted(combo_brands)[:5])}")

    if brands:
        record["brand_names"] = sorted(set(b for b in brands
                                           if b.lower() != key))[:10]
        record["synonyms"].extend(b.lower() for b in record["brand_names"])
        log.ok(f"FDA brand names: {', '.join(record['brand_names'][:5])}")
    if manufacturers:
        # Use the most common manufacturer as sponsor
        record["sponsor"] = manufacturers[0]
        log.ok(f"FDA manufacturer: {record['sponsor']}")

    # ClinicalTrials.gov sponsor overrides FDA manufacturer if available
    if ctgov_info["sponsor"] != "Unknown":
        record["sponsor"] = ctgov_info["sponsor"]
        log.ok(f"CTgov sponsor: {record['sponsor']}")

    if ctgov_info["dev_codes"]:
        record["dev_codes"] = ctgov_info["dev_codes"]
        record["synonyms"].extend(d.lower() for d in record["dev_codes"])
        log.ok(f"Dev codes: {', '.join(record['dev_codes'][:5])}")

    if ctgov_info["indications"]:
        record["indications"] = ctgov_info["indications"]
        log.ok(f"Indications: {', '.join(record['indications'][:5])}")

    # Deduplicate synonyms
    record["synonyms"] = list(OrderedDict.fromkeys(record["synonyms"]))

    return record
