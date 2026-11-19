"""Step 2: innovator / FDA product lookup."""

from ..utils import http_get, log
from .molecule_normalization import FDA_API
SPONSOR_DOMAINS = {
    "novo nordisk": "novonordisk.com", "eli lilly": "lilly.com",
    "regeneron": "regeneron.com", "sanofi": "sanofi.com",
    "merck": "merck.com", "pfizer": "pfizer.com",
    "roche": "roche.com", "novartis": "novartis.com",
    "astrazeneca": "astrazeneca.com", "bristol-myers squibb": "bms.com",
    "johnson & johnson": "jnj.com", "abbvie": "abbvie.com",
    "amgen": "amgen.com", "gilead": "gilead.com", "gsk": "gsk.com",
    "bayer": "bayer.com", "takeda": "takeda.com",
    "boehringer ingelheim": "boehringer-ingelheim.com",
}

def _sponsor_domain(sponsor):
    for key, domain in SPONSOR_DOMAINS.items():
        if key in sponsor.lower(): return domain
    return sponsor.lower().replace(" ", "").replace("/", "") + ".com"


def innovator_deep_dive(molecule):
    generic = molecule["generic_name"]
    sponsor = molecule.get("sponsor", "Unknown")
    domain = _sponsor_domain(sponsor)
    sp_short = sponsor.split("/")[0].strip().replace(" ", "+")

    result = {
        "sponsor": sponsor,
        "fda_product_entries": [],
        "search_queries_suggested": [
            f"{generic} trial results site:{domain}",
            f"{generic} clinical study site:{domain}",
            f"{generic} phase 2 results site:{domain}",
            f"{generic} phase 3 results site:{domain}",
            f"{generic} investor presentation site:{domain}",
        ],
        "manual_search_links": {
            "sponsor_website":        f"https://www.google.com/search?q={generic.replace(' ','+')}+trial+results+site:{domain}",
            "press_releases":         f"https://www.google.com/search?q={generic.replace(' ','+')}+press+release+{sp_short}",
            "investor_presentations": f"https://www.google.com/search?q={generic.replace(' ','+')}+investor+presentation+{sp_short}",
            "pipeline_page":          f"https://www.google.com/search?q={sp_short}+pipeline+{generic.replace(' ','+')}",
            "clinical_trial_updates": f"https://www.google.com/search?q={generic.replace(' ','+')}+clinical+trial+update+{sp_short}",
        },
    }
    log.info(f"Checking FDA for '{generic}'...")
    fda = _search_fda(generic)
    result["fda_product_entries"] = fda
    if fda: log.ok(f"Found {len(fda)} FDA product entries")
    else:   log.info("No FDA entries (may be pre-approval)")
    return result


def _search_fda(name):
    for field in ["generic_name", "brand_name"]:
        r = http_get(FDA_API, params={"search": f'openfda.{field}:"{name}"', "limit": 10})
        if r and r.status_code == 200:
            try:
                entries = []
                for res in r.json().get("results", []):
                    openfda = res.get("openfda", {})
                    for prod in res.get("products", []):
                        entries.append({
                            "brand_name": ", ".join(openfda.get("brand_name", ["N/A"])),
                            "manufacturer": ", ".join(openfda.get("manufacturer_name", ["N/A"])),
                            "application_no": res.get("application_number", "N/A"),
                            "dosage_form": prod.get("dosage_form", "N/A"),
                            "route": prod.get("route", "N/A"),
                            "marketing_status": prod.get("marketing_status", "N/A"),
                        })
                if entries: return entries
            except Exception: pass
    return []
