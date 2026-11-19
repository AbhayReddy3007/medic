"""Step 3: real-world evidence (RWE) synthesis grounded with Google Search.

Each RWE source is queried with a Google-Search-grounded Gemini call, so the summaries
rest on retrieved web pages instead of the model's memory. The grounding URLs are kept
per source and returned alongside the synthesis, which feeds the E and C scores.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from ..llm_runtime import call_gemini, call_gemini_grounded
from ..utils import log, submit_with_ctx

_RWE_SOURCES = [
    ("mimic",     "MIMIC-IV (ICU EHR, Beth Israel / MIT)"),
    ("cms",       "CMS Medicare Public Use Files"),
    ("resdac",    "ResDAC - Medicare & Medicaid longitudinal claims"),
    ("cdc",       "CDC Open Data / NHANES population surveys"),
    ("allofus",   "NIH All of Us Research Program (700k+ participants)"),
    ("innovator", "Innovator / Manufacturer Website (press releases, trials, pipeline)"),
]

_RWE_PROMPT_TEMPLATE = """You are an expert clinical data analyst specialising in real-world evidence research.

Drug under investigation: "{drug}"
Data source: {source_name}

Use Google Search to find published studies, reports, datasets or official pages that
describe real-world, post-launch evidence for "{drug}" from the perspective of {source_name}.
Then write a concise summary (150-200 words) focused ONLY on quantitative post-launch evidence:
- HbA1c reduction observed in this data source (mean change or % reaching <7.0%)
- Weight loss (%TBWL or kg) documented in this data source
- Cardiovascular or renal outcomes if available
- Any safety signals unique to real-world use
- Population size / cohort characteristics

Rules:
- Report only figures that appear in the pages you found; never estimate or invent statistics.
- If the search finds limited or no data on "{drug}" for this source, say so clearly in 2-3 sentences.
- Do not cite a source that you did not retrieve.
Keep the summary to 150-200 words."""

_SYNTHESIS_PROMPT_TEMPLATE = """You are a senior pharmaceutical researcher.

Based on the real-world evidence summaries below from {n_sources} data sources,
write an integrated RWE synthesis (300-400 words) for "{drug}".

Structure your synthesis around:
1. CONSISTENCY OF FINDINGS - do multiple independent sources confirm efficacy?
   Estimate what percentage of data sources report positive/consistent findings.
2. QUANTITATIVE EVIDENCE STRENGTH - cite the strongest numeric outcomes seen
   (HbA1c change, %TBWL, MACE HR) across sources.
3. POST-LAUNCH REAL-WORLD PERFORMANCE - does RWE align with or diverge from
   clinical trial results?
4. EVIDENCE GAPS - what outcomes or populations lack real-world data?

--- SUMMARIES FROM {n_sources} SOURCES ---
{combined}

Write in professional clinical language. Use only information present in the summaries.
Be specific about which sources provide strong evidence vs. which are limited or absent."""


def _dedupe_urls(records: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for record in records:
        url = record.get("url")
        if url and url not in seen:
            seen[url] = record
    return list(seen.values())


def run_rwe_summarizer(drug_name: str) -> dict:
    """Build the grounded RWE synthesis for a molecule.

    Returns:
        {
          "synthesis": str,                       # integrated synthesis (or raw summaries on failure)
          "source_summaries": {source_id: str},
          "grounding_urls_by_source": {source_id: [url records]},
          "grounding_urls": [url records],        # de-duplicated across sources
        }
    """
    log.sub("Collecting grounded RWE summaries from data sources...")

    def _fetch_rwe_source(src_id: str, src_name: str) -> tuple[str, list[dict]]:
        prompt = _RWE_PROMPT_TEMPLATE.format(drug=drug_name, source_name=src_name)
        try:
            summary, urls = call_gemini_grounded(prompt, max_tokens=1024, temperature=0.2)
            summary = summary.strip()
            log.ok(f"  {src_name}: {len(summary)} chars, {len(urls)} source(s)")
            return f"=== {src_name} ===\n{summary}", urls
        except Exception as e:
            log.warn(f"  {src_name} grounded summary failed: {e}")
            return f"=== {src_name} ===\nData unavailable.", []

    results: dict[int, tuple[str, list[dict]]] = {}
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {
            submit_with_ctx(executor, _fetch_rwe_source, src_id, src_name): idx
            for idx, (src_id, src_name) in enumerate(_RWE_SOURCES)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    source_summaries = {}
    grounding_by_source = {}
    for idx, (src_id, _) in enumerate(_RWE_SOURCES):
        source_summaries[src_id], grounding_by_source[src_id] = results[idx]
    combined = "\n\n".join(source_summaries.values())
    all_urls = _dedupe_urls([u for urls in grounding_by_source.values() for u in urls])
    log.ok(f"Grounding sources collected: {len(all_urls)} unique URL(s)")

    synthesis_prompt = _SYNTHESIS_PROMPT_TEMPLATE.format(
        n_sources=len(_RWE_SOURCES), drug=drug_name, combined=combined,
    )
    try:
        synthesis = call_gemini(synthesis_prompt, max_tokens=2048, temperature=0.2).strip()
        log.ok(f"RWE synthesis generated ({len(synthesis)} chars)")
    except Exception as e:
        log.warn(f"RWE synthesis failed: {e}")
        synthesis = combined  # raw summaries as fallback

    return {
        "synthesis": synthesis,
        "source_summaries": source_summaries,
        "grounding_urls_by_source": grounding_by_source,
        "grounding_urls": all_urls,
    }
