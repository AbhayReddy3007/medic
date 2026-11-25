"""Two-Step Gemini Clinical Trial Extraction - Core Function

This module provides the core extraction logic used by the market potential agent.
"""
import json
import asyncio
import aiohttp
from json_repair import repair_json
import re
import time

from medical_potential.config import GEMINI_FLASH_PREVIEW_MODEL
from medical_potential.gemini_utils import gemini_api_call
from medical_potential.grounding_sources import gemini_search_call_async



def safe_json_parse(json_str: str, context: str = "") -> dict | list | None:
    """Parse JSON with automatic repair using json_repair library.

    Args:
        json_str: The JSON string to parse
        context: Optional context for error messages

    Returns:
        Parsed JSON object/array, or None if parsing fails
    """
    if not json_str or not json_str.strip():
        return None

    # Clean up common issues
    json_str = json_str.strip()

    # Remove markdown code blocks if present
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        parts = json_str.split("```")
        if len(parts) >= 2:
            json_str = parts[1].strip()

    # Find first { or [ if there's text before JSON
    json_start = -1
    for i, c in enumerate(json_str):
        if c in '{[':
            json_start = i
            break
    if json_start > 0:
        json_str = json_str[json_start:]

    # Try json_repair first (handles most issues automatically)
    try:
        repaired = repair_json(json_str, return_objects=True)
        if repaired is not None:
            return repaired
    except Exception as e:
        if context:
            print(f"  ⚠️ json_repair failed for {context}: {e}")

    # Fallback to standard json.loads
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    return None

# Configuration
BATCH_SIZE = 6
MAX_WORKERS = 4
RATE_LIMIT_DELAY = 2.0

# Models
SEARCH_MODEL = GEMINI_FLASH_PREVIEW_MODEL
EXTRACTION_MODEL = GEMINI_FLASH_PREVIEW_MODEL




async def gemini_call_with_search(prompt: str, model: str, response_mime_type: str = None) -> str:
    """Make a Gemini call with Google Search enabled.

    Args:
        prompt: The prompt to send to Gemini
        model: Model name (e.g., "gemini-2.5-flash")
        response_mime_type: Optional mime type (e.g., "application/json").
                           Note: Using response_mime_type with Google Search tools may cause errors.
    """

    return await gemini_search_call_async(
        "trial_tolerability_extraction",
        prompt=prompt,
        model=model,
        response_mime_type=response_mime_type,
    )


def _extract_first_json_object(text: str) -> str:
    """Extract the first complete JSON object from text.

    Handles cases where Gemini returns extra data after valid JSON.
    """
    # Find the first opening brace
    start = text.find('{')
    if start == -1:
        # Maybe it's an array
        start = text.find('[')
        if start == -1:
            return ""
        closing = ']'
        opening = '['
    else:
        closing = '}'
        opening = '{'

    # Track brace depth to find matching close
    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start:i+1]

    # No complete JSON found, return from start to end
    return text[start:]


def _repair_truncated_json(response: str) -> str:
    """Attempt to repair truncated/malformed JSON by removing incomplete trials and closing structures.

    Strategy:
    1. Find the last complete trial object (ends with })
    2. Remove everything after it
    3. Close the trials array and outer object
    """
    if not response:
        return response

    # Find the "trials" array start
    trials_start = response.find('"trials"')
    if trials_start == -1:
        return response

    # Find the [ that starts the trials array
    array_start = response.find('[', trials_start)
    if array_start == -1:
        return response

    # Find all complete trial objects (pattern: },\n or }\n])
    # We want to find the last } that completes a trial object before truncation

    # Strategy: Find all positions where we have a complete trial (} followed by comma or array close)
    last_complete_pos = -1
    depth = 0
    in_string = False
    escape_next = False

    for i in range(array_start + 1, len(response)):
        char = response[i]

        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue

        # Track brace depth
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            # If depth is 0, we've closed a complete trial object
            if depth == 0:
                # Check if next non-whitespace is comma or array close
                next_idx = i + 1
                while next_idx < len(response) and response[next_idx] in ' \n\r\t':
                    next_idx += 1
                if next_idx < len(response) and response[next_idx] in ',]':
                    last_complete_pos = i

    # If we found at least one complete trial, cut there
    if last_complete_pos != -1:
        # Cut right after the last complete }
        response = response[:last_complete_pos + 1]

        # Add closing for trials array and outer object
        response += '\n  ]\n}'
        return response

    # Fallback: try to salvage what we can
    # Find last complete } at any depth
    last_brace = response.rfind('}')
    if last_brace != -1 and last_brace > array_start:
        response = response[:last_brace + 1]
        response += '\n  ]\n}'
        return response

    # Last resort: just close everything
    open_braces = response.count('{') - response.count('}')
    open_brackets = response.count('[') - response.count(']')
    response += ']' * open_brackets
    response += '}' * open_braces

    return response





def _convert_bigquery_to_trials(bq_rows: list) -> list:
    """Convert supplied trial rows into the canonical trial format used downstream."""
    return [
        {
            "Molecule": row.get("molecule_name") or "",
            "Trial ID": row.get("trial_id", ""),
            "NCT_ID": row.get("trial_id", ""),
            "Dosage": row.get("dosage") or "N/A",
            "Size": row.get("trial_size") or row.get("Size") or "N/A",
            "Phase": str(row.get("phase") or "N/A"),
            "Primary Region": row.get("trial_location") or "N/A", 
            "Secondary Countries": row.get("secondary_locations") or "N/A",
        }
        for row in bq_rows
        if row.get("trial_id")
    ]


def _normalize_trial_id(trial_id: str) -> str:
    """Normalize trial identifiers for matching between Step 1 and Step 2 outputs."""
    if not trial_id:
        return ""
    normalized = str(trial_id).strip()
    if not normalized:
        return ""
    normalized = normalized.split("(")[0].strip()
    normalized = normalized.split(" ")[0].strip()
    return normalized.upper()


async def get_trial_details(molecule: str, trial_batch: list,
                            extra_fields_prompt: str = None,
                            extra_fields_json: str = None) -> dict:
    """Step 2: Get detailed efficacy data for a batch of trials (async).

    Args:
        molecule: Molecule name
        trial_batch: List of trial dicts from Step 1
        extra_fields_prompt: Optional additional prompt for extra fields (e.g., tolerability)
        extra_fields_json: Optional additional JSON fields to include in schema
    """

    # Filter out trials without NCT_ID to avoid None in join
    nct_ids = [t.get("NCT_ID") for t in trial_batch if t.get("NCT_ID")]
    nct_list = ", ".join(nct_ids)

    print(f"\nQuerying details for: {nct_list[:80]}...")
    prompt = f"""You are a clinical data extraction engine with access to MULTIPLE international trial registries.

IMPORTANT VALIDATION: Before extracting data, verify each trial is a {molecule} MONOTHERAPY trial.
- ❌ SKIP trials testing combination drugs (e.g., CagriSema = Cagrilintide + Semaglutide)
- ❌ SKIP trials where {molecule} is only background therapy
- ✅ ONLY extract data for trials where {molecule} alone is the primary intervention

Task: Extract structured interventional clinical trial data for these {len(trial_batch)} {molecule} MONOTHERAPY trials:

{chr(10).join([f"- {t.get('NCT_ID', 'N/A')} ({t.get('Program_Name', 'N/A')})" for t in trial_batch])}

DATA SOURCES (search ALL for complete data):
1. ClinicalTrials.gov - https://clinicaltrials.gov/study/[TRIAL_ID]
2. PubMed/published papers linked to the trial
3. ChiCTR (China) - https://www.chictr.org.cn/ (for Chinese trials)
4. EU Clinical Trials Register - https://www.clinicaltrialsregister.eu/ (for European trials)
5. CTRI (India) - https://ctri.nic.in/ (for Indian trials)
6. JRCT (Japan) - https://rctportal.niph.go.jp/en/ (for Japanese trials)
7. Other international registries as available

MULTI-SOURCE STRATEGY:
- If a field is missing from ClinicalTrials.gov, check other registries
- Cross-reference data across sources for accuracy
- Prioritize: Published papers > Primary registry > Secondary registries

For EACH trial, extract:
- Molecule: {molecule}
- Trial ID: Use the provided trial ID
- Size: Total enrollment number

{extra_fields_prompt if extra_fields_prompt else ""}
IMPORTANT:
- One entry per trial
- Include ALL {len(trial_batch)} trials even if data is incomplete
- Use "N/A" for missing fields

Return JSON:

{{
    "trials": [
        {{
            "Molecule": "{molecule}",
            "Trial ID": "",
                        "Size": 0{',' if extra_fields_json else ''}
            {extra_fields_json if extra_fields_json else ""}
        }}
    ]
}}
"""

    try:
        response = await gemini_call_with_search(
            prompt,
            model=EXTRACTION_MODEL,
            response_mime_type="application/json",
        )
    except Exception as error:
        print(f"  ❌ Gemini call failed for batch {nct_list[:80]}...: {error}")
        return {"trials": []}

    if not response or not response.strip():
        print(f"  ❌ Empty response received for batch")
        return {"trials": []}

    # FIRST ATTEMPT: Use json_repair library (handles most issues automatically)
    data = safe_json_parse(response, context="batch response")

    if data is not None:
        # Ensure it returns the expected structure
        if isinstance(data, dict) and "trials" in data:
            print(f"  ✓ json_repair successful - {len(data.get('trials', []))} trials")
            return data
        elif isinstance(data, list):
            print(f"  ✓ json_repair successful - {len(data)} trials (list)")
            return {"trials": data}
        elif isinstance(data, dict):
            # Check if it's a single trial object (has Trial ID or Molecule)
            if any(k in data for k in ["Trial ID", "Molecule", "trial_id", "molecule"]):
                print(f"  ℹ️  Single trial object returned, wrapping in list")
                return {"trials": [data]}
            # Check if keys look like trial data fields
            if any(k in data for k in ["Phase", "Size", "Dosage", "Status"]):
                print(f"  ℹ️  Single trial-like object returned, wrapping in list")
                return {"trials": [data]}
            # Log what we got for debugging
            print(f"  ⚠️  Unexpected dict structure. Keys: {list(data.keys())[:5]}")
            # Fall through to manual methods
        else:
            print(f"  ⚠️  Unexpected response type: {type(data)}")
            # Fall through to manual methods

    # SECOND ATTEMPT: Manual cleanup and parsing
    print(f"  🔧 json_repair returned unexpected structure, trying manual methods...")

    response = response.strip()
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].split("```")[0].strip()

    # Find the first { that starts valid JSON
    json_start = response.find('{')
    if json_start > 0:
        response = response[json_start:]

    # Sometimes Gemini adds google_search arrays - try to extract just the trials array
    if '"google_search"' in response:
        trials_match = response.find('"trials"')
        if trials_match != -1:
            bracket_start = response.find('[', trials_match)
            if bracket_start != -1:
                depth = 0
                for i, char in enumerate(response[bracket_start:], start=bracket_start):
                    if char == '[':
                        depth += 1
                    elif char == ']':
                        depth -= 1
                        if depth == 0:
                            trials_json = response[bracket_start:i+1]
                            parsed = safe_json_parse(trials_json, "trials array")
                            if parsed is not None:
                                if isinstance(parsed, list):
                                    return {"trials": parsed}

    # Extract just the first JSON object
    json_str = _extract_first_json_object(response)
    if json_str:
        parsed = safe_json_parse(json_str, "first JSON object")
        if parsed is not None:
            if isinstance(parsed, dict) and "trials" in parsed:
                return parsed
            elif isinstance(parsed, list):
                return {"trials": parsed}

    # THIRD ATTEMPT: Last resort field-by-field extraction
    print(f"  🔧 Attempting field-by-field extraction...")
    import re
    trials = []

    # Method 1: Try to find complete trial objects
    trial_pattern = r'\{\s*"Molecule"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.finditer(trial_pattern, response, re.DOTALL)

    for match in matches:
        try:
            trial_obj = safe_json_parse(match.group(0), "trial object")
            if trial_obj and (trial_obj.get("Trial ID") or trial_obj.get("NCT_ID")):
                trials.append(trial_obj)
        except:
            continue

    if trials:
        print(f"  ✓ Last-resort extraction (complete objects) - recovered {len(trials)} trials")
        return {"trials": trials}

    # Method 2: Extract individual fields from truncated JSON
    # This handles cases where even a single trial is truncated mid-field
    print(f"  🔧 Attempting field-by-field extraction...")
    field_patterns = {
        # Base clinical efficacy fields
        "Molecule": r'"Molecule"\s*:\s*"([^"]*)"',
        "Trial ID": r'"Trial ID"\s*:\s*"([^"]*)"',
        "Trial Title": r'"Trial Title"\s*:\s*"([^"]*)"',
        "Phase": r'"Phase"\s*:\s*"?([^",}]*)"?',
        "Size": r'"Size"\s*:\s*(\d+)',
        "Dosage": r'"Dosage"\s*:\s*"([^"]*)"',
        "Status": r'"Status"\s*:\s*"([^"]*)"',
        "Primary Region": r'"Primary Region"\s*:\s*"([^"]*)"',
        "Secondary Countries": r'"Secondary Countries"\s*:\s*"([^"]*)"',
        "HbA1c Change (%)": r'"HbA1c Change \(%\)"\s*:\s*"?([^",}]*)"?',
        "HbA1c Duration": r'"HbA1c Duration"\s*:\s*"([^"]*)"',
        "Weight Loss (%)": r'"Weight Loss \(%\)"\s*:\s*"?([^",}]*)"?',
        "Weight Duration": r'"Weight Duration"\s*:\s*"([^"]*)"',
        "MASH Outcome (%)": r'"MASH Outcome \(%\)"\s*:\s*"?([^",}]*)"?',
        "MASH Duration": r'"MASH Duration"\s*:\s*"([^"]*)"',
        "ALT Reduction (%)": r'"ALT Reduction \(%\)"\s*:\s*"?([^",}]*)"?',
        "ALT Duration": r'"ALT Duration"\s*:\s*"([^"]*)"',
        "Start Date": r'"Start Date"\s*:\s*"([^"]*)"',
        "Completion Date": r'"Completion Date"\s*:\s*"([^"]*)"',
        "Company": r'"Company"\s*:\s*"([^"]*)"',
        "Source URL": r'"Source URL"\s*:\s*"([^"]*)"',
        # Tolerability-specific fields
        "Drug Arm Size (N)": r'"Drug Arm Size \(N\)"\s*:\s*(\d+)',
        "Placebo Arm Size (N)": r'"Placebo Arm Size \(N\)"\s*:\s*(\d+)',
        "Discontinued AE Drug (N)": r'"Discontinued AE Drug \(N\)"\s*:\s*(\d+)',
        "Discontinued AE Placebo (N)": r'"Discontinued AE Placebo \(N\)"\s*:\s*(\d+)',
        "Comparator Name": r'"Comparator Name"\s*:\s*"([^"]*)"',
        "Comparator Arm Size (N)": r'"Comparator Arm Size \(N\)"\s*:\s*(\d+)',
        "Discontinued AE Comparator (N)": r'"Discontinued AE Comparator \(N\)"\s*:\s*(\d+)',
        "Common AEs": r'"Common AEs"\s*:\s*"([^"]*)"',
        "AE Severity": r'"AE Severity"\s*:\s*"([^"]*)"',
        "AE Persistence": r'"AE Persistence"\s*:\s*"([^"]*)"',
        "Management Required": r'"Management Required"\s*:\s*"([^"]*)"',
    }

    # Split JSON by trial boundaries (look for "Molecule" as trial start marker)
    trial_chunks = re.split(r'(?=\{\s*"Molecule")', response)

    for chunk in trial_chunks:
        if '"Molecule"' not in chunk:
            continue

        trial_obj = {}
        for field, pattern in field_patterns.items():
            match = re.search(pattern, chunk)
            if match:
                val = match.group(1).strip()
                if val and val not in ["N/A", "null", ""]:
                    trial_obj[field] = val

        # Only add if we have at least Trial ID or Molecule
        if trial_obj.get("Trial ID") or trial_obj.get("Molecule"):
            trials.append(trial_obj)

    if trials:
        print(f"  ✓ Last-resort extraction (field-by-field) - recovered {len(trials)} partial trials")
        return {"trials": trials}

    print(f"  ❌ All recovery attempts failed")
    return {"trials": []}


async def fetch_ctgov_dates(nct_ids: list) -> dict:
    """Fetch Study Start and Primary Completion dates from ClinicalTrials.gov API v2.

    Uses the authoritative CT.gov API instead of relying on Gemini for date accuracy.

    Returns:
        dict: {nct_id: {"Start Date": "YYYY-MM-DD", "Completion Date": "YYYY-MM-DD"}}
    """
    if not nct_ids:
        return {}

    async def fetch_one(session: aiohttp.ClientSession, nct_id: str):
        url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
        params = {"fields": "NCTId,StartDate,PrimaryCompletionDate"}
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    status_mod = data.get("protocolSection", {}).get("statusModule", {})
                    start = status_mod.get("startDateStruct", {}).get("date") or "N/A"
                    completion = status_mod.get("primaryCompletionDateStruct", {}).get("date") or "N/A"
                    return nct_id, {"Start Date": start, "Completion Date": completion}
                else:
                    print(f"  ⚠️  CT.gov API {resp.status} for {nct_id}")
        except Exception as e:
            print(f"  ⚠️  CT.gov API error for {nct_id}: {e}")
        return nct_id, None

    print(f"\n🌐 Fetching dates from CT.gov API for {len(nct_ids)} NCT trials...")
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, nct_id) for nct_id in nct_ids]
        results = await asyncio.gather(*tasks)

    dates_map = {nct_id: dates for nct_id, dates in results if dates}
    print(f"  ✓ Retrieved dates for {len(dates_map)}/{len(nct_ids)} trials")
    return dates_map


async def extract_clinical_trial_data(molecule_name: str, max_trials: int = None,
                                       base_trials: list[dict] | None = None,
                                       extra_fields_prompt: str = None,
                                       extra_fields_json: str = None) -> dict:
    """Main extraction function - Two-step Gemini extraction with multi-source search.

    Args:
        molecule_name: Name of the molecule to search for
        max_trials: Optional limit on number of trials to process (None = all)
        base_trials: Optional pre-fetched trial rows to use instead of querying
            BigQuery inside this function.
        extra_fields_prompt: Optional additional prompt text for extracting extra fields
                            (e.g., tolerability fields for Dimension 6)
        extra_fields_json: Optional additional JSON schema fields to include in response

    Returns:
        dict with keys:
            - trials: list of trial data dictionaries
            - total_trials: total number of trials found
            - completeness: percentage of filled fields
    """
    import time
    start_time = time.time()

    print(f"\n{'='*80}")
    print("TWO-STEP GEMINI EXTRACTION")
    print("(BigQuery Step 1)")
    print(f"{'='*80}")
    print(f"Molecule: {molecule_name}")
    print(f"Models:")
    print(f"  Search (Step 1):     Supplied trial rows")
    print(f"  Extraction (Step 2): {EXTRACTION_MODEL}")
    print(f"Strategy:")
    print(f"  Step 1: Use supplied base trials")
    print(f"  Step 2: Query detailed data in batches of {BATCH_SIZE} (accurate extraction)")
    print(f"  Max concurrent workers: {MAX_WORKERS}")

    # Step 1: Use supplied base trial metadata rows.
    step1_start = time.time()
    trial_list = _convert_bigquery_to_trials(base_trials or [])
    step1_time = time.time() - step1_start
    print(f"\n⏱️  Step 1 time: {step1_time:.1f}s")

    if not trial_list:
        print("\n❌ No base trials supplied")
        return {"trials": [], "total_trials": 0, "completeness": 0}

    print(f"\n{'='*80}")
    print(f"STEP 1 SUMMARY - Base Trial Rows")
    print(f"{'='*80}")
    print(f"  Total step-1 rows found: {len(trial_list)}")

    # Deduplicate trial IDs for step-2 extraction (preserve order)
    unique_trial_ids = []
    seen_ids = set()
    for trial in trial_list:
        normalized_id = _normalize_trial_id(trial.get("NCT_ID") or trial.get("Trial ID"))
        if normalized_id and normalized_id not in seen_ids:
            seen_ids.add(normalized_id)
            unique_trial_ids.append(normalized_id)

    print(f"  Unique trial IDs for step 2: {len(unique_trial_ids)}")
    print(f"\n  Trial IDs (showing first 20):")
    for nct in unique_trial_ids[:20]:
        print(f"    {nct}")

    if len(unique_trial_ids) > 20:
        print(f"    ... and {len(unique_trial_ids) - 20} more")

    unique_trial_list = [{"NCT_ID": trial_id} for trial_id in unique_trial_ids]

    # Optional: Filter to top N trials for faster processing
    if max_trials and len(unique_trial_list) > max_trials:
        print(f"\n⚡ Limiting to top {max_trials} unique trial IDs (out of {len(unique_trial_list)}) for speed")
        selected_trial_ids = {_normalize_trial_id(t.get("NCT_ID")) for t in unique_trial_list[:max_trials]}
        unique_trial_list = unique_trial_list[:max_trials]
        trial_list = [
            trial for trial in trial_list
            if _normalize_trial_id(trial.get("NCT_ID") or trial.get("Trial ID")) in selected_trial_ids
        ]

    # Step 2: Get details in batches (ASYNC)
    print(f"\n{'='*80}")
    print(f"STEP 2: Getting detailed data for {len(trial_list)} trials")
    print(f"Using model: {EXTRACTION_MODEL} (optimized for accurate extraction)")
    print(f"Processing ASYNCHRONOUSLY with up to {MAX_WORKERS} concurrent batches")
    print(f"{'='*80}\n")

    step2_start = time.time()
    all_trials = []
    # Build batches from unique trial ids (one entry per NCT)
    num_batches = (len(unique_trial_list) + BATCH_SIZE - 1) // BATCH_SIZE

    # Create all batches
    batches = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(unique_trial_list))
        batch = unique_trial_list[start_idx:end_idx]
        batches.append((batch_idx, batch))

    async def process_batch_async(batch_idx: int, batch: list):
        """Process a single batch asynchronously with rate-limit staggering."""
        # Stagger starts to respect rate limits
        stagger_delay = (batch_idx % MAX_WORKERS) * (RATE_LIMIT_DELAY / MAX_WORKERS)
        if stagger_delay > 0:
            await asyncio.sleep(stagger_delay)

        batch_start = time.time()
        print(f"\nBatch {batch_idx + 1}/{num_batches} ({len(batch)} trials) - starting")

        try:
            batch_data = await get_trial_details(molecule_name, batch, extra_fields_prompt, extra_fields_json)
        except Exception as error:
            print(f"  ❌ Batch {batch_idx + 1}/{num_batches} failed: {error}")
            return []

        batch_trials = batch_data.get("trials", [])

        batch_time = time.time() - batch_start
        print(f"  ✓ Batch {batch_idx + 1}/{num_batches} - completed ({len(batch_trials)} trials) in {batch_time:.1f}s")
        return batch_trials

    # Process batches with controlled concurrency using semaphore
    semaphore = asyncio.Semaphore(MAX_WORKERS)

    async def process_with_semaphore(batch_idx: int, batch: list):
        async with semaphore:
            return await process_batch_async(batch_idx, batch)

    # Run all batches concurrently with semaphore limit
    tasks = [process_with_semaphore(batch_idx, batch) for batch_idx, batch in batches]
    results = await asyncio.gather(*tasks)

    # Flatten results (filter out non-dict entries from malformed JSON)
    for batch_trials in results:
        all_trials.extend([t for t in batch_trials if isinstance(t, dict)])

    step2_time = time.time() - step2_start
    print(f"\n⏱️  Step 2 time: {step2_time:.1f}s ({num_batches} batches, {step2_time/num_batches:.1f}s avg/batch)")

    # Step 3: Merge step-1 metadata with step-2 extracted details
    step2_by_trial_id = {}
    for trial in all_trials:
        if not isinstance(trial, dict):
            continue
        normalized_id = _normalize_trial_id(trial.get("Trial ID") or trial.get("NCT_ID"))
        if normalized_id and normalized_id not in step2_by_trial_id:
            step2_by_trial_id[normalized_id] = trial

    merged_trials = []
    for base_trial in trial_list:
        normalized_id = _normalize_trial_id(base_trial.get("NCT_ID") or base_trial.get("Trial ID"))
        merged_trial = dict(base_trial)
        step2_trial = step2_by_trial_id.get(normalized_id, {})

        for key, value in step2_trial.items():
            # Keep Step-1 metadata fields authoritative.
            if key in {
                "Molecule",
                "Trial ID",
                "NCT_ID",
                "Dosage",
                "Size",
                "Phase",
                "Primary Region",
                "Secondary Countries",
            }:
                continue
            merged_trial[key] = value

        merged_trials.append(merged_trial)

    print(f"  ✓ Merged {len(merged_trials)} trial rows using {len(step2_by_trial_id)} step-2 detail records")

    # Step 3.5: Override Start Date / Completion Date from CT.gov API v2
    # Gemini often guesses dates incorrectly — the API is the authoritative source
    nct_ids = [
        t.get("Trial ID", "").split("(")[0].strip().split(" ")[0].strip()
        for t in merged_trials
        if isinstance(t, dict) and t.get("Trial ID", "").upper().startswith("NCT")
    ]
    nct_ids = [nid for nid in nct_ids if nid]  # remove empty strings

    if nct_ids:
        dates_map = await fetch_ctgov_dates(nct_ids)
        overrides = 0
        for trial in merged_trials:
            if not isinstance(trial, dict):
                continue
            raw_id = trial.get("Trial ID", "")
            nct_id = raw_id.split("(")[0].strip().split(" ")[0].strip()
            if nct_id in dates_map:
                trial["Start Date"] = dates_map[nct_id]["Start Date"]
                trial["Completion Date"] = dates_map[nct_id]["Completion Date"]
                overrides += 1
        print(f"  ✓ Overrode dates for {overrides} trials from CT.gov API")

    # Calculate completeness - count non-N/A values across all fields
    # Exclude metadata fields that are always present
    metadata_fields = ["Molecule", "Trial ID", "Trial Title", "Source URL"]

    total_fields = 0
    filled_fields = 0

    for trial in merged_trials:
        if not isinstance(trial, dict):
            continue
        for field, value in trial.items():
            # Skip metadata fields
            if field in metadata_fields:
                continue

            total_fields += 1

            # Count as filled if not N/A, not empty, and not 0
            if value and str(value).strip() not in ["N/A", "n/a", "", "0"]:
                filled_fields += 1

    completeness = (filled_fields / total_fields * 100) if total_fields > 0 else 0

    print(f"\n📊 Data Completeness: {completeness:.1f}%")
    print(f"   Filled: {filled_fields}/{total_fields} fields (excluding Molecule, Trial ID, Source URL)")
    print(f"   N/A or empty: {total_fields - filled_fields} fields")

    # Final timing summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"✓ COMPLETED IN {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"{'='*80}")
    print(f"Timing Breakdown:")
    print(f"  Step 1 (BigQuery base metadata):   {step1_time:.1f}s")
    print(f"  Step 2 (batch extraction):         {step2_time:.1f}s")
    print(f"  Total:                             {total_time:.1f}s")
    print(f"{'='*80}\n")

    return {
        "trials": merged_trials,
        "total_trials": len(merged_trials),
        "completeness": completeness,
        "extraction_time": total_time
    }
