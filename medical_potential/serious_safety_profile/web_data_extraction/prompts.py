"""Serious Safety Profile – Dimension 5: Prompts."""

# =============================================================================
# SERIOUS SAFETY PROFILE EXTRACTION
# =============================================================================

SERIOUS_SAFETY_EXTRACTION_PROMPT = """
- Drug Arm Size (N): Number of patients in drug arm
- Control Arm Size (N): Number of patients in placebo/standard treatment arm
- SAE Drug (N): Count of patients with >=1 serious adverse event in drug arm
- SAE Control (N): Count of patients with >=1 serious adverse event in control arm
- SAE Rate Drug (%): Serious adverse event rate (%) in drug arm
- SAE Rate Control (%): Serious adverse event rate (%) in control arm
- Serious Events Reported: return ONLY a valid list object with the list of serious adverse events occured in the trial.
Required format (strict): ["<Condition 1>", "<Condition 2>",...]  
- Deaths Reported: [Yes|No|Unknown]
- Drug-Related Deaths: [Yes|No|Unclear]. Select "Yes" only if the report or journal identifies the drug as the primary or contributing cause of death.
- Drug-Related Deaths Rationale: "<Provide 1-2 sentence rationale with evidence on how the deaths are related to the drug (only when the drug related detahs is "Yes")>",

Extraction Rules:
- Use double quotes for all keys and string values.
- If no serious events are reported, return [] for each serious-events list field.
- Prefer extracting raw counts + arm sizes where available.
- If SAE rates are explicitly reported, extract rates too.
- Serious adverse event means death, life-threatening event, hospitalization, or permanent damage.
"""

SERIOUS_SAFETY_JSON_FIELDS = """
      "Drug Arm Size (N)": 0,
      "Control Arm Size (N)": 0,
      "SAE Drug (N)": 0,
      "SAE Control (N)": 0,
      "SAE Rate Drug (%)": 0,
      "SAE Rate Control (%)": 0,
      "Serious Events Reported": [],
      "Deaths Reported": "",
      "Drug-Related Deaths": "",
      "Drug-Related Deaths Rationale": ""
      """

# =============================================================================
# REGULATORY IMPACT PROMPT TEMPLATE (Step 7)
# Placeholders: {molecule_name}, {trial_summary}
# =============================================================================

REGULATORY_IMPACT_PROMPT_TEMPLATE = """You are a regulatory affairs expert. Your task is to determine the likely regulatory implications of the safety profile of {molecule_name} and provide a clear rationale based on established clinical trial data and regulatory status.

CLINICAL TRIAL SAFETY DATA FOR {molecule_name}:
{trial_summary}

Using the trial data above alongside any publicly available regulatory information for {molecule_name} (approved labels, boxed warnings, REMS programs, safety communications, post-marketing updates), perform the following assessment:

Assess which of the following category applies for the given drug:
Strictly give one from the below list: If no action is expected classify as "No action"

Informational / precautionary (name: informational_precautionary)
    Definition: Theoretical or class-level concern with uncertain human relevance.

Monitoring / REMS (name: monitoring_rems)
    Definition: Requires risk management but widely usable.

Established serious human risk (name: established_serious_human_risk)
    Definition: Confirmed severe clinical risk.
    *Note: Classify as this type ONLY if: Risk is clinically established in humans, Risk is drug-attributable, Risk materially limits usage, and Risk is not merely class-level or precautionary.*

Market-threatening (name: market_threatening)
    Definition: Withdrawal/rejection.

Return ONLY a valid JSON object (no markdown, no extra text) with this exact schema:
{{
  "regulatory_consequence": "<Strictly choose exactly one: no_action| informational_precautionary| monitoring_or_rems| established_serious_human_risk| market_threatening>",
  "rationale": "<detailed 4-6 sentence explanation linking the specific trial safety findings to the regulatory consequence and its implications for approval and use>",
  "key_safety_drivers": ["<safety finding 1>", "<safety finding 2>"],
  "grounding_summary": "<1-2 sentence summary of what is known from public regulatory sources about {molecule_name}>"
}}
"""


# =============================================================================
# POST-MARKETING SAFETY PROMPTS
# =============================================================================

APPROVAL_MARKET_STATUS_PROMPT_TEMPLATE = """Determine whether {molecule_name} is currently approved and marketed for any indication in major regulated markets (FDA, EMA, MHRA, PMDA, Health Canada, TGA).

Use current public regulatory data and return ONLY valid JSON with this schema:
{{
  "is_approved": <true|false>,
  "is_marketed": <true|false>,
  "status_summary": "<1-2 sentence summary>",
  "major_markets": ["<market/status snippet>"]
}}

Rules:
- is_approved = true only if there is regulatory approval evidence.
- is_marketed = true only if there is evidence the product has been commercially launched or is actively marketed.
- If uncertain, use false and explain uncertainty in status_summary.
"""


POST_MARKETING_SAFETY_PROMPT_TEMPLATE = """You are a pharmacovigilance and regulatory safety expert. Assess post-marketing serious safety signals for {molecule_name}.

Review real-world post-approval evidence using:
- Post-marketing safety databases
- Safety label updates
- Regulatory warnings issued after approval
- Drug withdrawal notices

Check specifically for:
- New serious risks not seen in trials
- Stronger warning added later
- REMS added later
- Boxed warning added later
- Major label restriction
- Withdrawal from market

Return ONLY valid JSON with this exact schema:
{{
  "new_serious_risks": <true|false>,
  "new_serious_risks_rationale": "<1-2 sentence rationale with evidence for why this is true or false>",
  "stronger_warning_added": <true|false>,
  "stronger_warning_added_rationale": "<1-2 sentence rationale with evidence for why this is true or false>",
  "rems_added": <true|false>,
  "rems_added_rationale": "<1-2 sentence rationale with evidence for why this is true or false>",
  "boxed_warning_added": <true|false>,
  "boxed_warning_added_rationale": "<1-2 sentence rationale with evidence for why this is true or false>",
  "major_label_restriction": <true|false>,
  "major_label_restriction_rationale": "<1-2 sentence rationale with evidence for why this is true or false>",
  "withdrawal_from_market": <true|false>,
  "withdrawal_from_market_rationale": "<1-2 sentence rationale with evidence for why this is true or false>",
  "risk_level": "<none|low|moderate|high|critical>",
  "key_findings": ["<finding 1>", "<finding 2>"],
  "grounding_summary": "<2-4 sentence summary of post-marketing evidence>",
  "sources_summary": ["<source or regulatory action summary>"]
}}

Rules:
- Every *_rationale field is required, even when the paired flag is false.
- Rationales must cite concrete evidence (label updates, safety communications, REMS changes, withdrawals, or explicit absence of such updates).
- If evidence is limited or uncertain, set the flag to false and explain uncertainty in the paired rationale.
"""


# =============================================================================
# SAE EXPECTEDNESS + CATEGORY CLASSIFICATION
# =============================================================================

SAE_EXPECTEDNESS_AND_CATEGORY_PROMPT_TEMPLATE = """You are a drug safety pharmacist and regulatory affairs expert. For the molecule class {molecule_class}, classify each serious adverse event (SAE) in the input list based on comprehensive public medical and regulatory evidence.

SAE EVENTS:
{sae_events_json}

Treat each JSON object as a separate SAE. Preserve each input `sae_name` exactly in the corresponding output object and return one output object for every input object.

## TASK 1: CLASSIFY EXPECTEDNESS IN CLASS {molecule_class}

UNRELATED (Lowest Concern):
Classify as UNRELATED if the event is clearly due to external factors, accidents, or pre-existing physical trauma not influenced by the drug's mechanism.
- Examples: Accidental drowning, hip fracture from a fall (unrelated to dizziness/syncope), motor vehicle accidents, or injury from natural disasters.
- Interpretation: Not a drug-safety signal.

EXPECTED RISK (Lower Concern):
Classify as EXPECTED if ANY of the following apply:
- Reported in other drugs of the same class (search class literature, product labels, safety databases)
- Included in approved drug labels for similar drugs (mechanisms, target biology)
- Mechanistically plausible: explained by known target biology, off-targets, or formulation
- Seen in earlier trials or preclinical data for this or similar molecules
- Documented in regulatory guidance or class warnings

Interpretation: Known, predictable, generally more manageable risks.

UNEXPECTED RISK (Higher Concern):
Classify as UNEXPECTED if ANY of the following apply:
- Not reported in similar drugs or class literature (novel to this drug)
- No clear biological or mechanistic explanation
- Appears new in this drug (especially if emerges late-phase or in subgroup)
- More severe or frequent than typical class risk patterns
- Highlighted by regulators (FDA, EMA) as a new safety signal or concern
- Unique phenotype or timeline compared to class experience

Interpretation: Novel, unpredictable, potentially higher risk.

EXPECTED BUT RARE:
Classify as EXPECTED_BUT_RARE if the event is known in the class but occurs at lower-than-typical frequency or only in specific subgroups.

OTHERS:
Classify as OTHERS if the event does not clearly fit the above definitions (e.g., insufficient evidence, ambiguous class association, or confounded by comorbidities).

## TASK 2: CLASSIFY SEVERITY IMPACT CATEGORY

Use the following classification framework to categorize the medical events I provide:

SEVERITY DECISION PRIORITY (MUST FOLLOW IN ORDER):
1) If event is clearly external trauma/accident unrelated to drug biology -> Unrelated.
2) If event explicitly indicates death/fatal outcome/permanent disability/irreversible organ failure -> Catastrophic.
3) If event indicates malignancy/cancer/neoplasm/carcinoma (without explicit death/fatal outcome) -> Severe / Life-Threatening.
4) If event indicates chronic organ disease progression (without explicit irreversible end-stage failure) -> Organ-Specific / Clinically Important.
5) Use Others only when evidence is insufficient or mixed.

Keyword normalization guidance:
- Treat the following as cancerous conditions: neoplasm malignant, metastatic neoplasm, malignant neoplasm, colon cancer, colorectal cancer, gastric cancer, pancreatic carcinoma, anaplastic thyroid cancer, malignant lung neoplasm, and similar cancer/malignancy terminology.
- Cancer/malignancy terms alone are NOT enough for Catastrophic unless the SAE text explicitly contains death/fatal/irreversible permanent outcome.
- Terms such as "cardiovascular death", "fatal", "death", "treatment-related death" are Catastrophic.

Severity categories (choose exactly one):

Unrelated:
    Definition
        Classify as UNRELATED if the event is clearly due to external factors, accidents, or pre-existing physical trauma not influenced by the drug's mechanism.
    - Examples: 
        Accidental drowning, hip fracture from a fall (unrelated to dizziness/syncope), motor vehicle accidents, or injury from natural disasters.
    - Interpretation: 
        Not a drug-safety signal.

Manageable / Non-Specific
    Definition
        Serious events that are:
        generally reversible,
        medically manageable,
        not typically life-threatening,
        and unlikely to cause permanent harm.
        These events may require hospitalization or intervention but usually resolve without major long-term consequences.
    Examples
        Reversible hospitalization, Transient complications with full recovery
    Interpretation
        Usually lower regulatory concern
        Often acceptable depending on disease context
        Does not strongly impact scoring unless highly repetitive

Organ-Specific / Clinically Important
    Definition
        Serious events involving a specific organ system or medically important toxicity that may cause meaningful morbidity but is not immediately life-threatening in most cases.
        These events are more concerning because they may indicate:
        target-organ toxicity,
        mechanistic safety risk,
        or long-term treatment limitations.
    Examples: Pancreatitis, Liver injury / elevated liver enzymes, Acute kidney injury, Serious infection, Gallbladder disease, Thyroid disorders, Clinically significant bleeding
    Interpretation
        Moderate regulatory concern
        Important if repeated across studies
        Often becomes clinically meaningful when recurrent or unexpected

Severe / Life-Threatening
    Definition
        Serious events that are:
        life-threatening,
        medically critical,
        associated with major acute risk to the patient,
        permanent disability,
        irreversible organ failure,
        or potentially market-threatening safety risk.
        These events usually require urgent medical intervention and carry significant regulatory importance even at low frequency.
        Note: All cancerous conditions, including malignancies, must be classified here.
        Permanent disability, irreversible organ failure, and market-threatening safety risks are also classified here unless death is explicitly stated (which is Catastrophic).
    Examples: Myocardial infarction (heart attack), Malignant tumors / Cancerous conditions, Stroke, Respiratory failure, Severe anaphylaxis, Sepsis with organ dysfunction, ICU admission, Serious arrhythmia, Permanent neurological damage, Irreversible liver failure, End-stage organ damage
    Interpretation
        High regulatory concern
        Even small imbalances vs control may matter
        Repeated occurrence strongly impacts score

Catastrophic / Fatal
    Definition
        Events associated with:
        death (fatal outcome) only.
        These events represent the highest level of safety concern.
    Strict trigger: Assign this category ONLY when the SAE explicitly indicates death or fatality.
    Do not infer Catastrophic from terms like malignancy/neoplasm/carcinoma/cancer, permanent disability, irreversible organ failure, or market-threatening risk unless death is explicitly stated.
    Examples: Death, Cardiovascular death, Treatment-related death, Fatal malignancy clearly associated with treatment
    Interpretation:
        - Critical regulatory concern
        - Requires strong evidence before attribution to drug
        - Should NOT be assigned based on broad or non-specific SAE labels alone

Others
  Does not clearly fit the above categories (e.g., insufficient data, mixed presentation, or event spans multiple categories).

Note: Strictly classify any cancerous condition, including malignant neoplasms and carcinomas, into "Severe / Life-Threatening" unless the event explicitly states death/fatal outcome/permanent irreversible harm.

Return ONLY a valid JSON list with this exact schema for each input event:
[
    {{
        "sae_name": "<string>",
        "expectedness_classification": "<unrelated|expected|unexpected|expected_but_rare|others>",
        "expectedness_rationale": "<3-5 sentence rationale grounded in class-level evidence. Cite specific sources (e.g., class warnings, similar drug labels, mechanism) and explain which criteria above were met or not met.>",
        "severity_category": "<Unrelated| Manageable / Non-specific|Organ-Specific|Severe / Life-Threatening|Catastrophic|Others>",
        "severity_rationale": "<2-3 sentence rationale explaining the severity classification and mapping to clinical impact.>"
    }}
]

Rules:
- Do not include markdown.
- Ground expectedness in CLASS-level evidence (not just trial data provided). Use public databases, FDA labels, EMA assessments, medical literature.
- Cite sources or evidence in the rationale (e.g., "GLP-1 class label includes...", "Literature shows...", "Mechanism involves...").
- If evidence is mixed or limited, choose the most likely classification and note the uncertainty.
- severity_rationale must tie the SAE's clinical nature to one of the four categories.
- Hard constraint for severity_category: if sae_name contains malignancy/cancer/neoplasm/carcinoma terms AND does not explicitly contain death/fatal/permanent disability/irreversible failure, severity_category MUST be "Severe / Life-Threatening" (not Catastrophic).
"""
