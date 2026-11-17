"""Prompt templates for unmet needs extraction."""


UNMET_NEEDS_EXTRACTION_PROMPT = """You are an expert clinical and pharma strategy analyst.

Your task is to analyze the provided SOC (Standard of Care) document and extract **Unmet Needs** in a flat list (not grouped by category), at the most granular level possible, in a way that allows evaluation against new molecule clinical trials.

---

### Definitions:
- **Unmet Need**: Any gap, limitation, or inadequacy in current treatment, outcomes, safety, access, or patient experience.
- These may be explicitly stated OR inferred.
- **Granularity**: Each unmet need should describe ONE specific, distinct clinical or operational gap — not a broad theme. Prefer narrow, precise statements over sweeping generalizations.

---

### Categories (assign based on the nature of the unmet need — do NOT look for these words in the document):
- **Achievement of Clinical Targets** — suboptimal attainment of HbA1c, weight, blood pressure, or other clinical goals
- **Durability of Disease Control** — loss of response over time, relapse, progression despite treatment, therapy switching due to waning benefit
- **Safety & Treatment-Related Risk** — adverse events, toxicity, tolerability concerns, interactions, long-term safety risks
- **Treatment Burden & Adherence Complexity** — regimen complexity, dosing burden, persistence issues, administration-related burden
- **Access, Affordability & Care Delivery** — cost, reimbursement barriers, formulary limits, geographic/service availability constraints
- **Diagnostic Accuracy & Early Detection** — delayed diagnosis, underdiagnosis, misclassification, poor screening pathways
- **Complication Prevention & Management** — persistent rates or poor control of cardiovascular, renal, retinal, neuropathic, or other complications
- **Quality of Life & Psychosocial Burden** — symptom burden, functional impairment, emotional distress, patient-reported quality-of-life impacts
- **Population-Specific Gaps & Inequities** — underserved subgroups with disproportionately poor outcomes or limited care options

> **Important**: Categories are your analytical lens — they will rarely (if ever) appear as explicit labels in the SOC document. Your job is to read the clinical content, identify every gap or limitation described, and assign the most appropriate category yourself.

---

### Instructions:

1. Read the entire attached SOC document thoroughly.

2. Identify ALL unmet needs present in the document — both explicitly stated and clinically inferable:
   - Limitations of existing therapies (response rates, durability, resistance)
   - Safety and tolerability issues mentioned for current treatments
   - Populations described as having fewer or no good treatment options
   - Statements about disease burden, quality of life, or symptom persistence
   - Diagnostic delays, lack of reliable biomarkers, or misdiagnosis
   - Adherence or administration challenges mentioned for existing drugs
   - Any phrase like "need for", "remains a challenge", "limited by", "lack of", "few options"

3. For each identified unmet need, assign it to the most fitting category from the list above and extract:

   #### a. unmet_need
   - A single, concise, and specific clinical statement
   - Avoid vague language like "limited efficacy" — be precise, e.g., "Less than 30% of patients achieve HbA1c targets with current GLP-1 therapies"
   - Should be directly comparable against clinical trial endpoints

   #### b. evidence
   - Exact verbatim sentence or phrase from the SOC document
   - Do NOT paraphrase — quote directly

4. Return the extracted unmet needs in one flat list.
  - Each unmet need must still include the assigned category field.
  - Be exhaustive — do not consolidate distinct gaps into one.
  - A single drug or treatment approach may contribute multiple unmet needs across different categories.

---

5. Deduplication rules:
   - Only merge needs if they are truly identical in meaning and population
   - Preserve distinct needs even if they seem similar — granularity is preferred

6. Quality checks:
   - No hallucinations — all evidence must come from the SOC text
   - Exclude trivial or non-clinical observations
   - Every unmet need must have supporting evidence from the document

---

### Output Format (STRICT JSON)

Return one flat list under unmet_needs.

{{
  "unmet_needs": [
    {{
      "unmet_need": "<specific, granular unmet need statement>",
      "category": "<one of: Achievement of Clinical Targets, Durability of Disease Control, Safety & Treatment-Related Risk, Treatment Burden & Adherence Complexity, Access, Affordability & Care Delivery, Diagnostic Accuracy & Early Detection, Complication Prevention & Management, Quality of Life & Psychosocial Burden, Population-Specific Gaps & Inequities>",
      "evidence": "<exact verbatim quote from SOC document>"
    }}
  ]
}}

---

### Additional Guidelines:
- If no unmet needs are found at all, return:
  {{ "unmet_needs": [] }}
- Do NOT include any text or explanation outside the JSON block.
- Do NOT hallucinate — only use information grounded in the SOC document.
- Aim for depth and specificity over brevity.

---

### Input:
Document: {document_name}

Use the attached PDF as the only source of truth."""

# ---------------------------------------------------------------------------
# 3-Step Multi-Turn Web Search Prompts
# ---------------------------------------------------------------------------

UNMET_NEEDS_WEB_STEP1_PROMPT = """Act as a principal health economist and clinical researcher specialising in diabetes.

Conduct a comprehensive web search regarding Type 2 Diabetes (T2D) in {country} and provide a detailed overview covering:
1. Current T2D prevalence, incidence trends, and undiagnosed rates with the latest figures available.
2. Primary macro public health challenges specific to {country} (e.g., regional disparities, rural vs. urban access, healthcare infrastructure constraints).
3. National healthcare system structure for diabetes management — how public and private coverage applies to T2D patients.
4. Key demographic and comorbidity patterns (obesity, hypertension, renal disease) that shape T2D burden in {country}.

Use country-specific search queries such as:
- "type 2 diabetes prevalence {country}"
- "{country} diabetes public health burden"
- "{country} healthcare system diabetes coverage"

Return a detailed, factual narrative. Cite statistics and sources where possible. Do NOT return JSON at this stage."""


UNMET_NEEDS_WEB_STEP2_PROMPT = """Based on the T2D country landscape just established for {country}, execute targeted follow-up web searches \
focusing on unmet needs across the following three dimensions only:

1. **Access, Affordability & Care Delivery**: What are the specific barriers to advanced therapeutics (SGLT2 inhibitors, GLP-1 receptor agonists, \
   modern insulins)? What are out-of-pocket costs, reimbursement restrictions, formulary exclusions, or geographic/service availability constraints \
   for T2D patients in {country}?

2. **Diagnostic Accuracy & Early Detection**: Is continuous glucose monitoring (CGM) or digital health widely available to T2D patients in {country}? \
   Are there delays in T2D diagnosis, underdiagnosis rates, misclassification issues, or poor screening pathways?

3. **Population-Specific Gaps & Inequities**: Are there underserved subgroups in {country} — elderly, low-income, rural, ethnic minorities, or \
   comorbid patients — with disproportionately poor T2D outcomes or limited care options?

Use country-specific search queries such as:
- "{country} GLP-1 SGLT2 access reimbursement"
- "{country} diabetes affordability out-of-pocket cost"
- "{country} diabetes diagnosis delay underdiagnosis"
- "{country} diabetes screening gaps"
- "{country} diabetes health disparities underserved population"

Return a detailed, factual narrative with evidence per dimension. Do NOT return JSON at this stage."""


UNMET_NEEDS_WEB_STEP3_PROMPT = """Synthesize all findings from our entire research conversation about Type 2 Diabetes in {country} into a \
structured, definitive list of unmet needs.

Do NOT execute any new web searches. Use ONLY the data and evidence gathered in the previous two research steps.

You are an expert clinical and pharma strategy analyst. Extract unmet needs ONLY from the three categories listed below and return them \
in the strict JSON schema below.

Definitions:
- Unmet need: A clinically meaningful gap where current T2D care in {country} does not adequately achieve desired patient, clinician, \
  or health-system outcomes.
- Each unmet need must be specific to {country}, granular, and supported by evidence from the research above.

Categories (extract ONLY from these three — use exactly as written):
• Access, Affordability & Care Delivery
• Diagnostic Accuracy & Early Detection
• Population-Specific Gaps & Inequities

Output JSON schema (REQUIRED — return ONLY this JSON, no other text):
{{
  "unmet_needs": [
    {{
      "category": "<one of the 3 categories exactly as written above>",
      "unmet_need": "<specific, {country}-relevant unmet need statement>",
      "evidence": "<evidence from the research gathered above>",
      "source_urls": ["<URL if available, else empty list>"]
    }}
  ]
}}

Rules:
- Extract ONLY unmet needs that fall under the three specified categories. Do NOT extract needs from any other category.
- Be exhaustive within those three categories — extract as many granular unmet needs as the research supports.
- Every unmet_need must be specific to {country}, not generic global information.
- Do NOT include any text or explanation outside the JSON block.
- If no unmet needs were found, return: {{ "unmet_needs": [] }}"""
