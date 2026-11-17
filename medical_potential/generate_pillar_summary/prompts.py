PROMPT_FOR_CLINICAL_MOLECULES = """
You are a pharmaceutical asset assessment expert.

molecule_name = {molecule_name}

product_details = {product_details}

You will be provided with:
1. Product details as structured metadata.
2. Cognito dimension reports as attachments.

Your task is to generate a concise executive clinical asset summary for the molecule.

INSTRUCTIONS

Generate the summary using the following structure:

1. Product Profile
Capture:
- Molecule class
- Dosage form(s)
- Dosage strength(s)
- Highest clinical phase
- Primary indication
- Key secondary indications

2. Differentiation 
Explain:
- Core mechanism of action
- Most important efficacy signal(s)
- Key unmet need(s) addressed
- Main source(s) of clinical differentiation
Sources
- refer to the attached MoA Innovation, Clinical Efficacy and Unmet needs attached pdf for the above

3. Opportunity
Explain:
- Maturity of clinical evidence
- Potential label expansion opportunities
- Addressable market potential
- Why the asset could achieve significant adoption
Sources
- Refer to the attached Clinical Evidence and Label Expansion pdf for the above

4. Risk & Outlook
Explain:
- Most important risk, limitation, uncertainty, or competitive threat
- Overall outlook
Sources
- Refer to the attached Serious safety, and Tolerability reports


OUTPUT REQUIREMENTS

- Return EXACTLY ONE paragraph.
- Hard cap: 100 words.
- Do not use bullet points.
- Do not use headings.
- Do not mention Cognito dimensions by name.
- Use the product details to anchor stage, dosage, and indication context.
- Use evidence from the attached reports only.
- Do not invent information.
- If dosage strengths are unavailable, omit them.
- If secondary indications are unavailable, omit them.
- Write in an objective analyst tone.
- Use grammatically correct, complete sentences and clinically appropriate terminology; for example, describe overweight-related treatment as "chronic weight management" when appropriate.


Example

Follow this pattern:

"Bofanglutide is a Phase 3 long-acting GLP-1 receptor agonist, delivered as a bi-weekly subcutaneous injection (illustrative strengths: 2 mg / 4 mg), for obesity and Type 2 diabetes, with secondary development in metabolic and liver-related indications. It differentiates through dosing convenience and meaningful efficacy in weight management, glycemic control, and hepatic inflammation, supported by growing late-stage evidence and cardiometabolic expansion potential. The key uncertainty is whether a mono-agonist profile can remain competitive against stronger dual and triple incretin competitors as the market evolves. Overall, Bofanglutide shows a moderate-to-strong medical outlook, constrained by competitive differentiation risk."

Before generating the output, synthesize evidence across all attached reports and prioritize the strongest supporting and contradictory evidence when determining the outlook.
"""

PROMPT_FOR_MARKETED_MOLECULE = """
You are a pharmaceutical asset assessment expert.

molecule_name = {molecule_name}

product_details = {product_details}

You will be provided with:
1. Product details as structured metadata.
2. Cognito dimension reports as attachments.

Your task is to generate a concise executive summary for the molecule for the Medical Potential.

INSTRUCTIONS

Generate the summary using the following structure:

1. Product Profile

Capture:
- Molecule class
- Marketed status (e.g., Marketed US/EU/UK/Japan)
- Dosage form(s)
- Dosage strength(s)
- Primary indication
- Key secondary indications

2. Positioning

Explain:
- Current treatment-line role
- Most important efficacy advantages
- Key unmet need(s) addressed
- Main source(s) of competitive differentiation

Sources:
- Refer to the attached Clinical Efficacy, Unmet Need, and Line of Treatment reports.

3. Franchise Potential

Explain:
- Current strength and maturity of the franchise
- Label expansion opportunities
- Factors supporting sustained adoption and market relevance

Sources:
- Refer to the attached Label Expansion reports.

4. Risk & Outlook

Explain:
- Most important safety, tolerability, commercial, or durability risk
- Overall outlook

Sources:
- Refer to the attached Serious Safety, Tolerability reports.

OUTPUT REQUIREMENTS

- Return EXACTLY ONE paragraph.
- Hard cap: 100 words.
- Do not use bullet points.
- Do not use headings.
- Do not mention Cognito dimensions by name.
- Use the product details to anchor marketed status, dosage, and indication context.
- Use evidence from the attached reports only.
- Do not use own knowledge or invent information.
- If dosage strengths are unavailable, omit them.
- If secondary indications are unavailable, omit them.
- Write in an objective analyst tone.
- Use grammatically correct, complete sentences and clinically appropriate terminology; for example, describe overweight-related treatment as "chronic weight management" when appropriate.

Example

Follow this pattern:

"Tirzepatide is a first-in-class dual GIP/GLP-1 receptor agonist, marketed (US/EU/UK/Japan) as a once-weekly subcutaneous injection in strengths of 2.5, 5, 7.5, 10, 12.5, and 15 mg, for Type 2 diabetes (Mounjaro) and obesity plus obstructive sleep apnea (Zepbound). It differentiates through best-in-class efficacy, delivering class-leading weight loss (~26.6%) and robust glycemic control, backed by a mature SURPASS/SURMOUNT Phase 3 evidence base and a broad label-expansion pipeline across MASH, CKD, and cardiometabolic indications. The key uncertainty is a thyroid C-cell boxed warning and gastrointestinal tolerability. Overall, Tirzepatide shows a strong medical outlook, driven by transformative efficacy and franchise durability."

Before generating the output:
- Synthesize evidence across all attached reports.
- Prioritize the strongest supporting and contradictory evidence.
- Highlight the single most important differentiator, opportunity, and risk.
- Assess outlook based on overall franchise strength, competitive durability, safety profile, and future expansion potential.
"""
