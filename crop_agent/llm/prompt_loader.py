from typing import List, Dict


# ---------------------------------------------------------------------------
# Disease knowledge base — soybean diseases supported by the CNN model.
# Each entry provides agronomic context injected into both prompts so the LLM
# can reason about disease biology, not just the label string.
# ---------------------------------------------------------------------------
DISEASE_KNOWLEDGE: Dict[str, Dict[str, str]] = {
    "Bacterial Blight": {
        "pathogen": "Pseudomonas savastanoi pv. glycinea (bacterium)",
        "symptoms": "Water-soaked, angular spots on leaves bounded by leaf veins; spots turn "
        "brown with yellow halos. Infected pods show reddish-brown streaks.",
        "spread": "Spreads via wind-driven rain, infected seed, and field equipment. "
        "Favoured by cool to warm (18–28 °C), wet, and humid conditions.",
        "urgency": "Moderate — rarely causes severe yield loss alone but can predispose "
        "plants to secondary infections. Monitor closely in wet cool seasons.",
        "management_notes": "No fully effective chemical control; copper-based bactericides "
        "offer partial suppression. Use certified disease-free seed. Remove infected debris.",
    },
    "Cercospora Leaf Blight": {
        "pathogen": "Cercospora kikuchii (fungus)",
        "symptoms": "Purple-bronze discolouration on upper leaf surface; leaves may curl "
        "and drop prematurely. Infected seed shows purple staining (purple seed stain).",
        "spread": "Airborne conidia; seed-borne. Favoured by hot (30–35 °C), humid late-season "
        "conditions during seed fill.",
        "urgency": "High during seed fill — seed infection reduces germination and introduces "
        "disease into next season's crop.",
        "management_notes": "Apply triazole fungicides at R3–R5 growth stages. Use disease-free "
        "seed; treat seed with fungicide before planting.",
    },
    "Downy Mildew": {
        "pathogen": "Peronospora manshurica (oomycete)",
        "symptoms": "Pale green to yellow spots on upper leaf surface with corresponding "
        "grey-white downy sporulation on the underside. Infected seed may be encrusted with white mycelium.",
        "spread": "Airborne sporangia and seed-borne oospores. Cool (15–22 °C), moist, "
        "overcast conditions with high humidity favour infection.",
        "urgency": "Moderate — rarely causes major yield loss but can reduce seed quality "
        "and persist through infected seed.",
        "management_notes": "Seed treatment with metalaxyl reduces seed-borne infection. "
        "Foliar fungicides offer limited benefit. Rotate crops and improve field drainage.",
    },
    "Frogeye Leaf Spot": {
        "pathogen": "Cercospora sojina (fungus)",
        "symptoms": "Circular to irregular grey-brown lesions with reddish-purple borders "
        "('frog-eye' appearance). Lesions merge under heavy pressure.",
        "spread": "Spores spread by wind and rain splash. Warm (25–28 °C), humid conditions "
        "and extended leaf wetness accelerate infection.",
        "urgency": "High in susceptible varieties — can cause 10–30 % yield loss. "
        "Act promptly when lesions appear during reproductive stages.",
        "management_notes": "Fungicide resistance to strobilurins reported in some regions; "
        "use triazoles or tank mixes. Rotate crops and use resistant cultivars.",
    },
    "Healthy": {
        "pathogen": "None",
        "symptoms": "No visible disease symptoms. Plant appears vigorous with normal "
        "leaf colour and structure.",
        "spread": "N/A",
        "urgency": "None — no intervention required.",
        "management_notes": "Continue routine scouting. Maintain good agronomic practices "
        "to preserve plant health.",
    },
    "Potassium Deficiency": {
        "pathogen": "Abiotic — nutrient deficiency (not infectious)",
        "symptoms": "Yellowing and scorching of leaf margins and tips starting on older lower "
        "leaves, progressing upward. Interveinal tissue remains green initially.",
        "spread": "Not contagious. Caused by low soil potassium, compacted soil, drought stress, "
        "or excessive leaching reducing K uptake.",
        "urgency": "Moderate to high depending on growth stage — deficiency during pod fill "
        "significantly reduces seed weight and protein content.",
        "management_notes": "Apply potassium fertiliser (muriate of potash / SOP) based on soil test. "
        "Correct soil pH (6.0–6.5) to improve K availability. Ensure adequate soil moisture.",
    },
    "Soybean Rust": {
        "pathogen": "Phakopsora pachyrhizi (fungus — Asian soybean rust)",
        "symptoms": "Small tan to reddish-brown pustules (uredinia) mainly on the undersides "
        "of leaves; upper surface shows small yellow-tan lesions. Rapid defoliation in severe cases.",
        "spread": "Wind-dispersed urediniospores; can travel hundreds of kilometres. "
        "Warm (15–28 °C), humid conditions with extended leaf wetness drive rapid spread.",
        "urgency": "Very high — one of the most economically damaging soybean diseases globally. "
        "Can cause 10–80 % yield loss. Immediate fungicide action is critical.",
        "management_notes": "Apply triazole fungicides (tebuconazole, propiconazole) or triazole + "
        "strobilurin mixtures at first sign. Scout undersides of leaves regularly. "
        "Early application before 25 % canopy infection is essential.",
    },
    "Target Spot": {
        "pathogen": "Corynespora cassiicola (fungus)",
        "symptoms": "Circular brown lesions with concentric rings (target-board pattern) "
        "and a yellow halo on leaves. Defoliation in severe cases.",
        "spread": "Airborne conidia; thrives in warm (25–30 °C), humid, overcast conditions. "
        "Canopy density and dense stands increase risk.",
        "urgency": "High during reproductive stages — defoliation of upper canopy reduces "
        "pod fill and yield. Act before 10 % canopy defoliation.",
        "management_notes": "Apply triazole or strobilurin fungicides at canopy closure. "
        "Avoid excessive canopy density; scout upper canopy regularly.",
    },
}


def _get_disease_block(disease: str) -> str:
    """Return a formatted disease information block for the given disease name."""
    info = DISEASE_KNOWLEDGE.get(disease)
    if not info:
        return f"Disease: {disease}\n(No detailed information available in knowledge base.)"
    return (
        f"Disease Information — {disease}:\n"
        f"  Pathogen      : {info['pathogen']}\n"
        f"  Symptoms      : {info['symptoms']}\n"
        f"  Spread        : {info['spread']}\n"
        f"  Urgency       : {info['urgency']}\n"
        f"  Mgmt Notes    : {info['management_notes']}"
    )


def build_prompt(context: Dict, memory: List[str], allowed_actions: List[str]) -> str:
    """
    Build the structured treatment-decision prompt.

    Usage:
        prompt = build_prompt(case_context, similar_cases, allowed_actions)
    """
    memory_block = (
        "\n".join(m for m in memory if m is not None) or "No similar historical cases."
    )

    actions_block = "\n".join([f"- {action}" for action in allowed_actions])
    disease_block = _get_disease_block(context["disease"])

    prompt = f"""
You are an expert agricultural treatment decision agent specialising in crop disease management.

Your task is to select ONE appropriate treatment action based on the current case,
the disease biology below, and past similar cases.

--- DISEASE BIOLOGY ---
{disease_block}

--- CURRENT CASE ---
Crop        : {context["crop"]}
Disease     : {context["disease"]}
Severity    : {context["severity"]}
Confidence  : {context["confidence"]}
Temperature : {context.get("temperature", "N/A")}
Humidity    : {context.get("humidity", "N/A")}

--- SIMILAR PAST CASES ---
{memory_block}

--- ALLOWED ACTIONS ---
{actions_block}

DECISION GUIDELINES:
- Use the disease urgency and management notes above to inform your decision.
- For diseases marked "Very high" or "High" urgency at medium/high severity, prefer MEDICINAL_TREATMENT.
- For "Healthy" or low-urgency diseases at low severity, prefer NO_TREATMENT or MONITOR.
- Factor in temperature and humidity where available — warm humid conditions accelerate fungal spread.

IMPORTANT:
- Choose exactly ONE action from the allowed list.
- Do NOT invent new actions.
- Respond ONLY in valid JSON — no extra text, no explanation outside JSON.
- Always respond in English.

JSON format:
{{
  "decision": "<ACTION>",
  "reason": "<short explanation in English referencing disease biology>"
}}
"""

    return prompt.strip()


def build_plan_prompt(
    case_context: Dict, decision: Dict, allowed_actions: List[str]
) -> str:
    """
    Build the structured treatment-plan prompt.

    Usage:
        prompt = build_plan_prompt(case_context, decision, allowed_plan_actions)
    """
    allowed = "\n".join([f"- {a}" for a in allowed_actions])
    disease_block = _get_disease_block(case_context["disease"])

    return f"""You are an expert agricultural treatment planner specialising in crop disease management.

--- DISEASE BIOLOGY ---
{disease_block}

--- CURRENT CASE ---
Crop        : {case_context["crop"]}
Disease     : {case_context["disease"]}
Severity    : {case_context["severity"]}

--- APPROVED DECISION ---
Action : {decision["decision"]}
Reason : {decision.get("reason", "N/A")}

Create a concise, biologically-informed step-by-step treatment plan using ONLY the allowed actions below.
Reference the disease management notes above when choosing and describing each step.

Allowed Plan Actions:
{allowed}

RULES:
- Use ONLY the actions listed above.
- Respond ONLY in valid JSON — no extra text before or after.
- Always respond in English.
- Keep the plan to 3-5 steps maximum.
- Each detail should be one practical, specific sentence referencing the disease.

Output format:
{{
  "plan": [
    {{"step": 1, "action": "<ACTION>", "details": "<one practical sentence referencing the disease>"}},
    {{"step": 2, "action": "<ACTION>", "details": "<one practical sentence referencing the disease>"}}
  ]
}}""".strip()
