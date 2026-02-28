from typing import List, Dict


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

    prompt = f"""
You are an agricultural treatment decision agent.

Your task is to select ONE appropriate treatment action
based on the current case and past similar cases.

Current Case:
Crop: {context["crop"]}
Disease: {context["disease"]}
Severity: {context["severity"]}
Confidence: {context["confidence"]}
Temperature: {context.get("temperature", "N/A")}
Humidity: {context.get("humidity", "N/A")}

Similar Past Cases:
{memory_block}

Allowed Actions:
{actions_block}

IMPORTANT:
- Choose exactly ONE action from the allowed list.
- Do NOT invent new actions.
- Respond ONLY in valid JSON — no extra text, no explanation outside JSON.
- Always respond in English.

JSON format:
{{
  "decision": "<ACTION>",
  "reason": "<short explanation in English>"
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

    return f"""You are an agricultural treatment planner.

Current Case:
Crop: {case_context["crop"]}
Disease: {case_context["disease"]}
Severity: {case_context["severity"]}

Approved Decision: {decision["decision"]}
Reason: {decision.get("reason", "N/A")}

Create a short, logical step-by-step treatment plan using ONLY the allowed actions below.

Allowed Plan Actions:
{allowed}

RULES:
- Use ONLY the actions listed above.
- Respond ONLY in valid JSON — no extra text before or after.
- Always respond in English.
- Keep the plan to 3-5 steps maximum.

Output format:
{{
  "plan": [
    {{"step": 1, "action": "<ACTION>", "details": "<one sentence in English>"}},
    {{"step": 2, "action": "<ACTION>", "details": "<one sentence in English>"}}
  ]
}}""".strip()
