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
- Choose exactly ONE action.
- Do NOT invent new actions.
- Respond ONLY in valid JSON.

JSON format:
{{
  "decision": "<ACTION>",
  "reason": "<short explanation>"
}}
"""

    return prompt.strip()
