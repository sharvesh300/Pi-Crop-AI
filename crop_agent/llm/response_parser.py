import json
from typing import Dict, List


def parse_response(response: str, allowed_actions: List[str]) -> Dict:
    """
    Extract and validate JSON from a raw LLM response string.

    Falls back to NO_TREATMENT if the output is invalid or the
    decision is not in allowed_actions.

    Usage:
        result = parse_response(raw_response, allowed_actions)
    """
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        json_str = response[start:end]
        parsed = json.loads(json_str)

        decision = parsed.get("decision")
        if decision not in allowed_actions:
            raise ValueError("Invalid decision from LLM.")

        return parsed

    except Exception:
        return {
            "decision": "NO_TREATMENT",
            "reason": "Fallback: invalid or unsafe LLM output.",
        }
