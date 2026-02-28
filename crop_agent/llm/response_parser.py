from typing import Dict, List

from crop_agent.utils import Utils


def parse_response(response: str, allowed_actions: List[str]) -> Dict:
    """
    Extract and validate a treatment decision JSON from a raw LLM response.

    Delegates to Utils.parse_response.

    Usage:
        result = parse_response(raw_response, allowed_actions)
    """
    return Utils.parse_response(response, allowed_actions)
