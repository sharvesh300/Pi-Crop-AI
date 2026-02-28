import json
import yaml
from pathlib import Path
from typing import Dict, List

# Root of the project (two levels up from this file)
_CONFIG_DIR = Path(__file__).parent.parent / "config"

# YAML config file paths
AGENT_CONFIG = _CONFIG_DIR / "agent.yaml"
DATABASE_CONFIG = _CONFIG_DIR / "database.yaml"
DEPLOYMENT_CONFIG = _CONFIG_DIR / "deployment.yaml"
MODEL_CONFIG = _CONFIG_DIR / "model.yaml"
SENSORS_CONFIG = _CONFIG_DIR / "sensors.yaml"
SYSTEM_CONFIG = _CONFIG_DIR / "system_config.yaml"


class Utils:
    """
    Collection of shared utility helpers used across the project.

    All methods are static — no instantiation required.

    Usage:
        config = Utils.load_yaml(SYSTEM_CONFIG)
        parsed = Utils.parse_json(raw_string)
        decision = Utils.parse_response(raw, allowed_actions)
        plan = Utils.parse_plan(raw, allowed_plan_actions)
    """

    @staticmethod
    def load_yaml(path: str | Path) -> dict:
        """
        Load a YAML file and return its contents as a dictionary.

        Args:
            path: Absolute or relative path to a YAML file.

        Returns:
            Parsed contents as a dict.

        Usage:
            config = Utils.load_yaml(SYSTEM_CONFIG)
            config = Utils.load_yaml("config/agent.yaml")
        """
        with open(path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def parse_json(response: str) -> dict:
        """
        Extract the first JSON object found in a raw string.

        Args:
            response: Raw string that may contain surrounding text or
                      markdown code fences (```json ... ```).

        Returns:
            Parsed dict from the extracted JSON block.

        Raises:
            ValueError: If no valid JSON object is found.

        Usage:
            data = Utils.parse_json(raw_llm_output)
        """
        import re

        # Strip markdown code fences if present
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))
        start = response.find("{")
        end = response.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in response.")
        return json.loads(response[start:end])

    @staticmethod
    def parse_response(response: str, allowed_actions: List[str]) -> Dict:
        """
        Extract and validate a treatment decision JSON from a raw LLM response.

        Falls back to NO_TREATMENT if the output is invalid or the decision
        is not in allowed_actions.

        Args:
            response: Raw LLM output string.
            allowed_actions: List of permitted decision action strings.

        Returns:
            A dict with 'decision' and 'reason' fields.

        Usage:
            result = Utils.parse_response(raw_response, allowed_actions)
        """
        try:
            parsed = Utils.parse_json(response)
            decision = parsed.get("decision")
            if decision not in allowed_actions:
                raise ValueError("Invalid decision from LLM.")
            return parsed
        except Exception:
            return {
                "decision": "NO_TREATMENT",
                "reason": "Fallback: invalid or unsafe LLM output.",
            }

    @staticmethod
    def parse_plan(response: str, allowed_plan_actions: List[str]) -> Dict:
        """
        Extract a treatment plan JSON from a raw LLM response.

        The plan steps are constrained at the prompt level; strict per-step
        action validation is intentionally omitted here to avoid silently
        discarding valid plans when the LLM uses slightly different casing
        or wording.

        Falls back to an empty plan only if JSON extraction entirely fails.

        Args:
            response: Raw LLM output string.
            allowed_plan_actions: Passed through for reference; not used for
                                  hard validation at parse time.

        Returns:
            A dict with a 'plan' key containing an ordered list of step dicts.

        Usage:
            plan = Utils.parse_plan(raw_response, allowed_plan_actions)
        """
        try:
            return Utils.parse_json(response)
        except Exception:
            return {
                "plan": [],
                "reason": "Fallback: could not parse plan from LLM output.",
            }


# Module-level shim so existing callers of load_yaml() continue to work
def load_yaml(path: str | Path) -> dict:
    return Utils.load_yaml(path)
