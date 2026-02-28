from typing import Dict

from crop_agent.utils import load_yaml, AGENT_CONFIG


class SafetyValidator:
    """
    Validates an LLM decision against the configured safety rules.

    Only blocks decisions that are entirely unknown (not in the allowed
    action list). All valid actions are passed through — severity-based
    overrides are intentionally omitted so the LLM retains full decision
    authority over recognised actions.

    Flags decisions that require human confirmation when configured.

    Usage:
        validator = SafetyValidator()
        result = validator.validate(case_context, llm_decision)
    """

    def __init__(self):
        """
        Load allowed actions and safety config from agent.yaml.

        Usage:
            validator = SafetyValidator()
        """
        agent_cfg = load_yaml(AGENT_CONFIG)["agent"]
        self.allowed_actions = agent_cfg["allowed_actions"]
        self.safety_cfg = agent_cfg["safety"]

    def validate(self, case_context: Dict, llm_decision: Dict) -> Dict:
        """
        Validate the LLM decision and enrich it with safety metadata.

        Blocks only decisions that are not in the allowed action list.
        All other decisions are passed through unchanged with safe=True.

        Args:
            case_context: The current crop case dict.
            llm_decision: The raw decision dict from the LLM.

        Returns:
            Decision dict enriched with 'safe', 'override', and optionally
            'requires_confirmation' fields.

        Usage:
            result = validator.validate(case_context, llm_decision)
        """
        decision = llm_decision.get("decision", "NO_TREATMENT")

        if (
            self.safety_cfg.get("block_unknown_actions")
            and decision not in self.allowed_actions
        ):
            return {
                **llm_decision,
                "decision": "NO_TREATMENT",
                "reason": "Decision not in allowed action list.",
                "safe": False,
                "override": True,
            }

        result = {**llm_decision, "safe": True, "override": False}
        if self.safety_cfg.get("require_human_confirmation"):
            result["requires_confirmation"] = True

        return result
