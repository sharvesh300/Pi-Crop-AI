from typing import List, Dict

from crop_agent.utils import Utils, load_yaml, MODEL_CONFIG, AGENT_CONFIG
from crop_agent.llm.model_backends import get_backend
from crop_agent.llm.prompt_runner import PromptRunner
from crop_agent.llm.prompt_loader import build_plan_prompt, build_prompt


class LLMEngine:
    """
    Lightweight agentic LLM engine for crop treatment decision support.

    Delegates backend execution to ModelBackend and prompt execution to
    PromptRunner, keeping decision/plan logic cleanly separated.

    Usage:
        engine = LLMEngine()
        decision = engine.generate_decision(case_context, similar_cases)
        plan = engine.generate_plan(case_context, decision, plan_actions)
    """

    def __init__(self):
        """
        Load model and agent configuration, then wire up the backend and runner.

        Usage:
            engine = LLMEngine()
        """
        self.model_config = load_yaml(MODEL_CONFIG)
        self.agent_config = load_yaml(AGENT_CONFIG)

        self.llm_cfg = self.model_config["llm"]
        self.allowed_actions = self.agent_config["agent"]["allowed_actions"]

        self.enforce_json = self.llm_cfg.get("enforce_json", True)

        backend = get_backend(
            self.llm_cfg["backend"],
            self.llm_cfg["model_name"],
        )
        self.runner = PromptRunner(backend)

    def generate_decision(
        self,
        case_context: Dict,
        similar_cases: List[str],
    ) -> Dict:
        """
        Generate a treatment decision for the given crop case.

        Builds a structured prompt via PromptRunner, then parses and validates
        the JSON response with Utils. Falls back to NO_TREATMENT on failure.

        Args:
            case_context: Dict with crop, disease, severity, confidence, and
                          optional temperature/humidity fields.
            similar_cases: List of historical case strings retrieved from memory.

        Returns:
            A dict with 'decision' and 'reason' fields.

        Usage:
            decision = engine.generate_decision(case_context, similar_cases)
        """
        raw = self.runner.run_with_builder(
            build_prompt, case_context, similar_cases, self.allowed_actions
        )

        if self.enforce_json:
            return Utils.parse_response(raw, self.allowed_actions)

        return {"raw_response": raw}

    def generate_plan(
        self,
        case_context: Dict,
        decision: Dict,
        allowed_plan_actions: List[str],
    ) -> Dict:
        """
        Generate a step-by-step treatment plan for an approved decision.

        Args:
            case_context: The current crop case dict.
            decision: The validated decision dict returned by generate_decision.
            allowed_plan_actions: List of permitted plan step action strings.

        Returns:
            A dict with a 'plan' key containing an ordered list of step dicts.

        Usage:
            plan = engine.generate_plan(case_context, decision, plan_actions)
        """
        raw = self.runner.run_with_builder(
            build_plan_prompt, case_context, decision, allowed_plan_actions
        )
        return Utils.parse_plan(raw, allowed_plan_actions)
