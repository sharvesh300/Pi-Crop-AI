from typing import Dict

from crop_agent.llm.llm_client import LLMEngine


class TreatmentPlanner:
    """
    Generates a step-by-step treatment plan for an approved decision.

    Uses the LLMEngine to produce a structured plan restricted to the
    allowed plan actions defined in agent.yaml.

    Usage:
        planner = TreatmentPlanner(llm_engine, agent_config)
        plan = planner.build_plan(case_context, decision)
    """

    def __init__(self, llm_engine: LLMEngine, agent_config: dict):
        """
        Args:
            llm_engine: Initialised LLMEngine instance.
            agent_config: Parsed contents of agent.yaml (top-level dict).

        Usage:
            planner = TreatmentPlanner(llm_engine, agent_config)
        """
        self.llm = llm_engine
        self.allowed_plan_actions = agent_config["planner"]["allowed_plan_actions"]

    def build_plan(self, case_context: Dict, decision: Dict) -> Dict:
        """
        Build a structured treatment plan for the given case and decision.

        Args:
            case_context: The current crop case dict (crop, disease, severity, …).
            decision: The validated decision dict returned by SafetyValidator.

        Returns:
            A dict with a 'plan' key containing an ordered list of step dicts,
            each with 'step', 'action', and 'details' fields.

        Usage:
            plan = planner.build_plan(case_context, decision)
        """
        return self.llm.generate_plan(
            case_context,
            decision,
            self.allowed_plan_actions,
        )
