from typing import Dict, List

from crop_agent.memory.vector_store import VectorStore
from crop_agent.memory.metadata_store import MetadataStore
from crop_agent.llm.llm_client import LLMEngine
from crop_agent.agent.planner import TreatmentPlanner
from crop_agent.utils import load_yaml, SYSTEM_CONFIG, AGENT_CONFIG


class DecisionAgent:
    """
    Orchestrates the crop treatment decision and planning pipeline.

    Retrieves similar historical cases from memory, asks the LLM for a
    decision, then generates a step-by-step treatment plan for that decision.

    Usage:
        agent = DecisionAgent()
        result = agent.decide(case_context)
    """

    def __init__(self):
        """
        Initialise all sub-components from configuration.

        Usage:
            agent = DecisionAgent()
        """
        self.system_config = load_yaml(SYSTEM_CONFIG)
        self.agent_config = load_yaml(AGENT_CONFIG)

        self.vector_store = VectorStore(SYSTEM_CONFIG)
        self.metadata_store = MetadataStore()

        self.llm_engine = LLMEngine()
        self.planner = TreatmentPlanner(self.llm_engine, self.agent_config)

        self.top_k = self.system_config["memory"]["top_k"]

    def decide(self, case_context: Dict) -> Dict:
        """
        Run the full agentic pipeline for a given crop case.

        Steps:
            1. Retrieve semantically similar historical cases from memory.
            2. Ask the LLM to select a treatment decision.
            3. Generate a step-by-step treatment plan for that decision.

        Args:
            case_context: Dict containing crop, disease, severity, confidence,
                          and optional temperature/humidity fields.

        Returns:
            A dict with 'decision', 'reason', and 'plan' fields.

        Usage:
            result = agent.decide({
                "crop": "Tomato",
                "disease": "Leaf Blight",
                "severity": "medium",
                "confidence": 0.88,
            })
        """
        similar_cases = self._retrieve_memory(case_context)

        decision = self.llm_engine.generate_decision(case_context, similar_cases)

        plan_result = self.planner.build_plan(case_context, decision)
        decision["plan"] = plan_result.get("plan", [])

        return decision

    def _retrieve_memory(self, case_context: Dict) -> List[str]:
        """
        Build a query string from the case context and retrieve the top-k
        most similar historical cases from the vector store.

        Args:
            case_context: The current crop case dict.

        Returns:
            A list of matching case text strings (None entries are excluded).

        Usage:
            cases = agent._retrieve_memory(case_context)
        """
        query_text = (
            f"{case_context['crop']} "
            f"{case_context['disease']} "
            f"{case_context['severity']} "
            f"{case_context.get('temperature', '')} "
            f"{case_context.get('humidity', '')}"
        )

        indices, _ = self.vector_store.search(query_text)

        return [
            case
            for idx in indices
            if (case := self.metadata_store.get_case(idx + 1)) is not None
        ]
