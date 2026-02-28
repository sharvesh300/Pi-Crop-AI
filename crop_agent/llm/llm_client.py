import subprocess
from typing import List, Dict

from crop_agent.utils import load_yaml, MODEL_CONFIG, AGENT_CONFIG
from crop_agent.llm.prompt_loader import build_prompt
from crop_agent.llm.response_parser import parse_response


class LLMEngine:
    """
    Lightweight Agentic LLM Engine for treatment decision support.

    Responsibilities:
    - Load LLM config
    - Enforce bounded action space
    - Build structured prompt
    - Call local LLM (Ollama)
    - Parse and validate JSON output
    """

    def __init__(self):
        self.model_config = load_yaml(MODEL_CONFIG)
        self.agent_config = load_yaml(AGENT_CONFIG)

        self.llm_cfg = self.model_config["llm"]
        self.allowed_actions = self.agent_config["agent"]["allowed_actions"]

        self.backend = self.llm_cfg["backend"]
        self.model_name = self.llm_cfg["model_name"]
        self.temperature = self.llm_cfg["temperature"]
        self.max_tokens = self.llm_cfg["max_tokens"]
        self.enforce_json = self.llm_cfg.get("enforce_json", True)

    # ============================================================
    # PUBLIC API
    # ============================================================

    def generate_decision(
        self,
        case_context: Dict,
        similar_cases: List[str],
    ) -> Dict:
        """
        Main entry point for decision generation.
        """

        prompt = build_prompt(case_context, similar_cases, self.allowed_actions)

        raw_response = self._infer(prompt)

        if self.enforce_json:
            return parse_response(raw_response, self.allowed_actions)

        return {"raw_response": raw_response}

    # ============================================================
    # INFERENCE BACKEND
    # ============================================================

    def _infer(self, prompt: str) -> str:
        if self.backend == "ollama":
            return self._infer_ollama(prompt)

        raise ValueError(f"Unsupported backend: {self.backend}")

    def _infer_ollama(self, prompt: str) -> str:
        """
        Calls local Ollama model.
        """

        result = subprocess.run(
            [
                "ollama",
                "run",
                self.model_name,
            ],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Ollama error: {result.stderr.decode()}")

        return result.stdout.decode()
