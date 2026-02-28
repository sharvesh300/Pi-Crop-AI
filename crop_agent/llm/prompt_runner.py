from typing import Callable

from crop_agent.llm.model_backends import ModelBackend


class PromptRunner:
    """
    Executes prompts against a model backend.

    Supports two modes:
    - Static: run a fully-built prompt string directly.
    - Injected: run a prompt template with runtime variables merged in at call time.

    Usage:
        runner = PromptRunner(backend)

        # Static
        raw = runner.run("Tell me about tomato blight.")

        # Injected template
        raw = runner.run_template(
            "Diagnose {disease} in {crop}.",
            crop="Tomato",
            disease="Leaf Blight",
        )

        # Builder function
        raw = runner.run_with_builder(build_prompt, case_context, memory, actions)
    """

    def __init__(self, backend: ModelBackend):
        """
        Args:
            backend: Any initialised ModelBackend (OllamaBackend, HuggingFaceBackend, …).

        Usage:
            runner = PromptRunner(backend)
        """
        self.backend = backend

    def run(self, prompt: str) -> str:
        """
        Execute a fully-built static prompt and return the raw response.

        Args:
            prompt: The complete prompt string to send to the model.

        Returns:
            Raw model output string.

        Usage:
            raw = runner.run("You are an agricultural agent. Diagnose...")
        """
        return self.backend.run(prompt)

    def run_template(self, template: str, **variables) -> str:
        """
        Inject runtime variables into a prompt template and execute it.

        Args:
            template: A str.format-style template, e.g. "Diagnose {disease} in {crop}.".
            **variables: Key-value pairs merged into the template at call time.

        Returns:
            Raw model output string.

        Usage:
            raw = runner.run_template(
                "Diagnose {disease} in {crop}.",
                crop="Tomato",
                disease="Leaf Blight",
            )
        """
        prompt = template.format(**variables)
        return self.backend.run(prompt)

    def run_with_builder(self, builder: Callable[..., str], *args, **kwargs) -> str:
        """
        Build a prompt using any callable builder function, then execute it.

        Args:
            builder: A callable that returns a prompt string when called with
                     the provided args/kwargs (e.g. build_prompt, build_plan_prompt).
            *args: Positional arguments forwarded to the builder.
            **kwargs: Keyword arguments forwarded to the builder.

        Returns:
            Raw model output string.

        Usage:
            raw = runner.run_with_builder(
                build_prompt, case_context, similar_cases, allowed_actions
            )
            raw = runner.run_with_builder(
                build_plan_prompt, case_context, decision, plan_actions
            )
        """
        prompt = builder(*args, **kwargs)
        return self.backend.run(prompt)
