import subprocess
from typing import Protocol


class ModelBackend(Protocol):
    """
    Interface that all model backends must satisfy.

    Any backend must implement a single `run(prompt)` method that
    accepts a prompt string and returns the raw model output string.
    """

    def run(self, prompt: str) -> str: ...


class OllamaBackend:
    """
    Runs a locally-served Ollama model via subprocess.

    Usage:
        backend = OllamaBackend("qwen2.5:1.5b")
        response = backend.run("Your prompt here")
    """

    def __init__(self, model_name: str):
        """
        Args:
            model_name: The Ollama model tag, e.g. 'qwen2.5:1.5b'.

        Usage:
            backend = OllamaBackend("qwen2.5:1.5b")
        """
        self.model_name = model_name

    def run(self, prompt: str) -> str:
        """
        Pass the prompt to the Ollama CLI and return stdout.

        Args:
            prompt: Fully-built prompt string.

        Returns:
            Decoded stdout from the Ollama process.

        Raises:
            RuntimeError: If the process exits with a non-zero code.

        Usage:
            raw = backend.run(prompt)
        """
        result = subprocess.run(
            ["ollama", "run", self.model_name],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Ollama error: {result.stderr.decode()}")

        return result.stdout.decode()


class HuggingFaceBackend:
    """
    Runs a HuggingFace text-generation pipeline locally.

    Requires the 'transformers' and 'torch' packages to be installed.

    Usage:
        backend = HuggingFaceBackend("microsoft/phi-2", max_new_tokens=256)
        response = backend.run("Your prompt here")
    """

    def __init__(self, model_name: str, max_new_tokens: int = 256):
        """
        Load the HuggingFace pipeline for the given model.

        Args:
            model_name: HuggingFace model repo ID, e.g. 'microsoft/phi-2'.
            max_new_tokens: Maximum number of tokens to generate.

        Usage:
            backend = HuggingFaceBackend("microsoft/phi-2")
        """
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "HuggingFaceBackend requires 'transformers' and 'torch'. "
                "Install them with: pip install transformers torch"
            ) from e

        self.max_new_tokens = max_new_tokens
        self._pipeline = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=max_new_tokens,
        )

    def run(self, prompt: str) -> str:
        """
        Run the HuggingFace pipeline on the given prompt.

        Args:
            prompt: Fully-built prompt string.

        Returns:
            Generated text string (excluding the original prompt).

        Usage:
            raw = backend.run(prompt)
        """
        outputs = self._pipeline(prompt)
        generated = outputs[0]["generated_text"]
        # Strip the input prompt from the output
        return generated[len(prompt) :].strip()


def get_backend(backend: str, model_name: str, **kwargs) -> ModelBackend:
    """
    Factory — return the correct backend instance for the given backend name.

    Args:
        backend: One of 'ollama' or 'huggingface'.
        model_name: Model identifier passed to the backend.
        **kwargs: Extra keyword arguments forwarded to the backend constructor.

    Returns:
        An initialised ModelBackend instance.

    Raises:
        ValueError: If the backend name is not recognised.

    Usage:
        backend = get_backend("ollama", "qwen2.5:1.5b")
        backend = get_backend("huggingface", "microsoft/phi-2", max_new_tokens=512)
    """
    backends = {
        "ollama": OllamaBackend,
        "huggingface": HuggingFaceBackend,
    }

    if backend not in backends:
        raise ValueError(
            f"Unsupported backend '{backend}'. Choose from: {list(backends)}"
        )

    return backends[backend](model_name, **kwargs)
