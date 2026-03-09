import json as _json
import os
import subprocess
from typing import Generator, Protocol


# Generous timeout for on-device inference (Pi is slow; two calls per request)
_OLLAMA_TIMEOUT_S = 180  # seconds per single LLM call

# Ollama base URL — resolved at import time.
# Priority: OLLAMA_HOST env var → model.yaml ollama_host → localhost:11434
def _resolve_ollama_host() -> str:
    env = os.environ.get("OLLAMA_HOST", "").strip()
    if env:
        return env.rstrip("/")
    # Try reading from model.yaml if available
    try:
        import yaml as _yaml
        from pathlib import Path as _Path
        _cfg_path = _Path(__file__).parent.parent.parent / "config" / "model.yaml"
        with open(_cfg_path) as _f:
            _cfg = _yaml.safe_load(_f)
        host = _cfg.get("llm", {}).get("ollama_host", "").strip()
        if host:
            return host.rstrip("/")
    except Exception:
        pass
    return "http://localhost:11434"


_OLLAMA_HOST = _resolve_ollama_host()


def _check_ollama_running() -> None:
    """Raise RuntimeError early if the Ollama daemon is not up."""
    import requests as _http
    try:
        resp = _http.get(f"{_OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
    except _http.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {_OLLAMA_HOST}. "
            "Run `ollama serve` on the target machine and check the host setting in config/model.yaml."
        )
    except _http.exceptions.Timeout:
        raise RuntimeError(f"Ollama health-check timed out at {_OLLAMA_HOST} — is the daemon running?")
    except Exception as exc:
        raise RuntimeError(f"Ollama health-check failed ({_OLLAMA_HOST}): {exc}")


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
            RuntimeError: If Ollama is not running, the process errors, or
                          the call exceeds _OLLAMA_TIMEOUT_S seconds.

        Usage:
            raw = backend.run(prompt)
        """
        _check_ollama_running()

        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name],
                input=prompt.encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=_OLLAMA_TIMEOUT_S,
                env={**os.environ, "OLLAMA_HOST": _OLLAMA_HOST},
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Ollama inference timed out after {_OLLAMA_TIMEOUT_S} s. "
                "The model may be too large for this device or Ollama is busy."
            )

        if result.returncode != 0:
            raise RuntimeError(f"Ollama error: {result.stderr.decode()}")

        return result.stdout.decode()

    def stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Stream tokens from the Ollama HTTP API (localhost:11434/api/generate).

        Yields individual token strings as they arrive, enabling real-time
        display in the UI without waiting for the full response.

        Args:
            prompt: Fully-built prompt string.

        Yields:
            Individual token strings from the model.

        Usage:
            for token in backend.stream(prompt):
                print(token, end="", flush=True)
        """
        import requests as _http

        _check_ollama_running()

        try:
            resp = _http.post(
                f"{_OLLAMA_HOST}/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": True},
                stream=True,
                timeout=_OLLAMA_TIMEOUT_S,
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = _json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
        except _http.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama streaming timed out after {_OLLAMA_TIMEOUT_S} s."
            )
        except _http.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama HTTP error: {exc}")


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

    def stream(self, prompt: str) -> Generator[str, None, None]:
        """HuggingFace pipelines don't support token streaming; yield full text."""
        yield self.run(prompt)


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
