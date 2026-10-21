"""LLM provider adapter.

The rest of the app only depends on the LLMClient protocol, so providers can be swapped.
Providers:
  * "ollama" (default): a local model served by Ollama, with JSON-schema-constrained output.
  * "anthropic": the Anthropic API (optional; needs the `anthropic` package and credentials).
  * "none": no LLM. Callers use the rule-based parser and template explanations instead
    (see src/query_parser.py and src/generation/explanations.py).
"""

from __future__ import annotations

from typing import Protocol, TypeVar

import requests
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Any failure to get a valid structured response. Callers fall back to deterministic logic."""


class LLMClient(Protocol):
    name: str

    def parse(self, system: str, user: str, schema: type[T], effort: str = "low") -> T: ...


class OllamaClient:
    """Calls Ollama's /api/chat with `format` set to the Pydantic model's JSON schema."""

    def __init__(self, model: str, host: str = "http://localhost:11434", num_ctx: int = 8192, timeout: float = 300.0):
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx  # Ollama's default (often 4096) is too small for the explanation prompt
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def check(self) -> None:
        """Raise LLMError unless the server is reachable and the model is pulled."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException as e:
            raise LLMError(f"Ollama is not reachable at {self.host}") from e
        names = {m["name"] for m in response.json().get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise LLMError(f"Ollama model {self.model!r} is not pulled (run: ollama pull {self.model})")

    def parse(self, system: str, user: str, schema: type[T], effort: str = "low") -> T:
        # `effort` is an Anthropic concept; local models ignore it.
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema.model_json_schema(),
            "stream": False,
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
            "keep_alive": "10m",
        }
        try:
            response = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        except requests.Timeout as e:
            raise LLMError(f"Ollama did not answer within {self.timeout:.0f}s") from e
        except requests.RequestException as e:
            raise LLMError(f"Could not reach Ollama at {self.host}") from e
        if response.status_code != 200:
            raise LLMError(f"Ollama error {response.status_code}: {response.text[:200]}")

        data = response.json()
        if data.get("done_reason") == "length":
            raise LLMError("The model ran out of context or output length")
        try:
            return schema.model_validate_json(data["message"]["content"])
        except (KeyError, ValidationError) as e:
            raise LLMError("The model's output did not match the expected schema") from e


class AnthropicClient:
    def __init__(self, model: str):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()  # credentials resolved from the environment
        self.model = model
        self.name = f"anthropic:{model}"

    def parse(self, system: str, user: str, schema: type[T], effort: str = "low") -> T:
        anthropic = self._anthropic
        try:
            response = self.client.beta.messages.parse(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
                output_config={"effort": effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as e:
            raise LLMError("Anthropic credentials were rejected") from e
        except anthropic.RateLimitError as e:
            raise LLMError("Anthropic rate limit reached") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Could not reach the Anthropic API") from e

        if response.stop_reason == "refusal":
            raise LLMError("The model declined this request")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise LLMError("The model did not return a complete structured response")
        return response.parsed_output


def get_client(
    provider: str, model: str, ollama_host: str = "http://localhost:11434", ollama_num_ctx: int = 8192
) -> tuple[LLMClient | None, str]:
    """Return (client, status). The client is None when the app should run without an LLM."""
    if provider == "none":
        return None, "LLM disabled (FULLCOURT_LLM_PROVIDER=none)"
    if provider == "ollama":
        client = OllamaClient(model, ollama_host, ollama_num_ctx)
        try:
            client.check()
        except LLMError as e:
            return None, f"LLM unavailable: {e}"
        return client, f"LLM: {client.name}"
    if provider == "anthropic":
        try:
            client = AnthropicClient(model)
        except Exception as e:  # missing package or credentials
            return None, f"LLM unavailable: Anthropic client could not start ({e})"
        return client, f"LLM: {client.name}"
    raise ValueError(f"unknown LLM provider {provider!r}")
