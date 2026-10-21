"""LLM provider adapter.

The rest of the app only depends on the LLMClient protocol, so providers can be swapped.
Providers:
  * "gemini": Google Gemini. Used first whenever GEMINI_API_KEY is set, ahead of the configured
    provider, which then serves as the backup.
  * Groq: used whenever GROQ_API_KEY is set, after Gemini (so it takes over when Gemini's quota
    runs out or Gemini fails) and before the configured provider.
  * "ollama" (default): a local model served by Ollama, with JSON-schema-constrained output.
  * "anthropic": the Anthropic API (optional; needs the `anthropic` package and credentials).
  * "none": no LLM, even if a Gemini key is set. Callers use the rule-based parser and template
    explanations instead (see src/query_parser.py and src/generation/explanations.py).
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


class GeminiClient:
    """Calls Gemini with the Pydantic model's JSON schema as the required response format.

    The SDK makes a single attempt unless told otherwise, so a brief overload (503) or rate limit
    (429) would go straight to the fallback. Those two statuses are retried up to
    GEMINI_RETRIES times with exponential backoff (about 1, 2 and 4 seconds by default).
    """

    RETRYABLE_STATUS_CODES = (429, 503)
    GEMINI_RETRIES = 3

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        retry_initial_delay: float = 1.0,
        retry_max_delay: float = 4.0,
        retry_jitter: float = 0.5,
        base_url: str | None = None,
    ):
        from google import genai
        from google.genai import errors, types

        self._errors = errors
        self._types = types
        self.http_options = types.HttpOptions(
            base_url=base_url,
            retry_options=types.HttpRetryOptions(
                attempts=self.GEMINI_RETRIES + 1,  # the SDK counts the first call as an attempt
                initial_delay=retry_initial_delay,
                max_delay=retry_max_delay,
                exp_base=2,
                jitter=retry_jitter,
                http_status_codes=list(self.RETRYABLE_STATUS_CODES),
            ),
        )
        self.client = genai.Client(api_key=api_key, http_options=self.http_options)
        self.model = model
        self.name = f"gemini:{model}"

    def parse(self, system: str, user: str, schema: type[T], effort: str = "low") -> T:
        # `effort` is an Anthropic concept; Gemini ignores it.
        config = self._types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema.model_json_schema(),
            temperature=0,
        )
        try:
            response = self.client.models.generate_content(model=self.model, contents=user, config=config)
        except self._errors.ClientError as e:
            # Only the status goes into the message because it is shown in the UI; the body stays in the traceback.
            retried = f" after {self.GEMINI_RETRIES} retries" if e.code in self.RETRYABLE_STATUS_CODES else ""
            raise LLMError(f"Gemini rejected the request ({e.code} {e.status}){retried}") from e
        except self._errors.ServerError as e:
            retried = f" after {self.GEMINI_RETRIES} retries" if e.code in self.RETRYABLE_STATUS_CODES else ""
            raise LLMError(f"Gemini server error ({e.code}){retried}") from e
        except Exception as e:  # network failures surface as httpx exceptions
            raise LLMError("Could not reach Gemini") from e

        if not response.text:
            reason = response.candidates[0].finish_reason if response.candidates else "no candidates"
            raise LLMError(f"Gemini returned no content ({reason})")
        try:
            return schema.model_validate_json(response.text)
        except ValidationError as e:
            raise LLMError("Gemini's output did not match the expected schema") from e


class GroqClient:
    """Calls Groq's OpenAI-compatible chat endpoint with the Pydantic model's JSON schema.

    Strict mode is off because it rejects schemas with optional fields, which ours have; the reply
    is validated against the Pydantic model instead. There are no retries: Groq sits in a fallback
    chain, so a rate limit should hand over to the next client rather than wait.
    """

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, model: str, api_key: str, timeout: float = 60.0, base_url: str = BASE_URL):
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self.name = f"groq:{model}"

    def parse(self, system: str, user: str, schema: type[T], effort: str = "low") -> T:
        # The caller's `effort` is not passed through. Groq's free tier caps a single request at
        # 8,000 tokens per minute and counts the reasoning budget against it; at gpt-oss's default
        # (medium) effort the explanation prompt is rejected with a 413, while "low" fits. Other
        # Groq models reject "low", so it is only sent to gpt-oss.
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(), "strict": False},
            },
            "temperature": 0,
        }
        if self.model.startswith("openai/gpt-oss"):
            payload["reasoning_effort"] = "low"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout)
        except requests.Timeout as e:
            raise LLMError(f"Groq did not answer within {self.timeout:.0f}s") from e
        except requests.RequestException as e:
            raise LLMError("Could not reach Groq") from e
        if response.status_code != 200:
            raise LLMError(f"Groq rejected the request ({response.status_code}{self._error_code(response)})")

        try:
            choice = response.json()["choices"][0]
        except (ValueError, KeyError, IndexError) as e:
            raise LLMError("Groq returned an unexpected response") from e
        if choice.get("finish_reason") == "length":
            raise LLMError("Groq ran out of output length")
        try:
            return schema.model_validate_json(choice["message"]["content"] or "")
        except (KeyError, ValidationError) as e:
            raise LLMError("Groq's output did not match the expected schema") from e

    @staticmethod
    def _error_code(response) -> str:
        # Only Groq's short error code goes into the message because it is shown in the UI.
        try:
            code = response.json()["error"]["code"]
        except (ValueError, KeyError, TypeError):
            return ""
        return f" {code}" if code else ""


class FallbackClient:
    """Tries each client in order and returns the first valid response."""

    def __init__(self, clients: list[LLMClient]):
        self.clients = clients
        self.name = ", falling back to ".join(c.name for c in clients)

    def parse(self, system: str, user: str, schema: type[T], effort: str = "low") -> T:
        failures = []
        for llm in self.clients:
            try:
                return llm.parse(system, user, schema, effort)
            except LLMError as e:
                failures.append(f"{llm.name}: {e}")
        raise LLMError("; ".join(failures))


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
    provider: str,
    model: str,
    ollama_host: str = "http://localhost:11434",
    ollama_num_ctx: int = 8192,
    gemini_api_key: str | None = None,
    gemini_model: str = "gemini-flash-latest",
    groq_api_key: str | None = None,
    groq_model: str = "openai/gpt-oss-120b",
) -> tuple[LLMClient | None, str]:
    """Return (client, status). The client is None when the app should run without an LLM.

    Clients are tried in this order: Gemini (with a Gemini key), Groq (with a Groq key), then the
    configured provider.
    """
    if provider == "none":
        return None, "LLM disabled (FULLCOURT_LLM_PROVIDER=none)"
    if provider not in ("ollama", "anthropic", "gemini"):
        raise ValueError(f"unknown LLM provider {provider!r}")

    clients: list[LLMClient] = []
    problems: list[str] = []
    if gemini_api_key:
        try:
            clients.append(GeminiClient(gemini_model, gemini_api_key))
        except Exception as e:  # missing google-genai package
            problems.append(f"Gemini client could not start ({type(e).__name__})")
    elif provider == "gemini":
        problems.append("FULLCOURT_LLM_PROVIDER=gemini but GEMINI_API_KEY is not set")

    if groq_api_key:
        clients.append(GroqClient(groq_model, groq_api_key))

    if provider == "ollama":
        ollama = OllamaClient(model, ollama_host, ollama_num_ctx)
        try:
            ollama.check()
            clients.append(ollama)
        except LLMError as e:
            problems.append(str(e))
    elif provider == "anthropic":
        try:
            clients.append(AnthropicClient(model))
        except Exception as e:  # missing package or credentials
            problems.append(f"Anthropic client could not start ({e})")

    notes = f" ({'; '.join(problems)})" if problems else ""
    if not clients:
        return None, f"LLM unavailable{notes}"
    client = clients[0] if len(clients) == 1 else FallbackClient(clients)
    return client, f"LLM: {client.name}{notes}"
