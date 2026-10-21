"""LLM adapter tests with HTTP and SDK calls mocked; no Ollama server or Gemini API is contacted."""

import json
from types import SimpleNamespace

import pytest
import requests

from src.llm import client as client_module
from src.llm.client import FallbackClient, GeminiClient, LLMError, OllamaClient, get_client
from src.schemas import LLMSearchPlan


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(self.status_code)


def chat_reply(content: str, done_reason: str = "stop") -> FakeResponse:
    return FakeResponse(payload={"message": {"content": content}, "done_reason": done_reason})


def test_parse_sends_schema_and_validates_reply(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(json)
        return chat_reply('{"semantic_query": "pass-first guard", "positions": ["PG"]}')

    monkeypatch.setattr(client_module.requests, "post", fake_post)
    plan = OllamaClient("some-model", num_ctx=8192).parse("system", "question", LLMSearchPlan)
    assert plan.positions[0].value == "PG"
    assert sent["format"]["properties"]["semantic_query"]
    assert sent["options"] == {"temperature": 0, "num_ctx": 8192}
    assert sent["stream"] is False


@pytest.mark.parametrize(
    "reply",
    [
        chat_reply('{"positions": ["XX"], "semantic_query": "x"}'),  # invalid enum
        chat_reply("not json"),
        chat_reply('{"semantic_query": "x"}', done_reason="length"),
        FakeResponse(status_code=500, text="model crashed"),
    ],
)
def test_bad_replies_raise_llm_error(monkeypatch, reply):
    monkeypatch.setattr(client_module.requests, "post", lambda *a, **k: reply)
    with pytest.raises(LLMError):
        OllamaClient("some-model").parse("system", "question", LLMSearchPlan)


def test_get_client_reports_missing_model_and_unreachable_server(monkeypatch):
    monkeypatch.setattr(client_module.requests, "get", lambda *a, **k: FakeResponse(payload={"models": [{"name": "other:latest"}]}))
    llm, status = get_client("ollama", "wanted-model")
    assert llm is None and "ollama pull wanted-model" in status

    def unreachable(*a, **k):
        raise requests.ConnectionError()

    monkeypatch.setattr(client_module.requests, "get", unreachable)
    llm, status = get_client("ollama", "wanted-model")
    assert llm is None and "not reachable" in status

    monkeypatch.setattr(client_module.requests, "get", lambda *a, **k: FakeResponse(payload={"models": [{"name": "wanted-model:latest"}]}))
    llm, status = get_client("ollama", "wanted-model")
    assert llm is not None and json.dumps(status)


class FakeGeminiModels:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(config)
        return SimpleNamespace(text=self.text, candidates=[])


def gemini_with_reply(text):
    gemini = GeminiClient("gemini-test", api_key="test-key")
    gemini.client = SimpleNamespace(models=FakeGeminiModels(text))
    return gemini


def test_gemini_parse_sends_schema_and_validates_reply():
    gemini = gemini_with_reply('{"semantic_query": "rim protector", "positions": ["C"]}')
    plan = gemini.parse("system", "question", LLMSearchPlan)
    assert plan.positions[0].value == "C"
    config = gemini.client.models.calls[0]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema["properties"]["semantic_query"]
    assert config.temperature == 0


@pytest.mark.parametrize("text", ['{"positions": ["XX"], "semantic_query": "x"}', "", "not json"])
def test_gemini_bad_replies_raise_llm_error(text):
    with pytest.raises(LLMError):
        gemini_with_reply(text).parse("system", "question", LLMSearchPlan)


class StubLLM:
    def __init__(self, name, result=None):
        self.name = name
        self.result = result

    def parse(self, system, user, schema, effort="low"):
        if self.result is None:
            raise LLMError(f"{self.name} failed")
        return self.result


def test_fallback_client_uses_the_first_working_client():
    plan = LLMSearchPlan(semantic_query="x")
    assert FallbackClient([StubLLM("a"), StubLLM("b", plan)]).parse("s", "u", LLMSearchPlan) is plan
    with pytest.raises(LLMError, match="a: a failed; b: b failed"):
        FallbackClient([StubLLM("a"), StubLLM("b")]).parse("s", "u", LLMSearchPlan)


def ollama_has(monkeypatch, *models):
    payload = {"models": [{"name": m} for m in models]}
    monkeypatch.setattr(client_module.requests, "get", lambda *a, **k: FakeResponse(payload=payload))


def test_gemini_key_puts_gemini_ahead_of_ollama(monkeypatch):
    ollama_has(monkeypatch, "qwen2.5:7b")
    llm, status = get_client("ollama", "qwen2.5:7b", gemini_api_key="test-key", gemini_model="gemini-test")
    assert isinstance(llm, FallbackClient)
    assert [c.name for c in llm.clients] == ["gemini:gemini-test", "ollama:qwen2.5:7b"]
    assert status == "LLM: gemini:gemini-test, falling back to ollama:qwen2.5:7b"


def test_gemini_alone_when_ollama_model_missing(monkeypatch):
    ollama_has(monkeypatch, "other:latest")
    llm, status = get_client("ollama", "qwen2.5:7b", gemini_api_key="test-key", gemini_model="gemini-test")
    assert isinstance(llm, GeminiClient)
    assert "ollama pull qwen2.5:7b" in status


def test_provider_none_disables_gemini_too():
    llm, status = get_client("none", "", gemini_api_key="test-key")
    assert llm is None and "disabled" in status


def test_gemini_provider_without_key_is_reported():
    llm, status = get_client("gemini", "", gemini_api_key=None)
    assert llm is None and "GEMINI_API_KEY is not set" in status
