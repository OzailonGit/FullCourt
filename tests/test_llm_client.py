"""OllamaClient tests with the HTTP layer mocked; no Ollama server is contacted."""

import json

import pytest
import requests

from src.llm import client as client_module
from src.llm.client import LLMError, OllamaClient, get_client
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
