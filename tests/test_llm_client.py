"""LLM adapter tests with HTTP and SDK calls mocked; no Ollama server, Gemini or Groq API is contacted."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest
import requests

from src.llm import client as client_module
from src.llm.client import FallbackClient, GeminiClient, GroqClient, LLMError, OllamaClient, get_client
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


def groq_reply(content, finish_reason="stop"):
    return FakeResponse(payload={"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]})


def test_groq_parse_sends_schema_and_validates_reply(monkeypatch):
    sent = {}

    def fake_post(url, json, headers, timeout):
        sent.update(url=url, payload=json, headers=headers)
        return groq_reply('{"semantic_query": "stretch big", "positions": ["PF"]}')

    monkeypatch.setattr(client_module.requests, "post", fake_post)
    plan = GroqClient("groq-test", api_key="test-key").parse("system", "question", LLMSearchPlan)
    assert plan.positions[0].value == "PF"
    assert sent["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert sent["headers"] == {"Authorization": "Bearer test-key"}
    response_format = sent["payload"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"]["properties"]["semantic_query"]
    assert sent["payload"]["temperature"] == 0


@pytest.mark.parametrize("model, expected", [("openai/gpt-oss-120b", "low"), ("qwen/qwen3.8-27b", None)])
def test_groq_sends_low_reasoning_effort_only_to_gpt_oss(monkeypatch, model, expected):
    sent = {}

    def fake_post(url, json, headers, timeout):
        sent.update(json)
        return groq_reply('{"semantic_query": "x"}')

    monkeypatch.setattr(client_module.requests, "post", fake_post)
    GroqClient(model, api_key="test-key").parse("system", "question", LLMSearchPlan, effort="medium")
    assert sent.get("reasoning_effort") == expected


@pytest.mark.parametrize(
    "reply",
    [
        groq_reply('{"positions": ["XX"], "semantic_query": "x"}'),  # invalid enum
        groq_reply("not json"),
        groq_reply('{"semantic_query": "x"}', finish_reason="length"),
        FakeResponse(payload={"choices": []}),
        FakeResponse(status_code=500, text="server error"),
    ],
)
def test_groq_bad_replies_raise_llm_error(monkeypatch, reply):
    monkeypatch.setattr(client_module.requests, "post", lambda *a, **k: reply)
    with pytest.raises(LLMError):
        GroqClient("groq-test", api_key="test-key").parse("system", "question", LLMSearchPlan)


def test_groq_rate_limit_reports_status_and_code_only(monkeypatch):
    body = {"error": {"message": "Rate limit reached for org_123 on tokens per day", "code": "rate_limit_exceeded"}}
    monkeypatch.setattr(client_module.requests, "post", lambda *a, **k: FakeResponse(status_code=429, payload=body))
    with pytest.raises(LLMError, match=r"^Groq rejected the request \(429 rate_limit_exceeded\)$"):
        GroqClient("groq-test", api_key="test-key").parse("system", "question", LLMSearchPlan)


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


def test_groq_sits_between_gemini_and_ollama(monkeypatch):
    ollama_has(monkeypatch, "qwen2.5:7b")
    llm, status = get_client(
        "ollama", "qwen2.5:7b", gemini_api_key="g-key", gemini_model="gemini-test", groq_api_key="q-key", groq_model="groq-test"
    )
    assert [c.name for c in llm.clients] == ["gemini:gemini-test", "groq:groq-test", "ollama:qwen2.5:7b"]


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


class FakeGeminiServer:
    """Local HTTP server that fails with `status` for the first `failures` requests, then answers."""

    def __init__(self, failures: int, status: int = 503):
        self.failures, self.status, self.requests = failures, status, 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                server.requests += 1
                if server.requests <= server.failures:
                    code, body = server.status, {"error": {"code": server.status, "message": "busy", "status": "UNAVAILABLE"}}
                else:
                    text = json.dumps({"semantic_query": "rim protector"})
                    code, body = 200, {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}]}
                payload = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/"

    def client(self):
        return GeminiClient(
            "gemini-test", api_key="test-key", base_url=self.url,
            retry_initial_delay=0.01, retry_max_delay=0.02, retry_jitter=0.01,
        )


@pytest.fixture
def fake_gemini_server():
    servers = []

    def start(failures, status=503):
        servers.append(FakeGeminiServer(failures, status))
        return servers[-1]

    yield start
    for server in servers:
        server.httpd.shutdown()


def test_gemini_retries_overloaded_requests_then_succeeds(fake_gemini_server):
    server = fake_gemini_server(failures=2)
    plan = server.client().parse("system", "question", LLMSearchPlan)
    assert plan.semantic_query == "rim protector"
    assert server.requests == 3


def test_gemini_gives_up_after_three_retries(fake_gemini_server):
    server = fake_gemini_server(failures=10)
    with pytest.raises(LLMError, match=r"503\) after 3 retries"):
        server.client().parse("system", "question", LLMSearchPlan)
    assert server.requests == 4


def test_gemini_retries_rate_limits(fake_gemini_server):
    server = fake_gemini_server(failures=1, status=429)
    server.client().parse("system", "question", LLMSearchPlan)
    assert server.requests == 2


def test_gemini_does_not_retry_bad_requests(fake_gemini_server):
    server = fake_gemini_server(failures=10, status=400)
    with pytest.raises(LLMError, match="400"):
        server.client().parse("system", "question", LLMSearchPlan)
    assert server.requests == 1
