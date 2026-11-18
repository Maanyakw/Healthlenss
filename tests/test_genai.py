"""Provider selection and request/response handling for Groq, Gemini and Anthropic (HTTP mocked; no network)."""
import os
os.environ.setdefault("RISK_BACKEND", "hgb")
import pytest
from backend import config, genai


class Resp:
    def __init__(self, status, data): self.status_code, self._d, self.text = status, data, str(data)
    def json(self): return self._d


def keys(monkeypatch, **kw):
    for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setattr(config, k, kw.get(k, ""))
    monkeypatch.setattr(config, "CHAT_PROVIDER", "auto"); monkeypatch.setattr(config, "REPORT_PROVIDER", "auto")


def test_provider_choice(monkeypatch):
    keys(monkeypatch, GROQ_API_KEY="g", GEMINI_API_KEY="m")
    assert genai.provider("chat") == "groq" and genai.provider("report") == "gemini"
    keys(monkeypatch, GEMINI_API_KEY="m")
    assert genai.provider("chat") == "gemini" and genai.provider("report") == "gemini"
    keys(monkeypatch)
    assert not genai.available("chat") and not genai.available("report")
    keys(monkeypatch, GROQ_API_KEY="g"); monkeypatch.setattr(config, "CHAT_PROVIDER", "none")
    assert genai.provider("chat") is None


def test_groq_request_and_parse(monkeypatch):
    keys(monkeypatch, GROQ_API_KEY="secret")
    seen = {}
    def fake_post(url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, body=json)
        return Resp(200, {"choices": [{"message": {"content": " hello "}}]})
    monkeypatch.setattr(genai.requests, "post", fake_post)
    out = genai.generate("sys", [{"role": "user", "content": "hi"}], purpose="chat", json_mode=True)
    assert out == "hello" and seen["url"].startswith("https://api.groq.com/openai/v1/")
    assert seen["headers"]["Authorization"] == "Bearer secret"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"} and seen["body"]["response_format"]["type"] == "json_object"


def test_error_and_empty(monkeypatch):
    keys(monkeypatch, GROQ_API_KEY="k")
    monkeypatch.setattr(genai.requests, "post", lambda *a, **k: Resp(429, {"error": {"message": "rate limit"}}))
    with pytest.raises(genai.GenAIError, match="429"):
        genai.generate("s", [{"role": "user", "content": "x"}])
    monkeypatch.setattr(genai.requests, "post", lambda *a, **k: Resp(200, {"choices": []}))
    with pytest.raises(genai.GenAIError):
        genai.generate("s", [{"role": "user", "content": "x"}])


def test_chat_uses_groq_then_falls_back(monkeypatch):
    from fastapi.testclient import TestClient
    from backend.app import app
    P = dict(gender="Female", age=54, hypertension=1, heart_disease=0, smoking_history="never", bmi=31.2, hba1c=6.8, glucose=190)
    keys(monkeypatch, GROQ_API_KEY="k")
    monkeypatch.setattr(genai.requests, "post", lambda *a, **k: Resp(200, {"choices": [{"message": {"content": "Paris is the capital of France."}}]}))
    with TestClient(app) as c:
        r = c.post("/api/chat", json=dict(patient=P, narrative="", history=[], message="What is the capital of France?")).json()
        assert r["mode"] == "genai" and r["provider"] == "groq"
        monkeypatch.setattr(genai.requests, "post", lambda *a, **k: Resp(500, {"error": {"message": "down"}}))
        r = c.post("/api/chat", json=dict(patient=P, narrative="", history=[], message="What does my HbA1c mean?")).json()
        assert r["mode"] == "local-fallback" and r["reply"]
        w = c.post("/api/chat", json=dict(patient=P, narrative="", history=[], message="What if my BMI were 24?")).json()
        assert w.get("whatif")
