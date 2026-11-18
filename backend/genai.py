"""Thin client for hosted generative-AI APIs: Groq (chat), Gemini (report), Anthropic (optional).

`purpose` is "chat" or "report". If no key is configured for a purpose, `available()` is False and
callers fall back to clearly labelled local behaviour.
"""
from __future__ import annotations

import requests

from . import config


class GenAIError(RuntimeError):
    pass


_KEYS = {"groq": lambda: config.GROQ_API_KEY, "gemini": lambda: config.GEMINI_API_KEY,
         "anthropic": lambda: config.ANTHROPIC_API_KEY}
_ORDER = {"chat": ("groq", "gemini", "anthropic"), "report": ("gemini", "anthropic", "groq")}


def provider(purpose: str = "chat") -> str | None:
    want = config.CHAT_PROVIDER if purpose == "chat" else config.REPORT_PROVIDER
    if want == "none":
        return None
    if want in _KEYS:
        return want if _KEYS[want]() else None
    for name in _ORDER[purpose]:
        if _KEYS[name]():
            return name
    return None


def model_name(purpose: str = "chat") -> str | None:
    return {"groq": config.GROQ_MODEL, "gemini": config.GEMINI_MODEL,
            "anthropic": config.ANTHROPIC_MODEL}.get(provider(purpose))


def available(purpose: str = "chat") -> bool:
    return provider(purpose) is not None


def _err(resp: requests.Response) -> str:
    try:
        j = resp.json()
        msg = j.get("error", {}).get("message") if isinstance(j.get("error"), dict) else None
        msg = msg or str(j)[:300]
    except Exception:
        msg = resp.text[:300]
    return f"{resp.status_code}: {msg}"


def generate(system: str, messages: list[dict], *, purpose: str = "chat", json_mode: bool = False,
             max_tokens: int = 2048, temperature: float = 0.3) -> str:
    """messages: [{"role": "user"|"assistant", "content": str}, ...] (must end with a user turn)."""
    p = provider(purpose)
    if p is None:
        raise GenAIError("No generative-AI provider configured.")
    try:
        if p == "groq":
            body = {"model": config.GROQ_MODEL, "temperature": temperature, "max_tokens": max_tokens,
                    "messages": [{"role": "system", "content": system}] + messages}
            if "gpt-oss" in config.GROQ_MODEL:
                # reasoning model: keep thinking short and leave room for the visible answer
                body["reasoning_effort"] = "low"
                body["max_tokens"] = max(max_tokens, 2000)
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                              headers={"Authorization": f"Bearer {config.GROQ_API_KEY}", "Content-Type": "application/json"},
                              json=body, timeout=config.GENAI_TIMEOUT_S)
            if r.status_code != 200:
                raise GenAIError(_err(r))
            choices = r.json().get("choices") or []
            text = ((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
            if not text:
                raise GenAIError("empty response")
            return text
        if p == "gemini":
            body = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
                             for m in messages],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max(max_tokens, 2048) * 2},
            }
            if json_mode:
                body["generationConfig"]["responseMimeType"] = "application/json"
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent",
                headers={"x-goog-api-key": config.GEMINI_API_KEY, "Content-Type": "application/json"},
                json=body, timeout=config.GENAI_TIMEOUT_S)
            if r.status_code != 200:
                raise GenAIError(_err(r))
            data = r.json()
            cands = data.get("candidates") or []
            if not cands:
                raise GenAIError(f"empty response ({data.get('promptFeedback', {}).get('blockReason', 'no candidates')})")
            parts = cands[0].get("content", {}).get("parts", [])
            text = "".join(x.get("text", "") for x in parts).strip()
            if not text:
                raise GenAIError(f"empty response (finishReason={cands[0].get('finishReason')})")
            return text
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": config.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": config.ANTHROPIC_MODEL, "max_tokens": max_tokens, "temperature": temperature,
                  "system": system, "messages": messages},
            timeout=config.GENAI_TIMEOUT_S)
        if r.status_code != 200:
            raise GenAIError(_err(r))
        blocks = r.json().get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        if not text:
            raise GenAIError("empty response")
        return text
    except requests.RequestException as e:
        raise GenAIError(f"network error: {type(e).__name__}") from e
