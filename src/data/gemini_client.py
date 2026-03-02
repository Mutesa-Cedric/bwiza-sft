"""Minimal Gemini REST client with retries for SFT distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import os
import ssl
import time
from urllib import error, request


@dataclass(frozen=True)
class GeminiConfig:
    model: str
    api_key: str
    base_url: str = "https://generativelanguage.googleapis.com"
    temperature: float = 0.4
    top_p: float = 0.95
    max_output_tokens: int = 1024
    max_retries: int = 5
    retry_backoff_sec: float = 1.5


class GeminiError(RuntimeError):
    pass


def _ssl_context() -> ssl.SSLContext:
    # Allow explicit override when users provide a custom enterprise/OS bundle.
    ca_bundle = os.environ.get("GEMINI_CA_BUNDLE", "").strip()
    if ca_bundle:
        return ssl.create_default_context(cafile=ca_bundle)

    # Prefer certifi bundle to avoid local trust-store issues on some Python builds.
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _extract_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""

    parts = candidates[0].get("content", {}).get("parts", [])
    texts: list[str] = []
    for p in parts:
        t = p.get("text")
        if isinstance(t, str) and t.strip():
            texts.append(t.strip())
    return "\n".join(texts).strip()


def generate_text(cfg: GeminiConfig, user_prompt: str, system_prompt: str = "") -> str:
    endpoint = f"{cfg.base_url.rstrip('/')}/v1beta/models/{cfg.model}:generateContent?key={cfg.api_key}"

    body: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": cfg.temperature,
            "topP": cfg.top_p,
            "maxOutputTokens": cfg.max_output_tokens,
        },
    }
    if system_prompt.strip():
        body["systemInstruction"] = {"parts": [{"text": system_prompt.strip()}]}

    raw = json.dumps(body).encode("utf-8")
    req = request.Request(
        endpoint,
        data=raw,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    last_err = ""
    ctx = _ssl_context()
    for attempt in range(1, cfg.max_retries + 1):
        try:
            with request.urlopen(req, timeout=120, context=ctx) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                text = _extract_text(payload)
                if not text:
                    raise GeminiError("empty_candidate_text")
                return text
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            last_err = f"http_{e.code}:{detail[:400]}"
            if e.code in (429, 500, 502, 503, 504) and attempt < cfg.max_retries:
                time.sleep(cfg.retry_backoff_sec * attempt)
                continue
            raise GeminiError(last_err) from e
        except (error.URLError, TimeoutError, json.JSONDecodeError, GeminiError) as e:
            last_err = str(e)
            if attempt < cfg.max_retries:
                time.sleep(cfg.retry_backoff_sec * attempt)
                continue
            raise GeminiError(last_err) from e

    raise GeminiError(last_err or "gemini_request_failed")
