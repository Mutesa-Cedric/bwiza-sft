"""Minimal OpenAI chat-completions client with retries for organizer passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import http.client
import json
import os
import ssl
import time
from urllib import error, request


@dataclass(frozen=True)
class OpenAIConfig:
    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.0
    max_output_tokens: int = 300
    max_retries: int = 5
    retry_backoff_sec: float = 1.5


class OpenAIError(RuntimeError):
    pass


def _ssl_context() -> ssl.SSLContext:
    ca_bundle = os.environ.get("OPENAI_CA_BUNDLE", "").strip()
    if ca_bundle:
        return ssl.create_default_context(cafile=ca_bundle)
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _extract_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip()
    return ""


def chat_completion(
    cfg: OpenAIConfig,
    *,
    system_prompt: str,
    user_prompt: str,
) -> str:
    endpoint = f"{cfg.base_url.rstrip('/')}/chat/completions"
    body = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "max_completion_tokens": cfg.max_output_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    raw = json.dumps(body).encode("utf-8")
    req = request.Request(
        endpoint,
        data=raw,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        },
    )

    last_err = ""
    ctx = _ssl_context()
    for attempt in range(1, cfg.max_retries + 1):
        try:
            with request.urlopen(req, timeout=120, context=ctx) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                text = _extract_text(payload)
                if not text:
                    raise OpenAIError("empty_completion_text")
                return text
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            last_err = f"http_{e.code}:{detail[:400]}"
            if e.code in (408, 409, 429, 500, 502, 503, 504) and attempt < cfg.max_retries:
                time.sleep(cfg.retry_backoff_sec * attempt)
                continue
            raise OpenAIError(last_err) from e
        except (
            error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            OpenAIError,
            http.client.RemoteDisconnected,
            ssl.SSLError,
            ConnectionResetError,
        ) as e:
            last_err = str(e)
            if attempt < cfg.max_retries:
                time.sleep(cfg.retry_backoff_sec * attempt)
                continue
            raise OpenAIError(last_err) from e

    raise OpenAIError(last_err or "openai_request_failed")
