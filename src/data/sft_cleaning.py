"""Cleaning rules for generated SFT records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.data.sft_records import normalize_text


@dataclass(frozen=True)
class CleanConfig:
    min_prompt_chars: int = 6
    min_response_chars: int = 24
    max_response_chars: int = 6000


@dataclass(frozen=True)
class CleanDecision:
    keep: bool
    reason: str


def normalize_pair(prompt: str, response: str) -> tuple[str, str]:
    return normalize_text(prompt), normalize_text(response)


def clean_decision(prompt: str, response: str, cfg: CleanConfig) -> CleanDecision:
    p, r = normalize_pair(prompt, response)

    if not p:
        return CleanDecision(False, "empty_prompt")
    if not r:
        return CleanDecision(False, "empty_response")
    if len(p) < cfg.min_prompt_chars:
        return CleanDecision(False, "prompt_too_short")
    if len(r) < cfg.min_response_chars:
        return CleanDecision(False, "response_too_short")
    if len(r) > cfg.max_response_chars:
        return CleanDecision(False, "response_too_long")

    return CleanDecision(True, "ok")


def dedup_key(prompt: str, response: str) -> str:
    p, r = normalize_pair(prompt, response)
    return f"{p.lower()}\n{r.lower()}"


def to_clean_record(raw: dict[str, Any], prompt: str, response: str) -> dict[str, Any]:
    p, r = normalize_pair(prompt, response)
    out = dict(raw)
    out["prompt"] = p
    out["response"] = r
    return out
