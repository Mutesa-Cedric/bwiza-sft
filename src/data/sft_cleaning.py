"""Cleaning rules for generated SFT records."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from src.data.sft_records import normalize_text

_WORD_RE = re.compile(r"[A-Za-z']+")

EN_STOPWORDS = {
    "the",
    "and",
    "is",
    "are",
    "to",
    "of",
    "in",
    "that",
    "for",
    "with",
    "on",
    "this",
    "it",
    "as",
    "be",
    "by",
    "an",
    "from",
}


@dataclass(frozen=True)
class CleanConfig:
    min_prompt_chars: int = 6
    min_response_chars: int = 24
    max_response_chars: int = 6000
    max_consecutive_repeat_words: int = 12
    min_unique_word_ratio: float = 0.12
    max_english_ratio_for_rw: float = 0.45
    reject_role_prefix_leakage: bool = True


@dataclass(frozen=True)
class CleanDecision:
    keep: bool
    reason: str


def normalize_pair(prompt: str, response: str) -> tuple[str, str]:
    return normalize_text(prompt), normalize_text(response)


def _word_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _max_consecutive_word_repeat(words: list[str]) -> int:
    if not words:
        return 0
    best = 1
    cur = 1
    prev = words[0]
    for w in words[1:]:
        if w == prev:
            cur += 1
            if cur > best:
                best = cur
        else:
            prev = w
            cur = 1
    return best


def _english_ratio(words: list[str]) -> float:
    if not words:
        return 0.0
    return sum(1 for w in words if w in EN_STOPWORDS) / len(words)


def clean_decision(prompt: str, response: str, cfg: CleanConfig, raw: dict[str, Any] | None = None) -> CleanDecision:
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
    if cfg.reject_role_prefix_leakage and ("User:" in r or "Assistant:" in r):
        return CleanDecision(False, "role_prefix_leakage")

    r_words = _word_tokens(r)
    if not r_words:
        return CleanDecision(False, "response_no_words")

    repeat_max = _max_consecutive_word_repeat(r_words)
    if repeat_max > cfg.max_consecutive_repeat_words:
        return CleanDecision(False, "response_word_repetition")

    unique_ratio = len(set(r_words)) / max(1, len(r_words))
    if unique_ratio < cfg.min_unique_word_ratio:
        return CleanDecision(False, "response_low_unique_ratio")

    lang_mode = normalize_text((raw or {}).get("lang_mode", "")).lower()
    if lang_mode in {"rw", "control"}:
        en_ratio = _english_ratio(r_words)
        if en_ratio > cfg.max_english_ratio_for_rw:
            return CleanDecision(False, "response_too_english_for_rw")

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
