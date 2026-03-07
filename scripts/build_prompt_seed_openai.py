#!/usr/bin/env python3
"""Generate prompt-seed JSONL via OpenAI with compact batched plain-text outputs."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import datetime, timezone
from hashlib import sha1
import json
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dedup_index import PromptDedupIndex
from src.data.openai_client import OpenAIConfig, OpenAIError, chat_completion
from src.data.sft_records import append_jsonl, normalize_text

DEFAULT_TOPICS = [
    "uburezi mu Rwanda",
    "ubuhinzi n'ubworozi",
    "ubuzima rusange",
    "ikoranabuhanga",
    "ubukungu bw'umuryango",
    "amakuru yo ku mbuga nkoranyambaga",
    "uburere bw'abana",
    "kwiga no gutegura ibizamini",
    "akazi n'imyuga",
    "iterambere ry'icyaro",
    "ubwikorezi bwo mu mujyi",
    "imihindagurikire y'ibihe",
    "ubucuruzi buto",
    "imiyoborere myiza",
    "ubukerarugendo",
    "kubungabunga umuco",
    "serivisi za leta",
    "uburenganzira bw'abaturage",
]

TASK_SPECS = [
    {
        "task_type": "rw_instruction",
        "lang_mode": "rw",
        "instruction": "Write natural Kinyarwanda-only user requests.",
    },
    {
        "task_type": "followup_clarification",
        "lang_mode": "rw",
        "instruction": "Write realistic follow-up or clarification questions a user would ask after an earlier answer.",
    },
    {
        "task_type": "transformation",
        "lang_mode": "rw",
        "instruction": "Write requests that ask to summarize, rewrite, simplify, compare, classify, or extract.",
    },
    {
        "task_type": "noisy_input_robustness",
        "lang_mode": "rw_mixed",
        "instruction": "Write realistic messy user prompts with minor typos, informal phrasing, or mixed register, but still understandable.",
    },
    {
        "task_type": "code_switch_instruction",
        "lang_mode": "rw_mixed",
        "instruction": "Write realistic Rwanda-style code-switched prompts mixing mostly Kinyarwanda with a few English technical terms.",
    },
    {
        "task_type": "multilingual_retention",
        "lang_mode": "en",
        "instruction": "Write English user prompts from Rwanda context that should still be preserved by a multilingual assistant.",
    },
    {
        "task_type": "language_control",
        "lang_mode": "control",
        "instruction": "Write prompts that explicitly instruct the assistant which language or style to respond in.",
    },
    {
        "task_type": "structured_output",
        "lang_mode": "rw",
        "instruction": "Write requests asking for bullet points, steps, checklists, tables, or JSON-like structured answers.",
    },
    {
        "task_type": "open_domain_chat",
        "lang_mode": "rw",
        "instruction": "Write casual day-to-day user questions, not academic or overly formal.",
    },
]

SYSTEM_PROMPT = (
    "You are a senior native Kinyarwanda linguist and SFT data curator. "
    "Generate compact, high-quality user prompts only. "
    "Do not answer the prompts. "
    "Follow the requested topic and task style exactly. "
    "Output plain text only, one prompt per line, no numbering, no bullets, no prose before or after."
)

# Keep this block byte-stable across requests to maximize cached-input reuse.
FIXED_USER_PREFIX = "\n".join(
    [
        "Task: generate user prompts for SFT seed collection.",
        "Return plain text only, one prompt per line.",
        "No numbering. No bullets. No blank lines. No commentary.",
        "Keep each prompt realistic and concise, usually under 180 characters.",
        "Do not answer the prompts.",
        "Do not repeat or lightly paraphrase the same intent.",
        "Prefer everyday phrasing over textbook phrasing unless the task explicitly requires formal wording.",
    ]
)

SMART_PUNCT_MAP = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "«": '"',
        "»": '"',
        "‹": "'",
        "›": "'",
        "–": "-",
        "—": "-",
        "‑": "-",
        "‒": "-",
        "−": "-",
        "…": "...",
        "∕": "/",
        "⁄": "/",
        "：": ":",
        "，": ",",
        "；": ";",
        "（": "(",
        "）": ")",
        "？": "?",
        "！": "!",
        "\u00A0": " ",
        "\u2007": " ",
        "\u202F": " ",
    }
)

_LINE_PREFIX_RE = re.compile(
    r"^\s*(?:[-*•◦▪●]\s+|\(?\d+\)?[.)]?\s+|\(?[A-Za-z]\)?[.)]\s+|(?:Prompt|Question|User)\s*:\s+)",
    re.IGNORECASE,
)
_DROP_LINE_RE = re.compile(
    r"^\s*(?:```|json\s*$|here are\b|sure\b|below are\b|these are\b)",
    re.IGNORECASE,
)
_ZERO_WIDTH_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SFT prompt seed via OpenAI compact batching")
    p.add_argument("--output_jsonl", default="outputs/sft/prompts.seed.gpt54.jsonl")
    p.add_argument("--errors_jsonl", default="")
    p.add_argument("--state_path", default="")
    p.add_argument(
        "--dedup_db",
        default="outputs/sft/prompts.seed.shared.dedup.sqlite",
        help="Path to shared SQLite dedup DB. Reuse the same file across reruns.",
    )
    p.add_argument("--topics_file", default="")
    p.add_argument("--target", type=int, default=30000)
    p.add_argument("--prompts_per_request", type=int, default=10)
    p.add_argument("--model", default="gpt-5.4")
    p.add_argument("--api_key_env", default="OPENAI_API_KEY")
    p.add_argument("--env_file", default=".env")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max_output_tokens", type=int, default=0)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--retry_backoff_sec", type=float, default=1.5)
    p.add_argument("--request_timeout_sec", type=float, default=120.0)
    p.add_argument("--avoid_topic_recent", type=int, default=8)
    p.add_argument("--avoid_global_recent", type=int, default=8)
    p.add_argument("--avoid_pattern_topk", type=int, default=6)
    p.add_argument("--near_dup_threshold", type=float, default=0.9)
    p.add_argument("--near_dup_topic_limit", type=int, default=200)
    p.add_argument("--near_dup_global_limit", type=int, default=400)
    p.add_argument("--sleep_sec", type=float, default=0.0)
    p.add_argument("--print_every", type=int, default=10)
    p.add_argument("--cb_consecutive_429", type=int, default=8)
    p.add_argument("--cb_consecutive_fail", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith("'") and value.endswith("'")) or (
            value.startswith('"') and value.endswith('"')
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _load_topics(path: str) -> list[str]:
    if not path:
        return DEFAULT_TOPICS
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing topics file: {p}")
    topics: list[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        t = normalize_text(raw)
        if t:
            topics.append(t)
    if not topics:
        raise RuntimeError("Topics file is empty after normalization")
    return topics


def _load_state(path: Path) -> dict[str, object]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "topic_index": 0,
        "requests": 0,
        "generated": 0,
        "accepted": 0,
        "failed_calls": 0,
        "rejected_prompts": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": "",
    }


def _save_state(path: Path, state: dict[str, object]) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def _prompt_id(prompt: str) -> str:
    return sha1(prompt.encode("utf-8")).hexdigest()[:16]


def _collect_avoid_memory(
    dedup: PromptDedupIndex,
    topic: str,
    topic_recent: int,
    global_recent: int,
    pattern_topk: int,
) -> tuple[list[str], list[str]]:
    topic_prompts = dedup.recent_prompts(limit=max(0, topic_recent), topic=topic)
    global_prompts = dedup.recent_prompts(limit=max(0, global_recent), topic="")
    merged: list[str] = []
    seen: set[str] = set()
    for prompt in topic_prompts + global_prompts:
        key = normalize_text(prompt).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(normalize_text(prompt))
    patterns = dedup.frequent_patterns(limit=max(0, pattern_topk))
    return merged, patterns


def _build_user_prompt(
    *,
    topic: str,
    prompts_per_request: int,
    spec: dict[str, str],
    avoid_prompts: list[str],
    frequent_patterns: list[str],
) -> str:
    lines = [
        FIXED_USER_PREFIX,
        f"Topic: {topic}",
        f"Task type: {spec['task_type']}",
        f"Lang mode: {spec['lang_mode']}",
        f"Need exactly {prompts_per_request} prompts.",
        spec["instruction"],
    ]
    if avoid_prompts:
        lines.append("Avoid prompts too similar to these recent ones:")
        lines.extend([f"- {p}" for p in avoid_prompts[:12]])
    if frequent_patterns:
        lines.append("Avoid these common openings:")
        lines.extend([f"- {p}" for p in frequent_patterns[:8]])
    return "\n".join(lines)


def _parse_prompts(text: str) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        if _DROP_LINE_RE.match(raw):
            continue
        line = normalize_text(_LINE_PREFIX_RE.sub("", raw))
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        prompts.append(line)
    return prompts


def _normalize_prompt_text(prompt: str) -> str:
    prompt = _ZERO_WIDTH_RE.sub("", prompt)
    return normalize_text(prompt.translate(SMART_PUNCT_MAP))


def _validate_prompt(prompt: str) -> tuple[bool, str]:
    if not prompt:
        return False, "empty_prompt"
    if len(prompt) < 10:
        return False, "prompt_too_short"
    if len(prompt) > 260:
        return False, "prompt_too_long"
    return True, "ok"


def _weighted_specs() -> list[tuple[dict[str, str], int]]:
    weights = {
        "rw_instruction": 28,
        "open_domain_chat": 18,
        "followup_clarification": 12,
        "transformation": 12,
        "structured_output": 10,
        "code_switch_instruction": 8,
        "noisy_input_robustness": 5,
        "language_control": 4,
        "multilingual_retention": 3,
    }
    return [(spec, weights.get(spec["task_type"], 1)) for spec in TASK_SPECS]


def _sample_spec(rng: random.Random, emitted_counts: dict[str, int]) -> dict[str, str]:
    weighted = _weighted_specs()
    scored: list[tuple[float, dict[str, str]]] = []
    total_emitted = sum(emitted_counts.values())
    for spec, weight in weighted:
        actual = emitted_counts.get(spec["task_type"], 0)
        expected = total_emitted * (weight / sum(w for _, w in weighted))
        deficit = max(0.15, 1.0 + (expected - actual) / max(expected, 1.0))
        score = weight * deficit * (0.85 + 0.3 * rng.random())
        scored.append((score, spec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _sample_topic(rng: random.Random, topics: list[str], emitted_counts: dict[str, int]) -> str:
    if not topics:
        raise RuntimeError("No topics available")
    min_count = min(emitted_counts.get(topic, 0) for topic in topics)
    candidates = [topic for topic in topics if emitted_counts.get(topic, 0) <= min_count + 1]
    return rng.choice(candidates)


def _classify_failure(err: str) -> bool:
    return "http_429" in normalize_text(err).lower()


def _circuit_break_reason(
    *,
    consecutive_429: int,
    consecutive_fail: int,
    max_consecutive_429: int,
    max_consecutive_fail: int,
) -> str:
    if max_consecutive_429 > 0 and consecutive_429 >= max_consecutive_429:
        return "consecutive_429_limit_reached"
    if max_consecutive_fail > 0 and consecutive_fail >= max_consecutive_fail:
        return "consecutive_fail_limit_reached"
    return ""


def _iter_existing_prompts(path: Path) -> set[str]:
    prompts: set[str] = set()
    if not path.exists():
        return prompts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        prompt = normalize_text(obj.get("prompt", ""))
        if prompt:
            prompts.add(prompt.lower())
    return prompts


def _bootstrap_dedup(
    dedup: PromptDedupIndex,
    output_path: Path,
    source_name: str,
) -> None:
    if dedup.count() > 0:
        return
    existing = _iter_existing_prompts(output_path)
    if not existing:
        return
    ts = datetime.now(timezone.utc).isoformat()
    for key in existing:
        dedup.add_if_new(
            prompt_key=key,
            first_prompt=key,
            first_source=source_name,
            topic="",
            created_at=ts,
        )


def main() -> int:
    args = parse_args()
    _load_env_file(Path(args.env_file))
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing API key in env var: {args.api_key_env} (checked env file: {args.env_file})"
        )

    output_path = Path(args.output_jsonl)
    errors_path = Path(args.errors_jsonl) if args.errors_jsonl else output_path.with_suffix(".errors.jsonl")
    state_path = Path(args.state_path) if args.state_path else output_path.with_suffix(".state.json")
    dedup_db_path = Path(args.dedup_db) if args.dedup_db else output_path.with_suffix(".dedup.sqlite")

    topics = _load_topics(args.topics_file)
    state = _load_state(state_path)
    topic_index = int(state.get("topic_index", 0))
    source_name = output_path.name
    rng = random.Random(int(args.seed))
    emitted_task_counts: dict[str, int] = {}
    emitted_topic_counts: dict[str, int] = {}

    cfg = OpenAIConfig(
        model=args.model,
        api_key=api_key,
        temperature=float(args.temperature),
        max_output_tokens=(int(args.max_output_tokens) if int(args.max_output_tokens) > 0 else 400),
        max_retries=int(args.max_retries),
        retry_backoff_sec=float(args.retry_backoff_sec),
        request_timeout_sec=float(args.request_timeout_sec),
    )

    with PromptDedupIndex(dedup_db_path) as dedup:
        _bootstrap_dedup(dedup, output_path, source_name)
        consecutive_429 = 0
        consecutive_fail = 0

        while int(state.get("accepted", 0)) < int(args.target):
            topic = _sample_topic(rng, topics, emitted_topic_counts)
            spec = _sample_spec(rng, emitted_task_counts)
            accepted_before = int(state.get("accepted", 0))
            remaining = int(args.target) - accepted_before
            prompts_per_request = max(1, min(int(args.prompts_per_request), remaining))
            avoid_prompts, frequent_patterns = _collect_avoid_memory(
                dedup=dedup,
                topic=topic,
                topic_recent=int(args.avoid_topic_recent),
                global_recent=int(args.avoid_global_recent),
                pattern_topk=int(args.avoid_pattern_topk),
            )
            user_prompt = _build_user_prompt(
                topic=topic,
                prompts_per_request=prompts_per_request,
                spec=spec,
                avoid_prompts=avoid_prompts,
                frequent_patterns=frequent_patterns,
            )
            try:
                text = chat_completion(cfg, system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
                state["requests"] = int(state.get("requests", 0)) + 1
                prompts = _parse_prompts(text)
                state["generated"] = int(state.get("generated", 0)) + len(prompts)

                for prompt in prompts:
                    if int(state.get("accepted", 0)) >= int(args.target):
                        break
                    prompt = _normalize_prompt_text(prompt)
                    ok, reason = _validate_prompt(prompt)
                    if not ok:
                        state["rejected_prompts"] = int(state.get("rejected_prompts", 0)) + 1
                        append_jsonl(
                            errors_path,
                            {"topic": topic, "reason": reason, "prompt": prompt, "raw_output": text[:500]},
                        )
                        continue

                    near_dup, near_match = dedup.has_near_duplicate(
                        prompt=prompt,
                        topic=topic,
                        threshold=float(args.near_dup_threshold),
                        topic_limit=int(args.near_dup_topic_limit),
                        global_limit=int(args.near_dup_global_limit),
                    )
                    if near_dup:
                        state["rejected_prompts"] = int(state.get("rejected_prompts", 0)) + 1
                        append_jsonl(
                            errors_path,
                            {
                                "topic": topic,
                                "reason": "near_duplicate_prompt",
                                "prompt": prompt,
                                "matched_prompt": near_match,
                            },
                        )
                        continue

                    inserted = dedup.add_if_new(
                        prompt_key=prompt.lower(),
                        first_prompt=prompt,
                        first_source=source_name,
                        topic=topic,
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )
                    if not inserted:
                        state["rejected_prompts"] = int(state.get("rejected_prompts", 0)) + 1
                        append_jsonl(errors_path, {"topic": topic, "reason": "duplicate_prompt", "prompt": prompt})
                        continue

                    rec = {
                        "id": f"oa_{_prompt_id(prompt)}",
                        "prompt": prompt,
                        "task_type": spec["task_type"],
                        "lang_mode": spec["lang_mode"],
                        "topic": topic,
                        "source": "openai_prompt_seed_v1",
                        "teacher_model": args.model,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    append_jsonl(output_path, rec)
                    state["accepted"] = int(state.get("accepted", 0)) + 1
                    emitted_task_counts[spec["task_type"]] = emitted_task_counts.get(spec["task_type"], 0) + 1
                    emitted_topic_counts[topic] = emitted_topic_counts.get(topic, 0) + 1

                consecutive_429 = 0
                consecutive_fail = 0
            except OpenAIError as e:
                state["failed_calls"] = int(state.get("failed_calls", 0)) + 1
                is_429 = _classify_failure(str(e))
                consecutive_429 = consecutive_429 + 1 if is_429 else 0
                consecutive_fail += 1
                append_jsonl(
                    errors_path,
                    {
                        "topic": topic,
                        "reason": "openai_prompt_generation_error",
                        "error": str(e),
                    },
                )

            topic_index += 1
            state["topic_index"] = topic_index
            state["dedup_db_entries"] = dedup.count()
            _save_state(state_path, state)

            if topic_index % max(1, int(args.print_every)) == 0:
                requests = int(state.get("requests", 0))
                accepted = int(state.get("accepted", 0))
                print(
                    "accepted="
                    f"{accepted} requests={requests} generated={state['generated']} "
                    f"rejected={state['rejected_prompts']} failed_calls={state['failed_calls']} "
                    f"topic_index={state['topic_index']} dedup_entries={state['dedup_db_entries']} "
                    f"accepted_per_request={round(accepted / max(requests, 1), 3)} cb429={consecutive_429} cbfail={consecutive_fail}"
                )

            cb_reason = _circuit_break_reason(
                consecutive_429=consecutive_429,
                consecutive_fail=consecutive_fail,
                max_consecutive_429=int(args.cb_consecutive_429),
                max_consecutive_fail=int(args.cb_consecutive_fail),
            )
            if cb_reason:
                append_jsonl(
                    errors_path,
                    {
                        "reason": "circuit_breaker_stop",
                        "circuit_reason": cb_reason,
                        "accepted": state["accepted"],
                        "requests": state["requests"],
                        "generated": state["generated"],
                        "failed_calls": state["failed_calls"],
                        "topic_index": state["topic_index"],
                    },
                )
                print(f"circuit_breaker_stop: {cb_reason}")
                break

            if args.sleep_sec > 0:
                time.sleep(float(args.sleep_sec))

    _save_state(state_path, state)
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
