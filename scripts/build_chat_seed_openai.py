#!/usr/bin/env python3
"""Generate multi-turn chat seed data via OpenAI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha1
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

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

CHAT_STYLE_SPECS = [
    {
        "style": "casual_day_to_day",
        "instruction": "Create a realistic everyday help conversation with practical user needs.",
    },
    {
        "style": "followup_resolution",
        "instruction": "Create a conversation where the user asks a follow-up after an initial assistant answer.",
    },
    {
        "style": "code_switch_chat",
        "instruction": "Create a conversation with natural Rwanda-style code-switching in some user turns.",
    },
    {
        "style": "short_mobile_chat",
        "instruction": "Create a conversation with some short, phone-like user turns and concise assistant replies.",
    },
    {
        "style": "guided_decision",
        "instruction": "Create a conversation where the user is uncertain and the assistant helps them choose or plan.",
    },
]

SYSTEM_PROMPT = (
    "You are a senior native Kinyarwanda linguist and SFT conversation curator. "
    "Generate one realistic assistant chat conversation for fine-tuning. "
    "Return strict JSON only, no markdown, no code fences, no prose."
)

FIXED_USER_PREFIX = "\n".join(
    [
        "Task: generate exactly one realistic multi-turn assistant conversation for SFT.",
        "Return one minified JSON object only.",
        "Schema: {\"messages\":[{\"role\":\"user|assistant\",\"content\":\"<text>\"}]}",
        "Messages must alternate user then assistant.",
        "Conversation must start with user and end with assistant.",
        "Keep the chat realistic, grounded, and natural for Rwanda context.",
        "Keep replies concise; avoid unnecessarily long assistant messages.",
        "Avoid robotic repetition and textbook phrasing unless the topic needs it.",
        "Allow occasional code-switching only when natural for the chosen style.",
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
_ZERO_WIDTH_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build multi-turn chat seed via OpenAI")
    p.add_argument("--output_prefix", default="outputs/sft/chat.seed.openai")
    p.add_argument(
        "--shared_output_prefix",
        default="",
        help="Optional shared prefix for canonical JSONL outputs. State remains under output_prefix.",
    )
    p.add_argument("--topics_file", default="")
    p.add_argument("--target_dialogues", type=int, default=5000)
    p.add_argument("--topic_index_start", type=int, default=0)
    p.add_argument("--model", default="gpt-5.4")
    p.add_argument("--api_key_env", default="OPENAI_API_KEY")
    p.add_argument("--env_file", default=".env")
    p.add_argument("--dedup_db", default="outputs/sft/chat.seed.shared.dedup.sqlite")
    p.add_argument("--min_turns", type=int, default=2)
    p.add_argument("--max_turns", type=int, default=4)
    p.add_argument("--history_window_messages", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max_output_tokens", type=int, default=0)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--retry_backoff_sec", type=float, default=1.5)
    p.add_argument("--request_timeout_sec", type=float, default=120.0)
    p.add_argument("--avoid_topic_recent", type=int, default=10)
    p.add_argument("--avoid_global_recent", type=int, default=10)
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


def _out_path(prefix: Path, suffix: str) -> Path:
    return Path(f"{prefix}.{suffix}")


def _load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "topic_index": 0,
        "requests": 0,
        "accepted": 0,
        "rejected": 0,
        "failed_calls": 0,
        "pairs_written": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": "",
    }


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def _extract_json_block(text: str) -> dict[str, Any]:
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
        if s.startswith("json"):
            s = s[4:].strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("no_json_object_found")
    block = s[start : end + 1]
    repaired = re.sub(r",(\s*[}\]])", r"\1", block)
    obj = json.loads(repaired)
    if not isinstance(obj, dict):
        raise ValueError("json_object_not_dict")
    return obj


def _extract_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    raw = payload.get("messages")
    if not isinstance(raw, list):
        raw = payload.get("conversation")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for x in raw:
        if not isinstance(x, dict):
            continue
        role = normalize_text(x.get("role", "")).lower()
        content = _normalize_chat_text(x.get("content", ""))
        if role not in {"user", "assistant"} or not content:
            continue
        out.append({"role": role, "content": content})
    return out


def _normalize_chat_text(value: str) -> str:
    value = _ZERO_WIDTH_RE.sub("", str(value))
    return normalize_text(value.translate(SMART_PUNCT_MAP))


def _validate_messages(messages: list[dict[str, str]], *, min_turns: int, max_turns: int) -> tuple[bool, str]:
    if not messages:
        return False, "empty_messages"
    if messages[0]["role"] != "user":
        return False, "must_start_user"
    if messages[-1]["role"] != "assistant":
        return False, "must_end_assistant"
    if len(messages) % 2 != 0:
        return False, "messages_not_even"
    turns = len(messages) // 2
    if turns < min_turns:
        return False, "too_few_turns"
    if turns > max_turns:
        return False, "too_many_turns"
    prev = None
    for m in messages:
        if prev == m["role"]:
            return False, "roles_not_alternating"
        if len(m["content"]) < 2:
            return False, "content_too_short"
        if any(ord(ch) > 127 for ch in m["content"]):
            return False, "non_ascii_content"
        prev = m["role"]
    return True, "ok"


def _conversation_signature(messages: list[dict[str, str]]) -> str:
    users = [m["content"] for m in messages if m["role"] == "user"]
    assistants = [m["content"] for m in messages if m["role"] == "assistant"]
    parts: list[str] = []
    parts.extend(users[:2])
    if assistants:
        parts.append(assistants[0])
    return normalize_text(" || ".join(parts))


def _conversation_id(messages: list[dict[str, str]]) -> str:
    raw = " | ".join([f"{m['role']}:{m['content']}" for m in messages])
    return f"ch_{sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _infer_lang_mode(text: str) -> str:
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return "rw"
    en = {"the", "and", "is", "are", "to", "for", "with", "how", "what", "when", "where"}
    en_ratio = sum(1 for w in words if w in en) / len(words)
    return "rw_mixed" if en_ratio >= 0.12 else "rw"


def _history_to_prompt(messages: list[dict[str, str]], *, window: int) -> str:
    clipped = messages[-window:] if window > 0 else messages
    lines = []
    for m in clipped:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def _collect_avoid_memory(
    dedup: PromptDedupIndex,
    topic: str,
    *,
    topic_recent: int,
    global_recent: int,
    pattern_topk: int,
) -> tuple[list[str], list[str]]:
    topic_prompts = dedup.recent_prompts(limit=max(0, topic_recent), topic=topic)
    global_prompts = dedup.recent_prompts(limit=max(0, global_recent), topic="")
    merged: list[str] = []
    seen: set[str] = set()
    for p in topic_prompts + global_prompts:
        key = normalize_text(p).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(normalize_text(p))
    patterns = dedup.frequent_patterns(limit=max(0, pattern_topk))
    return merged, patterns


def _build_user_prompt(
    *,
    topic: str,
    style_spec: dict[str, str],
    min_turns: int,
    max_turns: int,
    avoid_prompts: list[str],
    frequent_patterns: list[str],
) -> str:
    lines = [
        FIXED_USER_PREFIX,
        f"Topic: {topic}",
        f"Conversation style: {style_spec['style']}",
        style_spec["instruction"],
        f"Generate between {min_turns} and {max_turns} user-assistant pairs.",
        "Use helpful assistant replies that are concise but complete.",
    ]
    if avoid_prompts:
        lines.append("Avoid conversations too similar to these recent starters:")
        lines.extend([f"- {p}" for p in avoid_prompts[:6]])
    if frequent_patterns:
        lines.append("Avoid overusing these openings:")
        lines.extend([f"- {p}" for p in frequent_patterns[:4]])
    return "\n".join(lines)


def _weighted_styles() -> list[tuple[dict[str, str], int]]:
    weights = {
        "casual_day_to_day": 30,
        "followup_resolution": 24,
        "guided_decision": 18,
        "short_mobile_chat": 16,
        "code_switch_chat": 12,
    }
    return [(spec, weights.get(spec["style"], 1)) for spec in CHAT_STYLE_SPECS]


def _sample_style(rng: random.Random, emitted_counts: dict[str, int]) -> dict[str, str]:
    weighted = _weighted_styles()
    scored: list[tuple[float, dict[str, str]]] = []
    total_emitted = sum(emitted_counts.values())
    total_weight = sum(w for _, w in weighted)
    for spec, weight in weighted:
        actual = emitted_counts.get(spec["style"], 0)
        expected = total_emitted * (weight / total_weight)
        deficit = max(0.15, 1.0 + (expected - actual) / max(expected, 1.0))
        score = weight * deficit * (0.85 + 0.3 * rng.random())
        scored.append((score, spec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _sample_topic(rng: random.Random, topics: list[str], emitted_counts: dict[str, int]) -> str:
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


def _accept_dialogue(
    *,
    messages: list[dict[str, str]],
    topic: str,
    source: str,
    model_name: str,
    dedup: PromptDedupIndex,
    near_dup_threshold: float,
    near_dup_topic_limit: int,
    near_dup_global_limit: int,
    dialogues_path: Path,
    pairs_path: Path,
    history_window_messages: int,
) -> tuple[bool, str, int]:
    sig = _conversation_signature(messages)
    if not sig:
        return False, "empty_signature", 0
    near_dup, near_match = dedup.has_near_duplicate(
        prompt=sig,
        topic=topic,
        threshold=near_dup_threshold,
        topic_limit=near_dup_topic_limit,
        global_limit=near_dup_global_limit,
    )
    if near_dup:
        return False, f"near_duplicate:{near_match}", 0

    inserted = dedup.add_if_new(
        prompt_key=sig.lower(),
        first_prompt=sig,
        first_source=source,
        topic=topic,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    if not inserted:
        return False, "duplicate_signature", 0

    conv_id = _conversation_id(messages)
    dialogue_rec = {
        "id": conv_id,
        "conversation_id": conv_id,
        "topic": topic,
        "messages": messages,
        "source": source,
        "teacher_model": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl(dialogues_path, dialogue_rec)

    pairs_written = 0
    history: list[dict[str, str]] = []
    turn = 0
    for m in messages:
        if m["role"] == "user":
            history.append(m)
            continue
        turn += 1
        prompt = _history_to_prompt(history, window=history_window_messages)
        pair_rec = {
            "id": f"{conv_id}_t{turn:02d}",
            "conversation_id": conv_id,
            "turn": turn,
            "topic": topic,
            "prompt": prompt,
            "response": m["content"],
            "task_type": "chat_multiturn",
            "content_type": "chat_multiturn",
            "lang_mode": _infer_lang_mode(f"{prompt} {m['content']}"),
            "source": source,
            "teacher_model": model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        append_jsonl(pairs_path, pair_rec)
        pairs_written += 1
        history.append(m)

    return True, "ok", pairs_written


def main() -> int:
    args = parse_args()
    _load_env_file(Path(args.env_file))
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing API key in env var: {args.api_key_env} (checked env file: {args.env_file})"
        )

    out_prefix = Path(args.output_prefix)
    shared_prefix = Path(args.shared_output_prefix) if args.shared_output_prefix else out_prefix
    dialogues_path = _out_path(shared_prefix, "dialogues.jsonl")
    pairs_path = _out_path(shared_prefix, "pairs.jsonl")
    rejects_path = _out_path(shared_prefix, "rejects.jsonl")
    errors_path = _out_path(shared_prefix, "errors.jsonl")
    state_path = _out_path(out_prefix, "state.json")

    topics = _load_topics(args.topics_file)
    state = _load_state(state_path)
    if not state_path.exists():
        state["topic_index"] = max(0, int(args.topic_index_start))
    rng = random.Random(int(args.seed))
    emitted_topic_counts: dict[str, int] = {}
    emitted_style_counts: dict[str, int] = {}

    cfg = OpenAIConfig(
        model=args.model,
        api_key=api_key,
        temperature=float(args.temperature),
        max_output_tokens=(int(args.max_output_tokens) if int(args.max_output_tokens) > 0 else 2500),
        max_retries=int(args.max_retries),
        retry_backoff_sec=float(args.retry_backoff_sec),
        request_timeout_sec=float(args.request_timeout_sec),
    )

    with PromptDedupIndex(Path(args.dedup_db)) as dedup:
        consecutive_429 = 0
        consecutive_fail = 0

        while int(state.get("accepted", 0)) < int(args.target_dialogues):
            topic = _sample_topic(rng, topics, emitted_topic_counts)
            style_spec = _sample_style(rng, emitted_style_counts)
            avoid_prompts, frequent_patterns = _collect_avoid_memory(
                dedup,
                topic,
                topic_recent=int(args.avoid_topic_recent),
                global_recent=int(args.avoid_global_recent),
                pattern_topk=int(args.avoid_pattern_topk),
            )
            try:
                text = chat_completion(
                    cfg,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=_build_user_prompt(
                        topic=topic,
                        style_spec=style_spec,
                        min_turns=int(args.min_turns),
                        max_turns=int(args.max_turns),
                        avoid_prompts=avoid_prompts,
                        frequent_patterns=frequent_patterns,
                    ),
                )
                state["requests"] = int(state.get("requests", 0)) + 1
                payload = _extract_json_block(text)
                messages = _extract_messages(payload)
                ok, reason = _validate_messages(
                    messages,
                    min_turns=int(args.min_turns),
                    max_turns=int(args.max_turns),
                )
                if not ok:
                    state["rejected"] = int(state.get("rejected", 0)) + 1
                    append_jsonl(
                        rejects_path,
                        {"topic": topic, "style": style_spec["style"], "reason": reason, "raw_output": text[:1200]},
                    )
                else:
                    accepted, reason2, n_pairs = _accept_dialogue(
                        messages=messages,
                        topic=topic,
                        source="openai_chat_seed_v1",
                        model_name=args.model,
                        dedup=dedup,
                        near_dup_threshold=float(args.near_dup_threshold),
                        near_dup_topic_limit=int(args.near_dup_topic_limit),
                        near_dup_global_limit=int(args.near_dup_global_limit),
                        dialogues_path=dialogues_path,
                        pairs_path=pairs_path,
                        history_window_messages=int(args.history_window_messages),
                    )
                    if accepted:
                        state["accepted"] = int(state.get("accepted", 0)) + 1
                        state["pairs_written"] = int(state.get("pairs_written", 0)) + n_pairs
                        emitted_topic_counts[topic] = emitted_topic_counts.get(topic, 0) + 1
                        emitted_style_counts[style_spec["style"]] = emitted_style_counts.get(style_spec["style"], 0) + 1
                    else:
                        state["rejected"] = int(state.get("rejected", 0)) + 1
                        append_jsonl(
                            rejects_path,
                            {"topic": topic, "style": style_spec["style"], "reason": reason2, "raw_output": text[:1200]},
                        )
                consecutive_429 = 0
                consecutive_fail = 0
            except (OpenAIError, ValueError, json.JSONDecodeError) as e:
                state["failed_calls"] = int(state.get("failed_calls", 0)) + 1
                is_429 = _classify_failure(str(e))
                consecutive_429 = consecutive_429 + 1 if is_429 else 0
                consecutive_fail += 1
                append_jsonl(
                    errors_path,
                    {"topic": topic, "style": style_spec["style"], "reason": "openai_chat_generation_error", "error": str(e)},
                )

            state["topic_index"] = int(state.get("topic_index", 0)) + 1
            state["dedup_db_entries"] = dedup.count()
            _save_state(state_path, state)

            if int(state["topic_index"]) % max(1, int(args.print_every)) == 0:
                print(
                    "accepted="
                    f"{state['accepted']} requests={state['requests']} pairs_written={state['pairs_written']} "
                    f"rejected={state['rejected']} failed_calls={state['failed_calls']} "
                    f"topic_index={state['topic_index']} dedup_entries={state['dedup_db_entries']} "
                    f"accepted_per_request={round(int(state['accepted']) / max(int(state['requests']), 1), 3)} "
                    f"cb429={consecutive_429} cbfail={consecutive_fail}"
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
