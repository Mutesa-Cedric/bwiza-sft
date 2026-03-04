#!/usr/bin/env python3
"""Hybrid chat-data pipeline: Gemini Flash dialogue generation + OpenAI organizer normalization."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha1
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dedup_index import PromptDedupIndex
from src.data.gemini_client import GeminiConfig, GeminiError, generate_text
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
]

FLASH_SYSTEM_PROMPT = (
    "Generate natural, realistic chat conversations for assistant fine-tuning. "
    "Return STRICT JSON only with no markdown or code fences."
)

ORGANIZER_SYSTEM_PROMPT = (
    "You repair malformed conversation JSON into valid JSON. "
    "Return only minified JSON. "
    "If unusable, return: {\"reject_reason\":\"<short_reason>\"}."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid chat seed pipeline (Flash + organizer)")
    p.add_argument("--output_prefix", default="outputs/sft/chat.seed.hybrid")
    p.add_argument("--topics_file", default="")
    p.add_argument("--target_dialogues", type=int, default=5000)
    p.add_argument("--gemini_model", default="gemini-3-flash-preview")
    p.add_argument("--organizer_model", default="gpt-5.2")
    p.add_argument("--env_file", default=".env")
    p.add_argument("--gemini_api_key_env", default="GEMINI_API_KEY")
    p.add_argument("--openai_api_key_env", default="OPENAI_API_KEY")
    p.add_argument("--dedup_db", default="outputs/sft/chat.seed.shared.dedup.sqlite")
    p.add_argument("--min_turns", type=int, default=3, help="Minimum user-assistant pairs.")
    p.add_argument("--max_turns", type=int, default=6, help="Maximum user-assistant pairs.")
    p.add_argument("--history_window_messages", type=int, default=8)
    p.add_argument("--sleep_sec", type=float, default=0.2)
    p.add_argument("--print_every", type=int, default=10)
    p.add_argument("--max_output_tokens", type=int, default=0, help="Gemini max output tokens; 0 disables cap.")
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--retry_backoff_sec", type=float, default=1.5)
    p.add_argument("--near_dup_threshold", type=float, default=0.9)
    p.add_argument("--near_dup_topic_limit", type=int, default=200)
    p.add_argument("--near_dup_global_limit", type=int, default=400)
    p.add_argument("--avoid_topic_recent", type=int, default=10)
    p.add_argument("--avoid_global_recent", type=int, default=10)
    p.add_argument("--avoid_pattern_topk", type=int, default=6)
    p.add_argument("--cb_consecutive_429", type=int, default=12)
    p.add_argument("--cb_consecutive_parse", type=int, default=12)
    p.add_argument("--cb_consecutive_fail", type=int, default=30)
    return p.parse_args()


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
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
        content = normalize_text(x.get("content", ""))
        if role not in {"user", "assistant"} or not content:
            continue
        out.append({"role": role, "content": content})
    return out


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
        prev = m["role"]
    return True, "ok"


def _conversation_signature(messages: list[dict[str, str]]) -> str:
    # Use first user turns plus first assistant answer to reduce false duplicate hits on common openings.
    users = [m["content"] for m in messages if m["role"] == "user"]
    assistants = [m["content"] for m in messages if m["role"] == "assistant"]
    parts: list[str] = []
    parts.extend(users[:2])
    if assistants:
        parts.append(assistants[0])
    sig = " || ".join(parts)
    return normalize_text(sig)


def _conversation_id(messages: list[dict[str, str]]) -> str:
    raw = " | ".join([f"{m['role']}:{m['content']}" for m in messages])
    return f"ch_{sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _infer_lang_mode(text: str) -> str:
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return "rw"
    en = {
        "the",
        "and",
        "is",
        "are",
        "to",
        "for",
        "with",
        "how",
        "what",
        "when",
        "where",
    }
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


def _out_path(prefix: Path, suffix: str) -> Path:
    # Treat prefix as literal prefix; do not strip ".hybrid" with Path.with_suffix().
    return Path(f"{prefix}.{suffix}")


def _build_flash_user_prompt(
    topic: str,
    *,
    min_turns: int,
    max_turns: int,
    avoid_prompts: list[str],
    frequent_patterns: list[str],
) -> str:
    avoid_block = ""
    if avoid_prompts:
        avoid_block += "Avoid conversations very similar to these starters:\n"
        avoid_block += "\n".join([f"- {p}" for p in avoid_prompts]) + "\n"
    if frequent_patterns:
        avoid_block += "Avoid overusing these openings:\n"
        avoid_block += "\n".join([f"- {p}" for p in frequent_patterns]) + "\n"
    return (
        "Generate one realistic day-to-day chat conversation.\n"
        f"Topic: {topic}\n"
        "Output one minified JSON object with exact schema:\n"
        "{\"messages\":[{\"role\":\"user|assistant\",\"content\":\"<text>\"}]}\n"
        "Rules:\n"
        f"- total turns between {min_turns} and {max_turns} user-assistant pairs\n"
        "- messages must alternate user/assistant and start with user\n"
        "- natural style, include occasional short and informal user messages\n"
        "- allow natural code-switching in some user messages\n"
        + avoid_block
        + "- output JSON only"
    )


def _build_organizer_user_prompt(
    *,
    topic: str,
    raw_text: str,
    failure_reason: str,
    min_turns: int,
    max_turns: int,
    avoid_prompts: list[str],
    frequent_patterns: list[str],
) -> str:
    avoid_block = ""
    if avoid_prompts:
        avoid_block += "Do not produce chats with starters similar to:\n"
        avoid_block += "\n".join([f"- {p}" for p in avoid_prompts]) + "\n"
    if frequent_patterns:
        avoid_block += "Avoid these common openings:\n"
        avoid_block += "\n".join([f"- {p}" for p in frequent_patterns]) + "\n"
    return (
        "Repair/normalize the raw output into valid conversation JSON.\n"
        f"Topic: {topic}\n"
        f"Validation failure: {failure_reason}\n"
        "Return exactly one minified JSON object in one of these forms:\n"
        "{\"messages\":[{\"role\":\"user|assistant\",\"content\":\"<text>\"}]}\n"
        "OR\n"
        "{\"reject_reason\":\"<short_reason>\"}\n"
        f"- conversation must contain {min_turns} to {max_turns} user-assistant pairs\n"
        "- must start user, end assistant, alternating roles\n"
        + avoid_block
        + "Raw output follows:\n"
        f"{raw_text}"
    )


def _classify_failure(err: str) -> tuple[bool, bool]:
    e = normalize_text(err).lower()
    is_429 = "http_429" in e
    is_parse = (
        "invalid_json" in e
        or "json_object_not_dict" in e
        or "no_json_object_found" in e
        or "messages_not_even" in e
        or "roles_not_alternating" in e
    )
    return is_429, is_parse


def _circuit_break_reason(
    *,
    consecutive_429: int,
    consecutive_parse: int,
    consecutive_fail: int,
    max_consecutive_429: int,
    max_consecutive_parse: int,
    max_consecutive_fail: int,
) -> str:
    if max_consecutive_429 > 0 and consecutive_429 >= max_consecutive_429:
        return "consecutive_429_limit_reached"
    if max_consecutive_parse > 0 and consecutive_parse >= max_consecutive_parse:
        return "consecutive_parse_limit_reached"
    if max_consecutive_fail > 0 and consecutive_fail >= max_consecutive_fail:
        return "consecutive_fail_limit_reached"
    return ""


def _load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "topic_index": 0,
        "flash_calls": 0,
        "organizer_calls": 0,
        "accepted_local": 0,
        "accepted_organized": 0,
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

    key = sig.lower()
    inserted = dedup.add_if_new(
        prompt_key=key,
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
            "task_type": "followup_clarification",
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

    gemini_key = os.environ.get(args.gemini_api_key_env, "").strip()
    if not gemini_key:
        raise RuntimeError(
            f"Missing Gemini key in env var: {args.gemini_api_key_env} (checked env file: {args.env_file})"
        )
    openai_key = os.environ.get(args.openai_api_key_env, "").strip()
    if not openai_key:
        raise RuntimeError(
            f"Missing OpenAI key in env var: {args.openai_api_key_env} (checked env file: {args.env_file})"
        )

    out_prefix = Path(args.output_prefix)
    dialogues_path = _out_path(out_prefix, "dialogues.jsonl")
    pairs_path = _out_path(out_prefix, "pairs.jsonl")
    rejects_path = _out_path(out_prefix, "rejects.jsonl")
    errors_path = _out_path(out_prefix, "errors.jsonl")
    raw_path = _out_path(out_prefix, "raw.jsonl")
    state_path = _out_path(out_prefix, "state.json")

    topics = _load_topics(args.topics_file)
    state = _load_state(state_path)
    topic_index = int(state.get("topic_index", 0))

    flash_cfg = GeminiConfig(
        model=args.gemini_model,
        api_key=gemini_key,
        temperature=0.2,
        top_p=0.9,
        max_output_tokens=(int(args.max_output_tokens) if int(args.max_output_tokens) > 0 else None),
        max_retries=int(args.max_retries),
        retry_backoff_sec=float(args.retry_backoff_sec),
    )
    org_cfg = OpenAIConfig(
        model=args.organizer_model,
        api_key=openai_key,
        temperature=0.0,
        max_output_tokens=800,
        max_retries=int(args.max_retries),
        retry_backoff_sec=float(args.retry_backoff_sec),
    )

    consecutive_429 = 0
    consecutive_parse = 0
    consecutive_fail = 0

    with PromptDedupIndex(Path(args.dedup_db)) as dedup:
        while (int(state.get("accepted_local", 0)) + int(state.get("accepted_organized", 0))) < int(args.target_dialogues):
            topic = topics[topic_index % len(topics)]
            accepted_before = int(state.get("accepted_local", 0)) + int(state.get("accepted_organized", 0))
            avoid_prompts, frequent_patterns = _collect_avoid_memory(
                dedup,
                topic,
                topic_recent=int(args.avoid_topic_recent),
                global_recent=int(args.avoid_global_recent),
                pattern_topk=int(args.avoid_pattern_topk),
            )
            raw_text = ""
            local_reason = ""
            try:
                raw_text = generate_text(
                    flash_cfg,
                    user_prompt=_build_flash_user_prompt(
                        topic,
                        min_turns=int(args.min_turns),
                        max_turns=int(args.max_turns),
                        avoid_prompts=avoid_prompts,
                        frequent_patterns=frequent_patterns,
                    ),
                    system_prompt=FLASH_SYSTEM_PROMPT,
                )
                state["flash_calls"] = int(state.get("flash_calls", 0)) + 1
                append_jsonl(
                    raw_path,
                    {
                        "topic": topic,
                        "raw_output": raw_text,
                        "model": args.gemini_model,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )

                try:
                    payload = _extract_json_block(raw_text)
                    msgs = _extract_messages(payload)
                    ok, reason = _validate_messages(
                        msgs,
                        min_turns=int(args.min_turns),
                        max_turns=int(args.max_turns),
                    )
                    if ok:
                        accepted, reason2, n_pairs = _accept_dialogue(
                            messages=msgs,
                            topic=topic,
                            source="flash_local",
                            model_name=args.gemini_model,
                            dedup=dedup,
                            near_dup_threshold=float(args.near_dup_threshold),
                            near_dup_topic_limit=int(args.near_dup_topic_limit),
                            near_dup_global_limit=int(args.near_dup_global_limit),
                            dialogues_path=dialogues_path,
                            pairs_path=pairs_path,
                            history_window_messages=int(args.history_window_messages),
                        )
                        if accepted:
                            state["accepted_local"] = int(state.get("accepted_local", 0)) + 1
                            state["pairs_written"] = int(state.get("pairs_written", 0)) + n_pairs
                        else:
                            local_reason = reason2
                    else:
                        local_reason = reason
                except (ValueError, json.JSONDecodeError) as e:
                    local_reason = f"invalid_json_output:{e}"

                if local_reason and ((int(state.get("accepted_local", 0)) + int(state.get("accepted_organized", 0))) < int(args.target_dialogues)):
                    org_text = chat_completion(
                        org_cfg,
                        system_prompt=ORGANIZER_SYSTEM_PROMPT,
                        user_prompt=_build_organizer_user_prompt(
                            topic=topic,
                            raw_text=raw_text,
                            failure_reason=local_reason,
                            min_turns=int(args.min_turns),
                            max_turns=int(args.max_turns),
                            avoid_prompts=avoid_prompts,
                            frequent_patterns=frequent_patterns,
                        ),
                    )
                    state["organizer_calls"] = int(state.get("organizer_calls", 0)) + 1
                    org_payload = _extract_json_block(org_text)
                    if "reject_reason" in org_payload:
                        state["rejected"] = int(state.get("rejected", 0)) + 1
                        append_jsonl(
                            rejects_path,
                            {
                                "topic": topic,
                                "reason": normalize_text(org_payload.get("reject_reason", "organizer_reject")),
                                "raw_output": raw_text,
                                "organizer_output": org_text,
                            },
                        )
                    else:
                        org_msgs = _extract_messages(org_payload)
                        ok, reason = _validate_messages(
                            org_msgs,
                            min_turns=int(args.min_turns),
                            max_turns=int(args.max_turns),
                        )
                        if not ok:
                            state["rejected"] = int(state.get("rejected", 0)) + 1
                            append_jsonl(
                                rejects_path,
                                {
                                    "topic": topic,
                                    "reason": reason,
                                    "raw_output": raw_text,
                                    "organizer_output": org_text,
                                },
                            )
                        else:
                            accepted, reason2, n_pairs = _accept_dialogue(
                                messages=org_msgs,
                                topic=topic,
                                source="organized_gpt",
                                model_name=args.organizer_model,
                                dedup=dedup,
                                near_dup_threshold=float(args.near_dup_threshold),
                                near_dup_topic_limit=int(args.near_dup_topic_limit),
                                near_dup_global_limit=int(args.near_dup_global_limit),
                                dialogues_path=dialogues_path,
                                pairs_path=pairs_path,
                                history_window_messages=int(args.history_window_messages),
                            )
                            if accepted:
                                state["accepted_organized"] = int(state.get("accepted_organized", 0)) + 1
                                state["pairs_written"] = int(state.get("pairs_written", 0)) + n_pairs
                            else:
                                state["rejected"] = int(state.get("rejected", 0)) + 1
                                append_jsonl(
                                    rejects_path,
                                    {
                                        "topic": topic,
                                        "reason": reason2,
                                        "raw_output": raw_text,
                                        "organizer_output": org_text,
                                    },
                                )

            except (GeminiError, OpenAIError, ValueError, json.JSONDecodeError) as e:
                state["failed_calls"] = int(state.get("failed_calls", 0)) + 1
                is_429, is_parse = _classify_failure(str(e))
                consecutive_429 = consecutive_429 + 1 if is_429 else 0
                consecutive_parse = consecutive_parse + 1 if is_parse else 0
                consecutive_fail += 1
                append_jsonl(
                    errors_path,
                    {
                        "topic": topic,
                        "reason": "chat_hybrid_error",
                        "error": str(e),
                        "raw_output": raw_text,
                    },
                )

            accepted_after = int(state.get("accepted_local", 0)) + int(state.get("accepted_organized", 0))
            if accepted_after > accepted_before:
                consecutive_429 = 0
                consecutive_parse = 0
                consecutive_fail = 0

            topic_index += 1
            state["topic_index"] = topic_index
            state["dedup_db_entries"] = dedup.count()
            _save_state(state_path, state)

            if topic_index % max(1, int(args.print_every)) == 0:
                print(
                    "accepted_local="
                    f"{state['accepted_local']} accepted_organized={state['accepted_organized']} "
                    f"pairs_written={state['pairs_written']} rejected={state['rejected']} failed_calls={state['failed_calls']} "
                    f"topic_index={state['topic_index']} dedup_entries={state['dedup_db_entries']} "
                    f"cb429={consecutive_429} cbparse={consecutive_parse} cbfail={consecutive_fail}"
                )

            cb_reason = _circuit_break_reason(
                consecutive_429=consecutive_429,
                consecutive_parse=consecutive_parse,
                consecutive_fail=consecutive_fail,
                max_consecutive_429=int(args.cb_consecutive_429),
                max_consecutive_parse=int(args.cb_consecutive_parse),
                max_consecutive_fail=int(args.cb_consecutive_fail),
            )
            if cb_reason:
                append_jsonl(
                    errors_path,
                    {
                        "reason": "circuit_breaker_stop",
                        "circuit_reason": cb_reason,
                        "topic_index": state["topic_index"],
                        "failed_calls": state["failed_calls"],
                        "accepted_local": state["accepted_local"],
                        "accepted_organized": state["accepted_organized"],
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
