"""Persistent dedup index for prompt generation pipelines."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import re
from collections import Counter


_WORD_RE = re.compile(r"[\\w']+", re.UNICODE)


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    uni = len(a | b)
    return inter / uni if uni else 0.0


class PromptDedupIndex:
    """SQLite-backed prompt dedup index safe for resumable runs."""

    def __init__(self, db_path: str | Path, timeout_sec: float = 30.0) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=timeout_sec)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seen_prompts (
                prompt_key TEXT PRIMARY KEY,
                first_prompt TEXT NOT NULL,
                first_source TEXT NOT NULL,
                topic TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            """
        )
        # Migration for older DBs without topic column.
        try:
            self._conn.execute("ALTER TABLE seen_prompts ADD COLUMN topic TEXT NOT NULL DEFAULT '';")
        except sqlite3.OperationalError:
            pass
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_seen_prompts_created_at ON seen_prompts(created_at DESC);"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_seen_prompts_topic ON seen_prompts(topic);")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PromptDedupIndex":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def add_if_new(
        self,
        prompt_key: str,
        first_prompt: str,
        first_source: str,
        topic: str,
        created_at: str,
    ) -> bool:
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO seen_prompts(prompt_key, first_prompt, first_source, topic, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (prompt_key, first_prompt, first_source, topic, created_at),
        )
        self._conn.commit()
        return int(cur.rowcount) == 1

    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM seen_prompts")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def recent_prompts(self, limit: int, topic: str = "") -> list[str]:
        if limit <= 0:
            return []
        if topic:
            cur = self._conn.execute(
                """
                SELECT first_prompt FROM seen_prompts
                WHERE topic = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (topic, int(limit)),
            )
        else:
            cur = self._conn.execute(
                """
                SELECT first_prompt FROM seen_prompts
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (int(limit),),
            )
        return [str(r[0]) for r in cur.fetchall() if r and r[0]]

    def frequent_patterns(self, limit: int, prefix_words: int = 4, sample_size: int = 5000) -> list[str]:
        if limit <= 0:
            return []
        cur = self._conn.execute(
            """
            SELECT first_prompt FROM seen_prompts
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (int(sample_size),),
        )
        c: Counter[str] = Counter()
        for row in cur.fetchall():
            if not row or not row[0]:
                continue
            parts = str(row[0]).strip().lower().split()
            if len(parts) < 2:
                continue
            prefix = " ".join(parts[: min(prefix_words, len(parts))])
            c[prefix] += 1
        # keep meaningful repeated prefixes only
        items = [k for k, v in c.most_common() if v >= 2]
        return items[: int(limit)]

    def has_near_duplicate(
        self,
        prompt: str,
        topic: str = "",
        threshold: float = 0.9,
        topic_limit: int = 200,
        global_limit: int = 400,
    ) -> tuple[bool, str]:
        toks = _tokenize(prompt)
        if not toks:
            return False, ""

        candidates: list[str] = []
        if topic:
            candidates.extend(self.recent_prompts(limit=topic_limit, topic=topic))
        candidates.extend(self.recent_prompts(limit=global_limit, topic=""))

        seen: set[str] = set()
        deduped_candidates: list[str] = []
        for c in candidates:
            key = c.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped_candidates.append(c)

        best_score = 0.0
        best_prompt = ""
        for cand in deduped_candidates:
            s = _jaccard(toks, _tokenize(cand))
            if s > best_score:
                best_score = s
                best_prompt = cand
            if s >= threshold:
                return True, cand
        return False, best_prompt if best_score > 0 else ""
