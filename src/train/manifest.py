"""Run manifest for SFT runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


@dataclass
class RunManifest:
    run_id: str
    run_name: str
    git_commit: str
    config_fingerprint: str
    dataset_fingerprint: str
    tokenizer_name: str
    model_name: str
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ended_at: str = ""
    status: str = "running"
    supervised_tokens_seen: int = 0
    global_step: int = 0
    checkpoints: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def finish(self, status: str, supervised_tokens_seen: int, global_step: int) -> None:
        self.status = status
        self.supervised_tokens_seen = supervised_tokens_seen
        self.global_step = global_step
        self.ended_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
