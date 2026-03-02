"""Checkpoint helpers for resumable SFT runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re


_STEP_RE = re.compile(r"^step-(\d+)$")


@dataclass(frozen=True)
class ResumeState:
    checkpoint_dir: str
    global_step: int
    supervised_tokens_seen: int


def checkpoint_dir(output_dir: str | Path, global_step: int) -> Path:
    return Path(output_dir) / "checkpoints" / f"step-{global_step:08d}"


def list_checkpoints(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir) / "checkpoints"
    if not root.exists():
        return []
    items: list[tuple[int, Path]] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = _STEP_RE.match(p.name)
        if not m:
            continue
        items.append((int(m.group(1)), p))
    items.sort(key=lambda x: x[0])
    return [p for _, p in items]


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    ckpts = list_checkpoints(output_dir)
    return ckpts[-1] if ckpts else None


def save_checkpoint(
    output_dir: str | Path,
    model,
    tokenizer,
    optimizer,
    scheduler,
    global_step: int,
    supervised_tokens_seen: int,
) -> Path:
    import torch

    ckpt = checkpoint_dir(output_dir, global_step)
    tmp = ckpt.with_suffix(".tmp")
    tmp.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(tmp)
    tokenizer.save_pretrained(tmp)

    state = {
        "global_step": int(global_step),
        "supervised_tokens_seen": int(supervised_tokens_seen),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(state, tmp / "trainer_state.pt")

    meta = {
        "global_step": int(global_step),
        "supervised_tokens_seen": int(supervised_tokens_seen),
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if ckpt.exists():
        raise FileExistsError(f"Checkpoint already exists: {ckpt}")
    tmp.rename(ckpt)
    return ckpt


def load_resume_state(checkpoint_path: str | Path, optimizer, scheduler) -> ResumeState:
    import torch

    ckpt = Path(checkpoint_path)
    state_path = ckpt / "trainer_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing trainer state: {state_path}")

    state = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])

    return ResumeState(
        checkpoint_dir=str(ckpt),
        global_step=int(state.get("global_step", 0)),
        supervised_tokens_seen=int(state.get("supervised_tokens_seen", 0)),
    )
