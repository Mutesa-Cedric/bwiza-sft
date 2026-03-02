"""Lightweight W&B wrapper with local JSONL fallback."""

from __future__ import annotations

from pathlib import Path
import json


class WandbLogger:
    def __init__(
        self,
        enabled: bool,
        project: str,
        run_name: str,
        run_group: str,
        mode: str = "online",
        output_dir: str = "outputs",
        config: dict | None = None,
    ) -> None:
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.local_log = self.output_dir / "metrics.jsonl"
        self._run = None

        if enabled:
            try:
                import wandb

                self._run = wandb.init(
                    project=project,
                    name=run_name,
                    group=run_group,
                    mode=mode,
                    config=config or {},
                )
            except Exception:
                self._run = None

    def log(self, data: dict, step: int | None = None) -> None:
        if self._run is not None:
            try:
                self._run.log(data, step=step)
            except Exception:
                pass

        record = dict(data)
        if step is not None:
            record["step"] = int(step)
        with self.local_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def finish(self) -> None:
        if self._run is not None:
            try:
                self._run.finish()
            except Exception:
                pass
