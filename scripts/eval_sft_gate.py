#!/usr/bin/env python3
"""Apply strict gate on eval_sft report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.gating import GateConfig, evaluate_gate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SFT promotion gate")
    p.add_argument("--eval_report", required=True)
    p.add_argument("--output", default="outputs/reports/eval_sft_gate.json")
    p.add_argument("--require_better_val_ppl", action="store_true", default=False)
    p.add_argument("--require_better_test_ppl", action="store_true", default=False)
    p.add_argument("--max_english_drift_delta", type=float, default=0.0)
    p.add_argument("--min_rw_marker_density_delta", type=float, default=0.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rp = Path(args.eval_report)
    if not rp.exists():
        raise FileNotFoundError(f"Missing eval report: {rp}")

    report = json.loads(rp.read_text(encoding="utf-8"))
    cfg = GateConfig(
        require_better_val_ppl=bool(args.require_better_val_ppl),
        require_better_test_ppl=bool(args.require_better_test_ppl),
        max_english_drift_delta=float(args.max_english_drift_delta),
        min_rw_marker_density_delta=float(args.min_rw_marker_density_delta),
    )
    decision = evaluate_gate(report, cfg)

    out = {
        "passed": decision.passed,
        "checks": decision.checks,
        "reasons": decision.reasons,
        "delta": decision.delta,
        "config": {
            "require_better_val_ppl": cfg.require_better_val_ppl,
            "require_better_test_ppl": cfg.require_better_test_ppl,
            "max_english_drift_delta": cfg.max_english_drift_delta,
            "min_rw_marker_density_delta": cfg.min_rw_marker_density_delta,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if decision.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
