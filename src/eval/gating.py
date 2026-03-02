"""Promotion gate logic for SFT model selection."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class GateConfig:
    require_better_val_ppl: bool = True
    require_better_test_ppl: bool = True
    max_english_drift_delta: float = 0.0
    min_rw_marker_density_delta: float = 0.0


@dataclass
class GateDecision:
    passed: bool
    checks: dict[str, bool]
    reasons: list[str]
    delta: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_gate(report: dict[str, Any], cfg: GateConfig) -> GateDecision:
    d = report.get("delta", {})
    val_ppl_delta = float(d.get("val_ppl_delta", 0.0))
    test_ppl_delta = float(d.get("test_ppl_delta", 0.0))
    english_drift_delta = float(d.get("english_drift_delta", 0.0))
    rw_marker_density_delta = float(d.get("rw_marker_density_delta", 0.0))

    checks: dict[str, bool] = {}
    reasons: list[str] = []

    if cfg.require_better_val_ppl:
        checks["better_val_ppl"] = val_ppl_delta < 0
        if not checks["better_val_ppl"]:
            reasons.append(f"val_ppl_not_improved:{val_ppl_delta:+.6f}")
    else:
        checks["better_val_ppl"] = True

    if cfg.require_better_test_ppl:
        checks["better_test_ppl"] = test_ppl_delta < 0
        if not checks["better_test_ppl"]:
            reasons.append(f"test_ppl_not_improved:{test_ppl_delta:+.6f}")
    else:
        checks["better_test_ppl"] = True

    checks["english_drift_within_limit"] = english_drift_delta <= cfg.max_english_drift_delta
    if not checks["english_drift_within_limit"]:
        reasons.append(
            f"english_drift_delta_too_high:{english_drift_delta:+.6f}>{cfg.max_english_drift_delta:+.6f}"
        )

    checks["rw_marker_density_ok"] = rw_marker_density_delta >= cfg.min_rw_marker_density_delta
    if not checks["rw_marker_density_ok"]:
        reasons.append(
            f"rw_marker_density_too_low:{rw_marker_density_delta:+.6f}<{cfg.min_rw_marker_density_delta:+.6f}"
        )

    return GateDecision(
        passed=all(checks.values()),
        checks=checks,
        reasons=reasons,
        delta={
            "val_ppl_delta": val_ppl_delta,
            "test_ppl_delta": test_ppl_delta,
            "english_drift_delta": english_drift_delta,
            "rw_marker_density_delta": rw_marker_density_delta,
        },
    )
