from src.eval.gating import GateConfig, evaluate_gate


def test_gate_passes_good_delta() -> None:
    report = {
        "delta": {
            "val_ppl_delta": -1.0,
            "test_ppl_delta": -0.5,
            "english_drift_delta": -0.01,
            "rw_marker_density_delta": 0.02,
        }
    }
    cfg = GateConfig(
        require_better_val_ppl=True,
        require_better_test_ppl=True,
        max_english_drift_delta=0.0,
        min_rw_marker_density_delta=0.0,
    )
    d = evaluate_gate(report, cfg)
    assert d.passed


def test_gate_fails_bad_delta() -> None:
    report = {
        "delta": {
            "val_ppl_delta": 0.1,
            "test_ppl_delta": -0.2,
            "english_drift_delta": 0.05,
            "rw_marker_density_delta": -0.01,
        }
    }
    cfg = GateConfig(
        require_better_val_ppl=True,
        require_better_test_ppl=True,
        max_english_drift_delta=0.0,
        min_rw_marker_density_delta=0.0,
    )
    d = evaluate_gate(report, cfg)
    assert not d.passed
