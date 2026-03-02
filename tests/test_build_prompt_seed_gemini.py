from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "build_prompt_seed_gemini.py"
    spec = spec_from_file_location("build_prompt_seed_gemini", path)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_task_lang_mode_mismatch_rejected() -> None:
    mod = _load_module()
    ok, reason = mod._validate_item(
        {
            "prompt": "Nkeneye inama ku kubitsa amafaranga mu muryango.",
            "task_type": "language_control",
            "lang_mode": "rw",
        }
    )
    assert not ok
    assert reason == "task_lang_mode_mismatch"


def test_task_lang_mode_valid_accepts() -> None:
    mod = _load_module()
    ok, reason = mod._validate_item(
        {
            "prompt": "Subiza mu Kinyarwanda gusa ku bibazo by'uburezi.",
            "task_type": "language_control",
            "lang_mode": "control",
        }
    )
    assert ok
    assert reason == "ok"
