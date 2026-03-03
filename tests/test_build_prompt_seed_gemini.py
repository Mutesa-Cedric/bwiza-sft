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


def test_extract_json_block_handles_code_fence() -> None:
    mod = _load_module()
    text = """```json
{"items":[{"prompt":"Muraho","task_type":"rw_instruction","lang_mode":"rw"}]}
```"""
    payload = mod._extract_json_block(text)
    assert isinstance(payload, dict)
    assert "items" in payload


def test_extract_json_block_repairs_trailing_comma() -> None:
    mod = _load_module()
    text = (
        '{"items":[{"prompt":"Muraho","task_type":"rw_instruction","lang_mode":"rw",}],}'
    )
    payload = mod._extract_json_block(text)
    assert isinstance(payload, dict)
    assert isinstance(payload.get("items"), list)


def test_choose_batch_size_candidate_pool_for_single() -> None:
    mod = _load_module()
    assert mod._choose_batch_size(1, 3) == 3
    assert mod._choose_batch_size(1, 1) == 1
    assert mod._choose_batch_size(4, 3) == 4


def test_build_user_prompt_includes_soft_memory() -> None:
    mod = _load_module()
    p = mod._build_user_prompt(
        topic="uburezi mu Rwanda",
        n=1,
        avoid_prompts=["Sobanura uburezi mu Rwanda."],
        frequent_patterns=["Sobanura uburyo bwo..."],
    )
    assert "Avoid prompts that are same/very similar" in p
    assert "Avoid overusing these common prompt openings" in p
