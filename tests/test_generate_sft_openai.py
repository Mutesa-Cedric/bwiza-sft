from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "generate_sft_openai.py"
    spec = spec_from_file_location("generate_sft_openai", path)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_user_prompt_has_fixed_prefix() -> None:
    mod = _load_module()
    text = mod._build_user_prompt("Sobanura impamvu amazi ari ingenzi ku buzima.")
    assert mod.FIXED_USER_PREFIX in text
    assert "Items:" in text
    assert "id: single" in text
    assert "prompt: Sobanura impamvu amazi ari ingenzi ku buzima." in text
    assert "Bwiza, an AI assistant" in mod.FIXED_USER_PREFIX


def test_default_system_prompt_has_identity_and_safety_policy() -> None:
    mod = _load_module()
    prompt = mod.DEFAULT_SYSTEM_PROMPT
    assert "Bwiza is an AI assistant" in prompt
    assert "refuse briefly, calmly, and clearly" in prompt
    assert "do not invent details" in prompt


def test_build_batch_user_prompt_includes_multiple_items() -> None:
    mod = _load_module()
    text = mod._build_batch_user_prompt(
        [
            {"id": "a1", "prompt": "Muraho neza"},
            {"id": "a2", "prompt": "Sobanura diyabete mu magambo yoroshye"},
        ]
    )
    assert "id: a1" in text
    assert "prompt: Muraho neza" in text
    assert "id: a2" in text
    assert "prompt: Sobanura diyabete mu magambo yoroshye" in text


def test_extract_batch_items_reads_json_items() -> None:
    mod = _load_module()
    payload = {
        "items": [
            {"id": "a1", "response": "Muraho neza."},
            {"id": "a2", "response": "Diyabete ni indwara..."},
        ]
    }
    assert mod._extract_batch_items(payload) == {
        "a1": "Muraho neza.",
        "a2": "Diyabete ni indwara...",
    }


def test_load_existing_answer_ids_reads_only_answered_rows(tmp_path: Path) -> None:
    mod = _load_module()
    path = tmp_path / "answers.jsonl"
    rows = [
        {"id": "a1", "response": "Muraho neza."},
        {"id": "a2", "response": ""},
        {"id": "", "response": "Igisubizo"},
        {"id": "a3", "response": "Wi‑Fi y'u rugo"},
    ]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    assert mod._load_existing_answer_ids(path) == {"a1", "a3"}


def test_classify_failure_maps_known_cases() -> None:
    mod = _load_module()
    assert mod._classify_failure("http_429: rate limit") == "openai_429"
    assert mod._classify_failure("empty_completion_text:{...}") == "empty_completion_text"
    assert mod._classify_failure("no_json_object_found") == "invalid_json_output"
    assert mod._classify_failure("The read operation timed out") == "timeout"
    assert mod._classify_failure("other failure") == "openai_error"
