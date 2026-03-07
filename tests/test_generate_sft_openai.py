from importlib.util import module_from_spec, spec_from_file_location
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
    assert "User prompt:" in text
    assert "Sobanura impamvu amazi ari ingenzi ku buzima." in text
    assert "Bwiza, an AI assistant" in mod.FIXED_USER_PREFIX


def test_default_system_prompt_has_identity_and_safety_policy() -> None:
    mod = _load_module()
    prompt = mod.DEFAULT_SYSTEM_PROMPT
    assert "Bwiza is an AI assistant" in prompt
    assert "refuse briefly, calmly, and clearly" in prompt
    assert "do not invent details" in prompt


def test_classify_failure_maps_known_cases() -> None:
    mod = _load_module()
    assert mod._classify_failure("http_429: rate limit") == "openai_429"
    assert mod._classify_failure("empty_completion_text:{...}") == "empty_completion_text"
    assert mod._classify_failure("The read operation timed out") == "timeout"
    assert mod._classify_failure("other failure") == "openai_error"
