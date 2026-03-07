from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "build_chat_seed_openai.py"
    spec = spec_from_file_location("build_chat_seed_openai", path)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_validate_messages_ok() -> None:
    mod = _load_module()
    ok, reason = mod._validate_messages(
        [
            {"role": "user", "content": "Muraho"},
            {"role": "assistant", "content": "Muraho neza."},
            {"role": "user", "content": "Mpa inama ku kwizigamira."},
            {"role": "assistant", "content": "Tangirira ku ngengo y'imari nto."},
        ],
        min_turns=2,
        max_turns=4,
    )
    assert ok
    assert reason == "ok"


def test_validate_messages_rejects_non_ascii() -> None:
    mod = _load_module()
    ok, reason = mod._validate_messages(
        [
            {"role": "user", "content": "Muraho"},
            {"role": "assistant", "content": "Wi‑Fi yawe iri gukora."},
        ],
        min_turns=1,
        max_turns=4,
    )
    assert not ok
    assert reason == "non_ascii_content"


def test_normalize_chat_text_strips_unicode_variants() -> None:
    mod = _load_module()
    text = "Wi‑Fi yanjye… irakora\u200b neza"
    normalized = mod._normalize_chat_text(text)
    assert normalized == "Wi-Fi yanjye... irakora neza"


def test_history_to_prompt_formats_roles() -> None:
    mod = _load_module()
    prompt = mod._history_to_prompt(
        [
            {"role": "user", "content": "Muraho"},
            {"role": "assistant", "content": "Muraho neza."},
            {"role": "user", "content": "Mpa inama ku buhinzi."},
        ],
        window=8,
    )
    assert "User: Muraho" in prompt
    assert "Assistant: Muraho neza." in prompt
    assert "User: Mpa inama ku buhinzi." in prompt
