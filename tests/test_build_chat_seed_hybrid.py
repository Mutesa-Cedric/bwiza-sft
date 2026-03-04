from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "build_chat_seed_hybrid.py"
    spec = spec_from_file_location("build_chat_seed_hybrid", path)
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


def test_validate_messages_rejects_non_alternating_roles() -> None:
    mod = _load_module()
    ok, reason = mod._validate_messages(
        [
            {"role": "user", "content": "Muraho"},
            {"role": "user", "content": "Mpa amakuru."},
        ],
        min_turns=1,
        max_turns=4,
    )
    assert not ok
    assert reason == "must_end_assistant"


def test_conversation_signature_uses_assistant_for_specificity() -> None:
    mod = _load_module()
    sig = mod._conversation_signature(
        [
            {"role": "user", "content": "Muraho"},
            {"role": "assistant", "content": "Muraho neza."},
            {"role": "user", "content": "Mpa inama ku buhinzi."},
            {"role": "assistant", "content": "Tangira gutegura ubutaka neza."},
        ]
    )
    assert "Muraho" in sig
    assert "Mpa inama ku buhinzi." in sig
    assert "Muraho neza." in sig


def test_out_path_preserves_prefix_literal() -> None:
    mod = _load_module()
    p = mod._out_path(Path("outputs/sft/chat.seed.hybrid"), "pairs.jsonl")
    assert str(p).endswith("outputs/sft/chat.seed.hybrid.pairs.jsonl")
