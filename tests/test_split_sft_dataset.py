from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "split_sft_dataset.py"
    spec = spec_from_file_location("split_sft_dataset", path)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_key_for_split_prefers_conversation_id() -> None:
    mod = _load_module()
    key = mod._key_for_split(
        {
            "id": "turn-1",
            "conversation_id": "conv-abc",
            "prompt": "User: Muraho",
            "response": "Muraho neza",
        }
    )
    assert key == "conversation:conv-abc"
