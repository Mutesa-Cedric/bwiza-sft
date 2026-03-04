from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "build_prompt_seed_hybrid.py"
    spec = spec_from_file_location("build_prompt_seed_hybrid", path)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_payload_items_accepts_batch_and_single() -> None:
    mod = _load_module()
    batch = {"items": [{"prompt": "A", "task_type": "rw_instruction", "lang_mode": "rw"}]}
    single = {"prompt": "A", "task_type": "rw_instruction", "lang_mode": "rw"}
    assert len(mod._payload_items(batch)) == 1
    assert len(mod._payload_items(single)) == 1


def test_build_flash_user_prompt_mentions_exact_count() -> None:
    mod = _load_module()
    text = mod._build_flash_user_prompt(
        "uburezi mu Rwanda",
        4,
        required_task_type="rw_instruction",
        targeting_mode="soft",
        avoid_prompts=["A", "B"],
        frequent_patterns=["Sobanura ..."],
    )
    assert "Need exactly 4 items." in text
    assert '"items"' in text
    assert "Avoid same/very similar prompts as" in text
    assert "Avoid these common openings" in text


def test_out_path_preserves_prefix_literal() -> None:
    mod = _load_module()
    p = mod._out_path(Path("outputs/sft/prompts.seed.hybrid"), "final.jsonl")
    assert str(p).endswith("outputs/sft/prompts.seed.hybrid.final.jsonl")


def test_validate_item_task_lang_mode_mismatch() -> None:
    mod = _load_module()
    ok, reason = mod._validate_item(
        {
            "prompt": "Sobanura uburezi bw'ibanze.",
            "task_type": "rw_instruction",
            "lang_mode": "en",
        }
    )
    assert not ok
    assert reason == "task_lang_mode_mismatch"


def test_circuit_break_reason_prefers_429() -> None:
    mod = _load_module()
    reason = mod._circuit_break_reason(
        consecutive_429=12,
        consecutive_parse=50,
        consecutive_fail=50,
        max_consecutive_429=10,
        max_consecutive_parse=100,
        max_consecutive_fail=100,
    )
    assert reason == "consecutive_429_limit_reached"


def test_collect_avoid_memory_merges_unique_and_patterns() -> None:
    mod = _load_module()

    class _FakeDedup:
        def recent_prompts(self, limit: int, topic: str = ""):
            if topic:
                return ["Prompt A", "Prompt B"]
            return ["Prompt B", "Prompt C"]

        def frequent_patterns(self, limit: int):
            return ["Sobanura ...", "Ni gute ..."][:limit]

    avoid, patterns = mod._collect_avoid_memory(
        _FakeDedup(),
        "uburezi mu Rwanda",
        topic_recent=10,
        global_recent=10,
        pattern_topk=2,
    )
    assert avoid == ["Prompt A", "Prompt B", "Prompt C"]
    assert patterns == ["Sobanura ...", "Ni gute ..."]
