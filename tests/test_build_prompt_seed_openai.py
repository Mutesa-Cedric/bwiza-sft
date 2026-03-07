from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "build_prompt_seed_openai.py"
    spec = spec_from_file_location("build_prompt_seed_openai", path)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_prompts_strips_numbering_and_bullets() -> None:
    mod = _load_module()
    text = "\n".join(
        [
            "1. Kuki ubukungu bw'igihugu ari ingenzi?",
            "- Ni gute umusoro ufasha leta?",
            "* Isoko rikora rite mu kugena ibiciro?",
            "• Kuki kuzigama amafaranga ari ingenzi?",
            "(1) Ni gute abaturage bagira uruhare mu iterambere?",
            "a) Ni irihe tandukaniro riri hagati y'ibikenerwa n'ibyifuzwa?",
            "Prompt: Kuki ubucuruzi mpuzamahanga bufitiye igihugu akamaro?",
        ]
    )
    prompts = mod._parse_prompts(text)
    assert prompts == [
        "Kuki ubukungu bw'igihugu ari ingenzi?",
        "Ni gute umusoro ufasha leta?",
        "Isoko rikora rite mu kugena ibiciro?",
        "Kuki kuzigama amafaranga ari ingenzi?",
        "Ni gute abaturage bagira uruhare mu iterambere?",
        "Ni irihe tandukaniro riri hagati y'ibikenerwa n'ibyifuzwa?",
        "Kuki ubucuruzi mpuzamahanga bufitiye igihugu akamaro?",
    ]


def test_parse_prompts_dedups_identical_lines() -> None:
    mod = _load_module()
    prompts = mod._parse_prompts("Muraho neza\nMuraho neza\n")
    assert prompts == ["Muraho neza"]


def test_parse_prompts_drops_wrapper_lines() -> None:
    mod = _load_module()
    text = "\n".join(
        [
            "Here are 10 prompts:",
            "```",
            "json",
            "Ni gute twagabanya ibihuha ku mbuga nkoranyambaga?",
            "```",
        ]
    )
    prompts = mod._parse_prompts(text)
    assert prompts == ["Ni gute twagabanya ibihuha ku mbuga nkoranyambaga?"]


def test_build_user_prompt_mentions_exact_count() -> None:
    mod = _load_module()
    text = mod._build_user_prompt(
        topic="ubukungu bw'umuryango",
        prompts_per_request=10,
        spec=mod.TASK_SPECS[0],
        avoid_prompts=["A", "B"],
        frequent_patterns=["Ni gute ..."],
    )
    assert mod.FIXED_USER_PREFIX in text
    assert "Need exactly 10 prompts." in text
    assert "Avoid prompts too similar to these recent ones:" in text
    assert "Avoid these common openings:" in text


def test_normalize_prompt_text_converts_smart_punctuation() -> None:
    mod = _load_module()
    prompt = 'Sobanura impamvu uburezi bw’u Rwanda "bw’ingenzi" - Wi‑Fi\u200b vuba… （none）'
    normalized = mod._normalize_prompt_text(prompt.replace('"', "“", 1).replace('"', "”", 1))
    assert "bw'u Rwanda" in normalized
    assert '"' in normalized
    assert "“" not in normalized
    assert "”" not in normalized
    assert "Wi-Fi" in normalized
    assert "Wi‑Fi" not in normalized
    assert "\u200b" not in normalized
    assert "..." in normalized
    assert "(" in normalized and ")" in normalized


def test_validate_prompt_bounds() -> None:
    mod = _load_module()
    assert mod._validate_prompt("too short")[0] is False
    assert mod._validate_prompt("Iyi ni prompt ifite uburebure buhagije kandi isobanutse neza.")[0] is True


def test_classify_failure_detects_429() -> None:
    mod = _load_module()
    assert mod._classify_failure("http_429: rate limit") is True
    assert mod._classify_failure("The read operation timed out") is False


def test_sample_topic_prefers_underrepresented_topics() -> None:
    mod = _load_module()
    rng = mod.random.Random(42)
    topics = ["a", "b", "c"]
    chosen = mod._sample_topic(rng, topics, {"a": 9, "b": 1, "c": 1})
    assert chosen in {"b", "c"}


def test_sample_spec_avoids_overrepresented_task() -> None:
    mod = _load_module()
    rng = mod.random.Random(7)
    counts = {spec["task_type"]: 0 for spec in mod.TASK_SPECS}
    counts["rw_instruction"] = 50
    chosen = mod._sample_spec(rng, counts)
    assert chosen["task_type"] != "rw_instruction"
