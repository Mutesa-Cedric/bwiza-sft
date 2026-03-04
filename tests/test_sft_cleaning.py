from src.data.sft_cleaning import CleanConfig, clean_decision, dedup_key


def test_clean_decision_rejects_short() -> None:
    cfg = CleanConfig(min_prompt_chars=3, min_response_chars=10, max_response_chars=100)
    d = clean_decision("hi", "ok", cfg)
    assert not d.keep


def test_clean_decision_accepts_valid() -> None:
    cfg = CleanConfig(min_prompt_chars=3, min_response_chars=5, max_response_chars=100)
    d = clean_decision("Muraho neza", "Ni meza cyane", cfg)
    assert d.keep


def test_dedup_key_normalizes() -> None:
    a = dedup_key(" Muraho  ", "Ni meza")
    b = dedup_key("muraho", "Ni   meza")
    assert a == b


def test_clean_decision_rejects_word_repetition() -> None:
    cfg = CleanConfig(min_prompt_chars=3, min_response_chars=5, max_response_chars=200, max_consecutive_repeat_words=4)
    d = clean_decision("Sobanura", "ni ni ni ni ni ni", cfg)
    assert not d.keep
    assert d.reason == "response_word_repetition"
