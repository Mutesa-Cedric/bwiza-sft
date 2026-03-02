from src.data.sft_records import extract_prompt, extract_response, split_bucket


def test_extract_prompt_variants() -> None:
    assert extract_prompt({"prompt": " Muraho "}) == "Muraho"
    assert extract_prompt({"question": " ikibazo? "}) == "ikibazo?"
    assert extract_prompt({"instruction": "Sobanura", "input": "ibi", "output": "x"}) == "Sobanura\nibi"


def test_extract_response_variants() -> None:
    assert extract_response({"response": " Igisubizo "}) == "Igisubizo"
    assert extract_response({"output": " out "}) == "out"


def test_split_bucket_deterministic() -> None:
    b1 = split_bucket("id-123", train_ratio=0.9, val_ratio=0.05)
    b2 = split_bucket("id-123", train_ratio=0.9, val_ratio=0.05)
    assert b1 == b2
    assert b1 in {"train", "val", "test"}
