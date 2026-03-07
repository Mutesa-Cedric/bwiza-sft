from src.data.sft_records import extract_prompt, extract_response, normalize_ascii_text


def test_normalize_ascii_text_converts_unicode_punctuation() -> None:
    text = 'Wi‑Fi y’u Rwanda… “ni byiza”\u200b'
    out = normalize_ascii_text(text)
    assert out == 'Wi-Fi y\'u Rwanda... "ni byiza"'


def test_extract_prompt_response_apply_ascii_normalization() -> None:
    rec = {
        "prompt": "Sobanura uko Wi‑Fi ikora mu rugo.",
        "response": "Wi‑Fi y’u Rwanda irakora neza…",
    }
    assert extract_prompt(rec) == "Sobanura uko Wi-Fi ikora mu rugo."
    assert extract_response(rec) == "Wi-Fi y'u Rwanda irakora neza..."
