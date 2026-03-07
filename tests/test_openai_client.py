from src.data.openai_client import _extract_text


def test_extract_text_from_string_content() -> None:
    payload = {"choices": [{"message": {"content": "Muraho neza"}}]}
    assert _extract_text(payload) == "Muraho neza"


def test_extract_text_from_content_parts() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "output_text", "text": "Muraho"},
                        {"type": "output_text", "text": "neza"},
                    ]
                }
            }
        ]
    }
    assert _extract_text(payload) == "Muraho\nneza"


def test_extract_text_from_top_level_output_text() -> None:
    payload = {"output_text": "Sobanura ibi mu buryo bworoshye."}
    assert _extract_text(payload) == "Sobanura ibi mu buryo bworoshye."


def test_extract_text_from_choice_text_fallback() -> None:
    payload = {"choices": [{"text": "Andika urutonde rw'ingingo 3."}]}
    assert _extract_text(payload) == "Andika urutonde rw'ingingo 3."
