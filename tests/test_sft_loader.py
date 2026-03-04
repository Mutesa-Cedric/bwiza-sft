from pathlib import Path

from src.data.sft_loader import build_sft_tokens, summarize_jsonl


class _Tok:
    eos_token = "<eos>"

    def __call__(self, text: str, add_special_tokens: bool = False):
        # deterministic fake tokenizer: byte per char
        return {"input_ids": [ord(c) % 251 for c in text]}


def test_summarize_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "sft.jsonl"
    p.write_text(
        '{"prompt":"Muraho","response":"Ni meza"}\n'
        '{"instruction":"Sobanura","input":"","output":"Igisubizo"}\n',
        encoding="utf-8",
    )
    s = summarize_jsonl(p)
    assert s.rows == 2
    assert s.valid_rows == 2
    assert s.invalid_rows == 0


def test_build_sft_tokens_masks_prompt() -> None:
    tok = _Tok()
    out = build_sft_tokens(tok, prompt="Muraho", response="Ni meza", seq_len=128)
    assert out is not None
    input_ids, labels, supervised = out
    assert len(input_ids) == len(labels)
    assert supervised > 0
    assert -100 in labels


def test_summarize_messages_counts_multiple_turn_pairs(tmp_path: Path) -> None:
    p = tmp_path / "chat.jsonl"
    p.write_text(
        '{"messages":[{"role":"user","content":"Muraho"},{"role":"assistant","content":"Muraho neza"},{"role":"user","content":"Mpa inama"},{"role":"assistant","content":"Tangira ku ntego nto"}]}\n',
        encoding="utf-8",
    )
    s = summarize_jsonl(p)
    assert s.rows == 1
    assert s.valid_rows == 2
    assert s.invalid_rows == 0


def test_build_sft_tokens_accepts_preformatted_chat_prompt() -> None:
    tok = _Tok()
    out = build_sft_tokens(
        tok,
        prompt="User: Muraho\nAssistant: Muraho neza\nUser: Mpa inama",
        response="Tangira kwizigamira buhoro buhoro.",
        seq_len=256,
    )
    assert out is not None
    input_ids, labels, supervised = out
    assert len(input_ids) == len(labels)
    assert supervised > 0
