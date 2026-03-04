from pathlib import Path

from src.data.dedup_index import PromptDedupIndex


def test_add_if_new_and_recent_prompts(tmp_path: Path) -> None:
    db = tmp_path / "dedup.sqlite"
    with PromptDedupIndex(db) as idx:
        assert idx.add_if_new("k1", "Muraho neza", "w1.jsonl", "uburezi", "2026-03-03T00:00:00Z")
        assert not idx.add_if_new("k1", "Muraho neza", "w1.jsonl", "uburezi", "2026-03-03T00:00:01Z")
        assert idx.count() == 1
        rec = idx.recent_prompts(limit=5, topic="uburezi")
        assert rec == ["Muraho neza"]


def test_has_near_duplicate_detects_similar_prompt(tmp_path: Path) -> None:
    db = tmp_path / "dedup.sqlite"
    with PromptDedupIndex(db) as idx:
        idx.add_if_new(
            "k1",
            "Sobanura uburyo bwo kwizigamira amafaranga mu muryango.",
            "w1.jsonl",
            "ubukungu",
            "2026-03-03T00:00:00Z",
        )
        found, matched = idx.has_near_duplicate(
            prompt="Sobanura uburyo bwo kwizigamira amafaranga mu muryango neza.",
            topic="ubukungu",
            threshold=0.7,
            topic_limit=20,
            global_limit=20,
        )
        assert found
        assert "kwizigamira amafaranga" in matched


def test_has_near_duplicate_does_not_flag_unrelated_prompt(tmp_path: Path) -> None:
    db = tmp_path / "dedup.sqlite"
    with PromptDedupIndex(db) as idx:
        idx.add_if_new(
            "k1",
            "Ni ubuhe buryo nakoresha ngo menye niba amakuru kuri WhatsApp ari ukuri?",
            "w1.jsonl",
            "amakuru",
            "2026-03-03T00:00:00Z",
        )
        found, _ = idx.has_near_duplicate(
            prompt="Ni ubuhe buryo bwo gushaka akazi mu Rwanda ukoresheje interineti?",
            topic="akazi",
            threshold=0.9,
            topic_limit=20,
            global_limit=20,
        )
        assert not found
