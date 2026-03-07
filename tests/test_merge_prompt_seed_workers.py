from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "merge_prompt_seed_workers.py"
    spec = spec_from_file_location("merge_prompt_seed_workers", path)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_merge_keeps_unique_prompts_across_workers(tmp_path: Path) -> None:
    mod = _load_module()
    run_dir = tmp_path / "run"
    worker1 = run_dir / "worker_01"
    worker2 = run_dir / "worker_02"
    worker1.mkdir(parents=True)
    worker2.mkdir(parents=True)

    rec1 = {
        "prompt": "Muraho neza",
        "task_type": "rw_instruction",
        "lang_mode": "rw",
        "topic": "greeting",
        "source": "flash_local",
        "teacher_model": "gemini-3-flash-preview",
        "created_at": "2026-03-06T00:00:00+00:00",
    }
    rec2 = {
        "prompt": "Muraho neza",
        "task_type": "rw_instruction",
        "content_type": "rw_instruction",
        "lang_mode": "rw",
        "topic": "greeting",
        "source": "flash_local",
        "teacher_model": "gemini-3-flash-preview",
        "created_at": "2026-03-06T00:00:01+00:00",
    }
    rec3 = {
        "prompt": "Sobanura uburezi",
        "task_type": "rw_instruction",
        "lang_mode": "rw",
        "topic": "education",
        "source": "flash_local",
        "teacher_model": "gemini-3-flash-preview",
        "created_at": "2026-03-06T00:00:02+00:00",
    }
    (worker1 / "prompts.seed.hybrid.final.jsonl").write_text(
        json.dumps(rec1) + "\n" + json.dumps(rec3) + "\n", encoding="utf-8"
    )
    (worker2 / "prompts.seed.hybrid.final.jsonl").write_text(json.dumps(rec2) + "\n", encoding="utf-8")

    input_files = mod._iter_candidate_files(run_dir)
    assert len(input_files) == 2

    # Execute merge script internals through subprocess-like main inputs would be overkill here.
    best = {}
    for path in input_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            key = mod._norm_prompt(rec["prompt"])
            prev = best.get(key)
            if prev is None or mod._score(rec) > mod._score(prev):
                best[key] = rec

    assert len(best) == 2
    assert best["muraho neza"]["content_type"] == "rw_instruction"
