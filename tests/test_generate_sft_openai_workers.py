from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "generate_sft_openai.py"
    spec = spec_from_file_location("generate_sft_openai", path)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_should_process_line_for_shard() -> None:
    mod = _load_module()
    assert mod._should_process_line(1, worker_index=0, worker_stride=3) is True
    assert mod._should_process_line(2, worker_index=0, worker_stride=3) is False
    assert mod._should_process_line(3, worker_index=2, worker_stride=3) is True
    assert mod._should_process_line(4, worker_index=0, worker_stride=3) is True
