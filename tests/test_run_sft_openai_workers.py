from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "run_sft_openai_workers.py"
    spec = spec_from_file_location("run_sft_openai_workers", path)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_loads() -> None:
    mod = _load_module()
    assert callable(mod.parse_args)
