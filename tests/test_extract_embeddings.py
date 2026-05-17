from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path

import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "training" / "extract_embeddings.py"
    spec = importlib.util.spec_from_file_location("extract_embeddings", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_birdnet_extractor_hard_fails_without_birdnetlib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "birdnetlib" or name.startswith("birdnetlib."):
            raise ImportError("birdnetlib ontbreekt")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(RuntimeError, match="tensorflow-cpu"):
        module._load_birdnet_extractor()
