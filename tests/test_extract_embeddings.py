from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "training" / "extract_embeddings.py"
    spec = importlib.util.spec_from_file_location("extract_embeddings", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeInterpreter:
    def __init__(self) -> None:
        self.invoke_count = 0
        self.tensor = np.zeros((1, 6522), dtype=np.float32)

    def set_tensor(self, _index: int, _value: np.ndarray) -> None:
        pass

    def invoke(self) -> None:
        self.invoke_count += 1
        self.tensor.fill(float(self.invoke_count))

    def get_tensor(self, _index: int) -> np.ndarray:
        return self.tensor


def test_extract_embedding_from_audio_averages_chunks_and_uses_copy() -> None:
    module = _load_module()
    interpreter = _FakeInterpreter()
    audio = np.ones(module.CHUNK_SAMPLES + 123, dtype=np.float32)

    embedding = module._extract_embedding_from_audio(audio, interpreter, 0)

    assert interpreter.invoke_count == 2
    assert embedding.shape == (module.EMBEDDING_DIM,)
    assert np.allclose(embedding, np.full(module.EMBEDDING_DIM, 1.5, dtype=np.float32))


def test_extract_embedding_from_audio_empty_returns_zero_vector() -> None:
    module = _load_module()
    interpreter = _FakeInterpreter()

    embedding = module._extract_embedding_from_audio(np.array([], dtype=np.float32), interpreter, 0)

    assert interpreter.invoke_count == 0
    assert embedding.shape == (module.EMBEDDING_DIM,)
    assert np.count_nonzero(embedding) == 0
