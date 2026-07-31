"""CLIP embedder plumbing tests — fake ONNX sessions (no network / weights)."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from iris.embeddings.clip import CLIPEmbedder, preprocess_image


def test_preprocess_shape_and_dtype() -> None:
    arr = preprocess_image(Image.new("RGB", (400, 300), (128, 64, 200)))
    assert arr.shape == (3, 224, 224)
    assert arr.dtype == np.float32
    assert -5.0 < float(arr.min()) and float(arr.max()) < 5.0  # normalized range


class _FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    def __init__(self, input_name: str, dim: int) -> None:
        self._input_name = input_name
        self._dim = dim

    def get_inputs(self) -> list[Any]:
        return [_FakeInput(self._input_name)]

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]:
        batch = next(iter(input_feed.values()))
        return [np.full((batch.shape[0], self._dim), 3.0, dtype=np.float32)]


class _FakeTokenizer:
    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        return {"input_ids": np.zeros((len(texts), 77), dtype=np.int64)}


def _embedder() -> CLIPEmbedder:
    return CLIPEmbedder(
        _FakeSession("pixel_values", 512), _FakeSession("input_ids", 512), _FakeTokenizer(), dim=512
    )


def test_embed_images_are_normalized() -> None:
    out = _embedder().embed_images(
        [Image.new("RGB", (64, 64), (10, 20, 30)), Image.new("RGB", (80, 60), (200, 100, 50))]
    )
    assert out.shape == (2, 512)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_embed_texts_are_normalized() -> None:
    out = _embedder().embed_texts(["a cat", "a dog on a beach"])
    assert out.shape == (2, 512)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_embed_empty_inputs() -> None:
    emb = _embedder()
    assert emb.embed_images([]).shape == (0, 512)
    assert emb.embed_texts([]).shape == (0, 512)
