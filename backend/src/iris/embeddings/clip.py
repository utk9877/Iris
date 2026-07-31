"""CLIP ONNX embedder — image + text into a shared 512-d space (ARCHITECTURE §2).

Uses the pre-exported ``vision_model.onnx`` / ``text_model.onnx`` (which already
include the projection, output name ``image_embeds`` / ``text_embeds``) plus the CLIP
tokenizer, downloaded from HuggingFace on first use. Runs on the CoreML EP with a CPU
fallback. Outputs are L2-normalized so cosine == dot product.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from PIL import Image

logger = logging.getLogger("iris.embeddings")

Vectors = npt.NDArray[np.float32]

# CLIP preprocessing constants (ViT-B/32).
_IMAGE_SIZE = 224
_CONTEXT_LEN = 77
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

_MODEL_FILES = (
    "onnx/vision_model.onnx",
    "onnx/text_model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)
_PROVIDERS = ["CoreMLExecutionProvider", "CPUExecutionProvider"]


class OnnxSession(Protocol):
    """Structural type for an ONNX Runtime session (real or a test fake)."""

    def get_inputs(self) -> list[Any]: ...
    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]: ...


def _l2_normalize(vectors: Vectors) -> Vectors:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.asarray(vectors / norms, dtype=np.float32)


def preprocess_image(image: Image.Image) -> Vectors:
    """CLIP preprocess: resize shortest side to 224, center-crop, normalize -> CHW."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = _IMAGE_SIZE / min(width, height)
    resized = rgb.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
    left = (resized.width - _IMAGE_SIZE) // 2
    top = (resized.height - _IMAGE_SIZE) // 2
    cropped = resized.crop((left, top, left + _IMAGE_SIZE, top + _IMAGE_SIZE))
    arr = np.asarray(cropped, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)


def _ensure_model(models_dir: Path, model_id: str) -> Path:
    """Download the model + tokenizer into ``models_dir`` on first use; return its dir."""
    from huggingface_hub import hf_hub_download

    local = models_dir / model_id.replace("/", "__")
    for filename in _MODEL_FILES:
        hf_hub_download(model_id, filename, local_dir=str(local))
    return local


def _make_session(path: Path, providers: list[str]) -> OnnxSession:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3  # quiet the per-node CoreML partition warnings
    session: OnnxSession = ort.InferenceSession(
        str(path), sess_options=options, providers=providers
    )
    return session


class CLIPEmbedder:
    """Batched CLIP image/text embedder. Sessions are injectable for testing."""

    def __init__(
        self, vision: OnnxSession, text: OnnxSession, tokenizer: Any, dim: int = 512
    ) -> None:
        self._vision = vision
        self._text = text
        self._tokenizer = tokenizer
        self.dim = dim
        self._vision_input = vision.get_inputs()[0].name
        self._text_input = text.get_inputs()[0].name

    @classmethod
    def load(cls, models_dir: Path, model_id: str, dim: int = 512) -> CLIPEmbedder:
        from transformers import CLIPTokenizerFast  # type: ignore[attr-defined]

        local = _ensure_model(models_dir, model_id)
        # Vision encoder: CoreML (throughput-critical for embedding the library),
        # with per-node CPU fallback. Text encoder: CPU only — the CoreML EP fails at
        # inference time on the CLIP text graph (ARCHITECTURE §10 risk #4), and it is
        # tiny + runs once per query, so CoreML buys nothing there.
        vision = _make_session(local / "onnx" / "vision_model.onnx", _PROVIDERS)
        text = _make_session(local / "onnx" / "text_model.onnx", ["CPUExecutionProvider"])
        tokenizer = CLIPTokenizerFast.from_pretrained(str(local))
        return cls(vision, text, tokenizer, dim)

    def embed_images(self, images: list[Image.Image]) -> Vectors:
        if not images:
            return np.empty((0, self.dim), dtype=np.float32)
        batch = np.stack([preprocess_image(img) for img in images]).astype(np.float32)
        out = self._vision.run(None, {self._vision_input: batch})[0]
        return _l2_normalize(np.asarray(out, dtype=np.float32))

    def embed_texts(self, texts: list[str]) -> Vectors:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        encoded = self._tokenizer(
            texts,
            padding="max_length",
            max_length=_CONTEXT_LEN,
            truncation=True,
            return_tensors="np",
        )
        input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
        out = self._text.run(None, {self._text_input: input_ids})[0]
        return _l2_normalize(np.asarray(out, dtype=np.float32))
