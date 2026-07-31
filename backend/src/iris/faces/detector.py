"""Face detection + alignment + ArcFace embedding via insightface (ARCHITECTURE §6).

Wraps insightface's ``FaceAnalysis`` (SCRFD ``det_10g`` + ArcFace ``w600k_r50`` in the
``buffalo_l`` pack), which handles the 5-landmark alignment internally and returns
L2-normalized 512-d embeddings. The model pack downloads on first use. The app object
is injectable so the pipeline can be tested with a fake detector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps

logger = logging.getLogger("iris.faces")

Vector = npt.NDArray[np.float32]
_ARCFACE_PX = 112.0
_SHARP_REF = 500.0  # Laplacian-variance value treated as "sharp"


class FaceApp(Protocol):
    """Structural type for insightface FaceAnalysis (or a test fake)."""

    def get(self, img: Any) -> list[Any]: ...


def load_image_bgr(path: str, max_edge: int) -> npt.NDArray[np.uint8]:
    """Decode an image, apply EXIF orientation, downscale, and return BGR (HWC uint8).

    insightface expects BGR (OpenCV convention). Reading the original is non-destructive.
    """
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        width, height = image.size
        scale = min(1.0, max_edge / max(width, height))
        if scale < 1.0:
            image = image.resize((round(width * scale), round(height * scale)))
        rgb = np.asarray(image, dtype=np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])  # RGB -> BGR


@dataclass(slots=True)
class DetectedFace:
    bbox: tuple[float, float, float, float]  # normalized (x, y, w, h) in [0, 1]
    det_score: float
    quality: float
    embedding: Vector  # 512-d, L2-normalized
    landmarks: bytes  # 5x2 float32 (normalized), for later re-alignment/overlay


def _laplacian_variance(gray: npt.NDArray[np.float64]) -> float:
    # Discrete Laplacian via a 4-neighbour kernel (numpy only; no cv2 needed here).
    lap = (
        -4.0 * gray
        + np.roll(gray, 1, 0)
        + np.roll(gray, -1, 0)
        + np.roll(gray, 1, 1)
        + np.roll(gray, -1, 1)
    )
    return float(lap[1:-1, 1:-1].var()) if gray.size > 4 else 0.0


def _frontalness(kps: npt.NDArray[np.float64]) -> float:
    """1.0 = nose centered between the eyes; lower = more turned/profile."""
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_mid = (left_eye + right_eye) / 2.0
    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    if eye_dist < 1e-3:
        return 0.0
    offset = abs(float(nose[0] - eye_mid[0])) / (eye_dist / 2.0)
    return max(0.0, 1.0 - min(offset, 1.0))


def compute_quality(
    image_bgr: npt.NDArray[np.uint8],
    bbox_px: tuple[float, float, float, float],
    kps: npt.NDArray[np.float64],
    det_score: float,
) -> float:
    """Composite face quality in [0, 1] from detector score, size, sharpness, pose."""
    x, y, w, h = (int(v) for v in bbox_px)
    x, y = max(0, x), max(0, y)
    crop = image_bgr[y : y + h, x : x + w]
    if crop.size == 0:
        return 0.0
    gray = crop.astype(np.float64).mean(axis=2)
    blur_norm = min(_laplacian_variance(gray) / _SHARP_REF, 1.0)
    size_norm = min(min(w, h) / _ARCFACE_PX, 1.0)
    frontal = _frontalness(kps)
    return float(0.4 * det_score + 0.2 * frontal + 0.2 * blur_norm + 0.2 * size_norm)


class FaceDetector:
    def __init__(self, app: FaceApp, *, min_size: int, min_det_score: float) -> None:
        self._app = app
        self._min_size = min_size
        self._min_det_score = min_det_score

    @classmethod
    def load(
        cls,
        model_name: str,
        *,
        det_size: int,
        min_size: int,
        min_det_score: float,
        providers: list[str] | None = None,
    ) -> FaceDetector:
        from insightface.app import FaceAnalysis

        providers = providers or ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        app = FaceAnalysis(name=model_name, providers=providers)
        app.prepare(ctx_id=0, det_size=(det_size, det_size))
        return cls(app, min_size=min_size, min_det_score=min_det_score)

    def detect(self, image_bgr: npt.NDArray[np.uint8]) -> list[DetectedFace]:
        height, width = image_bgr.shape[:2]
        results: list[DetectedFace] = []
        for face in self._app.get(image_bgr):
            x1, y1, x2, y2 = (float(v) for v in face.bbox)
            fw, fh = x2 - x1, y2 - y1
            if min(fw, fh) < self._min_size or float(face.det_score) < self._min_det_score:
                continue
            kps = np.asarray(face.kps, dtype=np.float64)
            quality = compute_quality(image_bgr, (x1, y1, fw, fh), kps, float(face.det_score))
            norm_kps = (kps / np.array([width, height], dtype=np.float64)).astype(np.float32)
            results.append(
                DetectedFace(
                    bbox=(x1 / width, y1 / height, fw / width, fh / height),
                    det_score=float(face.det_score),
                    quality=quality,
                    embedding=np.asarray(face.normed_embedding, dtype=np.float32),
                    landmarks=norm_kps.tobytes(),
                )
            )
        return results
