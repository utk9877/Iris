"""DCT-based 64-bit perceptual hash (numpy only, no scipy).

Used for near-duplicate detection (grouping is Phase 4; here we just compute and
store a stable value). Stored as a signed 64-bit int to fit SQLite INTEGER.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import numpy.typing as npt
from PIL import Image

_DCT_N = 32  # image is downscaled to 32x32 before the transform
_HASH_N = 8  # keep the top-left 8x8 low-frequency block -> 64 bits


@lru_cache(maxsize=1)
def _dct_matrix() -> npt.NDArray[np.float64]:
    """Orthonormal DCT-II basis matrix (cached)."""
    n = _DCT_N
    k = np.arange(n)
    basis = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    basis *= np.sqrt(2.0 / n)
    basis[0, :] *= np.sqrt(0.5)
    return basis.astype(np.float64)


def _to_signed64(value: int) -> int:
    """Map an unsigned 64-bit value into SQLite's signed INTEGER range."""
    return value - (1 << 64) if value >= (1 << 63) else value


def perceptual_hash(image: Image.Image) -> int:
    """Return the signed 64-bit perceptual hash of an image."""
    gray = image.convert("L").resize((_DCT_N, _DCT_N), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    matrix = _dct_matrix()
    dct = matrix @ pixels @ matrix.T
    low = dct[:_HASH_N, :_HASH_N]
    median = float(np.median(low))
    value = 0
    for bit in (low > median).flatten():
        value = (value << 1) | int(bit)
    return _to_signed64(value)


def hamming_distance(a: int, b: int) -> int:
    """Bit differences between two (signed) 64-bit perceptual hashes."""
    return ((a & 0xFFFFFFFFFFFFFFFF) ^ (b & 0xFFFFFFFFFFFFFFFF)).bit_count()
