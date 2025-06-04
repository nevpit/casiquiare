"""Satellite imagery processing utilities."""

from __future__ import annotations

try:
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None  # type: ignore

try:
    import cv2
except Exception:  # pragma: no cover - library may be missing
    cv2 = None  # type: ignore


def compute_ndvi(red: "np.ndarray", nir: "np.ndarray") -> "np.ndarray":
    """Compute the normalized difference vegetation index (NDVI)."""
    if np is None:
        raise RuntimeError("NumPy is required.")

    red_arr = red.astype(float)
    nir_arr = nir.astype(float)
    denom = nir_arr + red_arr

    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir_arr - red_arr) / denom
        if isinstance(ndvi, np.ndarray):
            ndvi[denom == 0] = 0.0
        else:
            ndvi = 0.0

    return ndvi.astype(np.float32)


def enhance_contrast(image: "np.ndarray") -> "np.ndarray":
    """Enhance image contrast using CLAHE."""
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required.")

    arr = np.clip(image, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if arr.ndim == 2:
        result = clahe.apply(arr)
    else:
        channels = cv2.split(arr)
        enhanced = [clahe.apply(ch) for ch in channels]
        result = cv2.merge(enhanced)

    return result


__all__ = ["compute_ndvi", "enhance_contrast"]
