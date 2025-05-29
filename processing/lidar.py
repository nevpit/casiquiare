"""LiDAR processing utilities."""

from __future__ import annotations

try:  # Optional heavy dependencies
    import cv2
except Exception:  # pragma: no cover - library may be missing
    cv2 = None

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover - library may be missing
    gaussian_filter = None

try:
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None


def generate_lrm(dtm: np.ndarray, sigma: float = 5) -> np.ndarray:
    """Generate a local relief model from a DTM using Gaussian blur."""
    if np is None:
        raise RuntimeError("NumPy is required.")

    if cv2 is not None:
        blurred = cv2.GaussianBlur(dtm.astype(float), (0, 0), sigmaX=sigma)
    elif gaussian_filter is not None:
        blurred = gaussian_filter(dtm.astype(float), sigma=sigma)
    else:
        raise RuntimeError("OpenCV or SciPy is required to generate an LRM.")

    return dtm.astype(float) - blurred


__all__ = ["generate_lrm"]
