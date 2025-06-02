"""Edge detection helpers for LiDAR-derived images."""

from __future__ import annotations

try:  # Optional heavy dependencies
    import cv2
except Exception:  # pragma: no cover - library may be missing
    cv2 = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None  # type: ignore


def detect_edges(image: "np.ndarray") -> "np.ndarray":
    """Detect subtle edges using a tuned Canny filter."""
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required.")

    arr = np.asarray(image, dtype=np.float32)
    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    arr_u8 = arr.astype(np.uint8)

    blurred = cv2.GaussianBlur(arr_u8, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 90, apertureSize=3, L2gradient=True)
    edges = cv2.dilate(edges, None)
    return edges


__all__ = ["detect_edges"]
