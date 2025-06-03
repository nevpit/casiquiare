"""Satellite feature detection utilities."""

from __future__ import annotations

try:
    import cv2
except Exception:  # pragma: no cover - library may be missing
    cv2 = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None  # type: ignore

try:
    from processing.image import to_uint8
except Exception:  # pragma: no cover - library may be missing
    to_uint8 = None  # type: ignore


def detect_edges(image: "np.ndarray") -> "np.ndarray":
    """Detect edges in a satellite image using the Canny algorithm."""
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required.")

    arr = image
    if to_uint8 is not None:
        arr_u8 = to_uint8(arr)
    else:
        arr_u8 = arr.astype(np.uint8)

    blurred = cv2.GaussianBlur(arr_u8, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 90)
    return edges


__all__ = ["detect_edges"]
