"""LiDAR feature detection utilities."""

from __future__ import annotations

try:  # Optional heavy dependencies
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
    """Detect edges in a relief image using the Canny algorithm."""
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

def find_contours(edge_img: "np.ndarray") -> list["np.ndarray"]:
    """Convert an edge image into contours using OpenCV."""
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required.")

    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def filter_contours(
    contours: list["np.ndarray"], min_size: float = 50, max_size: float = 300
) -> list["np.ndarray"]:
    """Filter contours by bounding-box size."""
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required.")

    filtered: list["np.ndarray"] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if max(w, h) < min_size or max(w, h) > max_size:
            continue
        filtered.append(cnt)

    return filtered


__all__ = ["detect_edges", "find_contours", "filter_contours"]
