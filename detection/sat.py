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


from .lidar import detect_lines, filter_contours, find_contours


def detect_geometric_outlines(
    edge_img: "np.ndarray",
    min_size: float = 50.0,
    max_size: float = 300.0,
) -> dict:
    """Detect large geometric clearings from an edge image."""
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required.")

    contours = find_contours(edge_img)
    contours = filter_contours(contours, min_size=min_size, max_size=max_size)

    outline_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        outline_contours.append({"bbox": (int(x), int(y), int(w), int(h)), "contour": cnt})

    lines = detect_lines(edge_img)

    return {"contours": outline_contours, "lines": lines}


__all__ = ["detect_edges", "detect_geometric_outlines"]
