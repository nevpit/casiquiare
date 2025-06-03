"""LiDAR shape detection helpers."""

from __future__ import annotations

from typing import Dict

try:
    import cv2
except Exception:  # pragma: no cover - library may be missing
    cv2 = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None  # type: ignore


def shape_metrics(contour: "np.ndarray") -> Dict[str, float]:
    """Return basic geometric metrics for a contour.

    Args:
        contour: OpenCV contour array.

    Returns:
        Dictionary with area, perimeter, aspect ratio and circularity.
    """
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required.")

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h != 0 else 0.0

    circularity = (4 * np.pi * area / (perimeter * perimeter)) if perimeter != 0 else 0.0

    return {
        "area": float(area),
        "perimeter": float(perimeter),
        "aspect_ratio": float(aspect_ratio),
        "circularity": float(circularity),
    }


__all__ = ["shape_metrics"]
