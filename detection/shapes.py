"""Shape detection helper functions."""

from __future__ import annotations

try:
    import cv2
except Exception:  # pragma: no cover - library may be missing
    cv2 = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None  # type: ignore

from .lidar import shape_metrics
from typing import Any, Dict, List, Optional, Tuple


def detect_shapes(
    image: "np.ndarray",
    profile: Optional[Dict[str, Any]] = None,
    size_range: Tuple[float, float] = (50.0, 300.0),
) -> List[Dict[str, Any]]:
    """Detect geometric shapes in a relief image.

    Args:
        image: Grayscale relief image as a NumPy array.
        profile: Optional rasterio profile describing the pixel size.
        size_range: Minimum and maximum feature size in the units of the
            ``profile`` transform or pixels if ``profile`` is ``None``.

    Returns:
        A list of detected shape feature dictionaries.
    """
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required.")

    arr = image.astype(np.float32)
    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    arr_u8 = arr.astype(np.uint8)

    edges = cv2.Canny(arr_u8, 100, 200)
    edges = cv2.dilate(edges, None)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cellsize = None
    if profile is not None:
        cellsize = profile.get("transform", [1])[0]

    min_size, max_size = size_range
    target_center = (min_size + max_size) / 2.0
    target_half = max((max_size - min_size) / 2.0, 1.0)
    features: List[Dict[str, Any]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        width = w if cellsize is None else w * cellsize
        height = h if cellsize is None else h * cellsize
        if max(width, height) < min_size or max(width, height) > max_size:
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        shape = "circle"
        if len(approx) == 4:
            shape = "rectangle"
        elif len(approx) > 4:
            shape = "polygon"

        mask = np.zeros(edges.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        # Normalized mean of Canny edge values within the contour.
        # Higher values indicate sharper, well-defined boundaries.
        edge_strength = float(cv2.mean(edges, mask=mask)[0]) / 255.0

        metrics = shape_metrics(cnt) if shape_metrics is not None else {}
        if metrics:
            if shape == "circle":
                shape_reg = metrics.get("circularity", 0.0)
            elif shape == "rectangle":
                ratio_score = 1.0 - abs(1.0 - metrics.get("aspect_ratio", 0.0))
                vert_score = 1.0 - abs(len(approx) - 4) / 4.0
                shape_reg = (ratio_score + vert_score) / 2.0
            else:
                shape_reg = metrics.get("circularity", 0.0)
        else:
            shape_reg = 0.0
        # Regularity score of the shape; ensures we favor well-formed geometries.
        shape_reg = max(0.0, min(1.0, shape_reg))

        size = max(width, height)
        size_conf = 1.0 - abs(size - target_center) / target_half
        size_conf = max(0.0, min(1.0, size_conf))

        # Final detection confidence.  Edges and shape regularity are
        # considered equally important (40% each), while size contributes 20%.
        score = 0.4 * edge_strength + 0.4 * shape_reg + 0.2 * size_conf

        features.append(
            {
                "bbox": (x, y, w, h),
                "width": width,
                "height": height,
                "num_vertices": len(approx),
                "shape": shape,
                "score": float(score),
            }
        )

    return features


__all__ = ["detect_shapes"]
