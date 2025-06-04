"""LiDAR feature detection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot
import logging

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

from typing import Any, Dict

from log_config import setup_logger

logger = setup_logger(__name__)


def shape_metrics(contour: "np.ndarray") -> Dict[str, float]:
    """Return basic geometric metrics for a contour."""
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


@dataclass
class Line:
    """Representation of a detected line segment."""

    start: tuple[int, int]
    end: tuple[int, int]
    length: float
    angle: float


def detect_edges(image: "np.ndarray") -> "np.ndarray":
    """Detect edges in a relief image using the Canny algorithm."""
    if cv2 is None or np is None:
        logger.warning("OpenCV or NumPy missing; cannot detect edges")
        raise RuntimeError("OpenCV and NumPy are required.")

    logger.debug("Detecting edges")

    arr = image
    if to_uint8 is not None:
        arr_u8 = to_uint8(arr)
    else:
        arr_u8 = arr.astype(np.uint8)

    blurred = cv2.GaussianBlur(arr_u8, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 90)
    logger.debug("Edge detection complete")
    return edges


def shape_metrics(contour: "np.ndarray") -> dict:
    """Return basic geometric metrics for a contour."""
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

def find_contours(edge_img: "np.ndarray") -> list["np.ndarray"]:
    """Convert an edge image into contours using OpenCV."""
    if cv2 is None or np is None:
        logger.warning("OpenCV or NumPy missing; cannot find contours")
        raise RuntimeError("OpenCV and NumPy are required.")

    logger.debug("Finding contours")

    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.debug("Found %d contours", len(contours))
    return list(contours)


def filter_contours(
    contours: list["np.ndarray"], min_size: float = 50, max_size: float = 300
) -> list["np.ndarray"]:
    """Filter contours by bounding-box size."""
    if cv2 is None or np is None:
        logger.warning("OpenCV or NumPy missing; cannot filter contours")
        raise RuntimeError("OpenCV and NumPy are required.")

    logger.debug("Filtering %d contours", len(contours))

    filtered: list["np.ndarray"] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if max(w, h) < min_size or max(w, h) > max_size:
            continue
        filtered.append(cnt)

    logger.debug("%d contours remaining after filtering", len(filtered))
    return filtered


def detect_lines(edge_img: "np.ndarray") -> list[Line]:
    """Detect straight lines in an edge image using the probabilistic Hough transform."""
    if cv2 is None or np is None:
        logger.warning("OpenCV or NumPy missing; cannot detect lines")
        raise RuntimeError("OpenCV and NumPy are required.")

    logger.debug("Detecting lines")

    lines = cv2.HoughLinesP(edge_img, 1, np.pi / 180.0, 30, minLineLength=30, maxLineGap=5)
    result: list[Line] = []
    if lines is None:
        logger.debug("No lines detected")
        return result

    for x1, y1, x2, y2 in lines[:, 0]:
        length = float(hypot(x2 - x1, y2 - y1))
        angle = float(degrees(atan2(y2 - y1, x2 - x1)))
        result.append(Line(start=(int(x1), int(y1)), end=(int(x2), int(y2)), length=length, angle=angle))

    logger.debug("Detected %d lines", len(result))
    return result


__all__ = [
    "shape_metrics",
    "Line",
    "detect_edges",
    "find_contours",
    "filter_contours",
    "detect_lines",
    "shape_metrics",
]
