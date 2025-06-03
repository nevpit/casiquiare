"""Utilities for the standalone Eyes agent."""

from __future__ import annotations

from typing import Any, Dict, List

try:
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None  # type: ignore

from detection import Feature
from detection.shapes import detect_shapes
from io.raster import load_raster


def _to_feature(idx: int, item: Dict[str, Any], source: str) -> Feature:
    """Convert a detection dictionary to a :class:`Feature`."""
    bbox = item.get("bbox")
    geometry = {"type": "bbox", "bbox": bbox} if bbox else {}
    width = float(item.get("width", 0.0))
    height = float(item.get("height", 0.0))
    return Feature(
        id=idx,
        feature_type=str(item.get("shape", "unknown")),
        geometry=geometry,
        dimensions=(width, height),
        confidence=float(item.get("score", 0.0)),
        source=source,
    )


def _scan_image(image: "np.ndarray", profile: Dict[str, Any] | None = None, source: str = "raster") -> List[Feature]:
    """Run shape detection on a raster array."""
    if detect_shapes is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required.")

    results = detect_shapes(image, profile)
    return [_to_feature(i + 1, feat, source) for i, feat in enumerate(results)]


def scan_area(area_id: str | Dict[str, Any]) -> List[Feature]:
    """Scan a raster or image dictionary for archaeological features."""
    if isinstance(area_id, str):
        if load_raster is None or np is None:
            raise RuntimeError("rasterio and NumPy are required.")
        data, transform, crs = load_raster(area_id)
        profile = {"transform": transform, "crs": crs}
        image = data[0] if data.ndim == 3 else data
        return _scan_image(image, profile, source="raster")

    if isinstance(area_id, dict):
        image = area_id.get("image")
        if image is None:
            raise ValueError("area_id dictionary requires an 'image' entry")
        profile = area_id.get("profile")
        return _scan_image(image, profile, source=area_id.get("source", "raster"))

    raise TypeError("area_id must be a str path or a dictionary")


__all__ = ["scan_area"]

