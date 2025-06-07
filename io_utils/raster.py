"""Raster loading utilities."""

from __future__ import annotations

from typing import Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None  # type: ignore

try:
    import rasterio
    from rasterio.crs import CRS
    from affine import Affine
except Exception:  # pragma: no cover - library may be missing
    rasterio = None  # type: ignore
    CRS = None  # type: ignore
    Affine = None  # type: ignore


def load_raster(path: str) -> Tuple["np.ndarray", "Affine", "CRS"]:
    """Load all bands from a raster file with transform and CRS."""
    if rasterio is None or np is None:
        raise RuntimeError("rasterio and NumPy are required.")

    with rasterio.open(path) as src:
        data = src.read()
        transform = src.transform
        crs = src.crs

    return data, transform, crs


__all__ = ["load_raster"]
