"""LiDAR processing utilities."""

from __future__ import annotations

from typing import Any, Dict

try:
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None

try:
    import rasterio
except Exception:  # pragma: no cover - library may be missing
    rasterio = None


def write_geotiff(path: str, array: "np.ndarray", profile: Dict[str, Any]) -> None:
    """Write a NumPy array to a GeoTIFF file using rasterio.

    Parameters
    ----------
    path:
        Destination file path.
    array:
        Data array to write. If 2D, a single band is written; if 3D, the first
        dimension is treated as band count.
    profile:
        Rasterio profile containing CRS and transform information.
    """
    if rasterio is None:
        raise RuntimeError("rasterio is not installed.")
    if np is None:
        raise RuntimeError("NumPy is not installed.")

    arr = np.asarray(array)
    count = 1 if arr.ndim == 2 else arr.shape[0]

    prof = profile.copy()
    prof.update(driver="GTiff", count=count, dtype=arr.dtype)

    with rasterio.open(path, "w", **prof) as dst:
        if arr.ndim == 2:
            dst.write(arr, 1)
        else:
            dst.write(arr)
