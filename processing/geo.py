"""Geospatial helper utilities."""

from __future__ import annotations

from typing import Any

try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.crs import CRS
    from rasterio.warp import transform as rio_transform
except Exception:  # pragma: no cover - library may be missing
    rasterio = None  # type: ignore
    Affine = None  # type: ignore
    CRS = None  # type: ignore
    rio_transform = None  # type: ignore


def pixel_to_coords(
    row: int,
    col: int,
    transform: "Affine",
    crs: "CRS",
) -> tuple[float, float]:
    """Convert pixel indices to WGS-84 lon/lat coordinates."""
    if rasterio is None:
        raise RuntimeError("rasterio is not installed.")

    x, y = transform * (col + 0.5, row + 0.5)
    lon, lat = rio_transform(crs, "EPSG:4326", [x], [y])
    return round(float(lon[0]), 6), round(float(lat[0]), 6)


def contour_to_polygon(
    contour: "Any",
    transform: "Affine",
    crs: "CRS",
) -> list[list[float]]:
    """Convert an OpenCV contour to GeoJSON polygon coordinates."""
    if rasterio is None:
        raise RuntimeError("rasterio is not installed.")

    if contour is None:
        raise ValueError("contour is required")

    coords = contour.reshape(-1, 2)
    xs = []
    ys = []
    for col, row in coords:
        x, y = transform * (float(col) + 0.5, float(row) + 0.5)
        xs.append(x)
        ys.append(y)

    lon, lat = rio_transform(crs, "EPSG:4326", xs, ys)
    ring = [
        [round(float(lon[i]), 6), round(float(lat[i]), 6)]
        for i in range(len(lon))
    ]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


__all__ = ["pixel_to_coords", "contour_to_polygon"]
