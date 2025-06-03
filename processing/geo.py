"""Geospatial helper utilities."""

from __future__ import annotations

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
    return float(lon[0]), float(lat[0])


__all__ = ["pixel_to_coords"]
