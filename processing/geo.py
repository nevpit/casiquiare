"""Geospatial helper utilities."""

from __future__ import annotations

from typing import Any

try:
    from shapely.geometry import Polygon  # type: ignore
except Exception:  # pragma: no cover - library may be missing
    Polygon = None  # type: ignore

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


def bbox_to_polygon(
    bbox: tuple[int, int, int, int],
    transform: "Affine",
    crs: "CRS",
    *,
    as_shapely: bool = False,
) -> "Polygon" | list[list[float]]:
    """Convert a bounding box to a polygon in EPSG:4326 coordinates."""
    if rasterio is None:
        raise RuntimeError("rasterio is not installed.")

    if bbox is None:
        raise ValueError("bbox is required")

    x, y, w, h = bbox
    pts = [
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y + h),
        (x, y),
    ]

    xs = []
    ys = []
    for col, row in pts:
        xp, yp = transform * (float(col) + 0.5, float(row) + 0.5)
        xs.append(xp)
        ys.append(yp)

    lon, lat = rio_transform(crs, "EPSG:4326", xs, ys)
    ring = [
        [round(float(lon[i]), 6), round(float(lat[i]), 6)]
        for i in range(len(lon))
    ]

    if as_shapely:
        if Polygon is None:
            raise RuntimeError("Shapely is not installed.")
        return Polygon(ring)
    return ring


def bbox_centroid(
    bbox: tuple[int, int, int, int], transform: "Affine", crs: "CRS"
) -> tuple[float, float]:
    """Return the centroid of ``bbox`` in WGS-84 coordinates."""
    if rasterio is None:
        raise RuntimeError("rasterio is not installed.")

    x, y, w, h = bbox
    cx, cy = transform * (float(x) + w / 2.0, float(y) + h / 2.0)
    lon, lat = rio_transform(crs, "EPSG:4326", [cx], [cy])
    return round(float(lon[0]), 6), round(float(lat[0]), 6)


__all__ = [
    "pixel_to_coords",
    "contour_to_polygon",
    "bbox_to_polygon",
    "bbox_centroid",
]
