"""Utility functions for the Eyes agent."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

try:  # Optional heavy dependencies
    import cv2
except Exception:  # pragma: no cover - library may be missing
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None

try:
    import rasterio
except Exception:  # pragma: no cover - library may be missing
    rasterio = None

try:
    import pdal
except Exception:  # pragma: no cover - library may be missing
    pdal = None

try:
    from processing.lidar import build_dtm, generate_lrm, generate_hillshade
    from processing.image import to_uint8
except Exception:  # pragma: no cover - library may be missing
    build_dtm = None
    generate_lrm = None
    generate_hillshade = None
    to_uint8 = None  # type: ignore

try:
    from pyproj import Transformer
except Exception:  # pragma: no cover - library may be missing
    Transformer = None

try:
    from detection.lidar import filter_contours
except Exception:  # pragma: no cover - library may be missing
    filter_contours = None  # type: ignore


def analyze_lidar(path: str, pipeline: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Load and process a LiDAR point cloud using PDAL."""
    if pdal is None:
        raise RuntimeError("PDAL is not installed.")
    pipeline = pipeline or {"pipeline": [path]}
    pl = pdal.Pipeline(json.dumps(pipeline))
    pl.execute()
    arrays = pl.arrays
    return [arr.tolist() for arr in arrays]


def analyze_raster(path: str) -> Dict[str, Any]:
    """Read raster metadata using rasterio."""
    if rasterio is None:
        raise RuntimeError("rasterio is not installed.")
    with rasterio.open(path) as src:
        meta = src.meta.copy()
    return meta


def transform_coordinates(x: float, y: float, from_epsg: int = 4326, to_epsg: int = 3857) -> Tuple[float, float]:
    """Transform coordinates between projections using pyproj."""
    if Transformer is None:
        raise RuntimeError("pyproj is not installed.")
    transformer = Transformer.from_crs(from_epsg, to_epsg, always_xy=True)
    return transformer.transform(x, y)


def detect_image_features(path: str) -> Dict[str, Any]:
    """Detect simple features in an image using OpenCV."""
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required.")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image at {path}")
    edges = cv2.Canny(img, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return {"num_contours": len(contours)}


def lidar_tile_dtm(path: str, resolution: float = 1.0) -> Dict[str, Any]:
    """Generate a bare-earth DTM and visualizations from a LiDAR tile."""
    if pdal is None:
        raise RuntimeError("PDAL is not installed.")
    if rasterio is None or np is None:
        raise RuntimeError("rasterio and NumPy are required.")

    pipeline = {"pipeline": [path]}
    pl = pdal.Pipeline(json.dumps(pipeline))
    if build_dtm is None:
        raise RuntimeError("build_dtm utility is not available.")
    dtm, profile = build_dtm(pl, resolution)

    if to_uint8 is not None:
        dtm_u8 = to_uint8(dtm)
    else:
        dtm_u8 = dtm.astype(np.uint8) if np is not None else dtm

    cellsize = profile["transform"][0]
    gy, gx = np.gradient(dtm, cellsize)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(gx * gx + gy * gy))
    aspect = np.arctan2(-gx, gy)
    azimuth = np.deg2rad(315.0)
    altitude = np.deg2rad(45.0)
    hillshade = 255.0 * (np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect))
    if to_uint8 is not None:
        hillshade_u8 = to_uint8(hillshade)
    else:
        hillshade_u8 = np.clip(hillshade, 0, 255).astype(np.uint8)

    local_relief = None
    if generate_lrm is not None:
        try:
            local_relief = generate_lrm(dtm, sigma=5)
            if to_uint8 is not None:
                local_relief = to_uint8(local_relief)
        except Exception:  # pragma: no cover - dependency may be missing
            local_relief = None

    return {
        "dtm": dtm_u8,
        "hillshade": hillshade_u8,
        "local_relief": local_relief,
        "profile": profile,
    }


def lidar_feature_detection(
    path: str,
    resolution: float = 1.0,
    size_range: Tuple[float, float] = (50.0, 300.0),
) -> Dict[str, Any]:
    """Generate visualization rasters and detect shapes in a LiDAR tile."""
    if pdal is None:
        raise RuntimeError("PDAL is not installed.")
    if rasterio is None or np is None:
        raise RuntimeError("rasterio and NumPy are required.")
    if cv2 is None:
        raise RuntimeError("OpenCV is required.")
    if build_dtm is None or generate_lrm is None or generate_hillshade is None:
        raise RuntimeError("LiDAR utilities are not available.")
    if to_uint8 is None:
        raise RuntimeError("to_uint8 utility is not available.")

    pipeline = {"pipeline": [path]}
    pl = pdal.Pipeline(json.dumps(pipeline))
    dtm, profile = build_dtm(pl, resolution)

    hillshade = generate_hillshade(dtm)
    local_relief = generate_lrm(dtm, sigma=5)

    dtm_u8 = to_uint8(dtm)
    hillshade_u8 = to_uint8(hillshade)
    local_relief_u8 = to_uint8(local_relief)

    features = detect_shapes(local_relief_u8, profile, size_range)

    return {
        "dtm": dtm_u8,
        "hillshade": hillshade_u8,
        "local_relief": local_relief_u8,
        "features": features,
        "profile": profile,
    }


def detect_shapes(image: "np.ndarray", profile: Optional[Dict[str, Any]] = None, size_range: Tuple[float, float] = (50.0, 300.0)) -> List[Dict[str, Any]]:
    """Detect geometric shapes in a relief image."""
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
    if filter_contours is not None:
        px_min = min_size if cellsize is None else min_size / cellsize
        px_max = max_size if cellsize is None else max_size / cellsize
        contours = filter_contours(contours, min_size=px_min, max_size=px_max)
    features: List[Dict[str, Any]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        width = w if cellsize is None else w * cellsize
        height = h if cellsize is None else h * cellsize
        if filter_contours is None and (max(width, height) < min_size or max(width, height) > max_size):
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        shape = "circle"
        if len(approx) == 4:
            shape = "rectangle"
        elif len(approx) > 4:
            shape = "polygon"

        features.append({
            "bbox": (x, y, w, h),
            "width": width,
            "height": height,
            "num_vertices": len(approx),
            "shape": shape,
        })

    return features


TOOLS: Dict[str, Any] = {
    "analyze_lidar": analyze_lidar,
    "analyze_raster": analyze_raster,
    "transform_coordinates": transform_coordinates,
    "detect_image_features": detect_image_features,
    "lidar_tile_dtm": lidar_tile_dtm,
    "lidar_feature_detection": lidar_feature_detection,
    "detect_shapes": detect_shapes,
}

__all__ = [
    "analyze_lidar",
    "analyze_raster",
    "transform_coordinates",
    "detect_image_features",
    "lidar_tile_dtm",
    "lidar_feature_detection",
    "detect_shapes",
    "TOOLS",
]

