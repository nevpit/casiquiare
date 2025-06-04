"""Utility functions for the Eyes agent."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Iterable
from pathlib import Path

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
    from processing.lidar import (
        build_dtm,
        generate_lrm,
        generate_hillshade,
        write_geotiff,
    )
    from processing.image import to_uint8
    from io_utils.lidar import load_laz
    from io_utils.raster import load_raster
except Exception:  # pragma: no cover - library may be missing
    build_dtm = None
    generate_lrm = None
    generate_hillshade = None
    write_geotiff = None
    to_uint8 = None  # type: ignore
    load_laz = None
    load_raster = None

try:
    from pyproj import Transformer
except Exception:  # pragma: no cover - library may be missing
    Transformer = None

try:
    from detection.edges import multi_scale_canny
except Exception:  # pragma: no cover - library may be missing
    multi_scale_canny = None  # type: ignore

try:
    from detection import detect_shapes, shape_metrics
except Exception:  # pragma: no cover - library may be missing
    detect_shapes = None  # type: ignore
    shape_metrics = None  # type: ignore


def save_snippets(image: "np.ndarray", features: List[Dict[str, Any]], out_dir: str) -> List[str]:
    """Save cropped PNG snippets around detected features."""
    if cv2 is None or np is None or to_uint8 is None:
        raise RuntimeError("OpenCV, NumPy and to_uint8 are required.")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for idx, feat in enumerate(features):
        bbox = feat.get("bbox")
        if not bbox:
            continue
        x, y, w, h = bbox
        snippet = image[y : y + h, x : x + w]
        snippet_u8 = to_uint8(snippet) if snippet.dtype != np.uint8 else snippet
        file_path = out_path / f"snippet_{idx}.png"
        cv2.imwrite(str(file_path), snippet_u8)
        paths.append(str(file_path))
    return paths


def analyze_lidar(path: str, pipeline: Optional[Dict[str, Any]] = None) -> Iterable["np.ndarray"]:
    """Load and process a LiDAR point cloud using PDAL.

    Args:
        path: Path to the input LAZ or LAS file.
        pipeline: Optional PDAL pipeline specification. If ``None`` a simple
            reader pipeline is created.

    Returns:
        Sequence of point arrays in native PDAL ``numpy`` format.
    """
    if pdal is None:
        raise RuntimeError("PDAL is not installed.")
    pipeline = pipeline or {"pipeline": [path]}
    pl = pdal.Pipeline(json.dumps(pipeline))
    pl.execute()
    arrays = pl.arrays
    return arrays


def analyze_raster(path: str) -> Dict[str, Any]:
    """Read raster metadata using rasterio.

    Args:
        path: Path to the raster to inspect.

    Returns:
        Dictionary containing the raster's metadata profile.
    """
    if rasterio is None:
        raise RuntimeError("rasterio is not installed.")
    with rasterio.open(path) as src:
        meta = src.meta.copy()
    return meta


def transform_coordinates(x: float, y: float, from_epsg: int = 4326, to_epsg: int = 3857) -> Tuple[float, float]:
    """Transform coordinates between projections using pyproj.

    Args:
        x: X coordinate in the source CRS.
        y: Y coordinate in the source CRS.
        from_epsg: EPSG code of the source CRS.
        to_epsg: EPSG code of the destination CRS.

    Returns:
        Transformed ``(x, y)`` tuple.
    """
    if Transformer is None:
        raise RuntimeError("pyproj is not installed.")
    transformer = Transformer.from_crs(from_epsg, to_epsg, always_xy=True)
    x_out, y_out = transformer.transform(x, y)
    if to_epsg == 4326:
        return round(x_out, 6), round(y_out, 6)
    return x_out, y_out


def detect_image_features(path: str) -> Dict[str, Any]:
    """Detect simple features in an image using OpenCV.

    Args:
        path: Path to the image file to analyse.

    Returns:
        Dictionary describing detected contours.
    """
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required.")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image at {path}")

    edges = multi_scale_canny(img)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features: List[Dict[str, Any]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = float(cv2.contourArea(cnt))
        features.append({
            "bbox": (int(x), int(y), int(w), int(h)),
            "width": int(w),
            "height": int(h),
            "area": area,
        })

    return {"num_contours": len(contours), "features": features}


def lidar_tile_dtm(path: str, resolution: float = 1.0) -> Dict[str, Any]:
    """Generate a bare-earth DTM and visualizations from a LiDAR tile.

    Args:
        path: Path to the LiDAR tile.
        resolution: Resolution of the derived rasters in meters.

    Returns:
        Dictionary containing raster arrays and the rasterio profile.
    """
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

    if generate_hillshade is not None:
        try:
            hillshade = generate_hillshade(dtm)
        except Exception:  # pragma: no cover - dependency may be missing
            hillshade = None
    else:
        hillshade = None

    if hillshade is None:
        cellsize = profile["transform"][0]
        gy, gx = np.gradient(dtm, cellsize)
        slope = np.pi / 2.0 - np.arctan(np.sqrt(gx * gx + gy * gy))
        aspect = np.arctan2(-gx, gy)
        azimuth = np.deg2rad(315.0)
        altitude = np.deg2rad(45.0)
        hillshade = 255.0 * (
            np.sin(altitude) * np.sin(slope)
            + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
        )

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
    """Generate visualization rasters and detect shapes in a LiDAR tile.

    Args:
        path: Path to the LiDAR tile.
        resolution: Resolution of the derived rasters in meters.
        size_range: Tuple specifying the minimum and maximum feature size.

    Returns:
        Dictionary with rasters, features and the rasterio profile.
    """
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


def scan_area(
    path: str,
    resolution: float = 1.0,
    min_size: float = 50.0,
    max_size: float = 300.0,
) -> Dict[str, Any]:
    """Scan an area for geometric features using LiDAR data.

    Args:
        path: Path to the LiDAR tile.
        resolution: Output resolution for intermediate rasters.
        min_size: Minimum feature size to report.
        max_size: Maximum feature size to report.

    Returns:
        Dictionary with derived rasters and detected features.
    """
    size_range = (min_size, max_size)
    return lidar_feature_detection(path, resolution, size_range)


TOOLS: Dict[str, Any] = {
    "analyze_lidar": analyze_lidar,
    "analyze_raster": analyze_raster,
    "transform_coordinates": transform_coordinates,
    "detect_image_features": detect_image_features,
    "lidar_tile_dtm": lidar_tile_dtm,
    "lidar_feature_detection": lidar_feature_detection,
    "detect_shapes": detect_shapes,
    "save_snippets": save_snippets,
    "scan_area": scan_area,
}

__all__ = [
    "analyze_lidar",
    "analyze_raster",
    "transform_coordinates",
    "detect_image_features",
    "lidar_tile_dtm",
    "lidar_feature_detection",
    "detect_shapes",
    "save_snippets",
    "scan_area",
    "TOOLS",
]

