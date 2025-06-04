"""Utility functions for the Eyes agent."""

from __future__ import annotations

import json
import logging
from pathlib import Path
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
    from processing.lidar import (
        build_dtm,
        generate_lrm,
        generate_hillshade,
        write_geotiff,
    )
    from processing.image import to_uint8
    from io_helpers.lidar import load_laz
    from io.raster import load_raster
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
    from detection import detect_shapes, shape_metrics
except Exception:  # pragma: no cover - library may be missing
    detect_shapes = None  # type: ignore
    shape_metrics = None  # type: ignore

from log_config import setup_logger

logger = setup_logger(__name__)


def analyze_lidar(path: str, pipeline: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Load and process a LiDAR point cloud using PDAL.

    Args:
        path: Path to the input LAZ or LAS file.
        pipeline: Optional PDAL pipeline specification. If ``None`` a simple
            reader pipeline is created.

    Returns:
        List of point arrays in native PDAL format converted to Python lists.
    """
    if pdal is None:
        logger.warning("PDAL is not installed; cannot analyse %s", path)
        raise RuntimeError("PDAL is not installed.")

    logger.info("Starting LiDAR analysis for %s", path)
    pipeline = pipeline or {"pipeline": [path]}
    pl = pdal.Pipeline(json.dumps(pipeline))
    pl.execute()
    arrays = pl.arrays
    logger.info("Finished LiDAR analysis for %s", path)
    return [arr.tolist() for arr in arrays]


def analyze_raster(path: str) -> Dict[str, Any]:
    """Read raster metadata using rasterio.

    Args:
        path: Path to the raster to inspect.

    Returns:
        Dictionary containing the raster's metadata profile.
    """
    if rasterio is None:
        logger.warning("rasterio is not installed; cannot read %s", path)
        raise RuntimeError("rasterio is not installed.")
    logger.info("Reading raster metadata from %s", path)
    with rasterio.open(path) as src:
        meta = src.meta.copy()
    logger.debug("Raster %s profile: %s", path, meta)
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
        logger.warning("pyproj is not installed; cannot transform coordinates")
        raise RuntimeError("pyproj is not installed.")
    logger.debug(
        "Transforming coordinates (%s, %s) from EPSG:%s to EPSG:%s",
        x,
        y,
        from_epsg,
        to_epsg,
    )
    transformer = Transformer.from_crs(from_epsg, to_epsg, always_xy=True)
    x_out, y_out = transformer.transform(x, y)
    if to_epsg == 4326:
        result = (round(x_out, 6), round(y_out, 6))
    else:
        result = (x_out, y_out)
    logger.debug("Resulting coordinates: %s", result)
    return result


def detect_image_features(path: str) -> Dict[str, Any]:
    """Detect simple features in an image using OpenCV.

    Args:
        path: Path to the image file to analyse.

    Returns:
        Dictionary describing detected contours.
    """
    if cv2 is None or np is None:
        logger.warning("OpenCV or NumPy missing; cannot process %s", path)
        raise RuntimeError("OpenCV and NumPy are required.")

    logger.info("Detecting image features in %s", path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.error("Unable to read image at %s", path)
        raise ValueError(f"Unable to read image at {path}")

    edges = cv2.Canny(img, 100, 200)
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

    result = {"num_contours": len(contours), "features": features}
    logger.info("Found %d contours in %s", len(contours), path)
    return result


def lidar_tile_dtm(path: str, resolution: float = 1.0) -> Dict[str, Any]:
    """Generate a bare-earth DTM and visualizations from a LiDAR tile.

    Args:
        path: Path to the LiDAR tile.
        resolution: Resolution of the derived rasters in meters.

    Returns:
        Dictionary containing raster arrays and the rasterio profile.
    """
    if pdal is None:
        logger.warning("PDAL is not installed; cannot process %s", path)
        raise RuntimeError("PDAL is not installed.")
    if rasterio is None or np is None:
        logger.warning("rasterio or NumPy missing; cannot process %s", path)
        raise RuntimeError("rasterio and NumPy are required.")

    logger.info("Building DTM for %s", path)

    pipeline = {"pipeline": [path]}
    pl = pdal.Pipeline(json.dumps(pipeline))
    if build_dtm is None:
        logger.warning("build_dtm utility is not available")
        raise RuntimeError("build_dtm utility is not available.")
    dtm, profile = build_dtm(pl, resolution)

    if to_uint8 is not None:
        dtm_u8 = to_uint8(dtm)
    else:
        dtm_u8 = dtm.astype(np.uint8) if np is not None else dtm

    if generate_hillshade is not None:
        try:
            hillshade = generate_hillshade(dtm)
        except Exception as exc:  # pragma: no cover - dependency may be missing
            logger.warning("Hillshade generation failed: %s", exc)
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
        except Exception as exc:  # pragma: no cover - dependency may be missing
            logger.warning("LRM generation failed: %s", exc)
            local_relief = None

    logger.info("Finished building DTM for %s", path)
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
        logger.warning("PDAL is not installed; cannot analyse %s", path)
        raise RuntimeError("PDAL is not installed.")
    if rasterio is None or np is None:
        logger.warning("rasterio or NumPy missing; cannot analyse %s", path)
        raise RuntimeError("rasterio and NumPy are required.")
    if cv2 is None:
        logger.warning("OpenCV is required for feature detection")
        raise RuntimeError("OpenCV is required.")
    if build_dtm is None or generate_lrm is None or generate_hillshade is None:
        logger.warning("LiDAR utilities are not available")
        raise RuntimeError("LiDAR utilities are not available.")
    if to_uint8 is None:
        logger.warning("to_uint8 utility is not available")
        raise RuntimeError("to_uint8 utility is not available.")

    logger.info("Running feature detection on %s", path)

    pipeline = {"pipeline": [path]}
    pl = pdal.Pipeline(json.dumps(pipeline))
    dtm, profile = build_dtm(pl, resolution)

    hillshade = generate_hillshade(dtm)
    local_relief = generate_lrm(dtm, sigma=5)

    dtm_u8 = to_uint8(dtm)
    hillshade_u8 = to_uint8(hillshade)
    local_relief_u8 = to_uint8(local_relief)

    features = detect_shapes(local_relief_u8, profile, size_range)
    logger.info("Detected %d features in %s", len(features), path)

    logger.info("Finished feature detection for %s", path)
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
    logger.info(
        "Scanning %s with resolution=%s m and size range %.1f-%.1f", path, resolution, min_size, max_size
    )
    size_range = (min_size, max_size)
    result = lidar_feature_detection(path, resolution, size_range)
    logger.info("Finished scanning %s", path)
    return result


def save_snippets(image: "np.ndarray", features: List[Dict[str, Any]], out_dir: str) -> List[str]:
    """Save 256x256 PNG snippets around feature bounding boxes."""

    if cv2 is None or np is None or to_uint8 is None:
        logger.warning("Required dependencies missing; cannot save snippets")
        raise RuntimeError("OpenCV, NumPy and to_uint8 are required.")

    logger.info("Saving %d snippets to %s", len(features), out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    half = 128
    for idx, feat in enumerate(features):
        bbox = feat.get("bbox")
        if not bbox:
            continue
        x, y, w, h = [int(v) for v in bbox]
        cx, cy = x + w // 2, y + h // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(image.shape[1], cx + half)
        y2 = min(image.shape[0], cy + half)
        snippet = image[y1:y2, x1:x2]
        snippet_u8 = to_uint8(snippet)
        fname = Path(out_dir) / f"snippet_{idx}.png"
        cv2.imwrite(str(fname), snippet_u8)
        paths.append(str(fname))
        logger.debug("Saved snippet %s", fname)
    return paths


TOOLS: Dict[str, Any] = {
    "analyze_lidar": analyze_lidar,
    "analyze_raster": analyze_raster,
    "transform_coordinates": transform_coordinates,
    "detect_image_features": detect_image_features,
    "lidar_tile_dtm": lidar_tile_dtm,
    "lidar_feature_detection": lidar_feature_detection,
    "detect_shapes": detect_shapes,
    "scan_area": scan_area,
    "save_snippets": save_snippets,
}

__all__ = [
    "analyze_lidar",
    "analyze_raster",
    "transform_coordinates",
    "detect_image_features",
    "lidar_tile_dtm",
    "lidar_feature_detection",
    "detect_shapes",
    "scan_area",
    "save_snippets",
    "TOOLS",
]

