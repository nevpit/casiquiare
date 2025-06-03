"""Utility functions for the Eyes agent."""

from __future__ import annotations

import json
import os
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
    from io.lidar import load_laz
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
    x_out, y_out = transformer.transform(x, y)
    if to_epsg == 4326:
        return round(x_out, 6), round(y_out, 6)
    return x_out, y_out


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


def save_snippets(
    image: "np.ndarray",
    features: List[Dict[str, Any]],
    out_dir: str,
    size: int = 256,
) -> List[str]:
    """Save 2D crops around detected features as PNG files."""
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required.")
    if to_uint8 is None:
        raise RuntimeError("to_uint8 utility is not available.")

    arr = image.astype(np.float32)
    if arr.dtype != np.uint8:
        arr = to_uint8(arr)

    os.makedirs(out_dir, exist_ok=True)
    height, width = arr.shape[:2]
    half = size // 2
    paths: List[str] = []

    for idx, feat in enumerate(features):
        bbox = feat.get("bbox")
        if not bbox:
            continue
        x, y, w, h = bbox
        cx = int(x + w // 2)
        cy = int(y + h // 2)
        x_min = max(cx - half, 0)
        y_min = max(cy - half, 0)
        x_max = min(cx + half, width)
        y_max = min(cy + half, height)
        crop = arr[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)
        fname = os.path.join(out_dir, f"candidate_{idx:03d}.png")
        cv2.imwrite(fname, crop)
        paths.append(fname)

    return paths


def write_geotiff_wrapper(path: str, array: "np.ndarray", profile: Dict[str, Any]) -> None:
    """Write an array to GeoTIFF using processing.lidar.write_geotiff."""
    if write_geotiff is None:
        raise RuntimeError("write_geotiff utility is not available.")
    write_geotiff(path, array, profile)


def generate_lrm_wrapper(dtm: "np.ndarray", sigma: float = 5) -> "np.ndarray":
    """Wrapper around processing.lidar.generate_lrm."""
    if generate_lrm is None:
        raise RuntimeError("generate_lrm utility is not available.")
    return generate_lrm(dtm, sigma)


def generate_hillshade_wrapper(dtm: "np.ndarray", azimuth: int = 315, altitude: int = 45) -> "np.ndarray":
    """Wrapper around processing.lidar.generate_hillshade."""
    if generate_hillshade is None:
        raise RuntimeError("generate_hillshade utility is not available.")
    return generate_hillshade(dtm, azimuth, altitude)


def build_dtm_wrapper(pipeline: Any, resolution: float = 1.0) -> Tuple["np.ndarray", dict]:
    """Wrapper around processing.lidar.build_dtm."""
    if build_dtm is None:
        raise RuntimeError("build_dtm utility is not available.")
    return build_dtm(pipeline, resolution)


def load_laz_wrapper(path: str) -> Any:
    """Wrapper around io.lidar.load_laz."""
    if load_laz is None:
        raise RuntimeError("load_laz utility is not available.")
    return load_laz(path)


def load_raster_wrapper(
    path: str,
) -> Tuple["np.ndarray", "affine.Affine", "rasterio.crs.CRS"]:
    """Wrapper around io.raster.load_raster."""
    if load_raster is None:
        raise RuntimeError("load_raster utility is not available.")
    return load_raster(path)


TOOLS: Dict[str, Any] = {
    "analyze_lidar": analyze_lidar,
    "analyze_raster": analyze_raster,
    "transform_coordinates": transform_coordinates,
    "detect_image_features": detect_image_features,
    "lidar_tile_dtm": lidar_tile_dtm,
    "lidar_feature_detection": lidar_feature_detection,
    "detect_shapes": detect_shapes,
    "save_snippets": save_snippets,
    "write_geotiff": write_geotiff_wrapper,
    "generate_lrm": generate_lrm_wrapper,
    "generate_hillshade": generate_hillshade_wrapper,
    "build_dtm": build_dtm_wrapper,
    "load_laz": load_laz_wrapper,
    "load_raster": load_raster_wrapper,
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
    "write_geotiff",
    "generate_lrm",
    "generate_hillshade",
    "build_dtm",
    "load_laz",
    "load_raster",
    "TOOLS",
]

