"""LiDAR processing utilities."""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple

try:  # Optional heavy dependencies
    import cv2
except Exception:  # pragma: no cover - library may be missing
    cv2 = None  # type: ignore

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover - library may be missing
    gaussian_filter = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None  # type: ignore

try:
    import rasterio
except Exception:  # pragma: no cover - library may be missing
    rasterio = None  # type: ignore

try:
    from osgeo import gdal
    gdal.UseExceptions()  # Avoid FutureWarning in GDAL >= 3
except Exception:  # pragma: no cover - library may be missing
    gdal = None  # type: ignore

try:
    import pdal
except Exception:  # pragma: no cover - library may be missing
    pdal = None  # type: ignore


def write_geotiff(path: str, array: "np.ndarray", profile: Dict[str, Any]) -> None:
    """Write a NumPy array to a GeoTIFF file using rasterio.

    Args:
        path: Destination file path.
        array: Image data to write.
        profile: Rasterio profile describing the image.
    """
    if rasterio is None:
        raise RuntimeError("rasterio is not installed.")
    if np is None:
        raise RuntimeError("NumPy is not installed.")

    arr = np.asarray(array)
    count = 1 if arr.ndim == 2 else arr.shape[0]

    prof = profile.copy()
    prof.update(driver="GTiff", count=count, dtype=arr.dtype)
    
    nodata = prof.get("nodata")
    if nodata is not None and np is not None:
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            if nodata < info.min or nodata > info.max:
                prof.pop("nodata")

    with rasterio.open(path, "w", **prof) as dst:
        if arr.ndim == 2:
            dst.write(arr, 1)
        else:
            dst.write(arr)


def generate_lrm(dtm: "np.ndarray", sigma: float = 5) -> "np.ndarray":
    """Generate a local relief model from a DTM using Gaussian blur.

    Args:
        dtm: Digital terrain model array.
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        The local relief model as a float array.
    """
    if np is None:
        raise RuntimeError("NumPy is required.")

    if cv2 is not None:
        blurred = cv2.GaussianBlur(dtm.astype(float), (0, 0), sigmaX=sigma)
    elif gaussian_filter is not None:
        blurred = gaussian_filter(dtm.astype(float), sigma=sigma)
    else:
        raise RuntimeError("OpenCV or SciPy is required to generate an LRM.")

    return dtm.astype(float) - blurred

def generate_hillshade(dtm: "np.ndarray", azimuth: int = 315, altitude: int = 45) -> "np.ndarray":
    """Create a hillshade image from a digital terrain model using GDAL.

    Args:
        dtm: Digital terrain model array.
        azimuth: Light source direction.
        altitude: Light source altitude angle.

    Returns:
        Hillshade image array.
    """
    if gdal is None or np is None:
        raise RuntimeError("GDAL and NumPy are required.")

    rows, cols = dtm.shape
    driver = gdal.GetDriverByName("MEM")
    src = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
    src.GetRasterBand(1).WriteArray(dtm)
    src.SetGeoTransform([0, 1, 0, 0, 0, -1])

    options = gdal.DEMProcessingOptions(azimuth=azimuth, altitude=altitude)
    dst_ds = gdal.DEMProcessing(
        "",                  # Output filename (empty means in-memory)
        src,                 # Source dataset
        "hillshade",         # Processing type
        options + ["-of", "MEM"]  # Options with MEM driver
    )

    if dst_ds is None:
        raise RuntimeError("gdal.DEMProcessing failed to generate hillshade")

    hillshade = dst_ds.ReadAsArray()
    return hillshade.reshape(dtm.shape)


def build_dtm(pipeline: "pdal.Pipeline", resolution: float = 1.0) -> Tuple["np.ndarray", dict]:
    """Execute a PDAL pipeline to generate a bare-earth DTM.

    Args:
        pipeline: Configured PDAL pipeline instance.
        resolution: Output resolution in meters.

    Returns:
        The DTM array and its rasterio profile.
    """
    if pdal is None:
        raise RuntimeError("PDAL is not installed.")
    if rasterio is None or np is None:
        raise RuntimeError("rasterio and NumPy are required.")

    # ``Pipeline`` exposes the pipeline specification via :py:meth:`toJSON`.
    # Earlier versions of the code used ``pipeline.json`` which does not exist
    # in the upstream PDAL API and caused an ``AttributeError`` at runtime.
    spec = json.loads(pipeline.toJSON())
    
    if isinstance(spec, dict):
        stages = spec.setdefault("pipeline", [])
    else:  # ``spec`` is already a list of stages
        stages = spec
        spec = {"pipeline": stages}

    stages.extend( 
        [
            {"type": "filters.smrf"},
            {"type": "filters.range", "limits": "Classification[2:2]"},
            {
                "type": "writers.gdal",
                "filename": "/vsimem/dtm.tif",
                "resolution": resolution,
                "output_type": "mean",
            },
        ]
    )

    dtm_pl = pdal.Pipeline(json.dumps(spec))
    dtm_pl.execute()

    if gdal is not None:
        ds = gdal.Open("/vsimem/dtm.tif")
        if ds is None:
            raise RuntimeError("PDAL pipeline failed to generate DTM")
        ds = None

    with rasterio.open("/vsimem/dtm.tif") as src:
        dtm = src.read(1)
        profile = src.profile

    return dtm, profile


__all__ = [
    "write_geotiff",
    "generate_lrm",
    "generate_hillshade",
    "build_dtm",
]
