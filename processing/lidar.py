=======
from __future__ import annotations

from typing import Optional

try:
=======
"""LiDAR processing utilities."""

from __future__ import annotations

try:  # Optional heavy dependencies
    import cv2
except Exception:  # pragma: no cover - library may be missing
    cv2 = None

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover - library may be missing
    gaussian_filter = None

try:
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None


def generate_lrm(dtm: np.ndarray, sigma: float = 5) -> np.ndarray:
    """Generate a local relief model from a DTM using Gaussian blur."""
    if np is None:
        raise RuntimeError("NumPy is required.")

    if cv2 is not None:
        blurred = cv2.GaussianBlur(dtm.astype(float), (0, 0), sigmaX=sigma)
    elif gaussian_filter is not None:
        blurred = gaussian_filter(dtm.astype(float), sigma=sigma)
    else:
        raise RuntimeError("OpenCV or SciPy is required to generate an LRM.")

    return dtm.astype(float) - blurred


__all__ = ["generate_lrm"]
=======
import json
from typing import Any, Tuple

try:  # Optional heavy dependencies
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None  # type: ignore

try:
    from osgeo import gdal
except Exception:  # pragma: no cover - library may be missing
    gdal = None  # type: ignore


def generate_hillshade(dtm: "np.ndarray", azimuth: int = 315, altitude: int = 45) -> "np.ndarray":
    """Create a hillshade image from a digital terrain model using GDAL.

    Parameters
    ----------
    dtm : np.ndarray
        2D array containing elevation values.
    azimuth : int, optional
        Direction of the light source in degrees, by default 315.
    altitude : int, optional
        Altitude angle of the light source in degrees, by default 45.

    Returns
    -------
    np.ndarray
        Hillshade raster as a NumPy array with the same shape as ``dtm``.
    """
    if gdal is None or np is None:
        raise RuntimeError("GDAL and NumPy are required.")

    rows, cols = dtm.shape
    driver = gdal.GetDriverByName("MEM")
    src = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
    src.GetRasterBand(1).WriteArray(dtm)
    src.SetGeoTransform([0, 1, 0, 0, 0, -1])

    options = gdal.DEMProcessingOptions(azimuth=azimuth, altitude=altitude)
    dst = gdal.DEMProcessing("", src, "hillshade", options=options)

    hillshade = dst.ReadAsArray()
    return hillshade.reshape(dtm.shape)
=======
    import rasterio
except Exception:  # pragma: no cover - library may be missing
    rasterio = None  # type: ignore

try:
    import pdal
except Exception:  # pragma: no cover - library may be missing
    pdal = None  # type: ignore


def build_dtm(pipeline: "pdal.Pipeline", resolution: float = 1.0) -> Tuple["np.ndarray", dict]:
    """Execute a PDAL pipeline to generate a bare-earth DTM.

    Parameters
    ----------
    pipeline:
        A preconfigured :class:`pdal.Pipeline` with the data source steps
        already defined.
    resolution:
        Desired output resolution for the raster, in the units of the
        point cloud's coordinate system.

    Returns
    -------
    Tuple[np.ndarray, dict]
        The DTM array and its raster profile.
    """

    if pdal is None:
        raise RuntimeError("PDAL is not installed.")
    if rasterio is None or np is None:
        raise RuntimeError("rasterio and NumPy are required.")

    # Parse the existing pipeline specification so we can append steps
    spec = json.loads(pipeline.json)
    spec["pipeline"].extend(
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

    with rasterio.open("/vsimem/dtm.tif") as src:
        dtm = src.read(1)
        profile = src.profile

    return dtm, profile
