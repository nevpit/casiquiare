from __future__ import annotations

from typing import Optional

try:
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
