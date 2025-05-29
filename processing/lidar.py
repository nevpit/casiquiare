"""LiDAR processing utilities."""

from __future__ import annotations

import json
from typing import Any, Tuple

try:  # Optional heavy dependencies
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None  # type: ignore

try:
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
