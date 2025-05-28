"""Eyes agent for remote sensing and GIS analysis.

This module defines the `Eyes` agent that leverages geospatial libraries and
OpenAI's SDK to provide factual observations from LiDAR, raster, and imagery
data. The agent focuses on detection and description of potential
archaeological features without interpretation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
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
    from pyproj import Transformer
except Exception:  # pragma: no cover - library may be missing
    Transformer = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - library may be missing
    OpenAI = None


@dataclass
class Eyes:
    """Remote-sensing and GIS agent."""

    api_key: Optional[str] = None
    model: str = "gpt-4-turbo"
    client: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        if OpenAI is not None:
            self.client = OpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY"))

    def analyze_lidar(
        self, path: str, pipeline: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Load and process a LiDAR point cloud using PDAL."""
        if pdal is None:
            raise RuntimeError("PDAL is not installed.")
        pipeline = pipeline or {"pipeline": [path]}
        pl = pdal.Pipeline(json.dumps(pipeline))
        pl.execute()
        arrays = pl.arrays
        return [arr.tolist() for arr in arrays]

    def analyze_raster(self, path: str) -> Dict[str, Any]:
        """Read raster metadata using rasterio."""
        if rasterio is None:
            raise RuntimeError("rasterio is not installed.")
        with rasterio.open(path) as src:
            meta = src.meta.copy()
        return meta

    def transform_coordinates(
        self, x: float, y: float, from_epsg: int = 4326, to_epsg: int = 3857
    ) -> Tuple[float, float]:
        """Transform coordinates between projections using pyproj."""
        if Transformer is None:
            raise RuntimeError("pyproj is not installed.")
        transformer = Transformer.from_crs(from_epsg, to_epsg, always_xy=True)
        return transformer.transform(x, y)

    def detect_image_features(self, path: str) -> Dict[str, Any]:
        """Detect simple features in an image using OpenCV."""
        if cv2 is None or np is None:
            raise RuntimeError("OpenCV and NumPy are required.")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Unable to read image at {path}")
        edges = cv2.Canny(img, 100, 200)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return {"num_contours": len(contours)}

    def summarize(self, findings: str) -> str:
        """Generate a factual summary using an OpenAI model."""
        if self.client is None:
            raise RuntimeError("OpenAI SDK is not available.")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Eyes agent, an analytical assistant that "
                    "provides factual observations from remote sensing data. "
                    "Do not speculate or provide interpretations."
                ),
            },
            {"role": "user", "content": findings},
        ]
        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        return response.choices[0].message.content.strip()
