"""Eyes agent for remote sensing and GIS analysis."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .eyes_tools import (
    analyze_lidar,
    analyze_raster,
    transform_coordinates,
    detect_image_features,
    lidar_tile_dtm,
    lidar_feature_detection,
    detect_shapes,
    detect_edges,
    TOOLS,
)

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
    tools: Dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if OpenAI is not None:
            self.client = OpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY"))
        self.tools = TOOLS

    # Thin wrappers around utility functions ---------------------------------

    def analyze_lidar(self, path: str, pipeline: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return analyze_lidar(path, pipeline)

    def analyze_raster(self, path: str) -> Dict[str, Any]:
        return analyze_raster(path)

    def transform_coordinates(
        self, x: float, y: float, from_epsg: int = 4326, to_epsg: int = 3857
    ) -> Tuple[float, float]:
        return transform_coordinates(x, y, from_epsg, to_epsg)

    def detect_image_features(self, path: str) -> Dict[str, Any]:
        return detect_image_features(path)

    def lidar_tile_dtm(self, path: str, resolution: float = 1.0) -> Dict[str, Any]:
        return lidar_tile_dtm(path, resolution)

    def lidar_feature_detection(
        self,
        path: str,
        resolution: float = 1.0,
        size_range: Tuple[float, float] = (50.0, 300.0),
    ) -> Dict[str, Any]:
        return lidar_feature_detection(path, resolution, size_range)

    def detect_shapes(
        self,
        image: "Any",
        profile: Optional[Dict[str, Any]] = None,
        size_range: Tuple[float, float] = (50.0, 300.0),
    ) -> List[Dict[str, Any]]:
        return detect_shapes(image, profile, size_range)

    def detect_edges(self, image: "Any") -> "Any":
        return detect_edges(image)

    # -----------------------------------------------------------------------

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
        response = self.client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content.strip()
