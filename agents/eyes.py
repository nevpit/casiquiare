"""Eyes agent for remote sensing and GIS analysis."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Iterable, Optional

from .eyes_tools import (
    analyze_lidar,
    analyze_raster,
    transform_coordinates,
    detect_image_features,
    lidar_tile_dtm,
    lidar_feature_detection,
    detect_lines,
    detect_shapes,
    TOOLS,
)

try:
    import openai
except Exception:  # pragma: no cover - library may be missing
    openai = None


@dataclass
class Eyes:
    """Remote-sensing and GIS agent."""

    model: str = "gpt-4-turbo"
    tools: Dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if openai is not None:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        self.tools = TOOLS

    # Thin wrappers around utility functions ---------------------------------

    def analyze_lidar(
        self, path: str, pipeline: Optional[Dict[str, Any]] = None
    ) -> Iterable["np.ndarray"]:
        """Delegate to :func:`agents.eyes_tools.analyze_lidar`."""
        return analyze_lidar(path, pipeline)

    def analyze_raster(self, path: str) -> Dict[str, Any]:
        """Delegate to :func:`agents.eyes_tools.analyze_raster`."""
        return analyze_raster(path)

    def transform_coordinates(
        self, x: float, y: float, from_epsg: int = 4326, to_epsg: int = 3857
    ) -> Tuple[float, float]:
        """Delegate to :func:`agents.eyes_tools.transform_coordinates`."""
        return transform_coordinates(x, y, from_epsg, to_epsg)

    def detect_image_features(self, path: str) -> Dict[str, Any]:
        """Delegate to :func:`agents.eyes_tools.detect_image_features`."""
        return detect_image_features(path)

    def lidar_tile_dtm(
        self,
        path: str,
        resolution: float = 1.0,
        *,
        out_dir: str | None = None,
        return_paths: bool = False,
    ) -> Dict[str, Any]:
        """Delegate to :func:`agents.eyes_tools.lidar_tile_dtm`."""
        return lidar_tile_dtm(path, resolution, out_dir=out_dir, return_paths=return_paths)

    def lidar_feature_detection(
        self,
        path: str,
        resolution: float = 1.0,
        size_range: Tuple[float, float] = (50.0, 300.0),
        dilation_size: int = 3,
        *,
        out_dir: str | None = None,
        return_paths: bool = False,
    ) -> Dict[str, Any]:
        """Delegate to :func:`agents.eyes_tools.lidar_feature_detection`."""
        return lidar_feature_detection(
            path,
            resolution,
            size_range,
            dilation_size,
            out_dir=out_dir,
            return_paths=return_paths,
        )

    def detect_shapes(
        self,
        image: "Any",
        profile: Optional[Dict[str, Any]] = None,
        size_range: Tuple[float, float] = (50.0, 300.0),
        dilation_size: int = 3,
    ) -> List[Dict[str, Any]]:
        """Delegate to :func:`agents.eyes_tools.detect_shapes`."""
        return detect_shapes(image, profile, size_range, dilation_size=dilation_size)

    def detect_lines(self, edge_img: "Any") -> List[Dict[str, Any]]:
        """Delegate to :func:`agents.eyes_tools.detect_lines`."""
        return detect_lines(edge_img)

    # -----------------------------------------------------------------------

    def summarize(self, findings: str) -> str:
        """Generate a factual summary using an OpenAI model."""
        if openai is None:
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
        response = openai.ChatCompletion.create(model=self.model, messages=messages)
        return response.choices[0].message.content.strip()
