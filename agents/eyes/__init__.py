"""Eyes agent package combining the core agent class, tools and CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Iterable, Optional

from . import tools, cli

try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # pragma: no cover - optional
except Exception:  # pragma: no cover - library may be missing
    client = None


@dataclass
class Eyes:
    """Remote-sensing and GIS agent."""

    model: str = "gpt-4-turbo"
    tools: Dict[str, Any] = field(init=False, default_factory=dict)
    system_prompt: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if client is not None:
            self.tools = tools.TOOLS
            self.system_prompt = (
                "You are the Eyes agent, an analytical assistant that provides "
                "factual observations from remote sensing data. Do not "
                "speculate or interpret. Always respond with JSON formatted as "
                "{\\\"agent\\\": \\\"eyes\\\", \\\"type\\\": <type>, "
                "\\\"content\\\": <data>} inside a single markdown block."
            )

    def set_system_prompt(self, prompt: str) -> None:
        """Override the default system prompt."""
        self.system_prompt = prompt

    # Thin wrappers around utility functions ---------------------------------

    def analyze_lidar(
        self, path: str, pipeline: Optional[Dict[str, Any]] = None
    ) -> Iterable["np.ndarray"]:
        """Delegate to :func:`eyes.tools.analyze_lidar`."""
        return tools.analyze_lidar(path, pipeline)

    def analyze_raster(self, path: str) -> Dict[str, Any]:
        """Delegate to :func:`eyes.tools.analyze_raster`."""
        return tools.analyze_raster(path)

    def analyze_satellite_image(
        self,
        path: str,
        ndvi_bands: Optional[Tuple[int, int]] = None,
        ndwi_bands: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Delegate to :func:`eyes.tools.analyze_satellite_image`."""
        return tools.analyze_satellite_image(path, ndvi_bands, ndwi_bands)

    def transform_coordinates(
        self, x: float, y: float, from_epsg: int = 4326, to_epsg: int = 3857
    ) -> Tuple[float, float]:
        """Delegate to :func:`eyes.tools.transform_coordinates`."""
        return tools.transform_coordinates(x, y, from_epsg, to_epsg)

    def detect_image_features(self, path: str) -> Dict[str, Any]:
        """Delegate to :func:`eyes.tools.detect_image_features`."""
        return tools.detect_image_features(path)

    def lidar_tile_dtm(
        self,
        path: str,
        resolution: float = 1.0,
        *,
        out_dir: str | None = None,
        return_paths: bool = False,
    ) -> Dict[str, Any]:
        """Delegate to :func:`eyes.tools.lidar_tile_dtm`."""
        return tools.lidar_tile_dtm(path, resolution, out_dir=out_dir, return_paths=return_paths)

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
        """Delegate to :func:`eyes.tools.lidar_feature_detection`."""
        return tools.lidar_feature_detection(
            path,
            resolution,
            size_range,
            dilation_size,
            out_dir=out_dir,
            return_paths=return_paths,
        )

    def detect_shapes(
        self,
        image: Any,
        profile: Optional[Dict[str, Any]] = None,
        size_range: Tuple[float, float] = (50.0, 300.0),
        dilation_size: int = 3,
    ) -> List[Dict[str, Any]]:
        """Delegate to :func:`eyes.tools.detect_shapes`."""
        return tools.detect_shapes(image, profile, size_range, dilation_size=dilation_size)

    def detect_lines(self, edge_img: Any) -> List[Dict[str, Any]]:
        """Delegate to :func:`eyes.tools.detect_lines`."""
        return tools.detect_lines(edge_img)

    # -----------------------------------------------------------------------

    def summarize(self, findings: str) -> str:
        """Generate a factual summary using an OpenAI model."""
        if client is None:
            raise RuntimeError("OpenAI SDK is not available.")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": findings},
        ]
        response = client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content.strip()


# re-export tools module for convenience
__all__ = ["Eyes", "tools", "cli"]
