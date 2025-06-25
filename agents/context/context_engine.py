"""Context Engine agent providing environmental and geo-archaeological analysis."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict
import json

from log_config import setup_logger
from . import context_tools

try:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:  # pragma: no cover - library may be missing
    client = None

logger = setup_logger("casiquiare.context")


@dataclass
class ContextEngine:
    """Environmental & geo-archaeological analyst agent."""

    model: str = "gpt-4-turbo"
    tools: Dict[str, Any] = field(init=False, default_factory=dict)
    system_prompt: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if client is not None:
            self.tools = context_tools.TOOLS

            self.system_prompt = (
                "You are the Context Engine, an environmental and geo-archaeological "
                "analyst. Use geospatial tools (e.g. analyze_satellite_image, "
                "lidar_tile_dtm, detect_shapes) to process rasters and terrain data. "
                "Respond factually and analytically without speculation. "
                "Always provide your result as a JSON object in the form "
                "{\"agent\": \"context\", \"type\": <type>, \"content\": <data>} "
                "followed by a short explanatory paragraph.\n"
                "Example:\n"
                "```json\n"
                "{\"agent\": \"context\", \"type\": \"environment_summary\", "
                "\"content\": {\"mean_ndvi\": 0.58, \"river_distance_km\": 2.3}}\n"
                "```\n"
                "This paragraph interprets the JSON variables in plain language."
            )
            logger.info("ContextEngine initialized")

    def set_system_prompt(self, prompt: str) -> None:
        """Override the default system prompt."""
        self.system_prompt = prompt

    def ask(self, query: str) -> str:
        """Query the language model using the Context Engine persona."""
        if client is None:
            raise RuntimeError("OpenAI SDK is not available.")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        try:
            response = client.chat.completions.create(
                model=self.model, messages=messages
            )
            reply = response.choices[0].message.content.strip()
            logger.info("ask result length=%d", len(reply))
            from world_state import log_message

            log_message("context", "chat", reply)
            return reply
        except Exception as exc:  # pragma: no cover - runtime failure
            logger.warning("LLM chat failed: %s", exc)
            return json.dumps({"error": str(exc)})

    def sample_elevation(self, path: str, lat: float, lon: float) -> Dict[str, Any]:
        """Delegate to :func:`context_tools.sample_elevation` and log the result."""
        from world_state import log_message

        result = context_tools.sample_elevation(path, lat, lon)
        log_message("context", "sample_elevation", result)
        return result

    def land_cover_class(self, path: str, lat: float, lon: float) -> Dict[str, Any]:
        """Delegate to :func:`context_tools.land_cover_class` and log the result."""
        from world_state import log_message

        result = context_tools.land_cover_class(path, lat, lon)
        log_message("context", "land_cover_class", result)
        return result

    def soil_properties(self, path: str, lat: float, lon: float) -> Dict[str, Any]:
        """Delegate to :func:`context_tools.soil_properties` and log the result."""
        from world_state import log_message

        result = context_tools.soil_properties(path, lat, lon)
        log_message("context", "soil_properties", result)
        return result

    def distance_to_water(self, path: str, lat: float, lon: float) -> Dict[str, Any]:
        """Delegate to :func:`context_tools.distance_to_water` and log the result."""
        from world_state import log_message

        result = context_tools.distance_to_water(path, lat, lon)
        log_message("context", "distance_to_water", result)
        return result

    def climate_variables(self, path: str, lat: float, lon: float) -> Dict[str, Any]:
        """Delegate to :func:`context_tools.climate_variables` and log the result."""
        from world_state import log_message

        result = context_tools.climate_variables(path, lat, lon)
        log_message("context", "climate_variables", result)
        return result

    def check_context_layers(self, layers: Dict[str, str], lat: float, lon: float) -> Dict[str, bool]:
        """Delegate to :func:`context_tools.check_context_layers` and log the result."""
        from world_state import log_message

        result = context_tools.check_context_layers(layers, lat, lon)
        log_message("context", "check_context_layers", result)
        return result

    def analyze_environment(
        self,
        dem: str,
        land_cover: str,
        soil: str,
        distance: str,
        climate: str,
        context_layers: Dict[str, str],
        lat: float,
        lon: float,
    ) -> Dict[str, Any]:
        """Generate a full environmental summary for a coordinate."""

        elev = self.sample_elevation(dem, lat, lon)
        land = self.land_cover_class(land_cover, lat, lon)
        soil_props = self.soil_properties(soil, lat, lon)
        dist = self.distance_to_water(distance, lat, lon)
        clim = self.climate_variables(climate, lat, lon)
        ctx = self.check_context_layers(context_layers, lat, lon)

        return self.environment_summary(elev, land, soil_props, dist, clim, ctx)

    def environment_summary(
        self,
        elevation: Dict[str, Any],
        land_cover: Dict[str, Any],
        soil: Dict[str, Any],
        distance: Dict[str, Any],
        climate: Dict[str, Any],
        context_layers: Dict[str, bool],
    ) -> Dict[str, Any]:
        """Delegate to :func:`context_tools.environment_summary` and log the result."""
        from world_state import set_value, log_message

        result = context_tools.environment_summary(
            elevation, land_cover, soil, distance, climate, context_layers
        )
        set_value("context_environment", result)
        log_message("context", "analysis", result)
        return result

    def generate_context_summary(
        self, summary: Dict[str, Any], lat: float | None = None, lon: float | None = None
    ) -> str:
        """Convert a summary dict to a short narrative."""
        from world_state import log_message

        text = context_tools.generate_context_summary(summary, lat=lat, lon=lon)
        log_message("context", "generate_context_summary", text)
        return text

    def clear_cache(self) -> None:
        """Clear cached datasets used by context tools."""
        context_tools.clear_cache()


def analyze_environment(
    dem: str,
    land_cover: str,
    soil: str,
    distance: str,
    climate: str,
    context_layers: Dict[str, str],
    lat: float,
    lon: float,
) -> Dict[str, Any]:
    """Convenience wrapper creating a :class:`ContextEngine` and running analysis."""

    engine = ContextEngine()
    return engine.analyze_environment(
        dem,
        land_cover,
        soil,
        distance,
        climate,
        context_layers,
        lat,
        lon,
    )


__all__ = ["ContextEngine", "analyze_environment"]
