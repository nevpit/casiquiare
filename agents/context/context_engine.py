"""Context Engine agent providing environmental and geo-archaeological analysis."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict

from log_config import setup_logger

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
        response = client.chat.completions.create(model=self.model, messages=messages)
        reply = response.choices[0].message.content.strip()
        logger.info("ask result length=%d", len(reply))
        from world_state import log_message

        log_message("context", "chat", reply)
        return reply


__all__ = ["ContextEngine"]
