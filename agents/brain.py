"""Brain agent for data engineering and ML modeling."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict

from . import brain_tools

try:
    import openai
except Exception:  # pragma: no cover - library may be missing
    openai = None


@dataclass
class Brain:
    """Data engineer and ML modeller agent."""

    model: str = "gpt-4-turbo"
    tools: Dict[str, Any] = field(init=False, default_factory=dict)
    system_prompt: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if openai is not None:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        self.tools = brain_tools.TOOLS
        self.system_prompt = (
            "You are the Brain agent, a data engineer & ML modeler specializing in "
            "archaeological site prediction. You rigorously analyze data, run code, "
            "and return factual, reproducible results. Use tools for modeling, "
            "statistics, and clustering – avoid speculation."
        )

    def set_system_prompt(self, prompt: str) -> None:
        """Override the default system prompt."""
        self.system_prompt = prompt

    def ask(self, query: str) -> str:
        """Query the language model using the Brain persona."""
        if openai is None:
            raise RuntimeError("OpenAI SDK is not available.")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        response = openai.ChatCompletion.create(model=self.model, messages=messages)
        return response.choices[0].message.content.strip()


TOOLS: Dict[str, Any] = brain_tools.TOOLS

__all__ = ["Brain", "TOOLS"]
