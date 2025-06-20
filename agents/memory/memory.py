"""Memory Keeper agent handling historical sources."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from log_config import setup_logger

from . import memory_tools

try:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:  # pragma: no cover - library may be missing
    client = None


logger = setup_logger("casiquiare.memory")


@dataclass
class MemoryKeeper:
    """Ethno-historian and archivist agent."""

    model: str = "gpt-4-turbo"
    tools: Dict[str, Any] = field(init=False, default_factory=dict)
    system_prompt: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if client is not None:
            self.tools = memory_tools.TOOLS
            self.system_prompt = (
                "You are the Memory Keeper, an AI ethno-historian and archivist. "
                "You curate historical documents with meticulous citations and "
                "handle sensitive cultural knowledge with care. Answer factually "
                "and cite sources when available."
            )

    def set_system_prompt(self, prompt: str) -> None:
        """Override the default system prompt."""
        self.system_prompt = prompt

    def ask(self, query: str) -> str:
        """Query the language model using the Memory Keeper persona."""
        if client is None:
            raise RuntimeError("OpenAI SDK is not available.")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        response = client.chat.completions.create(model=self.model, messages=messages)
        reply = response.choices[0].message.content.strip()
        log_message("memory", "chat", reply)
        return reply

    # ------------------------------------------------------------------
    # Thin wrappers around utility functions that also update world_state

    def ocr_text(self, path: str) -> str:
        from world_state import world_state, log_message

        text = memory_tools.ocr_text(path)
        world_state["last_text"] = text
        log_message("memory", "ocr_text", text)
        return text

    def translate_text(self, text: str, target_lang: str = "en") -> str:
        from world_state import log_message

        result = memory_tools.translate_text(text, target_lang)
        log_message("memory", "translate_text", {"input": text, "output": result})
        return result

    def search_corpus(self, query: str):
        from world_state import world_state, log_message

        results = memory_tools.search_corpus(query)
        world_state["search_results"] = results
        log_message("memory", "search_corpus", results)
        return results

    def geocode_place(self, name: str):
        from world_state import world_state, log_message

        coords = memory_tools.geocode_place(name)
        world_state.setdefault("geocodes", {})[name] = coords
        log_message("memory", "geocode_place", {"place": name, "coords": coords})
        return coords

    def get_results(self, key: Optional[str] = None) -> Any:
        """Return stored results from ``world_state``."""
        from world_state import world_state

        if key is None:
            logger.info("Retrieving full world_state")
            return world_state
        logger.info("Retrieving %s from world_state", key)
        return world_state.get(key)


TOOLS: Dict[str, Any] = memory_tools.TOOLS

__all__ = ["MemoryKeeper", "TOOLS"]
