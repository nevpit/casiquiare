"""Memory Keeper agent handling historical sources."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Sequence

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
        logger.info("MemoryKeeper initialized with tools: %s", list(self.tools))

    def set_system_prompt(self, prompt: str) -> None:
        """Override the default system prompt."""
        self.system_prompt = prompt

    def ask(self, query: str) -> str:
        """Query the language model using the Memory Keeper persona."""
        if client is None:
            raise RuntimeError("OpenAI SDK is not available.")
        logger.info("ask: %s", query)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        response = client.chat.completions.create(model=self.model, messages=messages)
        reply = response.choices[0].message.content.strip()
        logger.info("ask result length=%d", len(reply))
        log_message("memory", "chat", reply)
        return reply

    # ------------------------------------------------------------------
    # Thin wrappers around utility functions that also update world_state

    def ocr_text(self, path: str) -> str:
        from world_state import world_state, log_message

        logger.info("ocr_text on %s", path)
        text = memory_tools.ocr_text(path)
        lang = memory_tools.detect_language(text)
        world_state["last_text"] = text
        world_state["last_language"] = lang
        world_state["last_source"] = path
        if lang in ("es", "pt"):
            text_en = memory_tools.translate_text(text, src_lang=lang)
            world_state["last_translation"] = text_en
        else:
            text_en = text
        log_message(
            "memory",
            "ocr_text",
            {"text": text_en, "lang": lang, "source": path},
        )
        logger.info("ocr_text result lang=%s chars=%d", lang, len(text_en))
        return text_en

    def ocr_image(self, image: "memory_tools.Image.Image") -> str:
        from world_state import world_state, log_message

        logger.info("ocr_image")
        text = memory_tools.ocr_image(image)
        lang = memory_tools.detect_language(text)
        world_state["last_text"] = text
        world_state["last_language"] = lang
        world_state["last_source"] = "image"
        if lang in ("es", "pt"):
            text_en = memory_tools.translate_text(text, src_lang=lang)
            world_state["last_translation"] = text_en
        else:
            text_en = text
        log_message(
            "memory",
            "ocr_image",
            {"text": text_en, "lang": lang, "source": "image"},
        )
        logger.info("ocr_image result lang=%s chars=%d", lang, len(text_en))
        return text_en

    def translate_text(self, text: str, src_lang: str, target_lang: str = "en") -> str:
        from world_state import world_state, log_message

        logger.info("translate_text from %s to %s", src_lang, target_lang)
        result = memory_tools.translate_text(text, src_lang, target_lang)
        entry = {
            "input": text,
            "output": result,
            "src": src_lang,
            "source": world_state.get("last_source"),
        }
        world_state.setdefault("translations", []).append(entry)
        log_message("memory", "translate_text", entry)
        logger.info("translate_text result chars=%d", len(result))
        return result

    def search_corpus(self, query: str):
        from world_state import world_state, log_message

        logger.info("search_corpus query=%s", query)
        results = memory_tools.search_corpus(query)
        world_state["search_results"] = results
        log_message("memory", "search_corpus", results)
        logger.info("search_corpus found %d results", len(results))
        return results

    def index_documents(self, docs: "Sequence[Dict[str, Any]]") -> None:
        from world_state import log_message

        logger.info("index_documents count=%d", len(docs))
        memory_tools.index_documents(docs)
        log_message("memory", "index_documents", {"count": len(docs)})

    def semantic_search(self, query: str, top_k: int = 5):
        from world_state import world_state, log_message

        logger.info("semantic_search query=%s", query)
        results = memory_tools.semantic_search(query, top_k=top_k)
        world_state["semantic_results"] = results
        log_message("memory", "semantic_search", results)
        logger.info("semantic_search returned %d results", len(results))
        return results

    def search_text(self, query: str, top_k: int = 5):
        from world_state import world_state, log_message

        logger.info("search_text query=%s", query)
        results = memory_tools.search_text(query, top_k=top_k)
        world_state["search_text_results"] = [asdict(r) for r in results]
        log_message(
            "memory",
            "search_text",
            [asdict(r) for r in results],
        )
        logger.info("search_text returned %d results", len(results))
        return results

    def extract_locations(self, text: str):
        from world_state import log_message

        logger.info("extract_locations len=%d", len(text))
        ents = memory_tools.extract_locations(text)
        log_message(
            "memory",
            "extract_locations",
            [asdict(e) for e in ents],
        )
        logger.info("extract_locations found %d entities", len(ents))
        return ents

    def search_location(self, name: str):
        from world_state import world_state, log_message

        logger.info("search_location name=%s", name)
        results = memory_tools.search_location(name)
        world_state["location_results"] = [asdict(r) for r in results]
        log_message(
            "memory",
            "search_location",
            [asdict(r) for r in results],
        )
        logger.info("search_location returned %d results", len(results))
        return results

    def search_distance_clues(self, name: str):
        from world_state import world_state, log_message

        logger.info("search_distance_clues name=%s", name)
        results = memory_tools.search_distance_clues(name)
        world_state["distance_clues"] = [asdict(c) for c in results]
        log_message(
            "memory",
            "search_distance_clues",
            [asdict(c) for c in results],
        )
        logger.info("search_distance_clues returned %d clues", len(results))
        return results

    def infer_relative_location(self, clue: "memory_tools.DistanceClue"):
        from world_state import world_state, log_message

        logger.info(
            "infer_relative_location from %s", asdict(clue)
        )
        loc = memory_tools.infer_relative_location(clue)
        if loc is not None:
            world_state.setdefault("inferred_locations", []).append(asdict(loc))
            coord_key = f"{loc.lon:.4f},{loc.lat:.4f}"
            summary = (
                f"{clue.distance} {clue.unit} {clue.direction} from {clue.ref_place}"
            )
            world_state.setdefault("historical_clues", {})[coord_key] = summary
        log_message(
            "memory",
            "infer_relative_location",
            asdict(loc) if loc else None,
        )
        logger.info("infer_relative_location result=%s", asdict(loc) if loc else None)
        return loc

    def geocode_place(self, name: str):
        from world_state import world_state, log_message

        logger.info("geocode_place %s", name)
        coords = memory_tools.geocode_place(name)
        world_state.setdefault("geocodes", {})[name] = coords
        if coords:
            coord_key = f"{coords[0]:.4f},{coords[1]:.4f}"
            world_state.setdefault("historical_clues", {})[coord_key] = name
        log_message("memory", "geocode_place", {"place": name, "coords": coords})
        logger.info("geocode_place result=%s", coords)
        return coords

    def summarize_clues(self, location_query: str, top_k: int = 5) -> str:
        from world_state import world_state, log_message

        logger.info("summarize_clues query=%s", location_query)
        summary = memory_tools.summarize_clues(location_query, top_k=top_k)
        world_state.setdefault("summaries", {})[location_query] = summary
        coords = memory_tools.geocode_place(location_query)
        if coords:
            coord_key = f"{coords[0]:.4f},{coords[1]:.4f}"
            world_state.setdefault("historical_clues", {})[coord_key] = summary
        log_message("memory", "summarize_clues", summary)
        logger.info("summarize_clues result chars=%d", len(summary))
        return summary

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
