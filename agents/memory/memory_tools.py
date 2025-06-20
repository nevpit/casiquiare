"""Utility functions for the Memory Keeper agent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from log_config import setup_logger

logger = setup_logger("casiquiare.memory")

try:
    import pytesseract
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency may be missing
    pytesseract = None  # type: ignore
    Image = None  # type: ignore

try:
    from googletrans import Translator  # type: ignore
except Exception:  # pragma: no cover - optional dependency may be missing
    Translator = None  # type: ignore

# Simple in-memory corpus used for demonstrations
CORPUS: Dict[str, str] = {
    "doc1": "The Orinoco river was a vital route for many communities.",
    "doc2": "El r\u00edo Casiquiare conecta el Orinoco con el Amazonas.",
}

# Minimal place name database for geocoding
PLACE_DB: Dict[str, Tuple[float, float]] = {
    "Orinoco": (-67.0, 4.0),
    "Casiquiare": (-66.5, 1.8),
}


def ocr_text(path: str) -> str:
    """Extract text from an image or text file."""
    if pytesseract is not None and Image is not None:
        try:
            return pytesseract.image_to_string(Image.open(path))
        except Exception as exc:  # pragma: no cover - runtime failure
            logger.warning("OCR failed: %s", exc)
    return Path(path).read_text(encoding="utf-8")


def translate_text(text: str, target_lang: str = "en") -> str:
    """Translate text using googletrans if available."""
    if Translator is None:
        logger.info("googletrans not available; returning original text")
        return text
    try:
        translator = Translator()
        return translator.translate(text, dest=target_lang).text
    except Exception as exc:  # pragma: no cover - runtime failure
        logger.warning("Translation failed: %s", exc)
        return text


def search_corpus(query: str) -> List[Dict[str, str]]:
    """Return documents containing the query string."""
    results = []
    for doc_id, text in CORPUS.items():
        if query.lower() in text.lower():
            results.append({"doc_id": doc_id, "snippet": text})
    return results


def geocode_place(name: str) -> Optional[Tuple[float, float]]:
    """Return coordinates for a place name if known."""
    return PLACE_DB.get(name)


TOOLS: Dict[str, Any] = {
    "ocr_text": ocr_text,
    "translate_text": translate_text,
    "search_corpus": search_corpus,
    "geocode_place": geocode_place,
}

__all__ = ["ocr_text", "translate_text", "search_corpus", "geocode_place", "TOOLS"]
