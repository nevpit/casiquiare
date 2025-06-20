"""Utility functions for the Memory Keeper agent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import os

from log_config import setup_logger

logger = setup_logger("casiquiare.memory")

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageOps, ImageFont, ImageDraw
except Exception:  # pragma: no cover - optional dependency may be missing
    pytesseract = None  # type: ignore
    Image = None  # type: ignore
    ImageEnhance = None  # type: ignore
    ImageOps = None  # type: ignore
    ImageFont = None  # type: ignore
    ImageDraw = None  # type: ignore

try:
    from langdetect import detect
except Exception:  # pragma: no cover - optional dependency may be missing
    detect = None  # type: ignore

try:
    from pdfminer.high_level import extract_text as pdf_extract_text  # type: ignore
except Exception:  # pragma: no cover - optional dependency may be missing
    pdf_extract_text = None  # type: ignore

try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:  # pragma: no cover - optional dependency may be missing
    openai_client = None

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
    """Extract text from an image, PDF, or text file."""
    # Bypass OCR for PDFs with embedded text
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf" and pdf_extract_text is not None:
        try:
            return pdf_extract_text(path)
        except Exception as exc:  # pragma: no cover - runtime failure
            logger.warning("PDF text extraction failed: %s", exc)

    if pytesseract is not None and Image is not None:
        try:
            return ocr_image(Image.open(path))
        except Exception as exc:  # pragma: no cover - runtime failure
            logger.warning("OCR failed: %s", exc)

    return Path(path).read_text(encoding="utf-8")


def ocr_image(image: "Image.Image") -> str:
    """Run OCR on a :class:`PIL.Image` with basic preprocessing."""
    if pytesseract is None or Image is None:
        return ""
    try:
        # Convert to grayscale and enhance contrast
        gray = image.convert("L")
        if ImageEnhance is not None:
            enhancer = ImageEnhance.Contrast(gray)
            gray = enhancer.enhance(2.0)
        if ImageOps is not None:
            gray = ImageOps.autocontrast(gray)
            gray = ImageOps.invert(gray)
            gray = gray.point(lambda x: 0 if x < 128 else 255, mode="1")
        return pytesseract.image_to_string(gray)
    except Exception as exc:  # pragma: no cover - runtime failure
        logger.warning("OCR image failed: %s", exc)
        return ""


def detect_language(text: str) -> Optional[str]:
    """Detect the language of ``text`` using ``langdetect`` if available."""
    if detect is None:
        logger.info("langdetect not available")
        return None
    try:
        return detect(text)
    except Exception as exc:  # pragma: no cover - runtime failure
        logger.warning("Language detection failed: %s", exc)
        return None


def translate_text(text: str, src_lang: str, target_lang: str = "en") -> str:
    """Translate text using an OpenAI model if available."""
    if openai_client is None:
        logger.info("OpenAI client not available; returning original text")
        return text
    messages = [
        {
            "role": "system",
            "content": (
                "You translate text while preserving proper nouns and archaic phrasing."
            ),
        },
        {
            "role": "user",
            "content": f"Translate the following text from {src_lang} to {target_lang}:\n{text}",
        },
    ]
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4-turbo", messages=messages
        )
        return resp.choices[0].message.content.strip()
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
    "ocr_image": ocr_image,
    "translate_text": translate_text,
    "detect_language": detect_language,
    "search_corpus": search_corpus,
    "geocode_place": geocode_place,
}

__all__ = [
    "ocr_text",
    "ocr_image",
    "translate_text",
    "detect_language",
    "search_corpus",
    "geocode_place",
    "TOOLS",
]
