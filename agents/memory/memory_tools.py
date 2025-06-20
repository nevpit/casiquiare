"""Utility functions for the Memory Keeper agent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import re

import os
from hashlib import md5

from log_config import setup_logger

logger = setup_logger("casiquiare.memory")


@dataclass
class Excerpt:
    """Segment of text stored in the semantic index."""

    doc_id: str
    title: str
    page: int
    text: str
    distance: float = 0.0


@dataclass
class Entity:
    """Named entity extracted from text."""

    text: str
    label: str
    start: int
    end: int


@dataclass
class DistanceClue:
    """Distance and direction reference extracted from text."""

    ref_place: str
    direction: str
    distance: float
    unit: str
    doc_id: str = ""
    page: int = 0

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

try:
    import spacy
    try:
        nlp = spacy.load("xx_ent_wiki_sm")
    except Exception:
        nlp = None
except Exception:  # pragma: no cover - optional dependency may be missing
    spacy = None  # type: ignore
    nlp = None

try:
    import faiss
    import numpy as np
except Exception:  # pragma: no cover - optional dependency may be missing
    faiss = None  # type: ignore
    np = None  # type: ignore

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

# Semantic search index
EMBED_DIM = 1536
SEMANTIC_INDEX: Optional[Any] = None
SEMANTIC_META: List[Dict[str, Any]] = []
KEYWORD_INDEX: Dict[str, List[int]] = {}
LOCATION_INDEX: Dict[str, List[int]] = {}
DISTANCE_CLUES: List[DistanceClue] = []


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


def extract_locations(text: str) -> List[Entity]:
    """Return named locations found in ``text``."""
    entities: List[Entity] = []
    if 'nlp' in globals() and nlp is not None:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in {"GPE", "LOC", "FAC", "ORG", "NORP"}:
                    entities.append(
                        Entity(
                            text=ent.text,
                            label=ent.label_,
                            start=ent.start_char,
                            end=ent.end_char,
                        )
                    )
            return entities
        except Exception as exc:  # pragma: no cover - runtime failure
            logger.warning("spaCy NER failed: %s", exc)
    # fallback simple regex for capitalized words
    pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
    for match in re.finditer(pattern, text):
        entities.append(
            Entity(
                text=match.group(1),
                label="UNK",
                start=match.start(1),
                end=match.end(1),
            )
        )
    return entities


NUM_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "uno": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "um": 1,
    "dois": 2,
    "tres": 3,
    "quatro": 4,
    "cinco": 5,
    "seis": 6,
    "sete": 7,
    "oito": 8,
    "nove": 9,
    "dez": 10,
}

DIR_MAP = {
    "norte": "north",
    "sur": "south",
    "este": "east",
    "oeste": "west",
    "noroeste": "northwest",
    "nordeste": "northeast",
    "sudeste": "southeast",
    "sudoeste": "southwest",
    "rio arriba": "upriver",
    "rio abajo": "downriver",
    "rio abaixo": "downriver",
    "upstream": "upriver",
    "downstream": "downriver",
}


def _parse_number(text: str) -> float:
    text = text.replace(",", ".")
    if text.isdigit() or re.match(r"\d+\.\d+", text):
        try:
            return float(text)
        except Exception:
            return 0.0
    return float(NUM_MAP.get(text.lower(), 0))


def parse_distance_clues(text: str) -> List[DistanceClue]:
    """Extract distance and direction relationships from ``text``."""

    clues: List[DistanceClue] = []

    unit_pattern = r"(?:leagues?|leguas?|km|kilometers?|kilometros?|miles?|millas?|days?|dias?)"
    dir_pattern = r"(?:north|south|east|west|northeast|northwest|southeast|southwest|upriver|downriver|upstream|downstream|norte|sur|este|oeste|noroeste|nordeste|sudeste|sudoeste|rio arriba|rio abaixo|rio abajo)"
    num_pattern = r"(\d+(?:[.,]\d+)?|(?:" + "|".join(NUM_MAP.keys()) + "))"
    pattern = re.compile(
        rf"{num_pattern}\s*({unit_pattern})(?:\s+(?:journey|travel|viaje|de\s+viaje))?\s+({dir_pattern})\s+(?:of|from|de|del)\s+([A-Z][A-Za-z\u00C0-\u017F\s]+)"
    )

    for match in pattern.finditer(text):
        num_str, unit, direction, place = match.groups()
        distance = _parse_number(num_str)
        unit = unit.lower()
        direction_norm = DIR_MAP.get(direction.lower(), direction.lower())
        clues.append(
            DistanceClue(
                ref_place=place.strip(),
                direction=direction_norm,
                distance=distance,
                unit=unit,
            )
        )
    return clues


def _compute_embedding(text: str) -> List[float]:
    """Return an embedding vector for ``text``."""
    if openai_client is not None:
        try:
            resp = openai_client.embeddings.create(
                model="text-embedding-ada-002", input=text
            )
            return list(resp.data[0].embedding)
        except Exception as exc:  # pragma: no cover - runtime failure
            logger.warning("Embedding failed: %s", exc)
    # deterministic fallback embedding
    h = md5(text.encode("utf-8")).digest()
    base = [(b / 255.0) for b in h]
    vec = base * (EMBED_DIM // len(base)) + base[: EMBED_DIM % len(base)]
    return vec


def index_documents(docs: List[Dict[str, Any]], chunk_size: int = 500) -> None:
    """Index documents for semantic and keyword search."""
    global SEMANTIC_INDEX, SEMANTIC_META, KEYWORD_INDEX, LOCATION_INDEX, DISTANCE_CLUES
    embeddings = []
    metadata = []
    KEYWORD_INDEX.clear()
    LOCATION_INDEX.clear()
    DISTANCE_CLUES.clear()
    if faiss is None or np is None:
        logger.info("faiss or numpy missing; building keyword index only")
    
    for doc in docs:
        doc_id = doc.get("id")
        title = doc.get("title", "")
        text = doc.get("text", "")
        page = doc.get("page", 0)
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        for pi, para in enumerate(paragraphs):
            words = para.split()
            for start in range(0, len(words), chunk_size):
                chunk = " ".join(words[start : start + chunk_size])
                if faiss is not None and np is not None:
                    emb = _compute_embedding(chunk)
                    embeddings.append(np.array(emb, dtype="float32"))
                locs = extract_locations(chunk)
                clues = parse_distance_clues(chunk)
                idx = len(metadata)
                metadata.append(
                    {
                        "doc_id": doc_id,
                        "title": title,
                        "page": page,
                        "chunk": pi,
                        "text": chunk,
                        "locations": [l.text for l in locs],
                        "distance_clues": [asdict(c) for c in clues],
                    }
                )
                for tok in set(re.findall(r"\w+", chunk.lower())):
                    KEYWORD_INDEX.setdefault(tok, []).append(idx)
                for loc in locs:
                    LOCATION_INDEX.setdefault(loc.text.lower(), []).append(idx)
                for c in clues:
                    DISTANCE_CLUES.append(
                        DistanceClue(
                            ref_place=c.ref_place,
                            direction=c.direction,
                            distance=c.distance,
                            unit=c.unit,
                            doc_id=doc_id or "",
                            page=page,
                        )
                    )
    if embeddings and faiss is not None and np is not None:
        index = faiss.IndexFlatL2(EMBED_DIM)
        index.add(np.vstack(embeddings))
        SEMANTIC_INDEX = index
    else:
        SEMANTIC_INDEX = None
    SEMANTIC_META = metadata


def semantic_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search the semantic index for ``query`` and return metadata."""
    if SEMANTIC_INDEX is None or faiss is None or np is None:
        return []
    vec = np.array(_compute_embedding(query), dtype="float32").reshape(1, -1)
    distances, indices = SEMANTIC_INDEX.search(vec, top_k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < 0 or idx >= len(SEMANTIC_META):
            continue
        meta = dict(SEMANTIC_META[idx])
        meta["distance"] = float(dist)
        results.append(meta)
    return results


def _keyword_search(query: str) -> List[int]:
    """Return chunk indices containing all words from ``query``."""
    tokens = re.findall(r"\w+", query.lower())
    if not tokens:
        return []
    counts: Dict[int, int] = {}
    for tok in tokens:
        for idx in KEYWORD_INDEX.get(tok, []):
            counts[idx] = counts.get(idx, 0) + 1
    sorted_indices = [idx for idx, _ in sorted(counts.items(), key=lambda x: -x[1])]
    return sorted_indices


def search_text(query: str, top_k: int = 5) -> List[Excerpt]:
    """Return top matching text chunks for ``query`` using semantic and keyword search."""
    kw_indices = _keyword_search(query)
    kw_excerpts: List[Excerpt] = []
    for idx in kw_indices:
        if idx < 0 or idx >= len(SEMANTIC_META):
            continue
        meta = SEMANTIC_META[idx]
        kw_excerpts.append(
            Excerpt(
                doc_id=meta.get("doc_id", ""),
                title=meta.get("title", ""),
                page=meta.get("page", 0),
                text=meta.get("text", ""),
                distance=0.0,
            )
        )

    sem_excerpts: List[Excerpt] = []
    if SEMANTIC_INDEX is not None and faiss is not None and np is not None:
        vec = np.array(_compute_embedding(query), dtype="float32").reshape(1, -1)
        distances, indices = SEMANTIC_INDEX.search(vec, top_k)
        seen = set(kw_indices)
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(SEMANTIC_META) or idx in seen:
                continue
            meta = SEMANTIC_META[idx]
            sem_excerpts.append(
                Excerpt(
                    doc_id=meta.get("doc_id", ""),
                    title=meta.get("title", ""),
                    page=meta.get("page", 0),
                    text=meta.get("text", ""),
                    distance=float(dist),
                )
            )

    results = kw_excerpts + sem_excerpts
    return results[:top_k]


def search_location(name: str) -> List[Excerpt]:
    """Return text excerpts mentioning ``name`` based on NER tags."""
    indices = LOCATION_INDEX.get(name.lower(), [])
    results: List[Excerpt] = []
    for idx in indices:
        if 0 <= idx < len(SEMANTIC_META):
            meta = SEMANTIC_META[idx]
            results.append(
                Excerpt(
                    doc_id=meta.get("doc_id", ""),
                    title=meta.get("title", ""),
                    page=meta.get("page", 0),
                    text=meta.get("text", ""),
                    distance=0.0,
                )
            )
    return results


def search_distance_clues(ref_place: str) -> List[DistanceClue]:
    """Return stored distance clues referencing ``ref_place``."""
    return [c for c in DISTANCE_CLUES if c.ref_place.lower() == ref_place.lower()]


TOOLS: Dict[str, Any] = {
    "ocr_text": ocr_text,
    "ocr_image": ocr_image,
    "translate_text": translate_text,
    "detect_language": detect_language,
    "search_corpus": search_corpus,
    "geocode_place": geocode_place,
    "index_documents": index_documents,
    "semantic_search": semantic_search,
    "search_text": search_text,
    "extract_locations": extract_locations,
    "search_location": search_location,
    "parse_distance_clues": parse_distance_clues,
    "search_distance_clues": search_distance_clues,
}

__all__ = [
    "Excerpt",
    "Entity",
    "DistanceClue",
    "ocr_text",
    "ocr_image",
    "translate_text",
    "detect_language",
    "search_corpus",
    "geocode_place",
    "index_documents",
    "semantic_search",
    "search_text",
    "extract_locations",
    "search_location",
    "parse_distance_clues",
    "search_distance_clues",
    "TOOLS",
]
