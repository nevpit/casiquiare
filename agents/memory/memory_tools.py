"""Utility functions for the Memory Keeper agent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import re

import os
from hashlib import md5
import math

from log_config import setup_logger

logger = setup_logger("casiquiare.memory")


def log_action(func):
    """Decorator to log entry and exit of a function."""

    def wrapper(*args, **kwargs):
        logger.info("%s called", func.__name__)
        res = func(*args, **kwargs)
        logger.info("%s returned", func.__name__)
        return res

    return wrapper


@dataclass
class Excerpt:
    """Segment of text stored in the semantic index."""

    doc_id: str
    title: str
    page: int
    text: str
    distance: float = 0.0
    source: str = ""


@dataclass
class Entity:
    """Named entity extracted from text."""

    text: str
    label: str
    start: int
    end: int
    doc_id: str = ""
    title: str = ""
    page: int = 0
    source: str = ""


@dataclass
class DistanceClue:
    """Distance and direction reference extracted from text."""

    ref_place: str
    direction: str
    distance: float
    unit: str
    doc_id: str = ""
    page: int = 0
    source: str = ""


@dataclass
class InferredLocation:
    """Approximate coordinates derived from a :class:`DistanceClue`."""

    lon: float
    lat: float
    ref_place: str
    distance_km: float
    direction: str
    uncertainty_km: float
    source: str = ""

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
    "Santa Isabel": (-65.0, 0.5),
}

# Semantic search index
EMBED_DIM = 1536
SEMANTIC_INDEX: Optional[Any] = None
SEMANTIC_META: List[Dict[str, Any]] = []

KEYWORD_INDEX: Dict[str, List[int]] = {}
LOCATION_INDEX: Dict[str, List[int]] = {}
DISTANCE_CLUES: List[DistanceClue] = []


def make_citation(title: str = "", author: str = "", date: str = "", page: Optional[int] = None) -> str:
    """Return a formatted citation string."""
    parts = []
    if title:
        parts.append(title)
    if author:
        parts.append(author)
    if date:
        parts.append(str(date))
    if page is not None and page != 0:
        parts.append(f"p.{page}")
    return ", ".join(parts)


@log_action
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


@log_action
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


@log_action
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


@log_action
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


@log_action
def search_corpus(query: str) -> List[Dict[str, str]]:
    """Return documents containing the query string."""
    results = []
    for doc_id, text in CORPUS.items():
        if query.lower() in text.lower():
            results.append({"doc_id": doc_id, "snippet": text, "citation": doc_id})
    return results


@log_action
def geocode_place(name: str) -> Optional[Tuple[float, float]]:
    """Return coordinates for a place name using fuzzy lookup and OSM."""

    if not name:
        return None

    # 1) exact and fuzzy match against built-in gazetteer
    normalized = name.lower().strip()
    if normalized in {n.lower() for n in PLACE_DB}:
        return PLACE_DB.get(next(k for k in PLACE_DB if k.lower() == normalized))

    # simple similarity search using difflib
    from difflib import SequenceMatcher

    best_score = 0.0
    best_coord: Optional[Tuple[float, float]] = None
    for place, coord in PLACE_DB.items():
        score = SequenceMatcher(None, normalized, place.lower()).ratio()
        if score > best_score:
            best_score = score
            best_coord = coord
    if best_score >= 0.8:
        return best_coord

    # 2) fall back to querying OpenStreetMap Nominatim
    try:
        import json
        from urllib import request, parse

        url = (
            "https://nominatim.openstreetmap.org/search?" +
            parse.urlencode({"q": name, "format": "json", "limit": 1})
        )
        req = request.Request(url, headers={"User-Agent": "casiquiare/1.0"})
        with request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            if data:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                return (lon, lat)
    except Exception as exc:  # pragma: no cover - network failure
        logger.warning("Geocoding API failed: %s", exc)

    return None


@log_action
def extract_locations(
    text: str,
    *,
    doc_id: str = "",
    title: str = "",
    page: int = 0,
    citation: str = "",
) -> List[Entity]:
    """Return named locations found in ``text`` with optional source info."""
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
                            doc_id=doc_id,
                            title=title,
                            page=page,
                            source=citation,
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
                doc_id=doc_id,
                title=title,
                page=page,
                source=citation,
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

# Direction bearings in degrees
BEARING_MAP = {
    "north": 0.0,
    "northeast": 45.0,
    "east": 90.0,
    "southeast": 135.0,
    "south": 180.0,
    "southwest": 225.0,
    "west": 270.0,
    "northwest": 315.0,
    "upriver": 0.0,  # default assumption if river orientation unknown
    "downriver": 180.0,
}

# Rough conversion from historical units to kilometers
UNIT_KM = {
    "km": 1.0,
    "kilometer": 1.0,
    "kilometers": 1.0,
    "kilometros": 1.0,
    "mile": 1.60934,
    "miles": 1.60934,
    "milla": 1.60934,
    "millas": 1.60934,
    "league": 5.0,
    "leagues": 5.0,
    "legua": 5.0,
    "leguas": 5.0,
}


def _parse_number(text: str) -> float:
    text = text.replace(",", ".")
    if text.isdigit() or re.match(r"\d+\.\d+", text):
        try:
            return float(text)
        except Exception:
            return 0.0
    return float(NUM_MAP.get(text.lower(), 0))


def _convert_to_km(distance: float, unit: str) -> float:
    """Return distance in kilometers for various units."""
    unit = unit.lower().rstrip("s")
    if "day" in unit or "dia" in unit:
        # assume one day of canoe travel is roughly 22.5 km
        return distance * 22.5
    factor = UNIT_KM.get(unit, 0.0)
    return distance * factor


@log_action
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


@log_action
def infer_relative_location(clue: DistanceClue) -> Optional[InferredLocation]:
    """Estimate coordinates from a :class:`DistanceClue`."""

    base = geocode_place(clue.ref_place)
    if base is None:
        return None

    lon, lat = base
    dist_km = _convert_to_km(clue.distance, clue.unit)
    bearing = BEARING_MAP.get(clue.direction.lower(), 0.0)
    rad = math.radians(bearing)
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(lat)) or 111.0

    delta_lat = (dist_km * math.cos(rad)) / km_per_deg_lat
    delta_lon = (dist_km * math.sin(rad)) / km_per_deg_lon
    uncertainty = max(1.0, dist_km * 0.25)

    return InferredLocation(
        lon=lon + delta_lon,
        lat=lat + delta_lat,
        ref_place=clue.ref_place,
        distance_km=dist_km,
        direction=clue.direction,
        uncertainty_km=uncertainty,
        source=clue.source,
    )


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
    """Index documents for semantic and keyword search using Milvus."""
    global SEMANTIC_INDEX, SEMANTIC_META, KEYWORD_INDEX, LOCATION_INDEX, DISTANCE_CLUES

    from milvus_client import connect_milvus, create_embeddings_collection

    KEYWORD_INDEX.clear()
    LOCATION_INDEX.clear()
    DISTANCE_CLUES.clear()
    SEMANTIC_META = []
    SEMANTIC_INDEX = None

    try:
        connect_milvus()
        collection = create_embeddings_collection()
    except Exception:  # pragma: no cover - optional dependency may be missing
        collection = None
        logger.info("Milvus not available; embeddings will not be stored", exc_info=True)

    doc_ids: List[str] = []
    titles: List[str] = []
    pages: List[int] = []
    chunks: List[int] = []
    embeddings: List[List[float]] = []

    for doc in docs:
        doc_id = str(doc.get("id", ""))
        title = doc.get("title", "")
        author = doc.get("author", "")
        date = doc.get("date", "")
        text = doc.get("text", "")
        page = doc.get("page", 0)
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        for pi, para in enumerate(paragraphs):
            words = para.split()
            for start in range(0, len(words), chunk_size):
                chunk = " ".join(words[start : start + chunk_size])
                emb = _compute_embedding(chunk)
                citation = make_citation(title, author, date, page)
                locs = extract_locations(chunk, doc_id=doc_id or "", title=title, page=page, citation=citation)
                clues = parse_distance_clues(chunk)
                idx = len(SEMANTIC_META)
                SEMANTIC_META.append(
                    {
                        "doc_id": doc_id,
                        "title": title,
                        "page": page,
                        "chunk": pi,
                        "text": chunk,
                        "locations": [l.text for l in locs],
                        "distance_clues": [asdict(c) for c in clues],
                        "citation": citation,
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
                            source=citation,
                        )
                    )

                if collection is not None:
                    doc_ids.append(doc_id)
                    titles.append(title)
                    pages.append(page)
                    chunks.append(pi)
                    embeddings.append(emb)

    if collection is not None and embeddings:
        collection.insert([doc_ids, titles, pages, chunks, embeddings])
        collection.flush()


@log_action
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


@log_action
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


@log_action
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
                source=meta.get("citation", ""),
            )
        )

    sem_excerpts: List[Excerpt] = []
    try:
        from milvus_client import connect_milvus, create_embeddings_collection

        connect_milvus()
        collection = create_embeddings_collection()

        vec = _compute_embedding(query)
        search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
        results = collection.search(
            [vec],
            "embedding",
            param=search_params,
            limit=top_k * 2,
            output_fields=["doc_id", "title", "page", "chunk"],
        )

        seen = set(kw_indices)
        for hit in results[0]:
            doc_id = str(hit.entity.get("doc_id", ""))
            page = int(hit.entity.get("page", 0))
            chunk = int(hit.entity.get("chunk", 0))
            idx = -1
            for i, meta in enumerate(SEMANTIC_META):
                if (
                    meta.get("doc_id") == doc_id
                    and meta.get("page") == page
                    and meta.get("chunk") == chunk
                ):
                    idx = i
                    break
            if idx < 0 or idx in seen:
                continue
            meta = SEMANTIC_META[idx]
            sem_excerpts.append(
                Excerpt(
                    doc_id=doc_id,
                    title=hit.entity.get("title", ""),
                    page=page,
                    text=meta.get("text", ""),
                    distance=float(hit.distance),
                    source=meta.get("citation", ""),
                )
            )
            if len(sem_excerpts) >= top_k:
                break
    except Exception:  # pragma: no cover - optional dependency may be missing
        pass

    results = kw_excerpts + sem_excerpts
    return results[:top_k]


@log_action
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
                    source=meta.get("citation", ""),
                )
            )
    return results


@log_action
def search_images(query: str | Any, top_k: int = 5) -> List[Dict[str, Any]]:
    """Return top matching images from Milvus for ``query``."""
    try:
        from milvus_client import search_images as _search_images

        return _search_images(query, top_k=top_k)
    except Exception:  # pragma: no cover - optional dependency may be missing
        logger.info("Image search unavailable", exc_info=True)
        return []


@log_action
def search_distance_clues(ref_place: str) -> List[DistanceClue]:
    """Return stored distance clues referencing ``ref_place``."""
    return [c for c in DISTANCE_CLUES if c.ref_place.lower() == ref_place.lower()]


@log_action
def summarize_clues(location_query: str, top_k: int = 5) -> str:
    """Return a short summary of historical clues about ``location_query``."""
    excerpts = search_text(location_query, top_k=top_k)
    clues = search_distance_clues(location_query)
    lines: List[str] = []
    for ex in excerpts:
        snippet = ex.text.strip()
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        citation = ex.source or ex.title
        lines.append(f"- {snippet} ({citation})")
    for cl in clues:
        lines.append(
            f"- {cl.distance} {cl.unit} {cl.direction} from {cl.ref_place} ({cl.source})"
        )
    summary = "\n".join(lines[:top_k])
    if openai_client is not None and summary:
        messages = [
            {"role": "system", "content": "Summarize historical clues with citations."},
            {
                "role": "user",
                "content": f"Summarize the following notes about {location_query}:\n{summary}",
            },
        ]
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4-turbo", messages=messages
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:  # pragma: no cover - runtime failure
            logger.warning("LLM summarization failed: %s", exc)
    return summary


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
    "search_images": search_images,
    "parse_distance_clues": parse_distance_clues,
    "search_distance_clues": search_distance_clues,
    "infer_relative_location": infer_relative_location,
    "summarize_clues": summarize_clues,
}

__all__ = [
    "Excerpt",
    "Entity",
    "DistanceClue",
    "InferredLocation",
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
    "search_images",
    "parse_distance_clues",
    "search_distance_clues",
    "infer_relative_location",
    "summarize_clues",
    "TOOLS",
]
