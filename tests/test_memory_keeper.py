from agents.memory import MemoryKeeper, memory_tools
from agents.brain import Brain  # ensure brain package loads before world_state
from world_state import world_state, reset

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional dependency may be missing
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore


def test_memory_keeper_basic():
    reset()
    mk = MemoryKeeper()
    results = mk.search_corpus("Orinoco")
    assert results and results[0]["doc_id"] == "doc1"

    coords = mk.geocode_place("Orinoco")
    assert coords == memory_tools.PLACE_DB["Orinoco"]

    messages = world_state.get("messages", [])
    if messages:
        msg = messages[0]
        assert msg["agent"] == "memory"
        assert "type" in msg and "content" in msg


def test_ocr_image_basic():
    reset()
    if Image is None:
        return
    mk = MemoryKeeper()
    img = Image.new("L", (100, 30), color=255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default() if ImageFont else None
    draw.text((10, 10), "test", fill=0, font=font)
    text = mk.ocr_image(img)
    assert isinstance(text, str)
    lang = world_state.get("last_language")
    assert lang is not None


def test_translate_text_basic():
    mk = MemoryKeeper()
    result = mk.translate_text("hola mundo", "es")
    assert isinstance(result, str)


def test_semantic_search_basic():
    reset()
    mk = MemoryKeeper()
    docs = [
        {"id": "d1", "title": "doc", "text": "The Amazon river basin is vast."}
    ]
    mk.index_documents(docs)
    results = mk.semantic_search("Amazon")
    if memory_tools.faiss is not None:
        assert results and results[0]["doc_id"] == "d1"


def test_search_text_basic():
    reset()
    mk = MemoryKeeper()
    docs = [
        {
            "id": "d1",
            "title": "doc",
            "text": "The Amazon river basin is vast. Many cultures lived here.",
        }
    ]
    mk.index_documents(docs)
    results = mk.search_text("Amazon basin")
    if memory_tools.faiss is not None:
        assert results and results[0].doc_id == "d1"


def test_search_text_keyword():
    reset()
    mk = MemoryKeeper()
    docs = [
        {"id": "d2", "title": "doc", "text": "The village of Yucuru sat by the river."}
    ]
    mk.index_documents(docs)
    results = mk.search_text("Yucuru")
    assert results and results[0].doc_id == "d2"


def test_extract_locations_basic():
    ents = memory_tools.extract_locations("The village of Yucuru sat by the river.")
    assert isinstance(ents, list)
    if ents:
        assert any(e.text == "Yucuru" for e in ents)


def test_search_location_basic():
    reset()
    mk = MemoryKeeper()
    docs = [
        {"id": "d3", "title": "doc", "text": "The village of Yucuru sat by the river."}
    ]
    mk.index_documents(docs)
    results = mk.search_location("Yucuru")
    assert results and results[0].doc_id == "d3"


def test_parse_distance_clues_basic():
    text = "two days upriver from Santa Isabel"
    clues = memory_tools.parse_distance_clues(text)
    assert clues and clues[0].ref_place == "Santa Isabel"
    assert clues[0].direction == "upriver"
    assert clues[0].distance == 2


def test_index_distance_clues():
    reset()
    mk = MemoryKeeper()
    docs = [
        {"id": "d4", "title": "doc", "text": "two days upriver from Santa Isabel"}
    ]
    mk.index_documents(docs)
    clues = memory_tools.search_distance_clues("Santa Isabel")
    if clues:
        assert clues[0].doc_id == "d4"

