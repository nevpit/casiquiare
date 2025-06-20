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

