from agents.memory import MemoryKeeper, memory_tools
import milvus_client
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
        {
            "id": "d1",
            "title": "doc",
            "author": "Smith",
            "date": "1920",
            "text": "The Amazon river basin is vast."
        }
    ]
    mk.index_documents(docs)
    results = mk.semantic_search("Amazon")
    assert results == []


def test_search_text_basic():
    reset()
    mk = MemoryKeeper()
    docs = [
        {
            "id": "d1",
            "title": "doc",
            "author": "Smith",
            "date": "1920",
            "text": "The Amazon river basin is vast. Many cultures lived here.",
        }
    ]
    mk.index_documents(docs)
    results = mk.search_text("Amazon basin")
    if milvus_client.connections is not None:
        assert results and results[0].doc_id == "d1"
        assert results[0].source


def test_search_text_keyword():
    reset()
    mk = MemoryKeeper()
    docs = [
        {
            "id": "d2",
            "title": "doc",
            "author": "Jones",
            "date": "1905",
            "text": "The village of Yucuru sat by the river."
        }
    ]
    mk.index_documents(docs)
    results = mk.search_text("Yucuru")
    assert results and results[0].doc_id == "d2"
    assert results[0].source


def test_extract_locations_basic():
    ents = memory_tools.extract_locations("The village of Yucuru sat by the river.")
    assert isinstance(ents, list)
    if ents:
        assert any(e.text == "Yucuru" for e in ents)


def test_search_location_basic():
    reset()
    mk = MemoryKeeper()
    docs = [
        {
            "id": "d3",
            "title": "doc",
            "author": "Jones",
            "date": "1905",
            "text": "The village of Yucuru sat by the river."
        }
    ]
    mk.index_documents(docs)
    results = mk.search_location("Yucuru")
    assert results and results[0].doc_id == "d3"
    assert results[0].source


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
        {
            "id": "d4",
            "title": "doc",
            "author": "Smith",
            "date": "1920",
            "text": "two days upriver from Santa Isabel"
        }
    ]
    mk.index_documents(docs)
    clues = memory_tools.search_distance_clues("Santa Isabel")
    if clues:
        assert clues[0].doc_id == "d4"
        assert clues[0].source


def test_geocode_place_fuzzy():
    coords = memory_tools.geocode_place("Santa Isabela")
    assert coords == memory_tools.PLACE_DB["Santa Isabel"]


def test_infer_relative_location_basic():
    clue = memory_tools.DistanceClue(
        ref_place="Santa Isabel", direction="north", distance=2, unit="days"
    )
    loc = memory_tools.infer_relative_location(clue)
    assert loc is not None
    base_lon, base_lat = memory_tools.PLACE_DB["Santa Isabel"]
    expected_lat = base_lat + (2 * 22.5) / 111.0
    assert abs(loc.lat - expected_lat) < 0.05


def test_summarize_clues_basic():
    reset()
    mk = MemoryKeeper()
    docs = [
        {
            "id": "d5",
            "title": "Journal",
            "author": "Fawcett",
            "date": "1920",
            "text": "two days upriver from Santa Isabel there are earth mounds.",
        }
    ]
    mk.index_documents(docs)
    summary = mk.summarize_clues("Santa Isabel")
    assert isinstance(summary, str)
    assert "Fawcett" in summary or "Journal" in summary


def test_historical_clues_update():
    reset()
    mk = MemoryKeeper()
    docs = [
        {
            "id": "d6",
            "title": "Diary",
            "author": "Smith",
            "date": "1930",
            "text": "camp two days north of Santa Isabel",
        }
    ]
    mk.index_documents(docs)
    clue = memory_tools.DistanceClue(
        ref_place="Santa Isabel", direction="north", distance=2, unit="days"
    )
    mk.infer_relative_location(clue)
    summary = mk.summarize_clues("Santa Isabel")
    coords = memory_tools.geocode_place("Santa Isabel")
    if coords:
        key = f"{coords[0]:.4f},{coords[1]:.4f}"
        assert key in world_state.get("historical_clues", {})


def test_memory_plan_and_act(monkeypatch):
    """MemoryKeeper should plan and call tools autonomously."""
    reset()

    class DummyClient:
        def __init__(self):
            self.calls = 0
            self.chat = self.Chat(self)

        class Chat:
            def __init__(self, outer):
                self.outer = outer
                self.completions = self

            def create(self, model, messages, tools=None, tool_choice=None):
                if self.outer.calls == 0:
                    self.outer.calls += 1

                    class ToolFunction:
                        name = "search_corpus"
                        arguments = '{"query": "Orinoco"}'

                    class ToolCall:
                        id = "1"
                        function = ToolFunction()

                    class Message:
                        content = ""
                        tool_calls = [ToolCall()]

                    class Completion:
                        choices = [type("C", (), {"message": Message()})]

                    return Completion()

                class Message:
                    content = "done"
                    tool_calls = None

                class Completion:
                    choices = [type("C", (), {"message": Message()})]

                return Completion()

    monkeypatch.setattr("agents.memory.memory.client", DummyClient())

    mk = MemoryKeeper()

    def stub_search(query: str):
        world_state["search_results"] = [query]
        return [query]

    mk.tools["search_corpus"] = stub_search

    answer = mk.plan_and_act("Find references")

    assert answer == "done"
    assert world_state.get("search_results")
    msgs = world_state.get("messages", [])
    assert any(m["type"] == "final_output" for m in msgs)


