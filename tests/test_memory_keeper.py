from agents.memory import MemoryKeeper, memory_tools
from agents.brain import Brain  # ensure brain package loads before world_state
from world_state import world_state, reset


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

