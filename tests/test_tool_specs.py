import pytest

from agents.brain import Brain


def test_tool_specs_list_and_dict():
    brain = Brain()

    def stub(numbers: list[int], opts: dict[str, int]):
        return numbers, opts

    brain.tools = {"stub": stub}
    specs = brain._tool_specs()
    assert specs
    params = specs[0]["function"]["parameters"]["properties"]
    assert params["numbers"]["type"] == "array"
    assert params["opts"]["type"] == "object"
