import pytest
from agents.brain import brain_tools


def test_exec_code_timeout():
    res = brain_tools.exec_code("while True: pass", timeout=1)
    assert res.get("error") == "timeout"


def test_exec_code_simple():
    res = brain_tools.exec_code("result = 2 + 2")
    assert res.get("result") == 4
    assert res.get("error") is None
