import pytest

from agents.brain import Brain
from agents.brain.brain_outputs import ExecutionResult
from world_state import reset, world_state


def test_brain_plan_and_act(monkeypatch):
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
                        name = "exec_code"
                        arguments = '{"code": "result = 2 + 2"}'

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

    monkeypatch.setattr("agents.brain.brain.client", DummyClient())

    brain = Brain()

    def stub_exec(code: str, local_vars=None, timeout=5):
        res = {"stdout": "", "result": 4, "figures": [], "locals": None, "error": None}
        world_state["last_exec"] = res
        return res

    brain.tools["exec_code"] = stub_exec

    answer = brain.plan_and_act("calculate")

    assert answer == "done"
    assert world_state.get("last_exec", {}).get("result") == 4
    msgs = world_state.get("messages", [])
    assert any(m["type"] == "final_output" for m in msgs)
