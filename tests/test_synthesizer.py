from agents.synthesizer import Synthesizer


def test_sub_agent_instantiation():
    syn = Synthesizer()
    assert syn.eyes.__class__.__name__ == "Eyes"
    assert syn.brain.__class__.__name__ == "Brain"
    assert syn.memory_keeper.__class__.__name__ == "MemoryKeeper"
    assert syn.context_engine.__class__.__name__ == "ContextEngine"


def test_tool_registration():
    syn = Synthesizer()
    assert "brain_train_model" in syn.tools
    assert syn.tools["brain_train_model"].__self__ is syn.brain
    assert "brain_exec_code" in syn.tools
    assert "eyes_scan_area" in syn.tools
    from agents.eyes import eyes_tools as eyes_tools
    assert syn.tools["eyes_scan_area"] is eyes_tools.scan_area


def test_world_state_integration():
    from world_state import reset, world_state

    reset()
    world_state["latest_model"] = {"model_type": "rf"}
    syn = Synthesizer()
    ctx = syn.gather_context()
    assert ctx["latest_model"]["model_type"] == "rf"

    res = syn.use_tool("memory_search_corpus", "Orinoco")
    assert res and world_state.get("search_results")
    messages = world_state.get("messages", [])
    assert any(m["agent"] == "synthesizer" for m in messages)


def test_plan_and_act(monkeypatch):
    from world_state import reset, world_state

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
                        name = "memory_search_corpus"
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
                else:
                    class Message:
                        content = "finished"
                        tool_calls = None

                    class Completion:
                        choices = [type("C", (), {"message": Message()})]

                    return Completion()

    monkeypatch.setattr("agents.synthesizer.synthesizer.client", DummyClient())

    syn = Synthesizer()
    answer = syn.plan_and_act("Find documents about Orinoco")
    assert answer == "finished"
    assert world_state.get("search_results")
    logs = world_state.get("messages", [])
    assert any(m["type"] == "delegate" for m in logs)
    assert any(m["type"] == "conclusion" for m in logs)
    assert any(m["type"] == "final_output" for m in logs)


def test_end_to_end_workflow(monkeypatch):
    """Run Synthesizer through a multi-tool workflow."""
    from world_state import reset, world_state, get_value, set_value

    reset()

    # Dummy OpenAI client to drive the planning loop
    class DummyClient:
        def __init__(self):
            self.step = 0
            self.chat = self.Chat(self)

        class Chat:
            def __init__(self, outer):
                self.outer = outer
                self.completions = self

            def create(self, model, messages, tools=None, tool_choice=None):
                step = self.outer.step
                self.outer.step += 1
                if step == 0:
                    return type("C", (), {
                        "choices": [type("S", (), {
                            "message": type("M", (), {
                                "content": "",
                                "tool_calls": [type("TC", (), {
                                    "id": "1",
                                    "function": type("F", (), {
                                        "name": "eyes_analyze_lidar",
                                        "arguments": '{"path": "region.laz"}'
                                    })()
                                })]
                            })()
                        })]
                    })()
                if step == 1:
                    return type("C", (), {
                        "choices": [type("S", (), {
                            "message": type("M", (), {
                                "content": "",
                                "tool_calls": [type("TC", (), {
                                    "id": "2",
                                    "function": type("F", (), {
                                        "name": "brain_cluster_features",
                                        "arguments": '{"features": [1,2]}'
                                    })()
                                })]
                            })()
                        })]
                    })()
                if step == 2:
                    return type("C", (), {
                        "choices": [type("S", (), {
                            "message": type("M", (), {
                                "content": "",
                                "tool_calls": [type("TC", (), {
                                    "id": "3",
                                    "function": type("F", (), {
                                        "name": "memory_search_corpus",
                                        "arguments": '{"query": "region x"}'
                                    })()
                                })]
                            })()
                        })]
                    })()
                if step == 3:
                    return type("C", (), {
                        "choices": [type("S", (), {
                            "message": type("M", (), {
                                "content": "",
                                "tool_calls": [type("TC", (), {
                                    "id": "4",
                                    "function": type("F", (), {
                                        "name": "context_analyze_environment",
                                        "arguments": '{"dem": "d.tif", "land_cover": "lc.tif", "soil": "s.tif", "distance": "dist.tif", "climate": "c.tif", "context_layers": {}, "lat": 0.0, "lon": 0.0}'
                                    })()
                                })]
                            })()
                        })]
                    })()
                return type("C", (), {
                    "choices": [type("S", (), {
                        "message": type("M", (), {"content": "summary", "tool_calls": None})()
                    })]
                })()

    monkeypatch.setattr("agents.synthesizer.synthesizer.client", DummyClient())

    syn = Synthesizer()

    # Patch tools with simple stubs that update world_state
    def stub_eyes(path: str):
        set_value("eyes_result", {"path": path})
        return {"features": 1}

    def stub_brain(features, eps=100.0, min_samples=2, area_id=None):
        set_value("latest_model", {"model_type": "stub"})
        set_value("clusters", {"labels": [0, 1]})
        return {"labels": [0, 1]}

    def stub_memory(query: str):
        set_value("search_results", [query])
        return [query]

    def stub_context(
        dem, land_cover, soil, distance, climate, context_layers, lat, lon
    ):
        set_value("context_environment", {"ndvi": 0.5})
        return {"ndvi": 0.5}

    syn.tools = {
        "eyes_analyze_lidar": stub_eyes,
        "brain_cluster_features": stub_brain,
        "memory_search_corpus": stub_memory,
        "context_analyze_environment": stub_context,
    }

    answer = syn.plan_and_act("Assess region", max_steps=6)

    assert answer == "summary"
    assert get_value("eyes_result")
    assert get_value("clusters")
    assert get_value("search_results")
    assert get_value("context_environment")
    msgs = world_state.get("messages", [])
    assert any(m["type"] == "final_output" for m in msgs)
