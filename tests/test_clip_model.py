from types import SimpleNamespace
import pytest

from clip_model import load_clip_model


def test_load_clip_model(monkeypatch):
    events = {}

    class DummyModel:
        def eval(self):
            events["eval"] = True

    def dummy_load(name="ViT-B/32", device="cpu"):
        events["name"] = name
        events["device"] = device
        return DummyModel(), "prep"

    monkeypatch.setattr("clip_model.clip", SimpleNamespace(load=dummy_load))

    model = load_clip_model("ViT-B/16", "cuda")
    assert isinstance(model, DummyModel)
    assert events["name"] == "ViT-B/16"
    assert events["device"] == "cuda"
    assert events.get("eval")


def test_load_clip_model_missing(monkeypatch):
    monkeypatch.setattr("clip_model.clip", None)
    with pytest.raises(RuntimeError):
        load_clip_model()
