from types import SimpleNamespace
import pytest
from clip_model import (
    load_clip_model,
    compute_image_embedding,
    compute_text_embedding,
)


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


def test_compute_image_embedding(monkeypatch):
    events = {}

    class DummyTensor:
        def __init__(self, label="tensor"):
            self.label = label

        def unsqueeze(self, dim):
            events["unsqueeze"] = dim
            return self

    class DummyArray:
        def __init__(self, arr):
            self.arr = arr

        def tolist(self):
            return self.arr

    class DummyModel:
        def eval(self):
            pass

        def encode_image(self, tensor):
            events["encode"] = tensor.label
            return [DummyArray([1.0, 2.0])]

    def dummy_load(name="ViT-B/32", device="cpu"):
        return DummyModel(), lambda img: DummyTensor(f"prep-{img}")

    class DummyNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    dummy_image = SimpleNamespace(convert=lambda mode: "img")

    monkeypatch.setattr("clip_model.clip", SimpleNamespace(load=dummy_load))
    monkeypatch.setattr(
        "clip_model.Image",
        SimpleNamespace(open=lambda path: dummy_image, Image=dummy_image.__class__),
    )
    monkeypatch.setattr("clip_model.torch", SimpleNamespace(no_grad=lambda: DummyNoGrad()))

    load_clip_model()
    vec = compute_image_embedding("foo.png")
    assert vec == [1.0, 2.0]
    assert events["unsqueeze"] == 0
    assert events["encode"] == "prep-img"

    vec2 = compute_image_embedding(dummy_image)
    assert vec2 == [1.0, 2.0]


def test_compute_text_embedding(monkeypatch):
    events = {}

    class DummyArray:
        def __init__(self, arr):
            self.arr = arr

        def tolist(self):
            return self.arr

    class DummyModel:
        def eval(self):
            pass

        def encode_text(self, tokens):
            events["encode"] = tokens
            return [DummyArray([3.0, 4.0])]

    def dummy_load(name="ViT-B/32", device="cpu"):
        return DummyModel(), "prep"

    class DummyNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "clip_model.clip",
        SimpleNamespace(load=dummy_load, tokenize=lambda x: "tok"),
    )
    monkeypatch.setattr("clip_model.torch", SimpleNamespace(no_grad=lambda: DummyNoGrad()))

    load_clip_model()
    vec = compute_text_embedding("hello")
    assert vec == [3.0, 4.0]
    assert events["encode"] == "tok"
