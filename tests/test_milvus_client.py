import types
from milvus_client import (
    connect_milvus,
    create_embeddings_collection,
    create_image_embeddings_collection,
    insert_image_embeddings,
)


def test_connect_milvus_env(monkeypatch):
    called = {}

    class DummyConns:
        def connect(self, alias="default", host=None, port=None):
            called["host"] = host
            called["port"] = port

        def get_connection(self, alias="default"):
            return "conn"  # simple marker

    dummy = DummyConns()
    monkeypatch.setattr("milvus_client.connections", dummy)
    monkeypatch.setenv("MILVUS_HOST", "10.0.0.1")
    monkeypatch.setenv("MILVUS_PORT", "12345")

    conn = connect_milvus()
    assert conn == "conn"
    assert called["host"] == "10.0.0.1"
    assert called["port"] == "12345"


def test_create_embeddings_collection(monkeypatch):
    events = {}

    class DummyUtility:
        @staticmethod
        def list_collections():
            return []

    def DummyFieldSchema(name, dtype, **kwargs):
        events.setdefault("fields", []).append((name, dtype, kwargs))
        return f"field-{name}"

    def DummyCollectionSchema(fields):
        events["schema"] = fields
        return "schema"

    class DummyCollection:
        def __init__(self, name=None, schema=None):
            events["name"] = name
            events["schema_passed"] = schema

        def create_index(self, field_name, params):
            events["index"] = params

    DataType = types.SimpleNamespace(
        INT64="INT64", FLOAT_VECTOR="FLOAT_VECTOR", VARCHAR="VARCHAR"
    )

    monkeypatch.setattr("milvus_client.utility", DummyUtility)
    monkeypatch.setattr("milvus_client.FieldSchema", DummyFieldSchema)
    monkeypatch.setattr("milvus_client.CollectionSchema", DummyCollectionSchema)
    monkeypatch.setattr("milvus_client.Collection", DummyCollection)
    monkeypatch.setattr("milvus_client.DataType", DataType)
    monkeypatch.setattr("milvus_client.connections", object())

    create_embeddings_collection()

    assert events["name"] == "text_embeddings"
    assert events["index"]["metric_type"] == "L2"
    assert len(events["fields"]) == 6
    assert events["fields"][-1][2]["dim"] == 1536


def test_create_image_embeddings_collection(monkeypatch):
    events = {}

    class DummyUtility:
        @staticmethod
        def list_collections():
            return []

    def DummyFieldSchema(name, dtype, **kwargs):
        events.setdefault("fields", []).append((name, dtype, kwargs))
        return f"field-{name}"

    def DummyCollectionSchema(fields):
        events["schema"] = fields
        return "schema"

    class DummyCollection:
        def __init__(self, name=None, schema=None):
            events["name"] = name
            events["schema_passed"] = schema

        def create_index(self, field_name, params):
            events["index"] = params

    DataType = types.SimpleNamespace(
        INT64="INT64", FLOAT_VECTOR="FLOAT_VECTOR", VARCHAR="VARCHAR"
    )

    monkeypatch.setattr("milvus_client.utility", DummyUtility)
    monkeypatch.setattr("milvus_client.FieldSchema", DummyFieldSchema)
    monkeypatch.setattr("milvus_client.CollectionSchema", DummyCollectionSchema)
    monkeypatch.setattr("milvus_client.Collection", DummyCollection)
    monkeypatch.setattr("milvus_client.DataType", DataType)
    monkeypatch.setattr("milvus_client.connections", object())

    create_image_embeddings_collection()

    assert events["name"] == "image_embeddings"
    assert events["index"]["metric_type"] == "L2"
    assert len(events["fields"]) == 4
    assert events["fields"][2][0] == "source"
    assert events["fields"][3][2]["dim"] == 512


def test_insert_image_embeddings(monkeypatch):
    inserted = {}

    class DummyCollection:
        def insert(self, data):
            inserted["data"] = data

        def flush(self):
            inserted["flushed"] = True

    monkeypatch.setattr(
        "milvus_client.create_image_embeddings_collection", lambda: DummyCollection()
    )
    monkeypatch.setattr(
        "milvus_client.compute_image_embedding", lambda img: [0.1, 0.2]
    )
    monkeypatch.setattr("milvus_client.connections", object())

    images = [
        {"image_id": 1, "path": "img1.jpg", "source": "scan"},
        {"image_id": 2, "image": "img", "description": "photo"},
    ]

    coll = insert_image_embeddings(images)

    assert inserted["data"][0] == [1, 2]
    assert inserted["data"][1] == ["scan", "photo"]
    assert inserted["data"][2] == [[0.1, 0.2], [0.1, 0.2]]
    assert inserted.get("flushed")
    assert isinstance(coll, DummyCollection)
