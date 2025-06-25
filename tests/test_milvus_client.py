import types
from milvus_client import (
    connect_milvus,
    create_embeddings_collection,
    create_image_embeddings_collection,
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

    DataType = types.SimpleNamespace(INT64="INT64", FLOAT_VECTOR="FLOAT_VECTOR")

    monkeypatch.setattr("milvus_client.utility", DummyUtility)
    monkeypatch.setattr("milvus_client.FieldSchema", DummyFieldSchema)
    monkeypatch.setattr("milvus_client.CollectionSchema", DummyCollectionSchema)
    monkeypatch.setattr("milvus_client.Collection", DummyCollection)
    monkeypatch.setattr("milvus_client.DataType", DataType)

    create_embeddings_collection()

    assert events["name"] == "text_embeddings"
    assert events["index"]["metric_type"] == "L2"
    assert events["fields"][2][2]["dim"] == 1536


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

    DataType = types.SimpleNamespace(INT64="INT64", FLOAT_VECTOR="FLOAT_VECTOR")

    monkeypatch.setattr("milvus_client.utility", DummyUtility)
    monkeypatch.setattr("milvus_client.FieldSchema", DummyFieldSchema)
    monkeypatch.setattr("milvus_client.CollectionSchema", DummyCollectionSchema)
    monkeypatch.setattr("milvus_client.Collection", DummyCollection)
    monkeypatch.setattr("milvus_client.DataType", DataType)

    create_image_embeddings_collection()

    assert events["name"] == "image_embeddings"
    assert events["index"]["metric_type"] == "L2"
    assert events["fields"][2][2]["dim"] == 512
