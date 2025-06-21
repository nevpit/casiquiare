import types
from milvus_client import connect_milvus


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
