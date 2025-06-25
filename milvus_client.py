from __future__ import annotations

"""Utility helpers for connecting to a Milvus vector database."""

import os
from typing import Any

try:
    from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
except Exception:  # pragma: no cover - optional dependency may be missing
    connections = None  # type: ignore


def connect_milvus(host: str | None = None, port: str | None = None, alias: str = "default") -> Any:
    """Connect to a Milvus instance and return the connection.

    Parameters
    ----------
    host:
        Hostname of the Milvus server. Defaults to the ``MILVUS_HOST``
        environment variable or ``localhost``.
    port:
        Port of the Milvus server. Defaults to the ``MILVUS_PORT``
        environment variable or ``19530``.
    alias:
        Connection alias to use. Defaults to ``"default"``.
    """
    if connections is None:
        raise RuntimeError("pymilvus is not installed")

    host = host or os.getenv("MILVUS_HOST", "localhost")
    port = port or os.getenv("MILVUS_PORT", "19530")
    connections.connect(alias=alias, host=host, port=port)
    return connections.get_connection(alias)


def create_embeddings_collection(
    name: str = "text_embeddings", dim: int = 1536, metric_type: str = "L2"
) -> Any:
    """Create or retrieve a collection for text embeddings."""
    if connections is None:
        raise RuntimeError("pymilvus is not installed")

    if name in utility.list_collections():
        return Collection(name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_id", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields)
    collection = Collection(name, schema=schema)
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": metric_type,
        "params": {"nlist": 1024},
    }
    collection.create_index("embedding", index)
    return collection


__all__ = ["connect_milvus", "create_embeddings_collection"]
