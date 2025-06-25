from __future__ import annotations

"""Utility helpers for connecting to a Milvus vector database."""

import os
from pathlib import Path
from typing import Any

from clip_model import compute_image_embedding, compute_text_embedding

try:
    from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
except Exception:  # pragma: no cover - optional dependency may be missing
    Collection = None  # type: ignore
    CollectionSchema = None  # type: ignore
    FieldSchema = None  # type: ignore
    DataType = None  # type: ignore
    connections = None  # type: ignore
    utility = None  # type: ignore


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
    """Create or retrieve a collection for text embeddings with metadata."""
    if connections is None:
        raise RuntimeError("pymilvus is not installed")

    if name in utility.list_collections():
        return Collection(name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="page", dtype=DataType.INT64),
        FieldSchema(name="chunk", dtype=DataType.INT64),
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


def create_image_embeddings_collection(
    name: str = "image_embeddings", dim: int = 512, metric_type: str = "L2"
) -> Any:
    """Create or retrieve a collection for image embeddings with metadata.

    The collection stores a ``FLOAT_VECTOR`` field named ``embedding`` as well
    as ``image_id`` and ``source`` columns describing the image source or
    provenance.
    """
    if connections is None:
        raise RuntimeError("pymilvus is not installed")

    if name in utility.list_collections():
        return Collection(name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="image_id", dtype=DataType.INT64),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
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


def insert_image_embeddings(
    images: list[dict[str, Any]],
    *,
    collection: Any | None = None,
) -> Any:
    """Insert image vectors into the Milvus image collection.

    Each item in ``images`` should provide ``image_id`` and either ``path`` or
    ``image`` along with an optional ``source`` or ``description`` field.
    """

    if connections is None:
        raise RuntimeError("pymilvus is not installed")

    if collection is None:
        collection = create_image_embeddings_collection()

    ids: list[int] = []
    sources: list[str] = []
    vectors: list[list[float]] = []

    for item in images:
        img_id = item.get("image_id") or item.get("id")
        if img_id is None:
            continue
        img = item.get("path") or item.get("image")
        if img is None:
            continue
        vec = compute_image_embedding(img)
        ids.append(int(img_id))
        sources.append(str(item.get("source") or item.get("description", "")))
        vectors.append(vec)

    if vectors:
        collection.insert([ids, sources, vectors])
        collection.flush()

    return collection


def search_images(
    query: str | "Image.Image",
    *,
    top_k: int = 5,
    collection: Any | None = None,
) -> list[dict[str, Any]]:
    """Search the image embeddings collection for ``query``.

    ``query`` may be an image path or :class:`PIL.Image` instance. If ``query``
    is text or a path that does not exist, a text embedding is computed using
    the CLIP model instead.
    """

    if connections is None:
        raise RuntimeError("pymilvus is not installed")

    if collection is None:
        collection = create_image_embeddings_collection()

    if isinstance(query, str) and not Path(query).is_file():
        vec = compute_text_embedding(query)
    else:
        vec = compute_image_embedding(query)

    search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
    results = collection.search(
        [vec],
        "embedding",
        param=search_params,
        limit=top_k,
        output_fields=["image_id", "source"],
    )

    matches: list[dict[str, Any]] = []
    for hit in results[0]:
        matches.append(
            {
                "image_id": int(hit.entity.get("image_id", 0)),
                "source": hit.entity.get("source", ""),
                "distance": float(hit.distance),
            }
        )
    return matches


__all__ = [
    "connect_milvus",
    "create_embeddings_collection",
    "create_image_embeddings_collection",
    "insert_image_embeddings",
    "search_images",
]
