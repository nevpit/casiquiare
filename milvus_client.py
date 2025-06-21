from __future__ import annotations

"""Utility helpers for connecting to a Milvus vector database."""

import os
from typing import Any

try:
    from pymilvus import connections
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


__all__ = ["connect_milvus"]
