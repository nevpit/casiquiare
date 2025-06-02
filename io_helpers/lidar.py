"""LiDAR I/O helpers."""

from __future__ import annotations

import json
from typing import Any

try:  # Optional heavy dependency
    import pdal
except Exception:  # pragma: no cover - library may be missing
    pdal = None


def load_laz(path: str) -> "pdal.Pipeline":
    """Create a PDAL pipeline that loads a LAZ file."""
    if pdal is None:
        raise RuntimeError("PDAL is not installed.")
    pipeline: Any = {"pipeline": [{"type": "readers.las", "filename": path}]}
    return pdal.Pipeline(json.dumps(pipeline))


__all__ = ["load_laz"]
