"""Utilities for loading configured data paths."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

try:
    import yaml
except Exception:  # pragma: no cover - dependency may be missing
    yaml = None  # type: ignore


_CONFIG_FILE = Path(__file__).resolve().parents[1] / "config" / "data_paths.yaml"


def load_mapping(path: Optional[str] = None) -> Dict[str, str]:
    """Load the data path mapping from a YAML file."""
    if yaml is None:
        raise RuntimeError("PyYAML is not installed.")
    cfg_path = Path(path) if path is not None else _CONFIG_FILE
    text = cfg_path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text) or {}
    except Exception:
        data = {}
        for line in text.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                data[k.strip()] = v.strip()
    if not isinstance(data, dict):
        raise ValueError("Mapping file must define a dictionary")
    return {str(k): str(v) for k, v in data.items()}


def get_data_path(
    *, tile_id: Optional[str] = None, bbox: Optional[Tuple[float, float, float, float]] = None,
    mapping: Optional[Dict[str, str]] = None
) -> str:
    """Return the data path for a tile ID or bounding box."""
    map_data = mapping or load_mapping()
    if tile_id is not None:
        try:
            return map_data[tile_id]
        except KeyError:
            raise KeyError(f"Tile ID {tile_id} not found")
    if bbox is not None:
        key = "[" + ",".join(str(x) for x in bbox) + "]"
        try:
            return map_data[key]
        except KeyError:
            raise KeyError(f"Bounding box {bbox} not found")
    raise ValueError("Either tile_id or bbox must be provided")


__all__ = ["load_mapping", "get_data_path"]
