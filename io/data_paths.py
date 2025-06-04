"""Utilities for loading configured data paths."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

from log_config import setup_logger

try:
    import yaml
except Exception:  # pragma: no cover - dependency may be missing
    yaml = None  # type: ignore


_CONFIG_FILE = Path(__file__).resolve().parents[1] / "config" / "data_paths.yaml"

logger = setup_logger(__name__)


def load_mapping(path: Optional[str] = None) -> Dict[str, str]:
    """Load the data path mapping from a YAML file."""
    if yaml is None:
        logger.warning("PyYAML is not installed; cannot load mapping")
        raise RuntimeError("PyYAML is not installed.")
    cfg_path = Path(path) if path is not None else _CONFIG_FILE
    logger.info("Loading mapping from %s", cfg_path)
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:  # type: ignore[attr-defined]
        logger.warning("YAML parse error: %s; falling back to naive parser", exc)
        data = {}
        with cfg_path.open("r", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                data[key.strip().replace(" ", "")] = val.strip()
    if not isinstance(data, dict):
        raise ValueError("Mapping file must define a dictionary")
    result = {str(k).replace(" ", ""): str(v) for k, v in data.items()}
    logger.debug("Loaded %d entries from mapping", len(result))
    return result


def get_data_path(
    *, tile_id: Optional[str] = None, bbox: Optional[Tuple[float, float, float, float]] = None,
    mapping: Optional[Dict[str, str]] = None
) -> str:
    """Return the data path for a tile ID or bounding box."""
    map_data = mapping or load_mapping()
    if tile_id is not None:
        try:
            path = map_data[tile_id]
            logger.info("Resolved tile %s to %s", tile_id, path)
            return path
        except KeyError:
            logger.warning("Tile ID %s not found", tile_id)
            raise KeyError(f"Tile ID {tile_id} not found")
    if bbox is not None:
        key = str(list(bbox)).replace(" ", "")
        try:
            path = map_data[key]
            logger.info("Resolved bbox %s to %s", bbox, path)
            return path
        except KeyError:
            logger.warning("Bounding box %s not found", bbox)
            raise KeyError(f"Bounding box {bbox} not found")
    raise ValueError("Either tile_id or bbox must be provided")


__all__ = ["load_mapping", "get_data_path"]
