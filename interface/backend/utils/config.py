from __future__ import annotations

"""Utility helpers for backend configuration."""

import os
from pathlib import Path
from typing import Dict


def load_backend_config() -> Dict[str, Path]:
    """Return configuration values for the Flask backend.

    Reads ``EYES_OUTPUT_DIR`` and ``EYES_TILE_DIR`` from the environment and
    converts them to :class:`~pathlib.Path` objects. The resulting dictionary can
    be passed directly to ``app.config.update``.
    """
    output_dir = os.getenv("EYES_OUTPUT_DIR")
    tile_dir = os.getenv("EYES_TILE_DIR")

    config: Dict[str, Path] = {}
    if output_dir:
        config["DETECTIONS_PATH"] = Path(output_dir)
    if tile_dir:
        config["TILES_ROOT"] = Path(tile_dir)
    return config


__all__ = ["load_backend_config"]
