from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from flask import Blueprint, jsonify, current_app

bp = Blueprint("eyes", __name__, url_prefix="/eyes")


@bp.route("/ping")
def ping():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


def _latest_geojson(base_dir: Path) -> Optional[Path]:
    files = list(base_dir.glob("*_features.geojson"))
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


@bp.route("/detections")
def detections():
    """Return the most recent <area>_features.geojson as JSON."""
    base_dir = Path(current_app.config.get("DETECTIONS_PATH", Path.cwd()))
    latest = _latest_geojson(base_dir)
    if latest is None:
        return jsonify({"error": "No detection file found"}), 404

    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return jsonify({"error": "Failed to read detections"}), 500
    return jsonify(data)


__all__ = ["bp"]
