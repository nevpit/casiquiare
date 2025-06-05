from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import collections
from flask import Blueprint, jsonify, current_app, request, Response, send_file

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


@bp.route("/logs")
def logs():
    """Return the last N lines of the casiquiare log file."""
    num_lines = request.args.get("n", 20)
    try:
        n = int(num_lines)
    except ValueError:
        n = 20

    log_file = Path(current_app.config.get("LOG_FILE", Path("logs/casiquiare.log")))
    if not log_file.exists():
        return jsonify({"error": "Log file not found"}), 404

    try:
        with log_file.open("r", encoding="utf-8") as fh:
            tail_lines = list(collections.deque(fh, maxlen=n))
    except Exception:
        return jsonify({"error": "Failed to read log file"}), 500

    return Response("".join(tail_lines), mimetype="text/plain")


@bp.route("/snippets/<path:filename>")
def snippet_file(filename: str):
    """Serve PNG snippet files from the <area>_snippets directories."""
    base_dir = Path(current_app.config.get("DETECTIONS_PATH", Path.cwd()))
    file_path = base_dir / filename
    if not file_path.is_file():
        return jsonify({"error": "Snippet not found"}), 404
    try:
        return send_file(file_path, mimetype="image/png")
    except Exception:
        return jsonify({"error": "Failed to read snippet"}), 500


__all__ = ["bp"]
