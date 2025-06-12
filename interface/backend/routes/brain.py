from __future__ import annotations

import collections
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, Response

bp = Blueprint("brain", __name__, url_prefix="/brain")


@bp.route("/logs")
def logs():
    """Return the last N lines of the Brain log file."""
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


__all__ = ["bp"]
