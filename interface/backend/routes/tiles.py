from __future__ import annotations

from pathlib import Path

from flask import Blueprint, current_app, jsonify, send_from_directory

bp = Blueprint("tiles", __name__, url_prefix="/tiles")


@bp.route("/<layer>/<int:z>/<int:x>/<int:y>.png")
def tile(layer: str, z: int, x: int, y: int):
    """Serve pre-generated map tiles."""
    base_dir = Path(
        current_app.config.get(
            "TILES_ROOT",
            Path(__file__).resolve().parents[2] / "static" / "tiles",
        )
    )
    tile_dir = base_dir / layer / str(z) / str(x)
    filename = f"{y}.png"
    if not (tile_dir / filename).is_file():
        return jsonify({"error": "Tile not found"}), 404
    try:
        return send_from_directory(tile_dir, filename)
    except Exception:
        return jsonify({"error": "Failed to read tile"}), 500


__all__ = ["bp"]
