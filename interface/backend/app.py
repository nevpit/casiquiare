"""Flask application factory for the casiquiare backend."""
from __future__ import annotations

from pathlib import Path

from flask import Flask, send_from_directory

from .utils.config import load_backend_config

from .routes.eyes import bp as eyes_bp
from .routes.tiles import bp as tiles_bp
from .routes.brain import bp as brain_bp


def create_app(static_path: Path | None = None) -> Flask:
    """Create and configure the Flask application."""
    if static_path is None:
        static_path = Path(__file__).resolve().parents[1] / "interface" / "frontend" / "build"

    app = Flask(__name__, static_folder=None)

    # Load environment-based configuration for route handlers
    app.config.update(load_backend_config())

    app.register_blueprint(eyes_bp)
    app.register_blueprint(tiles_bp)
    app.register_blueprint(brain_bp)

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def index(path: str):
        full_path = static_path / path
        if path and full_path.is_file():
            return send_from_directory(static_path, path)
        index_file = static_path / "index.html"
        if index_file.is_file():
            return send_from_directory(static_path, "index.html")
        return "", 404

    return app


__all__ = ["create_app"]
