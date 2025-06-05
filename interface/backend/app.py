"""Flask application factory for the casiquiare backend."""
from __future__ import annotations

from flask import Flask

from .eyes import bp as eyes_bp


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.register_blueprint(eyes_bp)
    return app


__all__ = ["create_app"]
