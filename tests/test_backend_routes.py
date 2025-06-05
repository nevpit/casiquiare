import json
import os
from interface import create_app


def test_detections_route_latest(tmp_path):
    older = tmp_path / "a_features.geojson"
    older.write_text('[{"id": 1}]', encoding="utf-8")
    os.utime(older, (0, 0))
    newer = tmp_path / "b_features.geojson"
    newer.write_text('[{"id": 2}]', encoding="utf-8")

    app = create_app()
    app.config["DETECTIONS_PATH"] = tmp_path

    with app.test_client() as client:
        resp = client.get("/eyes/detections")
        assert resp.status_code == 200
        assert resp.get_json() == [{"id": 2}]


def test_detections_route_missing(tmp_path):
    app = create_app()
    app.config["DETECTIONS_PATH"] = tmp_path
    with app.test_client() as client:
        resp = client.get("/eyes/detections")
        assert resp.status_code == 404


def test_logs_route(tmp_path):
    log_file = tmp_path / "casiquiare.log"
    log_file.write_text("line1\nline2\nline3\n", encoding="utf-8")

    app = create_app()
    app.config["LOG_FILE"] = log_file
    with app.test_client() as client:
        resp = client.get("/eyes/logs?n=2")
        assert resp.status_code == 200
        assert resp.data.decode() == "line2\nline3\n"


def test_logs_route_missing(tmp_path):
    app = create_app()
    app.config["LOG_FILE"] = tmp_path / "missing.log"
    with app.test_client() as client:
        resp = client.get("/eyes/logs")
        assert resp.status_code == 404


def test_snippet_route(tmp_path):
    snippet_dir = tmp_path / "area_snippets"
    snippet_dir.mkdir()
    snippet_file = snippet_dir / "snippet_0.png"
    snippet_file.write_bytes(b"data")

    app = create_app()
    app.config["DETECTIONS_PATH"] = tmp_path
    with app.test_client() as client:
        resp = client.get("/eyes/snippets/area_snippets/snippet_0.png")
        assert resp.status_code == 200
        assert resp.data == b"data"


def test_snippet_route_missing(tmp_path):
    app = create_app()
    app.config["DETECTIONS_PATH"] = tmp_path
    with app.test_client() as client:
        resp = client.get("/eyes/snippets/nope/snippet.png")
        assert resp.status_code == 404


def test_tile_route(tmp_path):
    tile_dir = tmp_path / "tiles" / "dtm" / "0" / "0"
    tile_dir.mkdir(parents=True)
    tile_file = tile_dir / "0.png"
    tile_file.write_bytes(b"data")

    app = create_app()
    app.config["TILES_ROOT"] = tmp_path / "tiles"
    with app.test_client() as client:
        resp = client.get("/tiles/dtm/0/0/0.png")
        assert resp.status_code == 200
        assert resp.data == b"data"


def test_tile_route_missing(tmp_path):
    app = create_app()
    app.config["TILES_ROOT"] = tmp_path / "tiles"
    with app.test_client() as client:
        resp = client.get("/tiles/dtm/0/0/0.png")
        assert resp.status_code == 404
