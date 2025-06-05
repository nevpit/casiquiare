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
