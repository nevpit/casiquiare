from interface import create_app


def test_env_config_paths(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    tile_dir = tmp_path / "tiles"
    output_dir.mkdir()
    tile_dir.mkdir()

    monkeypatch.setenv("EYES_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("EYES_TILE_DIR", str(tile_dir))

    app = create_app()

    assert app.config["DETECTIONS_PATH"] == output_dir
    assert app.config["TILES_ROOT"] == tile_dir
