import importlib.util
import sys
from pathlib import Path
import pytest

data_paths_path = Path(__file__).resolve().parents[1] / "io" / "data_paths.py"
spec = importlib.util.spec_from_file_location("io.data_paths", data_paths_path)
data_paths = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(data_paths)  # type: ignore[arg-type]
sys.modules["io.data_paths"] = data_paths


def test_load_mapping(tmp_path):
    cfg = tmp_path / "mapping.yaml"
    cfg.write_text("tile_a: /tmp/a.laz\n[1,2,3,4]: /tmp/b.laz")

    if data_paths.yaml is None:
        with pytest.raises(RuntimeError):
            data_paths.load_mapping(str(cfg))
    else:
        mapping = data_paths.load_mapping(str(cfg))
        assert mapping == {
            "tile_a": "/tmp/a.laz",
            "[1,2,3,4]": "/tmp/b.laz",
        }


def test_get_data_path(monkeypatch, tmp_path):
    cfg = tmp_path / "map.yaml"
    cfg.write_text("tile_a: /tmp/a.laz\n[1,2,3,4]: /tmp/b.laz")
    monkeypatch.setattr(data_paths, "_CONFIG_FILE", cfg)

    if data_paths.yaml is None:
        with pytest.raises(RuntimeError):
            data_paths.get_data_path(tile_id="tile_a")
    else:
        path1 = data_paths.get_data_path(tile_id="tile_a")
        assert path1 == "/tmp/a.laz"

        path2 = data_paths.get_data_path(bbox=(1, 2, 3, 4))
        assert path2 == "/tmp/b.laz"
