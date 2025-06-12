import importlib
import types
from unittest import mock

import numpy as np

from agents.eyes.cli import main as cli_main
from detection.feature import Feature


def test_scan_cli_basic(caplog):
    stub_raster_mod = types.ModuleType("io_utils.raster")
    stub_raster_mod.load_raster = lambda _: (np.zeros((1, 1)), None, None)
    real_import = importlib.import_module

    with (
        mock.patch("agents.eyes.cli.scan_area", return_value=[]) as scan,
        mock.patch("agents.eyes.cli.importlib.import_module") as import_mod,
    ):
        def fake_import(name, *args, **kwargs):
            if name == "io_utils.raster":
                return stub_raster_mod
            return real_import(name, *args, **kwargs)

        import_mod.side_effect = fake_import

        with caplog.at_level("INFO", logger="agents.eyes.cli"):
            cli_main(["scan", "area.tif"])

        scan.assert_called_once_with(np.zeros((1, 1)), None)
        assert "Detected 0 features" in caplog.text


def test_scan_cli_save_options(tmp_path):
    features = [
        Feature(1, "mound", {"type": "bbox", "bbox": (0, 0, 10, 10)}, (10.0, 5.0), 0.9, "test")
    ]
    stub_raster_mod = types.ModuleType("io_utils.raster")
    stub_raster_mod.load_raster = lambda _: (np.zeros((1, 1)), None, None)
    real_import = importlib.import_module

    with (
        mock.patch("agents.eyes.cli.scan_area", return_value=features) as scan,
        mock.patch("agents.eyes.cli.save_geojson") as save_geo,
        mock.patch("agents.eyes.cli.save_snippets", return_value=["area_snippets/snippet_0.png"]) as save_snip,
        mock.patch("agents.eyes.cli.importlib.import_module") as import_mod,
    ):
        def fake_import(name, *args, **kwargs):
            if name == "io_utils.raster":
                return stub_raster_mod
            return real_import(name, *args, **kwargs)

        import_mod.side_effect = fake_import

        cli_main(["scan", "area.tif", "--save-geojson", "--save-snippets"])

        scan.assert_called_once_with(np.zeros((1, 1)), None)
        save_geo.assert_called_once()
        save_snip.assert_called_once()
        assert features[0].snippet_path == "area_snippets/snippet_0.png"
