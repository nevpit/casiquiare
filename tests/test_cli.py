from unittest import mock
import numpy as np

from agents.eyes import cli
from detection.feature import Feature


def test_scan_cli_basic(caplog):
    with mock.patch("agents.eyes.cli.scan_area") as scan:
        scan.return_value = []
        with caplog.at_level("INFO", logger="agents.eyes.cli"):
            cli.main(["scan", "area.tif"])
        scan.assert_called_with("area.tif")
        assert "Detected 0 features" in caplog.text


def test_scan_cli_save_options(tmp_path):
    features = [
        Feature(1, "mound", {"type": "bbox", "bbox": (0, 0, 10, 10)}, (10.0, 5.0), 0.9, "test")
    ]
    with (
        mock.patch("agents.eyes.cli.scan_area", return_value=features) as scan,
        mock.patch("agents.eyes.cli.save_geojson") as save_geo,
        mock.patch(
            "io_utils.raster.load_raster",
            return_value=(np.zeros((1, 1)), None, None),
        ),
        mock.patch("agents.eyes.tools.save_snippets", return_value=["area_snippets/snippet_0.png"]) as save_snip,
    ):
        cli.main(["scan", "area.tif", "--save-geojson", "--save-snippets"])
        scan.assert_called_with("area.tif")
        assert save_geo.called
        assert save_snip.called
        assert features[0].snippet_path == "area_snippets/snippet_0.png"
