from unittest import mock
import numpy as np

from eyes_agent import cli
from detection.feature import Feature


def test_scan_cli_basic(capsys):
    with mock.patch("eyes_agent.cli.scan_area") as scan:
        scan.return_value = []
        cli.main(["scan", "area.tif"])
        scan.assert_called_with("area.tif")
        out = capsys.readouterr().out
        assert "Detected 0 features" in out


def test_scan_cli_save_options(tmp_path):
    features = [
        Feature(1, "mound", {"type": "bbox", "bbox": (0, 0, 10, 10)}, (10.0, 5.0), 0.9, "test")
    ]
    with (
        mock.patch("eyes_agent.cli.scan_area", return_value=features) as scan,
        mock.patch("eyes_agent.cli.save_geojson") as save_geo,
        mock.patch(
            "io.raster.load_raster",
            return_value=(np.zeros((1, 1)), None, None),
        ),
        mock.patch("agents.eyes_tools.save_snippets") as save_snip,
    ):
        cli.main(["scan", "area.tif", "--save-geojson", "--save-snippets"])
        scan.assert_called_with("area.tif")
        assert save_geo.called
        assert save_snip.called
