import pytest

import agents.eyes.eyes_tools as eyes_tools


def test_lidar_tile_dtm_runtime(tmp_path):
    fake_path = tmp_path / "dummy.laz"

    missing = (
        eyes_tools.pdal is None
        or eyes_tools.rasterio is None
        or eyes_tools.np is None
        or eyes_tools.build_dtm is None
    )

    if missing:
        with pytest.raises(RuntimeError):
            eyes_tools.lidar_tile_dtm(str(fake_path))
    else:
        pytest.skip("PDAL present but no sample data available")
