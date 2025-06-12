import pytest

import eyes.tools as eyes_tools


def test_lidar_feature_detection_runtime(tmp_path):
    fake_path = tmp_path / "dummy.laz"

    missing = (
        eyes_tools.pdal is None
        or eyes_tools.rasterio is None
        or eyes_tools.np is None
        or eyes_tools.cv2 is None
        or eyes_tools.to_uint8 is None
        or eyes_tools.generate_hillshade is None
        or eyes_tools.generate_lrm is None
    )

    if missing:
        with pytest.raises(RuntimeError):
            eyes_tools.lidar_feature_detection(str(fake_path))
    else:
        pytest.skip("PDAL present but no sample data available")

