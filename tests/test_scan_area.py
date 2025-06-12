import numpy as np
import pytest

import agents.eyes.tools as eyes_tools


def test_scan_area_runtime_dict():
    area = {"image": np.zeros((10, 10), dtype=np.uint8)}
    if eyes_tools.detect_shapes is None or eyes_tools.np is None:
        with pytest.raises(RuntimeError):
            eyes_tools.scan_area(area)
    else:
        result = eyes_tools.scan_area(area)
        assert isinstance(result, list)


def test_scan_area_runtime_path(tmp_path):
    fake = tmp_path / "area.tif"
    if (
        eyes_tools.load_raster is None
        or eyes_tools.np is None
        or eyes_tools.detect_shapes is None
    ):
        with pytest.raises(RuntimeError):
            eyes_tools.scan_area(str(fake))
    else:
        with pytest.raises(Exception):
            eyes_tools.scan_area(str(fake))
