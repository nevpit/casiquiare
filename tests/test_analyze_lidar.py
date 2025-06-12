import numpy as np
import pytest

import eyes.tools as eyes_tools


def test_analyze_lidar_runtime(tmp_path):
    fake_path = tmp_path / "dummy.laz"
    if eyes_tools.pdal is None:
        with pytest.raises(RuntimeError):
            next(iter(eyes_tools.analyze_lidar(str(fake_path))))
    else:
        pytest.skip("PDAL present but no sample data available")

