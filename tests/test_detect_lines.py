import numpy as np
import pytest

from detection import lidar


def test_detect_lines_runtime():
    edges = np.zeros((10, 10), dtype=np.uint8)
    if lidar.cv2 is None or lidar.np is None:
        with pytest.raises(RuntimeError):
            lidar.detect_lines(edges)
    else:
        lines = lidar.detect_lines(edges)
        assert isinstance(lines, list)
