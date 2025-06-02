import numpy as np
import pytest

from detection import lidar


def test_find_contours_runtime():
    edges = np.zeros((10, 10), dtype=np.uint8)
    if lidar.cv2 is None or lidar.np is None:
        with pytest.raises(RuntimeError):
            lidar.find_contours(edges)
    else:
        contours = lidar.find_contours(edges)
        assert isinstance(contours, list)

