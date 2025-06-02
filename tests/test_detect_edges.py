import numpy as np
import pytest

from detection import lidar


def test_detect_edges_runtime():
    arr = np.zeros((10, 10), dtype=np.uint8)
    if lidar.cv2 is None or lidar.np is None:
        with pytest.raises(RuntimeError):
            lidar.detect_edges(arr)
    else:
        result = lidar.detect_edges(arr)
        assert result.shape == arr.shape
        assert result.dtype == np.uint8
