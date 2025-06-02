import numpy as np
import pytest

from detection import lidar


def test_filter_contours_runtime():
    cnt_large = np.array([[[0, 0]], [[0, 10]], [[10, 10]], [[10, 0]]], dtype=np.int32)
    cnt_small = np.array([[[0, 0]], [[0, 2]], [[2, 2]], [[2, 0]]], dtype=np.int32)
    if lidar.cv2 is None or lidar.np is None:
        with pytest.raises(RuntimeError):
            lidar.filter_contours([cnt_large])
    else:
        filtered = lidar.filter_contours([cnt_large, cnt_small], min_size=5, max_size=15)
        assert isinstance(filtered, list)
        assert len(filtered) == 1

