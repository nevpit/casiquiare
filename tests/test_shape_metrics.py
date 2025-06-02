import numpy as np
import pytest

import detection.lidar as d_lidar


def test_shape_metrics_runtime():
    cnt = np.array([[[0, 0]], [[4, 0]], [[4, 2]], [[0, 2]]], dtype=np.int32)
    if d_lidar.cv2 is None or d_lidar.np is None:
        with pytest.raises(RuntimeError):
            d_lidar.shape_metrics(cnt)
    else:
        metrics = d_lidar.shape_metrics(cnt)
        assert set(metrics) == {"area", "perimeter", "aspect_ratio", "circularity"}

