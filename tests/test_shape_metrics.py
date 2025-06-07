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
        assert set(
            ["area", "perimeter", "aspect_ratio", "circularity", "num_vertices", "shape"]
        ).issubset(metrics)


def test_shape_classification():
    if d_lidar.cv2 is None or d_lidar.np is None:
        pytest.skip("OpenCV not installed")

    tri = np.array([[[0, 0]], [[5, 0]], [[2, 4]]], dtype=np.int32)
    metrics_tri = d_lidar.shape_metrics(tri)
    assert metrics_tri["shape"] == "triangle"

    line = np.array([[[0, 0]], [[50, 0]], [[50, 2]], [[0, 2]]], dtype=np.int32)
    metrics_line = d_lidar.shape_metrics(line)
    assert metrics_line["shape"] == "linear"

