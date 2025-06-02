import numpy as np
import pytest

from detection import lidar as detection_lidar


def test_detect_edges_runtime():
    arr = np.zeros((10, 10), dtype=np.uint8)
    if detection_lidar.cv2 is None or detection_lidar.np is None:
        with pytest.raises(RuntimeError):
            detection_lidar.detect_edges(arr)
    else:
        result = detection_lidar.detect_edges(
            arr,
            blur_kernel_size=(3, 3),
            canny_threshold1=30,
            canny_threshold2=90,
            dilation_iterations=1,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == arr.shape
