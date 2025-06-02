import numpy as np
import pytest

import agents.eyes_tools as eyes_tools


def test_detect_shapes_runtime(tmp_path):
    arr = np.zeros((10, 10), dtype=np.uint8)
    if eyes_tools.cv2 is None or eyes_tools.np is None:
        with pytest.raises(RuntimeError):
            eyes_tools.detect_shapes(arr)
    else:
        result = eyes_tools.detect_shapes(arr)
        assert isinstance(result, list)


def test_detect_shapes_scoring():
    if eyes_tools.cv2 is None or eyes_tools.np is None:
        pytest.skip("OpenCV not installed")

    img = np.zeros((60, 60), dtype=np.uint8)
    eyes_tools.cv2.rectangle(img, (10, 10), (50, 50), 255, -1)
    res = eyes_tools.detect_shapes(img, size_range=(0, 60))

    if res:
        feature = res[0]
        assert "score" in feature
        assert 0.0 <= feature["score"] <= 1.0
    else:
        pytest.fail("No shape detected")
