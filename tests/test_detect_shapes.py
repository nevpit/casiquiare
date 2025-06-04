import numpy as np
import pytest

from detection import detect_shapes
import detection.shapes as d_shapes


def test_detect_shapes_runtime(tmp_path):
    arr = np.zeros((10, 10), dtype=np.uint8)
    if d_shapes.cv2 is None or d_shapes.np is None:
        with pytest.raises(RuntimeError):
            detect_shapes(arr)
    else:
        result = detect_shapes(arr)
        assert isinstance(result, list)


def test_detect_shapes_scoring():
    if d_shapes.cv2 is None or d_shapes.np is None:
        pytest.skip("OpenCV not installed")

    img = np.zeros((60, 60), dtype=np.uint8)
    d_shapes.cv2.rectangle(img, (10, 10), (50, 50), 255, -1)
    res = detect_shapes(img, size_range=(0, 60))

    if res:
        feature = res[0]
        assert "score" in feature
        assert 0.0 <= feature["score"] <= 1.0
    else:
        pytest.fail("No shape detected")


def test_detect_shapes_threshold():
    if d_shapes.cv2 is None or d_shapes.np is None:
        pytest.skip("OpenCV not installed")

    img = np.zeros((60, 60), dtype=np.uint8)
    d_shapes.cv2.rectangle(img, (10, 10), (50, 50), 255, -1)

    # detection without threshold should return the rectangle
    res = detect_shapes(img, size_range=(0, 60))
    assert res

    # using a high threshold should filter out the detection
    res_high = detect_shapes(img, size_range=(0, 60), score_threshold=0.9)
    assert res_high == []
