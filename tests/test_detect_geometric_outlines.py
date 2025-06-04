import numpy as np
import pytest

import detection.sat as d_sat


def test_detect_geometric_outlines_runtime():
    edges = np.zeros((10, 10), dtype=np.uint8)
    if d_sat.cv2 is None or d_sat.np is None:
        with pytest.raises(RuntimeError):
            d_sat.detect_geometric_outlines(edges)
    else:
        result = d_sat.detect_geometric_outlines(edges)
        assert isinstance(result, dict)
        assert set(result) == {"contours", "lines"}


def test_detect_geometric_outlines_detection():
    if d_sat.cv2 is None or d_sat.np is None:
        pytest.skip("OpenCV not installed")

    img = np.zeros((60, 60), dtype=np.uint8)
    d_sat.cv2.rectangle(img, (5, 5), (55, 55), 255, 1)

    res = d_sat.detect_geometric_outlines(img)
    assert res["contours"] or res["lines"]


def test_detect_geometric_outlines_size_filter():
    if d_sat.cv2 is None or d_sat.np is None:
        pytest.skip("OpenCV not installed")

    img = np.zeros((400, 400), dtype=np.uint8)
    d_sat.cv2.rectangle(img, (10, 10), (390, 390), 255, 1)

    res = d_sat.detect_geometric_outlines(img)
    assert len(res["contours"]) == 0

