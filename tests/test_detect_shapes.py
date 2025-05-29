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
