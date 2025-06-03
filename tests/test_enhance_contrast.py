import numpy as np
import pytest

from processing import sat


def test_enhance_contrast_runtime():
    arr = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
    if sat.cv2 is None or sat.np is None:
        with pytest.raises(RuntimeError):
            sat.enhance_contrast(arr)
    else:
        result = sat.enhance_contrast(arr)
        assert result.shape == arr.shape
        assert result.dtype == np.uint8
