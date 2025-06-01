import numpy as np
import pytest

from processing import image


def test_to_uint8_runtime():
    arr = np.linspace(0, 100, 9).reshape(3, 3).astype(np.float32)
    if image.np is None:
        with pytest.raises(RuntimeError):
            image.to_uint8(arr)
    else:
        result = image.to_uint8(arr)
        assert result.dtype == np.uint8
        assert result.min() >= 0 and result.max() <= 255

