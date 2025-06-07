import numpy as np
import pytest

from processing import sat


def test_compute_ndwi_runtime():
    arr = np.ones((5, 5), dtype=np.uint8)
    if sat.np is None:
        with pytest.raises(RuntimeError):
            sat.compute_ndwi(arr, arr)
    else:
        res = sat.compute_ndwi(arr, arr)
        assert res.shape == arr.shape
        assert res.dtype == np.float32
