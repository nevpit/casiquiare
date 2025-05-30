import numpy as np
import pytest

from processing import lidar


def test_write_geotiff(tmp_path):
    arr = np.zeros((10, 10), dtype=np.float32)
    profile = {
        "driver": "GTiff",
        "height": 10,
        "width": 10,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": (1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    }

    output_path = tmp_path / "test.tif"

    if lidar.rasterio is None or lidar.np is None:
        with pytest.raises(RuntimeError):
            lidar.write_geotiff(str(output_path), arr, profile)
    else:
        lidar.write_geotiff(str(output_path), arr, profile)
        assert output_path.exists()
