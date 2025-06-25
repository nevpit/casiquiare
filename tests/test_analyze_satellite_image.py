import numpy as np
import pytest

import agents.eyes.eyes_tools as eyes_tools


def test_analyze_satellite_image_runtime(tmp_path):
    img_path = tmp_path / "img.tif"
    if (
        eyes_tools.rasterio is None
        or eyes_tools.np is None
        or eyes_tools.load_raster is None
        or eyes_tools.compute_ndvi is None
        or eyes_tools.compute_ndwi is None
    ):
        with pytest.raises(RuntimeError):
            eyes_tools.analyze_satellite_image(str(img_path))
    else:
        import rasterio

        data = np.zeros((4, 5, 5), dtype=np.uint8)
        with rasterio.open(
            img_path,
            "w",
            driver="GTiff",
            height=5,
            width=5,
            count=4,
            dtype="uint8",
        ) as dst:
            for i in range(4):
                dst.write(data[i], i + 1)

        res = eyes_tools.analyze_satellite_image(
            str(img_path), ndvi_bands=(3, 4), ndwi_bands=(2, 4)
        )
        assert "meta" in res
        assert "ndvi" in res
        assert "ndwi" in res
        assert res["ndvi"].shape == (5, 5)
        assert res["ndvi"].dtype == np.float32
