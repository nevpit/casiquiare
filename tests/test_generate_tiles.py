import numpy as np
import pytest

from processing import tiles


def _create_raster(path):
    import rasterio

    data = np.zeros((10, 10), dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="uint8",
    ) as dst:
        dst.write(data, 1)


def test_generate_tile_pyramid_runtime(tmp_path):
    tif = tmp_path / "img.tif"
    _create_raster(tif)

    exe = tiles._gdal2tiles_path()
    if exe is None:
        with pytest.raises(RuntimeError):
            tiles.generate_tile_pyramid(str(tif), "dtm", tiles_root=tmp_path)
    else:
        tiles.generate_tile_pyramid(str(tif), "dtm", zoom="0", tiles_root=tmp_path)
        assert (tmp_path / "dtm" / "0" / "0" / "0.png").exists()
