import importlib.util
import sys
from pathlib import Path
import pytest

module_path = Path(__file__).resolve().parents[1] / "io_utils" / "raster.py"
spec = importlib.util.spec_from_file_location("io_utils.raster", module_path)
raster = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(raster)  # type: ignore[arg-type]
sys.modules["io_utils.raster"] = raster


def test_load_raster_runtime(tmp_path):
    fake_path = tmp_path / "image.tif"

    if raster.rasterio is None or raster.np is None:
        with pytest.raises(RuntimeError):
            raster.load_raster(str(fake_path))
    else:
        with pytest.raises(Exception):
            raster.load_raster(str(fake_path))
