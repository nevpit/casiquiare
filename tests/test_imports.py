import importlib
import importlib.util
import sys
from pathlib import Path


def test_imports():
    """Modules should import without errors."""

    raster_path = Path(__file__).resolve().parents[1] / "io" / "raster.py"
    raster_spec = importlib.util.spec_from_file_location("io.raster", raster_path)
    raster_module = importlib.util.module_from_spec(raster_spec)
    assert raster_spec and raster_spec.loader
    raster_spec.loader.exec_module(raster_module)  # type: ignore[arg-type]
    sys.modules["io.raster"] = raster_module

    importlib.import_module("agents.eyes_tools")

    # Import io_helpers.lidar explicitly from file to ensure local module is used
    lidar_path = Path(__file__).resolve().parents[1] / "io_helpers" / "lidar.py"
    spec = importlib.util.spec_from_file_location("io_helpers.lidar", lidar_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    sys.modules["io_helpers.lidar"] = module

    importlib.import_module("io_helpers.lidar")
    importlib.import_module("processing.lidar")
    importlib.import_module("processing.image")
    importlib.import_module("detection.lidar")
    importlib.import_module("detection.sat")
    importlib.import_module("io.raster")
