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
    sys.modules["io.lidar"] = module

    # Import io.data_paths explicitly from file to avoid stdlib conflict
    paths_path = Path(__file__).resolve().parents[1] / "io" / "data_paths.py"
    spec_paths = importlib.util.spec_from_file_location("io.data_paths", paths_path)
    paths_module = importlib.util.module_from_spec(spec_paths)
    assert spec_paths and spec_paths.loader
    spec_paths.loader.exec_module(paths_module)  # type: ignore[arg-type]
    sys.modules["io.data_paths"] = paths_module

    importlib.import_module("io.lidar")
    importlib.import_module("io.data_paths")
    importlib.import_module("processing.lidar")
    importlib.import_module("processing.image")
    importlib.import_module("detection.lidar")
    importlib.import_module("detection.sat")
    importlib.import_module("io.raster")
