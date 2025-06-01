import importlib
import importlib.util
import sys
from pathlib import Path


def test_imports():
    """Modules should import without errors."""
    importlib.import_module("agents.eyes_tools")

    # Import io.lidar explicitly from file to avoid conflict with stdlib "io"
    lidar_path = Path(__file__).resolve().parents[1] / "io" / "lidar.py"
    spec = importlib.util.spec_from_file_location("io.lidar", lidar_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    sys.modules["io.lidar"] = module

    importlib.import_module("io.lidar")
    importlib.import_module("processing.lidar")
    importlib.import_module("processing.image")
