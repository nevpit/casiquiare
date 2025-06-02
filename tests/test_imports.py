import importlib


def test_imports():
    """Modules should import without errors."""
    importlib.import_module("agents.eyes_tools")

    importlib.import_module("io_helpers.lidar")
    importlib.import_module("processing.lidar")
    importlib.import_module("processing.image")
