from unittest import mock

from detection.feature import Feature
from agents.eyes.eyes_tools import scan_tiles_concurrent


def test_scan_tiles_concurrent_order():
    features1 = [Feature(1, "a", {}, (0.0, 0.0), 0.9, "test")]
    features2 = [Feature(2, "b", {}, (0.0, 0.0), 0.8, "test")]
    with mock.patch("agents.eyes.eyes_tools.scan_area") as scan:
        scan.side_effect = [features1, features2]
        results = scan_tiles_concurrent(["tile1", "tile2"], max_workers=2)
        assert scan.call_count == 2
        assert results[0] == features1
        assert results[1] == features2
