import pytest
from agents.brain import brain_tools


def test_validate_features_runtime():
    feats = [
        {
            "id": 1,
            "geometry": {"bbox": (0, 0, 50, 100)},
            "width": 50.0,
            "height": 100.0,
            "shape": "mound",
        },
        {
            "id": 2,
            "geometry": {"bbox": (0, 0, 300, 10)},
            "width": 300.0,
            "height": 10.0,
            "shape": "linear",
        },
    ]

    if brain_tools.np is None:
        with pytest.raises(RuntimeError):
            brain_tools.validate_features(feats)
    else:
        results = brain_tools.validate_features(feats)
        assert isinstance(results, list)
        assert len(results) == 2
        assert "size_match" in results[0]

