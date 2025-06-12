import pytest
from agents import brain_tools


def test_cluster_features_runtime():
    features = [
        {"id": 1, "bbox": (0, 0, 10, 10)},
        {"id": 2, "bbox": (12, 0, 10, 10)},
        {"id": 3, "bbox": (60, 60, 10, 10)},
    ]

    missing = brain_tools.DBSCAN is None or brain_tools.np is None
    if missing:
        with pytest.raises(RuntimeError):
            brain_tools.cluster_features(features)
    else:
        res = brain_tools.cluster_features(features, eps=20.0, min_samples=1)
        assert isinstance(res, dict)
        assert "labels" in res
        assert "summary" in res
        assert len(res["labels"]) == len(features)
        assert res["summary"]["num_clusters"] >= 1
