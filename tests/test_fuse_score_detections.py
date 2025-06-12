import pytest

from agents.brain import brain_tools

class DummyModel:
    def predict_proba(self, X):
        import numpy as np
        return np.tile([0.2, 0.8], (len(X), 1))


def test_fuse_score_detections():
    lidar = [{"bbox": (0, 0, 10, 10), "width": 10.0, "height": 10.0}]
    sat = [{"bbox": (5, 0, 10, 10), "width": 10.0, "height": 10.0}]

    if (
        brain_tools.np is None
        or brain_tools.merge_detections is None
        or brain_tools._feature_vector is None
    ):
        with pytest.raises(RuntimeError):
            brain_tools.fuse_score_detections(DummyModel(), lidar, sat, threshold=10)
    else:
        res = brain_tools.fuse_score_detections(DummyModel(), lidar, sat, threshold=10)
        assert len(res["merged"]) == 1
        assert len(res["scores"]) == 1
        assert res["merged"][0]["sources"] == ["lidar", "satellite"]
