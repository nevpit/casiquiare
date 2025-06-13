import pytest
from agents.brain import brain_tools


class DummyModel:
    def predict_proba(self, X):
        import numpy as np
        return np.tile([0.3, 0.7], (len(X), 1))

    feature_importances_ = [0.5, 0.5]


def test_predict_sites_vector_runtime():
    data = [[1.0, 2.0], [2.0, 3.0]]
    if brain_tools.np is None:
        with pytest.raises(RuntimeError):
            brain_tools.predict_sites(DummyModel(), data)
    else:
        res = brain_tools.predict_sites(DummyModel(), data)
        assert "scores_map" in res
        assert "summary" in res
        assert len(res["scores_map"]) == 2


def test_predict_sites_region_runtime():
    bbox = [0.0, 0.0, 0.02, 0.02]
    if brain_tools.np is None:
        with pytest.raises(RuntimeError):
            brain_tools.predict_sites(DummyModel(), bbox, grid_size=0.01)
    else:
        res = brain_tools.predict_sites(DummyModel(), bbox, grid_size=0.01)
        assert "scores_map" in res
        assert res["scores_map"].shape == (3, 3)
        assert "summary" in res


def test_predict_sites_rejects_dict_list():
    """Ensure lists of dictionaries are not treated as feature vectors."""
    data = [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 3.0}]
    if brain_tools.np is None:
        pytest.skip("NumPy is required")
    with pytest.raises(TypeError):
        brain_tools.predict_sites(DummyModel(), data)
