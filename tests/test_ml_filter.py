import pytest
import numpy as np

from detection import ml_filter


def _make_det(score=0.5):
    return {
        "width": 10.0,
        "height": 10.0,
        "num_vertices": 4,
        "score": score,
    }


def test_ml_filter_runtime():
    dets = [_make_det(0.9), _make_det(0.1)]
    labels = [1, 0]
    if ml_filter.RandomForestClassifier is None or ml_filter.np is None:
        with pytest.raises(RuntimeError):
            ml_filter.train_post_filter(dets, labels)
    else:
        clf = ml_filter.train_post_filter(dets, labels, n_estimators=5)
        filtered = ml_filter.apply_post_filter(clf, dets)
        assert isinstance(filtered, list)
        assert len(filtered) <= len(dets)


def test_apply_post_filter_invalid():
    dets = [_make_det(0.7)]
    if ml_filter.RandomForestClassifier is None or ml_filter.np is None:
        pytest.skip("scikit-learn not installed")
    with pytest.raises(ValueError):
        ml_filter.apply_post_filter(object(), dets)

