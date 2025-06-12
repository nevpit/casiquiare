import pytest

from agents.brain import brain_tools


def test_update_model_runtime():
    if brain_tools.np is None:
        with pytest.raises(RuntimeError):
            brain_tools.update_model(object(), [], [])
    else:
        from sklearn.linear_model import SGDClassifier

        clf = SGDClassifier(loss="log_loss")
        clf.partial_fit(brain_tools.np.array([[0.0, 0.0]]), [0], classes=[0, 1])

        updated = brain_tools.update_model(
            clf, [[0.1, 0.2], [0.2, 0.3]], [0, 1]
        )
        assert updated is clf


def test_update_model_requires_partial_fit():
    if brain_tools.np is None:
        pytest.skip("NumPy is required")
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=5)
    with pytest.raises(ValueError):
        brain_tools.update_model(clf, [[0.1, 0.2]], [1])
