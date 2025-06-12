import pytest

from agents.brain import brain_tools


def test_train_model_runtime(tmp_path):
    if brain_tools.pd is not None:
        df = brain_tools.pd.DataFrame(
            {
                "f1": [0.1, 0.2, 0.3, 0.4],
                "f2": [1.0, 0.0, 1.0, 0.0],
                "label": [1, 0, 1, 0],
            }
        )
    else:
        df = None

    if (
        brain_tools.RandomForestClassifier is None
        or brain_tools.np is None
        or brain_tools.pd is None
    ):
        with pytest.raises(RuntimeError):
            brain_tools.train_model(df)
    else:
        info = brain_tools.train_model(df, n_estimators=5, random_state=0, model_path=str(tmp_path / "model.joblib"))
        assert isinstance(info, dict)
        assert "model_type" in info
        assert "metrics" in info
        assert "feature_importances" in info
