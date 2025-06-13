import importlib.util
from pathlib import Path
import pytest

from agents.brain import brain_tools


def test_load_model_runtime(tmp_path):
    fake = tmp_path / "missing.joblib"
    has_joblib = importlib.util.find_spec("joblib") is not None
    if not has_joblib:
        with pytest.raises(RuntimeError):
            brain_tools.load_model(str(fake))
    else:
        with pytest.raises(Exception):
            brain_tools.load_model(str(fake))


def test_load_model_after_training(tmp_path):
    missing = (
        brain_tools.RandomForestClassifier is None
        or brain_tools.pd is None
        or brain_tools.np is None
    )
    if missing:
        pytest.skip("training dependencies not installed")
    df = brain_tools.pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.3, 0.4],
            "f2": [1.0, 0.0, 1.0, 0.0],
            "label": [1, 0, 1, 0],
        }
    )
    info = brain_tools.train_model(
        df,
        n_estimators=5,
        random_state=0,
        model_path=str(tmp_path / "model.joblib"),
    )
    model = brain_tools.load_model(info["model_path"])
    assert Path(info["model_card"]).exists()
    assert hasattr(model, "predict")


