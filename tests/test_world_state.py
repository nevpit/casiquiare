import pytest

from agents.brain import Brain
from agents.brain import brain_tools
from world_state import world_state, reset


def test_brain_updates_world_state(tmp_path):
    reset()
    agent = Brain()
    if (
        brain_tools.RandomForestClassifier is None
        or brain_tools.np is None
        or brain_tools.pd is None
    ):
        with pytest.raises(RuntimeError):
            agent.train_model(None)
    else:
        df = brain_tools.pd.DataFrame(
            {
                "f1": [0.1, 0.2, 0.3, 0.4],
                "f2": [1.0, 0.0, 1.0, 0.0],
                "label": [1, 0, 1, 0],
            }
        )
        res = agent.train_model(
            df,
            n_estimators=5,
            random_state=0,
            model_path=str(tmp_path / "model.joblib"),
        )
        assert res["model_type"] == "RandomForestClassifier"
        assert world_state["latest_model"]["model_type"] == "RandomForestClassifier"
