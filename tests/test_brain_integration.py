import pytest

from agents.brain import Brain
from agents.brain import brain_tools
from world_state import world_state, reset


@pytest.mark.usefixtures("tmp_path")
def test_brain_synthesizer_integration(tmp_path):
    reset()
    brain = Brain()

    feats = [
        {"id": "a", "width": 60.0, "height": 50.0, "shape": "mound"},
        {"id": "b", "width": 80.0, "height": 40.0, "shape": "rectangle"},
    ]

    if brain_tools.np is None:
        with pytest.raises(RuntimeError):
            brain.validate_features(feats)
        return

    validations = brain.validate_features(feats)
    assert world_state["validation"]
    assert validations[0]["feature_id"] == "a"

    missing = (
        brain_tools.RandomForestClassifier is None
        or brain_tools.pd is None
        or brain_tools.np is None
    )
    if missing:
        with pytest.raises(RuntimeError):
            brain.train_model(None)
    else:
        df = brain_tools.pd.DataFrame(
            {
                "f1": [0.1, 0.2, 0.3, 0.4],
                "f2": [1.0, 0.0, 1.0, 0.0],
                "label": [1, 0, 1, 0],
            }
        )
        model = brain.train_model(
            df,
            n_estimators=5,
            random_state=0,
            model_path=str(tmp_path / "model.joblib"),
        )
        assert world_state["latest_model"]["model_type"] == "RandomForestClassifier"
        brain.predict_sites(model["model"], [[0.5, 0.5], [0.2, 0.1]], top_n=1)
        assert "scores_map" in world_state["latest_prediction"]

    messages = world_state.get("messages", [])
    if messages:
        msg = messages[0]
        assert msg["agent"] == "brain"
        assert "type" in msg and "content" in msg

