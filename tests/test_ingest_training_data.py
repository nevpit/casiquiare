import pytest

from agents.brain import brain_tools


def test_ingest_training_data_runtime(tmp_path):
    csv = tmp_path / "train.csv"
    csv.write_text(
        "red,nir,elevation,label\n" "10,20,5,1\n" "15,25,10,0\n",
        encoding="utf-8",
    )

    if (
        brain_tools.pd is None
        or brain_tools.np is None
        or brain_tools.compute_ndvi is None
    ):
        with pytest.raises(RuntimeError):
            brain_tools.ingest_training_data(str(csv))
    else:
        df = brain_tools.ingest_training_data(str(csv))
        assert len(df) == 2
        assert "ndvi" in df.columns
        assert "slope" in df.columns
