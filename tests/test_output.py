import json
from detection import Feature
from output import serialize_features, save_geojson

def make_feature(idx=1):
    return Feature(
        id=idx,
        feature_type="mound",
        geometry={"type": "Point", "coordinates": [idx, idx + 1]},
        dimensions=(10.0, 5.0),
        confidence=0.9,
        source="test",
    )

def test_serialize_features(tmp_path):
    feats = [make_feature()]
    out_file = tmp_path / "feats.json"
    serialize_features(feats, str(out_file))
    assert out_file.exists()
    data = json.load(open(out_file))
    assert isinstance(data, list)
    assert data[0]["id"] == 1

def test_save_geojson(tmp_path):
    feats = [make_feature()]
    out_file = tmp_path / "feats.geojson"
    save_geojson(feats, str(out_file))
    assert out_file.exists()
    coll = json.load(open(out_file))
    assert coll["type"] == "FeatureCollection"
    assert coll["features"][0]["geometry"]["type"] == "Point"
