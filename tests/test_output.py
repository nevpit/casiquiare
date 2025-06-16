import json
from detection import Feature
import pytest
from output import (
    serialize_features,
    save_geojson,
    save_polygon_geojson,
    features_to_shapely,
)
from processing import geo
import numpy as np

def make_feature(idx=1):
    return Feature(
        id=idx,
        feature_type="mound",
        geometry={"type": "Point", "coordinates": [idx, idx + 1]},
        dimensions=(10.0, 5.0),
        confidence=0.9,
        source="test",
    )


def make_bbox_feature():
    return Feature(
        id=1,
        feature_type="mound",
        geometry={"type": "bbox", "bbox": (0, 0, 10, 10)},
        dimensions=(10.0, 10.0),
        confidence=0.9,
        source="test",
    )


def make_contour_feature():
    cnt = np.array([[[0, 0]], [[1, 0]], [[1, 1]], [[0, 1]]], dtype=np.int32)
    return Feature(
        id=2,
        feature_type="outline",
        geometry={"type": "contour", "contour": cnt},
        dimensions=(1.0, 1.0),
        confidence=0.8,
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


def test_save_geojson_bbox(tmp_path):
    feats = [make_bbox_feature()]
    out_file = tmp_path / "bbox.geojson"
    save_geojson(feats, str(out_file))
    assert out_file.exists()
    coll = json.load(open(out_file))
    assert coll["features"][0]["geometry"]["type"] == "Polygon"


def test_save_geojson_dict(tmp_path):
    feat = {
        "id": 1,
        "feature_type": "mound",
        "geometry": {"type": "Point", "coordinates": [0, 1]},
        "dimensions": (10.0, 5.0),
        "confidence": 0.9,
        "source": "test",
    }
    out_file = tmp_path / "dict.geojson"
    save_geojson([feat], str(out_file))
    assert out_file.exists()
    coll = json.load(open(out_file))
    assert coll["features"][0]["properties"]["id"] == 1


def test_save_polygon_geojson(tmp_path):
    transform = geo.Affine(1, 0, 10, 0, -1, 20)
    crs = geo.CRS.from_epsg(4326)
    feats = [make_bbox_feature()]
    out_file = tmp_path / "polys.geojson"
    if geo.rasterio is None:
        with pytest.raises(RuntimeError):
            save_polygon_geojson(feats, transform, crs, str(out_file))
    else:
        save_polygon_geojson(feats, transform, crs, str(out_file))
        data = json.load(open(out_file))
        assert data["crs"] == crs.to_string()
        assert data["features"][0]["geometry"]["type"] == "Polygon"
        assert "centroid" in data["features"][0]["properties"]


def test_save_polygon_geojson_contour(tmp_path):
    transform = geo.Affine(1, 0, 0, 0, -1, 0)
    crs = geo.CRS.from_epsg(4326)
    feats = [make_contour_feature()]
    out_file = tmp_path / "contours.geojson"
    if geo.rasterio is None:
        with pytest.raises(RuntimeError):
            save_polygon_geojson(feats, transform, crs, str(out_file))
    else:
        save_polygon_geojson(feats, transform, crs, str(out_file))
        data = json.load(open(out_file))
        assert data["features"][0]["geometry"]["type"] == "Polygon"


def test_features_to_shapely(tmp_path):
    transform = geo.Affine(1, 0, 10, 0, -1, 20)
    crs = geo.CRS.from_epsg(4326)
    feats = [make_bbox_feature()]
    if geo.rasterio is None or features_to_shapely.__globals__["Polygon"] is None:
        with pytest.raises(RuntimeError):
            features_to_shapely(feats, transform, crs)
    else:
        polys = features_to_shapely(feats, transform, crs)
        assert polys
        assert polys[0].geom_type == "Polygon"


def test_features_to_shapely_contour(tmp_path):
    transform = geo.Affine(1, 0, 0, 0, -1, 0)
    crs = geo.CRS.from_epsg(4326)
    feats = [make_contour_feature()]
    if (
        geo.rasterio is None
        or features_to_shapely.__globals__["Polygon"] is None
    ):
        with pytest.raises(RuntimeError):
            features_to_shapely(feats, transform, crs)
    else:
        polys = features_to_shapely(feats, transform, crs)
        assert polys
        assert polys[0].geom_type == "Polygon"
