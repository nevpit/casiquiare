from __future__ import annotations

"""Utilities for exporting detected features."""

import json
from dataclasses import asdict, is_dataclass

from detection.feature import Feature
from typing import Any, Dict


def _feature_to_dict(feature: Feature) -> dict:
    """Convert a :class:`Feature` to a JSON-serializable dictionary."""
    if is_dataclass(feature):
        return asdict(feature)
    # Fallback if feature is already a mapping like object
    if hasattr(feature, "__dict__"):
        return dict(feature.__dict__)
    raise TypeError("feature must be a dataclass instance")


def serialize_features(features: list[Feature], out_path: str) -> None:
    """Serialize a list of :class:`Feature` objects to JSON.

    Parameters
    ----------
    features:
        List of features to serialize.
    out_path:
        Path to the output JSON file.
    """
    feature_dicts = [_feature_to_dict(f) for f in features]
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(feature_dicts, fh, ensure_ascii=False, indent=2)


def save_geojson(features: list[Feature], out_path: str) -> None:
    """Export features as a GeoJSON ``FeatureCollection``.

    Parameters
    ----------
    features:
        List of features to export.
    out_path:
        Path to the GeoJSON file.
    """
    geo_features = []
    for feat in features:
        data = _feature_to_dict(feat)
        geometry = data.get("geometry")
        props = {k: v for k, v in data.items() if k != "geometry"}
        geo_features.append({"type": "Feature", "geometry": geometry, "properties": props})

    collection: Dict[str, Any] = {"type": "FeatureCollection", "features": geo_features}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(collection, fh, ensure_ascii=False, indent=2)


__all__ = ["serialize_features", "save_geojson"]

