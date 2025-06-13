from __future__ import annotations

"""Utilities for exporting detected features."""

import json
from dataclasses import asdict, is_dataclass
from collections import Counter
from pathlib import Path

from detection.feature import Feature
from typing import Any, Dict, List

try:
    from shapely.geometry import Polygon  # type: ignore
except Exception:  # pragma: no cover - library may be missing
    Polygon = None  # type: ignore

try:
    from rasterio.transform import Affine
    from rasterio.crs import CRS
except Exception:  # pragma: no cover - library may be missing
    Affine = None  # type: ignore
    CRS = None  # type: ignore

from processing.geo import bbox_to_polygon, bbox_centroid, contour_to_polygon


def _feature_to_dict(feature: Feature) -> dict:
    """Convert a :class:`Feature` to a JSON-serializable dictionary."""
    if is_dataclass(feature):
        return asdict(feature)
    # Fallback if feature is already a mapping like object
    if hasattr(feature, "__dict__"):
        return dict(feature.__dict__)
    raise TypeError("feature must be a dataclass instance")


def _write_summary(features: list[Feature], summary_path: str) -> None:
    """Write a simple summary text file for ``features``."""
    counts = Counter(f.feature_type for f in features)
    lines = [f"Features found: {len(features)}"]
    for name, count in sorted(counts.items()):
        lines.append(f"{name}: {count}")
    Path(summary_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    summary_path = Path(out_path).with_name(Path(out_path).stem + "_summary.txt")
    _write_summary(features, str(summary_path))


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
        if isinstance(geometry, dict) and geometry.get("type") == "bbox":
            bbox = geometry.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x, y, w, h = bbox
                if w == 0 and h == 0:
                    geometry = {"type": "Point", "coordinates": [x, y]}
                elif w == 0 or h == 0:
                    geometry = {
                        "type": "LineString",
                        "coordinates": [[x, y], [x + w, y + h]],
                    }
                else:
                    ring = [
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h],
                        [x, y],
                    ]
                    geometry = {"type": "Polygon", "coordinates": [ring]}
        props = {k: v for k, v in data.items() if k != "geometry"}
        geo_features.append({"type": "Feature", "geometry": geometry, "properties": props})

    collection: Dict[str, Any] = {"type": "FeatureCollection", "features": geo_features}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(collection, fh, ensure_ascii=False, indent=2)

    summary_path = Path(out_path).with_name(Path(out_path).stem + "_summary.txt")
    _write_summary(features, str(summary_path))


def save_polygon_geojson(
    features: list[Feature],
    transform: "Affine",
    crs: "CRS",
    out_path: str,
) -> None:
    """Export bounding-box features as polygons with centroids."""

    if Affine is None or CRS is None:
        raise RuntimeError("rasterio is not installed.")

    geo_features = []
    for feat in features:
        data = _feature_to_dict(feat)
        geom = data.get("geometry", {})
        ring = None
        centroid = None
        if isinstance(geom, dict) and "contour" in geom:
            contour = geom.get("contour")
            if contour is None:
                continue
            ring = contour_to_polygon(contour, transform, crs)
            xs = [pt[0] for pt in ring[:-1]] if ring else []
            ys = [pt[1] for pt in ring[:-1]] if ring else []
            if xs and ys:
                centroid = (
                    round(sum(xs) / len(xs), 6),
                    round(sum(ys) / len(ys), 6),
                )
        else:
            bbox = geom.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            ring = bbox_to_polygon(tuple(int(v) for v in bbox), transform, crs)
            centroid = bbox_centroid(tuple(int(v) for v in bbox), transform, crs)
        props = {k: v for k, v in data.items() if k != "geometry"}
        if centroid is not None:
            props["centroid"] = centroid
        geo_features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": props,
            }
        )

    collection: Dict[str, Any] = {
        "type": "FeatureCollection",
        "crs": crs.to_string() if hasattr(crs, "to_string") else str(crs),
        "features": geo_features,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(collection, fh, ensure_ascii=False, indent=2)

    summary_path = Path(out_path).with_name(Path(out_path).stem + "_summary.txt")
    _write_summary(features, str(summary_path))


def features_to_shapely(
    features: list[Feature], transform: "Affine", crs: "CRS"
) -> List["Polygon"]:
    """Return shapely polygons for the given features."""

    if Polygon is None:
        raise RuntimeError("Shapely is not installed.")
    if Affine is None or CRS is None:
        raise RuntimeError("rasterio is not installed.")

    polygons: List["Polygon"] = []
    for feat in features:
        geom = feat.geometry if isinstance(feat.geometry, dict) else {}
        if "contour" in geom:
            contour = geom.get("contour")
            if contour is None:
                continue
            ring = contour_to_polygon(contour, transform, crs)
            polygons.append(Polygon(ring))
        else:
            bbox = geom.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            poly = bbox_to_polygon(tuple(int(v) for v in bbox), transform, crs, as_shapely=True)
            polygons.append(poly)
    return polygons


__all__ = [
    "serialize_features",
    "save_geojson",
    "save_polygon_geojson",
    "features_to_shapely",
]

