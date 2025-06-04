"""Utilities for the standalone Eyes agent."""

from __future__ import annotations

from typing import Any, Dict, List
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from log_config import setup_logger

try:
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None  # type: ignore

from detection import Feature
from detection.shapes import detect_shapes
from io_utils.raster import load_raster

logger = setup_logger(__name__)


def _to_feature(idx: int, item: Dict[str, Any], source: str) -> Feature:
    """Convert a detection dictionary to a :class:`Feature`."""
    bbox = item.get("bbox")
    geometry = {"type": "bbox", "bbox": bbox} if bbox else {}
    width = float(item.get("width", 0.0))
    height = float(item.get("height", 0.0))
    return Feature(
        id=idx,
        feature_type=str(item.get("shape", "unknown")),
        geometry=geometry,
        dimensions=(width, height),
        confidence=float(item.get("score", 0.0)),
        source=source,
    )


def _feature_summary(features: List[Feature]) -> Dict[str, Any]:
    """Return detection summary stats for logging."""
    counts = len(features)
    type_counts = Counter(f.feature_type for f in features)
    scores = sorted((f.confidence for f in features), reverse=True)[:3]
    return {"count": counts, "types": dict(type_counts), "top_scores": scores}


def _scan_image(image: "np.ndarray", profile: Dict[str, Any] | None = None, source: str = "raster") -> List[Feature]:
    """Run shape detection on a raster array."""
    if detect_shapes is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required.")

    logger.debug("Running shape detection")
    results = detect_shapes(image, profile)
    logger.debug("Detected %d shapes", len(results))
    return [_to_feature(i + 1, feat, source) for i, feat in enumerate(results)]


def scan_area(area_id: str | Dict[str, Any]) -> List[Feature]:
    """Scan a raster or image dictionary for archaeological features."""
    if isinstance(area_id, str):
        if load_raster is None or np is None:
            raise RuntimeError("rasterio and NumPy are required.")
        data, transform, crs = load_raster(area_id)
        profile = {"transform": transform, "crs": crs}
        image = data[0] if data.ndim == 3 else data
        logger.info("Scanning raster %s", area_id)
        features = _scan_image(image, profile, source="raster")
        logger.info("%d features found in %s", len(features), area_id)
        summary = _feature_summary(features)
        logger.info(
            "Summary for %s: types=%s, top_scores=%s",
            area_id,
            summary["types"],
            summary["top_scores"],
        )
        return features

    if isinstance(area_id, dict):
        image = area_id.get("image")
        if image is None:
            raise ValueError("area_id dictionary requires an 'image' entry")
        profile = area_id.get("profile")
        source = area_id.get("source", "raster")
        logger.info("Scanning %s image", source)
        features = _scan_image(image, profile, source=source)
        logger.info("%d features found in %s image", len(features), source)
        summary = _feature_summary(features)
        logger.info(
            "Summary for %s image: types=%s, top_scores=%s",
            source,
            summary["types"],
            summary["top_scores"],
        )
        return features

    raise TypeError("area_id must be a str path or a dictionary")


def scan_tiles_concurrent(
    area_ids: List[str | Dict[str, Any]], max_workers: int | None = None
) -> List[List[Feature]]:
    """Scan multiple areas concurrently.

    Parameters
    ----------
    area_ids:
        Sequence of raster paths or image dictionaries accepted by
        :func:`scan_area`.
    max_workers:
        Maximum number of worker threads. Defaults to ``None`` for the
        executor to decide.

    Returns
    -------
    list of list of :class:`Feature`
        Detection results in the same order as ``area_ids``.
    """

    logger.info("Scanning %d tiles", len(area_ids))
    results: List[List[Feature]] = [list() for _ in area_ids]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(scan_area, aid): idx for idx, aid in enumerate(area_ids)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                feats = future.result()
            except Exception as exc:
                logger.error("Tile %s failed: %s", area_ids[idx], exc)
                feats = []
            results[idx] = feats
            summary = _feature_summary(feats)
            tile_label = area_ids[idx]
            logger.info(
                "Tile %s summary: %d features, types=%s, top_scores=%s",
                tile_label,
                summary["count"],
                summary["types"],
                summary["top_scores"],
            )
            logger.debug("Finished tile %d", idx)
    logger.info("Completed scanning tiles")
    return results


__all__ = ["scan_area", "scan_tiles_concurrent"]

