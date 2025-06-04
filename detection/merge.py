"""Utilities for merging detection results from multiple sensors."""

from __future__ import annotations

from math import hypot
from typing import Dict, List


def _centroid_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Compute the centroid of an integer bounding box."""
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def merge_detections(
    lidar: List[Dict[str, object]],
    satellite: List[Dict[str, object]],
    threshold: float = 25.0,
) -> List[Dict[str, object]]:
    """Merge detections when centroids are within ``threshold`` units.

    Parameters
    ----------
    lidar
        List of detections from LiDAR passes.
    satellite
        List of detections from satellite imagery.
    threshold
        Maximum centroid distance for two detections to be considered the same.

    Returns
    -------
    list
        Combined detection list with ``sources`` noting originating sensors.
    """

    merged: List[Dict[str, object]] = []
    used_sat: set[int] = set()

    for ld in lidar:
        bbox_ld = ld.get("bbox")
        if not isinstance(bbox_ld, (list, tuple)) or len(bbox_ld) != 4:
            merged_feat = ld.copy()
            merged_feat.setdefault("sources", ["lidar"])
            merged.append(merged_feat)
            continue
        c_ld = _centroid_from_bbox(tuple(int(v) for v in bbox_ld))

        match_idx = None
        for idx, st in enumerate(satellite):
            if idx in used_sat:
                continue
            bbox_st = st.get("bbox")
            if not isinstance(bbox_st, (list, tuple)) or len(bbox_st) != 4:
                continue
            c_st = _centroid_from_bbox(tuple(int(v) for v in bbox_st))
            if hypot(c_ld[0] - c_st[0], c_ld[1] - c_st[1]) <= threshold:
                match_idx = idx
                break

        if match_idx is not None:
            used_sat.add(match_idx)
            st = satellite[match_idx]
            bbox_st = st.get("bbox")
            if isinstance(bbox_st, (list, tuple)) and len(bbox_st) == 4:
                x1 = min(bbox_ld[0], bbox_st[0])
                y1 = min(bbox_ld[1], bbox_st[1])
                x2 = max(bbox_ld[0] + bbox_ld[2], bbox_st[0] + bbox_st[2])
                y2 = max(bbox_ld[1] + bbox_ld[3], bbox_st[1] + bbox_st[3])
                merged_bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            else:
                merged_bbox = bbox_ld
            merged_feat = {**ld, **st, "bbox": merged_bbox}
            sources = set(ld.get("sources", ["lidar"])) | set(st.get("sources", ["satellite"]))
            merged_feat["sources"] = sorted(sources)
            merged.append(merged_feat)
        else:
            merged_feat = ld.copy()
            merged_feat["sources"] = sorted(ld.get("sources", ["lidar"]))
            merged.append(merged_feat)

    for idx, st in enumerate(satellite):
        if idx in used_sat:
            continue
        merged_feat = st.copy()
        merged_feat["sources"] = sorted(st.get("sources", ["satellite"]))
        merged.append(merged_feat)

    return merged


__all__ = ["merge_detections"]
