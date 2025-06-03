#!/usr/bin/env python3
"""Command line interface for Eyes agent utilities."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from agents.eyes_tools import scan_area
from io.data_paths import get_data_path

try:  # Optional dependencies for output helpers
    import rasterio
    from rasterio.transform import xy
except Exception:  # pragma: no cover - library may be missing
    rasterio = None
    xy = None  # type: ignore

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    cv2 = None  # type: ignore
    np = None  # type: ignore


def _save_geojson(features: List[Dict[str, Any]], profile: Dict[str, Any], path: str) -> None:
    """Save detected features to a GeoJSON file."""
    if rasterio is None or xy is None:
        raise RuntimeError("rasterio is required to write GeoJSON")

    transform = profile.get("transform")
    if transform is None:
        raise ValueError("Profile missing transform information")

    items = []
    for feat in features:
        x, y, w, h = feat.get("bbox", (0, 0, 0, 0))
        ul = xy(transform, y, x, offset="ul")
        lr = xy(transform, y + h, x + w, offset="ul")
        poly = [
            [ul[0], ul[1]],
            [lr[0], ul[1]],
            [lr[0], lr[1]],
            [ul[0], lr[1]],
            [ul[0], ul[1]],
        ]
        props = {k: v for k, v in feat.items() if k != "bbox"}
        items.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Polygon", "coordinates": [poly]},
            }
        )

    fc = {"type": "FeatureCollection", "features": items}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)


def _save_snippets(features: List[Dict[str, Any]], image: "np.ndarray", out_dir: str, prefix: str) -> None:
    """Write PNG snippets around detected features."""
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required to save snippets")
    os.makedirs(out_dir, exist_ok=True)
    for idx, feat in enumerate(features):
        x, y, w, h = feat.get("bbox", (0, 0, 0, 0))
        crop = image[y : y + h, x : x + w]
        out = os.path.join(out_dir, f"{prefix}_feat_{idx}.png")
        cv2.imwrite(out, crop)


def _cmd_scan(args: argparse.Namespace) -> None:
    """Handle the ``scan`` subcommand."""
    try:
        lidar_path = get_data_path(tile_id=args.area_id)
    except Exception as exc:  # pragma: no cover - configuration errors
        raise SystemExit(str(exc))

    result = scan_area(lidar_path)
    features = result.get("features", [])
    print(f"Detected {len(features)} features in {args.area_id}")

    if args.save_geojson:
        profile = result.get("profile")
        if profile is None:
            print("No profile available; skipping GeoJSON export")
        else:
            out_file = f"{args.area_id}_features.geojson"
            _save_geojson(features, profile, out_file)
            print(f"Saved GeoJSON to {out_file}")

    if args.save_snippets:
        image = result.get("local_relief") or result.get("hillshade")
        if image is None:
            print("No snippet image available; skipping snippet export")
        else:
            _save_snippets(features, image, ".", args.area_id)
            print("Saved image snippets to current directory")


def main(argv: List[str] | None = None) -> None:
    """Run the Eyes agent CLI."""
    parser = argparse.ArgumentParser(prog="eyes-agent", description="Eyes agent utilities")
    sub = parser.add_subparsers(dest="command")

    scan = sub.add_parser("scan", help="Scan an area for geometric features")
    scan.add_argument("area_id", help="Tile identifier to scan")
    scan.add_argument("--save-geojson", action="store_true", help="Export features as GeoJSON")
    scan.add_argument("--save-snippets", action="store_true", help="Save PNG snippets around features")

    args = parser.parse_args(argv)

    if args.command == "scan":
        _cmd_scan(args)
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
