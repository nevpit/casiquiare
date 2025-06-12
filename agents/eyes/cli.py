"""Command line interface for the Eyes agent."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

# Ensure the local io_utils.raster module is importable without conflicting with
# Python's built-in ``io`` module.
_raster_path = Path(__file__).resolve().parents[2] / "io_utils" / "raster.py"
_spec = importlib.util.spec_from_file_location("io_utils.raster", _raster_path)
if _spec and _spec.loader:
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)  # type: ignore[arg-type]
    sys.modules.setdefault("io_utils.raster", _module)

from .tools import scan_area
from output import save_geojson
from log_config import setup_logger

logger = setup_logger(__name__)



def _run_scan(args: argparse.Namespace) -> None:
    """Execute the scan command."""
    logger.info("Scanning %s", args.area_id)
    features = scan_area(args.area_id)
    logger.info("Detected %d features", len(features))

    if args.save_snippets:
        from importlib import import_module

        load_raster = import_module("io_utils.raster").load_raster
        save_snippets = import_module("agents.eyes.tools").save_snippets

        data, _, _ = load_raster(args.area_id)
        image = data[0] if data.ndim == 3 else data
        features_with_bbox = [
            f for f in features if f.geometry.get("bbox") is not None
        ]
        feat_dicts = [{"bbox": f.geometry.get("bbox")} for f in features_with_bbox]
        out_dir = Path(f"{Path(args.area_id).stem}_snippets")
        paths = save_snippets(image, feat_dicts, str(out_dir))
        for feat, path in zip(features_with_bbox, paths):
            feat.snippet_path = path
        logger.info("Snippets saved to %s", out_dir)

    if args.save_geojson:
        out_file = Path(f"{Path(args.area_id).stem}_features.geojson")
        save_geojson(features, str(out_file))
        logger.info("GeoJSON saved to %s", out_file)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logger.info("Eyes CLI started")
    parser = argparse.ArgumentParser(prog="eyes-agent")
    subparsers = parser.add_subparsers(dest="command")

    scan_parser = subparsers.add_parser("scan", help="Scan an area for features")
    scan_parser.add_argument("area_id", help="Raster path or image descriptor")
    scan_parser.add_argument("--save-geojson", action="store_true", help="Save detections as GeoJSON")
    scan_parser.add_argument("--save-snippets", action="store_true", help="Save PNG snippets around detections")

    args = parser.parse_args(argv)

    if args.command == "scan":
        _run_scan(args)
    else:
        parser.print_help()
    logger.info("Eyes CLI finished")


if __name__ == "__main__":
    main()
