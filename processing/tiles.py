"""Tile pyramid generation using gdal2tiles."""

from __future__ import annotations

import subprocess
import shutil
from pathlib import Path


# Path to project root -> processing/.. -> root
_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "interface" / "backend" / "static" / "tiles"


def _gdal2tiles_path() -> str | None:
    """Return gdal2tiles executable path if available."""
    return shutil.which("gdal2tiles.py") or shutil.which("gdal2tiles")


def generate_tile_pyramid(
    raster: str,
    layer: str,
    zoom: str = "0-5",
    tiles_root: str | Path = _DEFAULT_ROOT,
) -> Path:
    """Generate map tiles for ``raster`` using gdal2tiles.

    Parameters
    ----------
    raster:
        Input raster path.
    layer:
        Layer name (e.g., ``dtm`` or ``hillshade``).
    zoom:
        Zoom levels passed to gdal2tiles (e.g., ``"0-14"``).
    tiles_root:
        Base directory to store tiles. Defaults to the backend static tiles
        folder.

    Returns
    -------
    Path
        The directory containing generated tiles.
    """
    exe = _gdal2tiles_path()
    if exe is None:
        raise RuntimeError("gdal2tiles.py is not installed")

    out_dir = Path(tiles_root) / layer
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        exe,
        "-z",
        zoom,
        "-w",
        "none",
        "--processes=1",
        "--tile-format",
        "png",
        raster,
        str(out_dir),
    ]
    subprocess.check_call(cmd)
    return out_dir


def generate_default_tiles(
    dtm: str | None = None,
    hillshade: str | None = None,
    ndvi: str | None = None,
    zoom: str = "0-5",
    tiles_root: str | Path = _DEFAULT_ROOT,
) -> None:
    """Generate tile pyramids for common layers.

    Any ``None`` input layers will be skipped.
    """
    if dtm:
        generate_tile_pyramid(dtm, "dtm", zoom, tiles_root)
    if hillshade:
        generate_tile_pyramid(hillshade, "hillshade", zoom, tiles_root)
    if ndvi:
        generate_tile_pyramid(ndvi, "ndvi", zoom, tiles_root)


__all__ = ["generate_tile_pyramid", "generate_default_tiles"]
