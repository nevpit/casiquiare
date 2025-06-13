"""Environmental feature engineering utilities."""

from __future__ import annotations

from typing import Dict

try:
    import numpy as np
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - library may be missing
    np = None  # type: ignore
    cKDTree = None  # type: ignore

try:
    from rasterio.transform import Affine
except Exception:  # pragma: no cover - library may be missing
    Affine = None  # type: ignore


def hydrological_distance(
    points: "np.ndarray",
    water_points: "np.ndarray",
) -> "np.ndarray":
    """Return distance from each point to the nearest water location."""
    if np is None or cKDTree is None:
        raise RuntimeError("NumPy and SciPy are required.")

    tree = cKDTree(water_points)
    dist, _ = tree.query(points)
    return dist.astype(float)


def sample_raster_values(
    raster: "np.ndarray",
    transform: "Affine",
    coords: "np.ndarray",
) -> "np.ndarray":
    """Sample ``raster`` values at ``coords`` using nearest-neighbour lookup."""
    if np is None or Affine is None:
        raise RuntimeError("rasterio and NumPy are required.")

    cols, rows = ~transform * (coords[:, 0], coords[:, 1])
    rows = np.clip(np.round(rows).astype(int), 0, raster.shape[0] - 1)
    cols = np.clip(np.round(cols).astype(int), 0, raster.shape[1] - 1)
    return raster[rows, cols]


def sample_climate_layers(
    layers: Dict[str, "np.ndarray"],
    transform: "Affine",
    coords: "np.ndarray",
) -> Dict[str, "np.ndarray"]:
    """Extract climate values from multiple raster ``layers`` at ``coords``."""
    if np is None:
        raise RuntimeError("NumPy is required.")

    samples = {name: sample_raster_values(arr, transform, coords) for name, arr in layers.items()}
    return samples


def hydro_cost_surface(
    distance: "np.ndarray",
    slope: "np.ndarray",
    weight: float = 0.5,
) -> "np.ndarray":
    """Compute a simple hydro-cost surface from distance and slope."""
    if np is None:
        raise RuntimeError("NumPy is required.")

    return distance * (1.0 + weight * slope)


__all__ = [
    "hydrological_distance",
    "sample_raster_values",
    "sample_climate_layers",
    "hydro_cost_surface",
]
