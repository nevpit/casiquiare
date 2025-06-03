"""Satellite imagery processing utilities."""

from __future__ import annotations

try:
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None  # type: ignore


def compute_ndvi(red: "np.ndarray", nir: "np.ndarray") -> "np.ndarray":
    """Compute the normalized difference vegetation index (NDVI)."""
    if np is None:
        raise RuntimeError("NumPy is required.")

    red_arr = red.astype(float)
    nir_arr = nir.astype(float)
    denom = nir_arr + red_arr

    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir_arr - red_arr) / denom
        if isinstance(ndvi, np.ndarray):
            ndvi[denom == 0] = 0.0
        else:
            ndvi = 0.0

    return ndvi.astype(np.float32)


__all__ = ["compute_ndvi"]
