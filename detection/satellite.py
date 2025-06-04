"""Satellite raster indices computations."""

from __future__ import annotations

try:
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None  # type: ignore

try:
    import rasterio
    from rasterio.enums import ColorInterp
    from rasterio import DatasetReader
except Exception:  # pragma: no cover - library may be missing
    rasterio = None  # type: ignore
    ColorInterp = None  # type: ignore
    DatasetReader = None  # type: ignore

try:
    from processing.sat import compute_ndvi, compute_ndwi
except Exception:  # pragma: no cover - library may be missing
    compute_ndvi = None  # type: ignore
    compute_ndwi = None  # type: ignore


def _compute_savi(red: "np.ndarray", nir: "np.ndarray", L: float = 0.5) -> "np.ndarray":
    """Compute the soil adjusted vegetation index (SAVI)."""
    if np is None:
        raise RuntimeError("NumPy is required.")

    red_arr = red.astype(float)
    nir_arr = nir.astype(float)
    denom = nir_arr + red_arr + L

    with np.errstate(divide="ignore", invalid="ignore"):
        savi = (nir_arr - red_arr) / denom * (1.0 + L)
        if isinstance(savi, np.ndarray):
            savi[denom == 0] = 0.0
        else:
            savi = 0.0

    return savi.astype(np.float32)


def _band_index(src: "DatasetReader", name: str, default: int | None) -> int | None:
    """Return band index matching ``name`` from color interpretation or ``descriptions``."""
    if rasterio is None:
        return default

    if hasattr(src, "colorinterp"):
        for idx, interp in enumerate(src.colorinterp, start=1):
            if interp is None:
                continue
            interp_name = getattr(interp, "name", str(interp)).lower()
            if name == "nir" and interp_name in {"nir", "nearinfrared", "undefined"}:
                return idx
            if interp_name == name:
                return idx

    if hasattr(src, "descriptions") and src.descriptions:
        for idx, desc in enumerate(src.descriptions, start=1):
            if desc is None:
                continue
            d = desc.lower()
            if name == "nir" and ("nir" in d or "near" in d and "infrared" in d):
                return idx
            if name in d:
                return idx

    return default


def compute_indices(src: "DatasetReader") -> dict[str, "np.ndarray"]:
    """Compute NDVI, NDWI and SAVI indices from a multi-band dataset."""
    if rasterio is None or np is None:
        raise RuntimeError("rasterio and NumPy are required.")
    if compute_ndvi is None or compute_ndwi is None:
        raise RuntimeError("Satellite utilities are not available.")

    red_idx = _band_index(src, "red", 3 if src.count >= 3 else None)
    green_idx = _band_index(src, "green", 2 if src.count >= 2 else None)
    nir_idx = _band_index(src, "nir", 4 if src.count >= 4 else None)

    if red_idx is None or green_idx is None or nir_idx is None:
        raise ValueError("Unable to determine band indices for red, green and NIR.")

    red = src.read(red_idx)
    green = src.read(green_idx)
    nir = src.read(nir_idx)

    ndvi = compute_ndvi(red, nir)
    ndwi = compute_ndwi(green, nir)
    savi = _compute_savi(red, nir)

    return {"ndvi": ndvi, "ndwi": ndwi, "savi": savi}


__all__ = ["compute_indices"]

