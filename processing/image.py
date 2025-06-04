"""Raster image utilities."""

from __future__ import annotations

try:
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None  # type: ignore


def to_uint8(
    image: "np.ndarray",
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> "np.ndarray":
    """Convert an array to 8-bit using contrast stretching.

    Args:
        image: The array to rescale.
        lower_percentile: Values below this percentile are clipped.
        upper_percentile: Values above this percentile are clipped.

    Returns:
        The rescaled image as ``uint8``.
    """
    if np is None:
        raise RuntimeError("NumPy is not installed.")

    arr = image.astype(float)
    vmin, vmax = np.percentile(arr, [lower_percentile, upper_percentile])
    if vmax - vmin <= 0:
        scaled = np.zeros_like(arr, dtype=np.uint8)
    else:
        clipped = np.clip(arr, vmin, vmax)
        scaled = (clipped - vmin) / (vmax - vmin) * 255.0
    return scaled.astype(np.uint8)


__all__ = ["to_uint8"]

