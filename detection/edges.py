"""Edge detection helpers."""

from __future__ import annotations

try:
    import cv2
except Exception:  # pragma: no cover - library may be missing
    cv2 = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None  # type: ignore


def multi_scale_canny(
    image: "np.ndarray",
    scales: tuple[int, ...] = (3, 5),
    sigma: float = 0.33,
) -> "np.ndarray":
    """Detect edges using Canny at multiple blur scales.

    Parameters
    ----------
    image : ndarray
        Grayscale uint8 image.
    scales : tuple[int, ...], optional
        Gaussian blur kernel sizes. Defaults to ``(3, 5)``.
    sigma : float, optional
        Threshold adjustment factor. Defaults to ``0.33``.

    Returns
    -------
    ndarray
        Combined edge mask.
    """
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required.")

    edges = np.zeros_like(image)
    for k in scales:
        if k % 2 == 0:
            k += 1  # kernel size must be odd
        blurred = cv2.GaussianBlur(image, (k, k), 0)
        med = np.median(blurred)
        lower = int(max(0, (1.0 - sigma) * med))
        upper = int(min(255, (1.0 + sigma) * med))
        edges = cv2.bitwise_or(edges, cv2.Canny(blurred, lower, upper))
    return edges


__all__ = ["multi_scale_canny"]
