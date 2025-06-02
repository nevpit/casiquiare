"""Edge detection helpers for LiDAR-derived images."""

from __future__ import annotations

try:  # Optional heavy dependencies
    import cv2
except Exception:  # pragma: no cover - library may be missing
    cv2 = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None  # type: ignore


def detect_edges(
    image: "np.ndarray",
    *,
    blur_kernel_size: tuple[int, int] = (3, 3),
    canny_threshold1: int = 30,
    canny_threshold2: int = 90,
    dilation_iterations: int = 1,
) -> "np.ndarray":
    """Detect subtle edges using a tuned Canny filter.

    Parameters
    ----------
    image:
        Input image array.
    blur_kernel_size:
        Kernel size for Gaussian blurring.
    canny_threshold1:
        First threshold passed to :func:`cv2.Canny`.
    canny_threshold2:
        Second threshold passed to :func:`cv2.Canny`.
    dilation_iterations:
        Number of times to dilate the resulting edge mask.
    """

    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required.")

    arr = np.asarray(image, dtype=np.float32)
    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    arr_u8 = arr.astype(np.uint8)

    blurred = cv2.GaussianBlur(arr_u8, blur_kernel_size, 0)
    edges = cv2.Canny(
        blurred, canny_threshold1, canny_threshold2, apertureSize=3, L2gradient=True
    )
    edges = cv2.dilate(edges, None, iterations=dilation_iterations)
    return edges


__all__ = ["detect_edges"]
