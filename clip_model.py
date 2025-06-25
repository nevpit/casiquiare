from __future__ import annotations

"""Helpers for loading a CLIP image encoder and computing embeddings."""

from typing import Any

try:  # optional dependency
    import clip  # type: ignore
    from PIL import Image  # type: ignore
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency may be missing
    clip = None  # type: ignore
    Image = None  # type: ignore
    torch = None  # type: ignore

CLIP_MODEL: Any = None
CLIP_PREPROCESS: Any = None


def load_clip_model(model_name: str = "ViT-B/32", device: str = "cpu") -> Any:
    """Load a CLIP model and return the image encoder."""
    if clip is None:
        raise RuntimeError("CLIP library is not installed")

    global CLIP_MODEL, CLIP_PREPROCESS
    CLIP_MODEL, CLIP_PREPROCESS = clip.load(model_name, device=device)
    CLIP_MODEL.eval()
    return CLIP_MODEL


def image_embedding(path: str) -> list[float]:
    """Return the embedding vector for an image file."""
    if clip is None or torch is None or Image is None:
        raise RuntimeError("CLIP dependencies are not installed")
    if CLIP_MODEL is None or CLIP_PREPROCESS is None:
        raise RuntimeError("CLIP model has not been loaded")

    img = Image.open(path).convert("RGB")
    tensor = CLIP_PREPROCESS(img).unsqueeze(0)
    with torch.no_grad():
        vec = CLIP_MODEL.encode_image(tensor)
    return vec[0].tolist()


__all__ = ["load_clip_model", "image_embedding"]
