"""Core dataclasses for detection results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional


@dataclass
class Feature:
    """Structured representation of a detected feature."""

    id: int
    feature_type: str
    geometry: Dict[str, Any]
    dimensions: Tuple[float, float]
    confidence: float
    source: str
    snippet_path: Optional[str] = None


__all__ = ["Feature"]
