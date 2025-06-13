"""Dataclasses describing structured Brain results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from time import time


@dataclass
class ModelResult:
    """Summary information for a trained model."""

    model_type: str
    metrics: Dict[str, float]
    feature_importances: Dict[str, float]
    model_path: str
    model_card: str
    # Actual model object kept for internal use only
    model: Any = field(repr=False, default=None)


@dataclass
class PredictionResult:
    """Likelihood scores or heatmap predictions."""

    scores_map: Any
    summary: Dict[str, Any]
    feature_importances: Optional[List[float]] = None


@dataclass
class ClusterResult:
    """Result from DBSCAN clustering."""

    labels: List[int]
    summary: Dict[str, Any]


@dataclass
class ExecutionResult:
    """Output from executing a code snippet."""

    stdout: str
    result: Any
    figures: List[str]
    locals: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class AgentMessage:
    """Structured message exchanged between agents."""

    agent: str
    type: str
    content: Dict[str, Any]
    timestamp: float = field(default_factory=time)


__all__ = [
    "ModelResult",
    "PredictionResult",
    "ClusterResult",
    "ExecutionResult",
    "AgentMessage",
]
