"""ML-based post-filter utilities for detections."""

from __future__ import annotations

from typing import Dict, List, Sequence, Optional

try:
    from sklearn.ensemble import RandomForestClassifier
except Exception:  # pragma: no cover - library may be missing
    RandomForestClassifier = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None  # type: ignore


def _feature_vector(det: Dict[str, object]) -> List[float]:
    """Extract numerical features from a detection dictionary."""
    width = float(det.get("width", 0.0))
    height = float(det.get("height", 0.0))
    vertices = float(det.get("num_vertices", 0.0))
    score = float(det.get("score", 0.0))
    return [width, height, vertices, score]


def train_post_filter(
    detections: Sequence[Dict[str, object]],
    labels: Sequence[int],
    n_estimators: int = 50,
    random_state: Optional[int] = None,
) -> "RandomForestClassifier":
    """Train a random-forest classifier for filtering detections."""
    if RandomForestClassifier is None or np is None:
        raise RuntimeError("scikit-learn and NumPy are required.")
    X = np.array([_feature_vector(d) for d in detections])
    y = np.array(list(labels))
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X, y)
    return clf


def apply_post_filter(
    classifier: "RandomForestClassifier",
    detections: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Filter detections using a trained classifier."""
    if RandomForestClassifier is None or np is None:
        raise RuntimeError("scikit-learn and NumPy are required.")
    if not hasattr(classifier, "predict"):
        raise ValueError("Invalid classifier instance.")
    X = np.array([_feature_vector(d) for d in detections])
    preds = classifier.predict(X)
    return [d for d, p in zip(detections, preds) if p == 1]


__all__ = ["train_post_filter", "apply_post_filter"]

