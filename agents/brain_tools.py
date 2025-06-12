"""Utility functions for the Brain agent."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Sequence, Optional

from log_config import setup_logger

logger = setup_logger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover - library may be missing
    RandomForestClassifier = None  # type: ignore
    KMeans = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None  # type: ignore

try:
    import pandas as pd
except Exception:  # pragma: no cover - library may be missing
    pd = None  # type: ignore

try:
    from io_utils.data_paths import load_mapping
except Exception:  # pragma: no cover - library may be missing
    load_mapping = None  # type: ignore

try:
    from processing.sat import compute_ndvi
except Exception:  # pragma: no cover - library may be missing
    compute_ndvi = None  # type: ignore


def ingest_training_data(
    csv_path: Optional[str] = None,
    mapping_file: Optional[str] = None,
) -> "pd.DataFrame":
    """Load and prepare the training dataset for model fitting.

    The function reads a CSV file with site/non-site samples and computes
    derived environmental indices such as NDVI and slope. If ``csv_path`` is
    not provided the ``training_data`` entry from ``data_paths.yaml`` is used.

    Returns a :class:`pandas.DataFrame` with feature columns and a ``label``
    column indicating archaeological sites (``1``) or background (``0``).
    """

    if pd is None or np is None or compute_ndvi is None:
        raise RuntimeError("pandas, NumPy and processing utilities are required.")

    if csv_path is None:
        if load_mapping is None:
            raise RuntimeError("PyYAML is required to read the data mapping.")
        mapping = load_mapping(mapping_file)
        try:
            csv_path = mapping["training_data"]
        except KeyError:
            raise ValueError("training_data path not specified")

    df = pd.read_csv(csv_path)

    if "red" in df.columns and "nir" in df.columns:
        ndvi = compute_ndvi(df["red"].to_numpy(), df["nir"].to_numpy())
        df["ndvi"] = ndvi

    if "elevation" in df.columns and "slope" not in df.columns:
        elev = df["elevation"].astype(float).to_numpy()
        df["slope"] = np.gradient(elev)

    if "label" not in df.columns:
        raise ValueError("Training data must include a 'label' column")

    logger.info("Loaded %d samples with %d features", len(df), len(df.columns) - 1)
    return df


def train_model(
    data: Optional["pd.DataFrame"] = None,
    *,
    n_estimators: int = 100,
    random_state: Optional[int] = None,
    model_path: str = "brain_model.joblib",
) -> Dict[str, Any]:
    """Train a random-forest classifier on feature data.

    Parameters
    ----------
    data:
        Optional pandas DataFrame containing feature columns and a ``label``
        column. When ``None`` the dataset is loaded using
        :func:`ingest_training_data`.
    n_estimators:
        Number of trees in the forest.
    random_state:
        Optional random seed for reproducibility.
    model_path:
        File path where the trained model will be saved using ``joblib``.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the trained model object, metrics and feature
        importances.
    """

    if (
        RandomForestClassifier is None
        or np is None
        or pd is None
    ):
        raise RuntimeError("scikit-learn, pandas and NumPy are required.")

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, roc_auc_score
        import joblib
    except Exception as exc:  # pragma: no cover - library may be missing
        raise RuntimeError("scikit-learn is required for training") from exc

    if data is None:
        data = ingest_training_data()

    if "label" not in data.columns:
        raise ValueError("Input data must include a 'label' column")

    feature_names = [c for c in data.columns if c != "label"]
    X = data[feature_names].to_numpy()
    y = data["label"].to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    logger.info("Loaded %d samples, training RandomForest", len(y_train))
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_val)
    acc = float(accuracy_score(y_val, preds))
    try:
        roc = float(roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1]))
    except Exception:  # pragma: no cover - may fail with single class
        roc = float("nan")

    joblib.dump(clf, model_path)

    importances = {name: float(val) for name, val in zip(feature_names, clf.feature_importances_)}
    logger.info("Model accuracy = %.3f", acc)
    logger.info("Important features = %s", importances)

    summary = {
        "model": clf,
        "model_type": "RandomForestClassifier",
        "metrics": {"accuracy": acc, "roc_auc": roc},
        "feature_importances": importances,
        "model_path": model_path,
    }
    return summary


def score_likelihood(model: Any, data: Sequence[Sequence[float]]) -> List[float]:
    """Score feature vectors using a trained model."""
    if np is None:
        raise RuntimeError("NumPy is required.")
    X = np.array(list(data))
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.predict(X)
    logger.info("Scored %d samples", X.shape[0])
    return scores.tolist()


def validate_features(features: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return features that contain a valid geometry entry."""
    valid: List[Dict[str, Any]] = []
    for feat in features:
        geom = feat.get("geometry")
        if isinstance(geom, dict) and ("bbox" in geom or "contour" in geom):
            valid.append(feat)
    logger.info("Validated %d/%d features", len(valid), len(features))
    return valid


def cluster_features(
    features: Sequence[Sequence[float]],
    n_clusters: int = 2,
    random_state: Optional[int] = None,
) -> List[int]:
    """Cluster feature vectors using k-means."""
    if KMeans is None or np is None:
        raise RuntimeError("scikit-learn and NumPy are required.")
    X = np.array(list(features))
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(X)
    logger.info("Clustered %d samples into %d groups", X.shape[0], n_clusters)
    return labels.tolist()


def exec_code(code: str, local_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute Python code dynamically and return local variables."""
    env: Dict[str, Any] = dict(local_vars or {})
    exec(code, {}, env)
    logger.info("Executed code block with %d vars", len(env))
    return env


TOOLS: Dict[str, Any] = {
    "ingest_training_data": ingest_training_data,
    "train_model": train_model,
    "score_likelihood": score_likelihood,
    "validate_features": validate_features,
    "cluster_features": cluster_features,
    "exec_code": exec_code,
}

__all__ = [
    "ingest_training_data",
    "train_model",
    "score_likelihood",
    "validate_features",
    "cluster_features",
    "exec_code",
    "TOOLS",
]
