"""Utility functions for the Brain agent."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Sequence, Optional
from collections.abc import Mapping
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from log_config import setup_logger

# Use a dedicated logger name so Brain operations are easy to trace
logger = setup_logger("casiquiare.brain")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import DBSCAN
except Exception:  # pragma: no cover - library may be missing
    RandomForestClassifier = None  # type: ignore
    DBSCAN = None  # type: ignore

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

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - library may be missing
    plt = None  # type: ignore


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

    logger.info("Reading training data from %s", csv_path)

    df = pd.read_csv(csv_path)

    if "red" in df.columns and "nir" in df.columns:
        ndvi = compute_ndvi(df["red"].to_numpy(), df["nir"].to_numpy())
        df["ndvi"] = ndvi

    if "elevation" in df.columns and "slope" not in df.columns:
        elev = df["elevation"].astype(float).to_numpy()
        df["slope"] = np.gradient(elev)

    if "label" not in df.columns:
        raise ValueError("Training data must include a 'label' column")

    sites = int(df["label"].sum()) if "label" in df.columns else 0
    non_sites = int(len(df) - sites)
    logger.info(
        "Loaded %d samples (%d sites, %d non-sites) with %d features",
        len(df),
        sites,
        non_sites,
        len(df.columns) - 1,
    )
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

    logger.info(
        "Starting model training with %d estimators (random_state=%s)",
        n_estimators,
        str(random_state),
    )

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

    strat = y if len(y) >= len(np.unique(y)) * 2 else None
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=strat
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=None
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
    try:
        size_kb = Path(model_path).stat().st_size / 1024.0
        logger.info("Saved model to %s (%.1f kB)", model_path, size_kb)
    except Exception:
        logger.info("Saved model to %s", model_path)

    importances = {name: float(val) for name, val in zip(feature_names, clf.feature_importances_)}
    logger.info("Validation accuracy = %.3f, ROC AUC = %.3f", acc, roc)
    logger.info("Important features = %s", importances)

    summary = {
        "model": clf,
        "model_type": "RandomForestClassifier",
        "metrics": {"accuracy": acc, "roc_auc": roc},
        "feature_importances": importances,
        "model_path": model_path,
    }
    logger.info("Training completed")
    return summary


def load_model(model_path: str) -> Any:
    """Load a model saved with :func:`train_model`."""

    try:
        import joblib
    except Exception as exc:  # pragma: no cover - library may be missing
        raise RuntimeError("joblib is required to load models") from exc

    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)
    logger.info("Loaded model of type %s", type(model).__name__)
    return model


def update_model(
    model: Any,
    new_data: Any,
    labels: Optional[Sequence[int]] = None,
) -> Any:
    """Incrementally update ``model`` with ``new_data`` using ``partial_fit``."""

    if np is None:
        raise RuntimeError("NumPy is required.")

    if not hasattr(model, "partial_fit"):
        raise ValueError("Model does not support incremental updates")

    X: "np.ndarray"
    y: "np.ndarray"

    if pd is not None and hasattr(new_data, "__array__") and hasattr(new_data, "columns"):
        df = new_data  # type: ignore[assignment]
        if labels is None:
            if "label" not in df.columns:
                raise ValueError("DataFrame must include a 'label' column or labels must be provided")
            y = df["label"].to_numpy()
            X = df.drop(columns=["label"]).to_numpy()
        else:
            y = np.array(list(labels))
            X = df.to_numpy()
    else:
        X = np.array(list(new_data))
        if labels is None:
            raise ValueError("labels must be provided when data is not a DataFrame with a 'label' column")
        y = np.array(list(labels))

    logger.info("Updating model with %d samples", len(y))
    if not hasattr(model, "classes_"):
        classes = np.unique(y)
        model.partial_fit(X, y, classes=classes)
    else:
        model.partial_fit(X, y)
    logger.info("Model updated")
    return model


def score_likelihood(model: Any, data: Sequence[Sequence[float]]) -> List[float]:
    """Score feature vectors using a trained model."""
    if np is None:
        raise RuntimeError("NumPy is required.")
    X = np.array(list(data))
    logger.debug("Scoring %d samples", X.shape[0])
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.predict(X)
    logger.info("Scored %d samples", X.shape[0])
    return scores.tolist()


def _parse_bbox(spec: Any) -> tuple[float, float, float, float]:
    """Return a bounding box tuple from ``spec``."""
    if isinstance(spec, (list, tuple)) and len(spec) == 4:
        return tuple(float(v) for v in spec)
    if isinstance(spec, str):
        import json
        from pathlib import Path

        path = Path(spec)
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                obj = json.load(fh)
            return _parse_bbox(obj)
    if isinstance(spec, dict):
        if "bbox" in spec and len(spec["bbox"]) == 4:
            return tuple(float(v) for v in spec["bbox"])
        if spec.get("type") == "Feature" and "geometry" in spec:
            return _parse_bbox(spec["geometry"])
        if spec.get("type") == "FeatureCollection":
            boxes = [
                _parse_bbox(f.get("geometry") or f.get("bbox"))
                for f in spec.get("features", [])
                if f.get("geometry") or f.get("bbox")
            ]
            xs = [b[0] for b in boxes] + [b[2] for b in boxes]
            ys = [b[1] for b in boxes] + [b[3] for b in boxes]
            if xs and ys:
                return min(xs), min(ys), max(xs), max(ys)
        if spec.get("type") in {"Polygon", "MultiPolygon"}:
            import numpy as _np

            coords = _np.array(spec.get("coordinates", [])).reshape(-1, 2)
            xs = coords[:, 0]
            ys = coords[:, 1]
            return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    raise ValueError("Unable to parse region spec")


def predict_sites(
    model: Any,
    input_data: Any,
    *,
    grid_size: float = 0.01,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Predict site likelihoods for feature vectors or a region.

    Parameters
    ----------
    model:
        Trained classifier supporting ``predict_proba`` or ``predict``.
    input_data:
        Either a sequence of feature vectors or a region specification. A region
        can be provided as a ``bbox`` tuple, a GeoJSON mapping or the path to a
        GeoJSON file.
    grid_size:
        Spacing in degrees between sampled points when ``input_data`` describes
        a region.
    top_n:
        Number of highest scoring locations to include in the summary.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys ``scores_map`` and ``summary`` containing the
        predicted probabilities and a summary of results.
    """

    if np is None:
        raise RuntimeError("NumPy is required.")

    input_type = "region" if not isinstance(input_data, Iterable) or isinstance(input_data, (str, bytes)) else "feature vectors"
    logger.info("Predicting sites for %s", input_type)

    # Pre-scored feature vectors
    if isinstance(input_data, Iterable) and input_data and not isinstance(input_data, (str, bytes)):
        first = next(iter(input_data))
        if isinstance(first, Mapping):
            raise TypeError(
                "input_data must be numeric feature vectors or a region specification"
            )
        if isinstance(first, Iterable) and not isinstance(first, (str, bytes)):
            scores = np.array(score_likelihood(model, input_data))
            summary = {
                "min_score": float(scores.min()),
                "max_score": float(scores.max()),
            }
            top_idx = scores.argsort()[::-1][:top_n]
            summary["top_indices"] = top_idx.tolist()
            logger.info("Scored %d feature vectors", len(scores))
            result: Dict[str, Any] = {"scores_map": scores.tolist(), "summary": summary}
            if hasattr(model, "feature_importances_"):
                result["feature_importances"] = [float(v) for v in model.feature_importances_]
            return result

    bbox = _parse_bbox(input_data)
    xs = np.arange(bbox[0], bbox[2] + grid_size, grid_size)
    ys = np.arange(bbox[1], bbox[3] + grid_size, grid_size)
    grid = np.array([(x, y) for y in ys for x in xs], dtype=float)

    scores = np.array(score_likelihood(model, grid))
    heatmap = scores.reshape(len(ys), len(xs))
    top_idx = scores.argsort()[::-1][:top_n]
    top_coords = [grid[i].tolist() for i in top_idx]
    summary = {
        "bbox": bbox,
        "min_score": float(scores.min()),
        "max_score": float(scores.max()),
        "top_coords": top_coords,
    }

    logger.info(
        "Scoring region %s – produced heatmap with shape %s", bbox, heatmap.shape
    )
    logger.debug("Top coordinates: %s", top_coords)

    result = {"scores_map": heatmap, "summary": summary}
    if hasattr(model, "feature_importances_"):
        result["feature_importances"] = [float(v) for v in model.feature_importances_]
    return result


def validate_features(features: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate Eyes detections against simple archaeological statistics.

    Parameters
    ----------
    features:
        Sequence of detection dictionaries or :class:`~detection.feature.Feature`
        objects produced by the Eyes agent.

    Returns
    -------
    list of dict
        Validation information for each feature including area ``z``-scores and
        shape checks.  Each returned dictionary contains at least the keys
        ``feature_id`` and ``size_match``.
    """

    if np is None:
        raise RuntimeError("NumPy is required.")

    logger.info("Validating %d features", len(features))

    # Reference statistics for typical archaeological mounds.  These values are
    # intentionally broad and would normally be computed from a training
    # dataset of confirmed sites.
    stats = {
        "area_mean": 5000.0,
        "area_std": 2000.0,
        "aspect_ratio_mean": 1.0,
        "aspect_ratio_std": 0.3,
    }
    typical_shapes = {"mound", "circle", "rectangle"}

    results: List[Dict[str, Any]] = []
    for item in features:
        # Support both dataclass instances and plain dictionaries.
        if hasattr(item, "__dict__") and not isinstance(item, dict):
            feat = item.__dict__  # type: ignore[assignment]
        else:
            feat = item

        fid = feat.get("id")
        geom = feat.get("geometry", {})

        # Width/height can be provided via ``dimensions`` or separate keys.
        dims = feat.get("dimensions")
        if dims and len(dims) == 2:
            width, height = float(dims[0]), float(dims[1])
        else:
            width = float(feat.get("width", 0.0))
            height = float(feat.get("height", 0.0))

        area = width * height
        aspect = width / height if height else 0.0
        z_area = (area - stats["area_mean"]) / stats["area_std"]
        z_ar = (aspect - stats["aspect_ratio_mean"]) / stats["aspect_ratio_std"]

        size_match = abs(z_area) <= 2.0
        aspect_match = abs(z_ar) <= 2.0

        shape = feat.get("shape") or feat.get("feature_type")
        shape_match = str(shape).lower() in typical_shapes if shape else False

        info = {
            "feature_id": fid,
            "area": area,
            "area_zscore": float(z_area),
            "size_match": size_match,
            "aspect_ratio": aspect,
            "aspect_zscore": float(z_ar),
            "aspect_match": aspect_match,
            "shape": shape,
            "shape_match": shape_match,
        }
        results.append(info)

        if not size_match:
            logger.info("Feature %s area %.1f is outside expected range", fid, area)
        if not shape_match:
            logger.debug("Feature %s shape %s atypical", fid, shape)

    matches = sum(1 for r in results if r["size_match"] and r["shape_match"])
    logger.info("Validated %d features, %d match typical profiles", len(results), matches)
    return results


def cluster_features(
    features: Sequence[Any],
    *,
    eps: float = 100.0,
    min_samples: int = 2,
) -> Dict[str, Any]:
    """Cluster geographic features using DBSCAN.

    Parameters
    ----------
    features:
        Sequence of feature dictionaries, dataclass instances or ``(x, y)``
        coordinate pairs.
    eps:
        Maximum distance in the same units as the coordinates for two points to
        be considered neighbors.
    min_samples:
        Minimum number of samples required to form a cluster.

    Returns
    -------
    dict
        Mapping containing ``labels`` for each feature and ``summary`` with
        cluster statistics.
    """

    if DBSCAN is None or np is None:
        raise RuntimeError("scikit-learn and NumPy are required.")

    logger.info("Clustering %d features", len(features))

    def _centroid(obj: Any) -> tuple[float, float]:
        if isinstance(obj, (list, tuple)) and len(obj) == 2:
            return float(obj[0]), float(obj[1])
        if hasattr(obj, "__dict__") and not isinstance(obj, dict):
            obj = obj.__dict__
        if isinstance(obj, dict):
            if "bbox" in obj and isinstance(obj["bbox"], (list, tuple)) and len(obj["bbox"]) == 4:
                x, y, w, h = obj["bbox"]
                return float(x + w / 2.0), float(y + h / 2.0)
            geom = obj.get("geometry", {})
            bbox = geom.get("bbox")
            if bbox and len(bbox) == 4:
                x, y, w, h = bbox
                return float(x + w / 2.0), float(y + h / 2.0)
            coords = geom.get("coordinates")
            if coords is not None:
                arr = np.array(coords, dtype=float).reshape(-1, 2)
                return float(arr[:, 0].mean()), float(arr[:, 1].mean())
            if "x" in obj and "y" in obj:
                return float(obj["x"]), float(obj["y"])
        raise ValueError("Feature lacks coordinate information")

    coords = np.array([_centroid(f) for f in features], dtype=float)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(coords)

    summary: Dict[str, Any] = {
        "num_clusters": int(len({l for l in labels if l != -1})),
        "noise": int(np.sum(labels == -1)),
        "clusters": [],
    }

    for label in sorted(set(labels)):
        if label == -1:
            continue
        idxs = np.where(labels == label)[0]
        cluster_pts = coords[idxs]
        centroid = cluster_pts.mean(axis=0)
        dists = np.linalg.norm(cluster_pts - centroid, axis=1)
        bbox = [
            float(cluster_pts[:, 0].min()),
            float(cluster_pts[:, 1].min()),
            float(cluster_pts[:, 0].max()),
            float(cluster_pts[:, 1].max()),
        ]
        ids: List[Any] = []
        for i in idxs:
            f = features[i]
            if hasattr(f, "__dict__") and not isinstance(f, dict):
                f = f.__dict__
            ids.append(f.get("id", i))
        summary["clusters"].append(
            {
                "cluster_id": int(label),
                "num_features": int(len(idxs)),
                "centroid": centroid.tolist(),
                "avg_distance": float(dists.mean()) if len(dists) > 0 else 0.0,
                "bbox": bbox,
                "features": ids,
            }
        )

    if summary["clusters"]:
        largest = max(summary["clusters"], key=lambda c: c["num_features"])
        avg_spacing = np.mean([c["avg_distance"] for c in summary["clusters"]])
        logger.info(
            "Identified %d clusters, largest has %d features, avg spacing %.1f",
            summary["num_clusters"],
            largest["num_features"],
            avg_spacing,
        )
    else:
        logger.info("No clusters found; %d noise points", summary["noise"])

    return {"labels": labels.tolist(), "summary": summary}


import multiprocessing


def _run_snippet(code: str, local_vars: Optional[Dict[str, Any]], queue: "multiprocessing.Queue") -> None:
    """Helper executed in a subprocess to run code snippets."""
    safe_builtins = {
        "print": print,
        "range": range,
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "enumerate": enumerate,
        "list": list,
        "dict": dict,
        "set": set,
        "float": float,
        "int": int,
        "str": str,
    }

    safe_globals: Dict[str, Any] = {"__builtins__": safe_builtins}
    if np is not None:
        safe_globals["np"] = np
    if pd is not None:
        safe_globals["pd"] = pd
    if plt is not None:
        safe_globals["plt"] = plt

    # Expose Brain tools except exec_code itself
    safe_globals.update({k: v for k, v in TOOLS.items() if k != "exec_code"})

    env: Dict[str, Any] = dict(local_vars or {})

    stdout_buf = StringIO()
    result: Any = None
    figures: List[str] = []
    try:
        with redirect_stdout(stdout_buf):
            exec(code, safe_globals, env)
        result = env.get("result")
        if plt is not None and plt.get_fignums():
            for num in plt.get_fignums():
                fig = plt.figure(num)
                path = f"figure_{num}.png"
                fig.savefig(path)
                figures.append(path)
            plt.close("all")
        queue.put(
            {
                "stdout": stdout_buf.getvalue(),
                "result": result,
                "figures": figures,
                "locals": env,
            }
        )
    except Exception as exc:  # pragma: no cover - runtime errors
        queue.put(
            {
                "stdout": stdout_buf.getvalue(),
                "error": str(exc),
                "result": None,
                "figures": figures,
            }
        )


def exec_code(
    code: str,
    local_vars: Optional[Dict[str, Any]] = None,
    *,
    timeout: int = 5,
) -> Dict[str, Any]:
    """Execute Python code in a restricted sandbox with a time limit."""

    snippet = code if len(code) <= 200 else code[:200] + "..."
    logger.info("Executing code snippet: %s", snippet)
    queue: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_run_snippet, args=(code, local_vars, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        logger.error("Execution timed out after %d seconds", timeout)
        proc.terminate()
        proc.join()
        return {"stdout": "", "result": None, "error": "timeout", "figures": []}

    if not queue.empty():
        return queue.get()

    return {"stdout": "", "result": None, "error": "no result", "figures": []}


TOOLS: Dict[str, Any] = {
    "ingest_training_data": ingest_training_data,
    "train_model": train_model,
    "load_model": load_model,
    "update_model": update_model,
    "score_likelihood": score_likelihood,
    "predict_sites": predict_sites,
    "validate_features": validate_features,
    "cluster_features": cluster_features,
    "exec_code": exec_code,
}

__all__ = [
    "ingest_training_data",
    "train_model",
    "load_model",
    "update_model",
    "score_likelihood",
    "predict_sites",
    "validate_features",
    "cluster_features",
    "exec_code",
    "TOOLS",
]
