"""Utilities for training and evaluating linear SEP probes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeSelection:
    layer: int
    model: Any
    validation_auc: float


def fit_layer_probe(X: np.ndarray, y: np.ndarray) -> Any:
    """Fit a scaled logistic-regression probe on one layer's features."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    if len(np.unique(y)) < 2:
        raise ValueError("y must contain both classes 0 and 1")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y must contain only labels 0 and 1")
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="liblinear", max_iter=1000, random_state=42),
    ).fit(X, y)


def select_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Choose a score threshold maximizing F1 on validation data.

    Prediction is class 1 when score >= threshold. Ties favor the larger threshold.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if y_true.ndim != 1 or scores.ndim != 1 or len(y_true) != len(scores):
        raise ValueError("y_true and scores must be 1D arrays of equal length")
    if len(y_true) == 0:
        raise ValueError("y_true and scores must not be empty")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only labels 0 and 1")
    candidates = np.unique(np.concatenate(([0.0, 1.0], scores)))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        pred = (scores >= threshold).astype(np.int64)
        value = f1_score(y_true, pred, zero_division=0)
        if value > best_f1 or (value == best_f1 and threshold > best_threshold):
            best_threshold, best_f1 = float(threshold), float(value)
    return best_threshold, best_f1


def select_layer(
    X_by_layer: np.ndarray, y: np.ndarray, layers: list[int] | None = None
) -> ProbeSelection:
    """Train a probe per layer and choose the best validation ROC AUC.

    X_by_layer shape is (samples, layers, hidden_size). Use validation data
    for layer selection; do not pass test data for this purpose.
    """
    X_by_layer = np.asarray(X_by_layer)
    y = np.asarray(y, dtype=np.int64)
    if X_by_layer.ndim != 3:
        raise ValueError("X_by_layer must have shape (samples, layers, hidden_size)")
    if X_by_layer.shape[0] != len(y):
        raise ValueError("X_by_layer and y must have the same number of samples")
    if len(np.unique(y)) != 2:
        raise ValueError("y must contain both classes 0 and 1")
    if not np.isfinite(X_by_layer).all():
        raise ValueError("X_by_layer contains non-finite values")
    layer_ids = list(range(X_by_layer.shape[1])) if layers is None else list(layers)
    if len(layer_ids) != X_by_layer.shape[1]:
        raise ValueError("layers length must match X_by_layer's layer dimension")
    best = None
    for index, layer_id in enumerate(layer_ids):
        model = fit_layer_probe(X_by_layer[:, index, :], y)
        scores = model.predict_proba(X_by_layer[:, index, :])[:, 1]
        auc = float(roc_auc_score(y, scores))
        candidate = ProbeSelection(int(layer_id), model, auc)
        if best is None or candidate.validation_auc > best.validation_auc:
            best = candidate
    if best is None:
        raise ValueError("No layers were provided")
    return best
