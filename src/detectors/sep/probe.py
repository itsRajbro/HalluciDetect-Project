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
    X_train_by_layer: np.ndarray,
    y_train: np.ndarray,
    X_val_by_layer: np.ndarray,
    y_val: np.ndarray,
    layers: list[int] | None = None,
) -> ProbeSelection:
    """Fit each layer's probe on training data and select by validation ROC AUC.

    Both feature arrays must have shape (samples, layers, hidden_size).
    Validation data is used only to compare trained probes, never to fit them.
    Do not use test data for layer selection.
    """
    X_train_by_layer = np.asarray(X_train_by_layer, dtype=np.float64)
    X_val_by_layer = np.asarray(X_val_by_layer, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)

    if X_train_by_layer.ndim != 3 or X_val_by_layer.ndim != 3:
        raise ValueError(
            "Training and validation features must be 3D arrays "
            "(samples, layers, hidden_size)"
        )

    if X_train_by_layer.shape[0] != len(y_train):
        raise ValueError("Training features and labels have different sample counts")

    if X_val_by_layer.shape[0] != len(y_val):
        raise ValueError("Validation features and labels have different sample counts")

    if X_train_by_layer.shape[1:] != X_val_by_layer.shape[1:]:
        raise ValueError("Training and validation layer/hidden dimensions must match")

    if len(y_train) == 0 or len(y_val) == 0:
        raise ValueError("Training and validation data must not be empty")

    if not np.isin(y_train, [0, 1]).all():
        raise ValueError("y_train must contain only labels 0 and 1")

    if not np.isin(y_val, [0, 1]).all():
        raise ValueError("y_val must contain only labels 0 and 1")

    if len(np.unique(y_train)) != 2:
        raise ValueError("y_train must contain both classes 0 and 1")

    if len(np.unique(y_val)) != 2:
        raise ValueError("y_val must contain both classes 0 and 1")

    if not np.isfinite(X_train_by_layer).all():
        raise ValueError("Training features contain non-finite values")

    if not np.isfinite(X_val_by_layer).all():
        raise ValueError("Validation features contain non-finite values")

    n_layers = X_train_by_layer.shape[1]
    layer_ids = list(range(n_layers)) if layers is None else list(layers)

    if len(layer_ids) != n_layers:
        raise ValueError("layers length must match the layer dimension")

    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError("layers must contain unique layer IDs")

    best = None

    for index, layer_id in enumerate(layer_ids):
        model = fit_layer_probe(X_train_by_layer[:, index, :], y_train)

        val_scores = model.predict_proba(
            X_val_by_layer[:, index, :]
        )[:, 1]

        val_auc = float(roc_auc_score(y_val, val_scores))
        candidate = ProbeSelection(int(layer_id), model, val_auc)

        if best is None or candidate.validation_auc > best.validation_auc:
            best = candidate

    if best is None:
        raise ValueError("No layers were provided")

    return best
