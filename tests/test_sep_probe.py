import numpy as np
import pytest

from src.detectors.sep.probe import fit_layer_probe, select_threshold


def test_fit_layer_probe_returns_probabilities():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 10))
    y = np.array([0, 1] * 50)
    X[y == 1, 0] += 2.0
    model = fit_layer_probe(X, y)
    probabilities = model.predict_proba(X)[:, 1]
    assert probabilities.shape == (100,)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_select_threshold_returns_valid_threshold():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    threshold, f1 = select_threshold(y_true, scores)
    assert 0.0 <= threshold <= 1.0
    assert f1 == pytest.approx(1.0)


def test_select_threshold_rejects_non_finite_scores():
    with pytest.raises(ValueError):
        select_threshold([0, 1], [0.2, np.nan])
