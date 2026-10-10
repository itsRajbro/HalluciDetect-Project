"""Tests for src/detectors/baselines/length_only.py. Run: python -m pytest tests -q"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.common import evaluator as E  # noqa: E402
from src.common.config import get_config  # noqa: E402
from src.common.schema import read_detector_output  # noqa: E402
from src.detectors.baselines import length_only as LO  # noqa: E402
from tests.test_evaluator import make_samples  # noqa: E402


def flip_lengths(samples):
    """Make the hallucinated answer the SHORT one (inverted confound)."""
    out = []
    for s in samples:
        n = 130 - s.answer_length_chars
        out.append(replace(s, answer="a" * n, answer_length_chars=n))
    return out


def test_direction_positive_when_hallucinations_are_longer():
    val = make_samples(200, "validation", seed=1)
    m = LO.fit(val)
    assert m.direction == 1 and m.val_auroc_raw > 0.98 and m.feature == "chars"
    assert 29 < m.threshold < 31


def test_direction_flips_when_hallucinations_are_shorter_and_score_convention_holds():
    val = flip_lengths(make_samples(200, "validation", seed=1))
    test = flip_lengths(make_samples(200, "test", seed=2, start=5000))
    m = LO.fit(val)
    assert m.direction == -1 and m.val_auroc_raw < 0.02
    # directed score: HIGHER = more likely hallucinated, so AUROC is high (not inverted)
    y = np.array([s.label for s in test])
    assert E.auroc(y, LO.score(test, m)) > 0.98


def test_fit_uses_validation_only_and_requires_two_classes():
    val = make_samples(100, "validation", seed=3)
    m1 = LO.fit(val)
    m2 = LO.fit(val)
    assert m1 == m2                                    # deterministic
    with pytest.raises(ValueError):
        LO.fit([s for s in val if s.label == 1])       # single class


def test_tokens_feature_requires_attached_lengths():
    val = make_samples(50, "validation", seed=4)
    with pytest.raises(ValueError, match="attach_token_lengths"):
        LO.fit(val, feature="tokens")
    val_tok = [replace(s, answer_length_tokens=s.answer_length_chars // 2) for s in val]
    assert LO.fit(val_tok, feature="tokens").feature == "tokens"
    with pytest.raises(ValueError):
        LO.lengths(val, "words")


def test_run_writes_schema_runs_with_validation_threshold_and_evaluates(tmp_path):
    cfg = get_config()
    val = make_samples(200, "validation", seed=5)
    test = make_samples(200, "test", seed=6, start=7000)
    out = LO.run(val, test, out_dir=tmp_path, cfg=cfg)
    recs, meta = read_detector_output(out["test_path"])
    assert meta.detector == "length_only" and meta.variant == "chars"
    assert meta.threshold_source == "validation" and meta.threshold == out["model"].threshold
    assert meta.extra["model"]["direction"] == 1
    assert all(r.n_generations == 0 and r.latency_s is not None for r in recs)
    rep = E.evaluate_run(val_path=out["val_path"], test_path=out["test_path"], val_samples=val,
                         test_samples=test, include_length_controlled=False)
    assert rep["detector"] == "length_only/chars" and rep["test"]["metrics"]["auroc"] > 0.98
    assert rep["threshold"]["selected_on"] == "validation"


def test_test_data_cannot_change_the_model(tmp_path):
    cfg = get_config()
    val = make_samples(150, "validation", seed=7)
    a = LO.run(val, make_samples(100, "test", seed=8, start=9000), out_dir=tmp_path / "a", cfg=cfg)
    b = LO.run(val, make_samples(100, "test", seed=9, start=9000), out_dir=tmp_path / "b", cfg=cfg)
    assert a["model"] == b["model"]
