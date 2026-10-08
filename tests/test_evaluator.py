"""Tests for src/common/evaluator.py (Step 9). Run: python -m pytest tests -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.common import evaluator as E  # noqa: E402
from src.common.data import Sample  # noqa: E402


def make_samples(n_pairs: int, split: str, seed: int = 0, overlap: bool = False, start: int = 0):
    """Synthetic pairs. Factual answers are short; hallucinated are longer
    (the HaluEval length confound) unless overlap=True (equal-length ranges)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(start, start + n_pairs):
        lf = int(rng.integers(3, 30))
        lh = int(rng.integers(3, 30)) if overlap else int(rng.integers(30, 120))
        for lab, ln in ((0, lf), (1, lh)):
            out.append(Sample(sample_id=f"S{i:06d}_{lab}", pair_id=f"P{i:05d}", split=split, language="en",
                              question=f"Question {i}?", knowledge="k", answer="a" * ln, label=lab,
                              answer_length_chars=ln))
    return out


def recs(samples, fn):
    return [{"sample_id": s.sample_id, "score": float(fn(s)), "runtime_sec": 0.01} for s in samples]


# ---- metric core vs scikit-learn ------------------------------------------------
def test_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    x = rng.normal(size=500) + y
    pred = (x > 0.4).astype(int)
    m = E.compute_metrics(y, pred, x)
    assert m["accuracy"] == pytest.approx(accuracy_score(y, pred))
    assert m["precision"] == pytest.approx(precision_score(y, pred))
    assert m["recall"] == pytest.approx(recall_score(y, pred))
    assert m["f1"] == pytest.approx(f1_score(y, pred))
    assert m["auroc"] == pytest.approx(roc_auc_score(y, x))
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    assert m["confusion_matrix"] == {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def test_auroc_ties_and_single_class():
    y = np.array([0, 0, 1, 1])
    assert E.auroc(y, np.array([1.0, 1.0, 1.0, 1.0])) == 0.5
    assert E.auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert E.auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert E.auroc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])) is None


def test_undefined_metrics_are_none_not_zero():
    m = E.compute_metrics(np.array([0, 1, 1]), np.array([0, 0, 0]))
    assert m["precision"] is None and m["f1"] is None and m["recall"] == 0.0


# ---- threshold discipline ---------------------------------------------------------
def test_threshold_selected_on_validation_separates_classes():
    y = np.array([0, 0, 0, 1, 1, 1])
    x = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    for crit in E.CRITERIA:
        t = E.select_threshold(y, x, crit)
        assert 3.0 < t["value"] <= 10.0 and t["selected_on"] == "validation"


def test_threshold_requires_two_classes():
    with pytest.raises(ValueError):
        E.select_threshold(np.array([1, 1]), np.array([0.2, 0.3]))


def test_evaluate_requires_threshold_or_validation():
    s = make_samples(20, "test")
    with pytest.raises(ValueError):
        E.evaluate_detector("x", test_samples=s, test_records=recs(s, lambda a: a.answer_length_chars))


def test_test_set_never_influences_threshold():
    val = make_samples(200, "validation", seed=1)
    test_a = make_samples(200, "test", seed=2, start=1000)
    test_b = make_samples(200, "test", seed=3, start=1000)
    f = lambda a: a.answer_length_chars
    ra = E.evaluate_detector("len", val_samples=val, val_records=recs(val, f),
                             test_samples=test_a, test_records=recs(test_a, f))
    rb = E.evaluate_detector("len", val_samples=val, val_records=recs(val, f),
                             test_samples=test_b, test_records=recs(test_b, f))
    assert ra["threshold"]["value"] == rb["threshold"]["value"]


# ---- validation of detector output --------------------------------------------------
def test_record_validation():
    s = make_samples(5, "test")
    with pytest.raises(ValueError):
        E.records_to_scores([{"sample_id": "a", "score": float("nan")}])
    with pytest.raises(ValueError):
        E.records_to_scores([{"sample_id": "a", "score": 1.0}, {"sample_id": "a", "score": 2.0}])
    with pytest.raises(ValueError):
        E.records_to_scores([{"sample_id": "a", "score": 1.0, "pred_label": 2}])
    scored = E.records_to_scores(recs(s[:-1], lambda a: 1))
    with pytest.raises(ValueError):
        E.align(s, scored)                               # one sample unscored
    scored_extra = E.records_to_scores(recs(s, lambda a: 1) + [{"sample_id": "zzz", "score": 1.0}])
    with pytest.raises(ValueError):
        E.align(s, scored_extra)                         # unknown id


def test_records_can_be_objects():
    class R:
        def __init__(self, sid, sc): self.sample_id, self.score = sid, sc
    out = E.records_to_scores([R("a", 0.3), R("b", 1)])
    assert out["a"]["score"] == 0.3 and out["b"]["pred_label"] is None


# ---- sanity detectors --------------------------------------------------------------
def test_random_detector_is_chance():
    val = make_samples(1500, "validation", seed=4)
    test = make_samples(1500, "test", seed=5, start=5000)
    rng = np.random.default_rng(0)
    rv = [{"sample_id": s.sample_id, "score": rng.random()} for s in val]
    rt = [{"sample_id": s.sample_id, "score": rng.random()} for s in test]
    rep = E.evaluate_detector("random", val_samples=val, val_records=rv, test_samples=test, test_records=rt)
    assert abs(rep["test"]["metrics"]["auroc"] - 0.5) < 0.05
    assert abs(rep["test"]["pair_level"]["pair_accuracy"] - 0.5) < 0.05


def test_length_only_exploits_confound_and_collapses_when_controlled():
    f = lambda a: a.answer_length_chars
    # confounded data: length separates perfectly
    val = make_samples(300, "validation", seed=6)
    test = make_samples(300, "test", seed=7, start=2000)
    rep = E.evaluate_detector("length_only", val_samples=val, val_records=recs(val, f),
                              test_samples=test, test_records=recs(test, f))
    assert rep["test"]["metrics"]["auroc"] > 0.98
    # same detector, length-overlapping data: no longer separates well
    val2 = make_samples(300, "validation", seed=8, overlap=True)
    test2 = make_samples(300, "test", seed=9, overlap=True, start=2000)
    rep2 = E.evaluate_detector("length_only", val_samples=val2, val_records=recs(val2, f),
                               test_samples=test2, test_records=recs(test2, f))
    assert rep2["test"]["metrics"]["auroc"] < 0.6


def test_inverted_detector_triggers_warning():
    val = make_samples(200, "validation", seed=10)
    test = make_samples(200, "test", seed=11, start=3000)
    inv = lambda a: -a.answer_length_chars
    rep = E.evaluate_detector("inverted", threshold=-50.0, test_samples=test, test_records=recs(test, inv))
    assert rep["test"]["metrics"]["auroc"] < 0.05
    assert any("inverted" in w for w in rep["warnings"])


# ---- analyses ---------------------------------------------------------------------
def test_pair_level_and_ties():
    s = make_samples(50, "test")
    scored = E.records_to_scores(recs(s, lambda a: a.label))          # hallucinated always higher
    assert E.pair_level(s, scored)["pair_accuracy"] == 1.0
    scored = E.records_to_scores(recs(s, lambda a: 0.0))              # all tied
    pl = E.pair_level(s, scored)
    assert pl["pair_accuracy"] == 0.5 and pl["n_ties"] == 50


def test_length_stratified_counts_and_single_class_bins():
    s = make_samples(400, "test", seed=12)
    y, x = E.align(s, E.records_to_scores(recs(s, lambda a: a.answer_length_chars)))
    bins = E.length_stratified(s, y, x, threshold=29.5, n_bins=4)
    assert sum(b["n"] for b in bins) == len(s)
    assert any(b["auroc"] is None for b in bins if b["n"])           # confound: near single-class bins
    assert all(b["n_pos"] + b["n_neg"] == b["n"] for b in bins if b["n"])


def test_error_analysis_finds_the_errors():
    s = make_samples(100, "test", seed=13)
    y, x = E.align(s, E.records_to_scores(recs(s, lambda a: a.label)))
    x = x.copy()
    x[0], x[1] = 1.0, 0.0            # force one FP (index0: factual scored high) and one FN (index1)
    out = E.error_analysis(s, y, x, threshold=0.5, top_k=3)
    assert out["top_false_positives"][0]["sample_id"] == s[0].sample_id
    assert out["top_false_negatives"][0]["sample_id"] == s[1].sample_id
    assert out["length_by_outcome"]["FP"]["n"] == 1 and out["length_by_outcome"]["FN"]["n"] == 1


def test_length_controlled_and_full_report_is_json():
    f = lambda a: a.answer_length_chars
    val = make_samples(200, "validation", seed=14)
    test = make_samples(200, "test", seed=15, start=4000)
    # controlled subset: first 40 pairs, each pair is one match (as in the loader tests)
    ctrl = []
    for i in range(0, 80, 2):
        for s in test[i:i + 2]:
            ctrl.append(Sample(**{**s.__dict__, "match_id": f"M{i // 2:04d}"}))
    rep = E.evaluate_detector("length_only", val_samples=val, val_records=recs(val, f),
                              test_samples=test, test_records=recs(test, f), ctrl_samples=ctrl,
                              config_hash="abc123")
    assert rep["length_controlled"]["metrics"]["n"] == 80
    assert rep["length_controlled"]["match_level"]["n_matches"] == 40
    assert rep["efficiency"]["test_total_runtime_sec"] == pytest.approx(len(test) * 0.01)
    text = json.dumps(rep)                                            # must be serializable
    assert json.loads(text)["config_hash"] == "abc123"
    E.print_summary(rep)
