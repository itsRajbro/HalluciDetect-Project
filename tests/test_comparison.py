"""Tests for src/common/comparison.py and the evaluator additions (CIs, bin reliability, full-text errors)."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.common import comparison as C  # noqa: E402
from src.common import evaluator as E  # noqa: E402
from tests.test_evaluator import make_samples  # noqa: E402


def scored_from(samples, fn):
    return E.records_to_scores([{"sample_id": s.sample_id, "score": float(fn(i, s))} for i, s in enumerate(samples)])


def perfect(samples, flip=()):
    flip = set(flip)
    return scored_from(samples, lambda i, s: 10.0 * ((1 - s.label) if i in flip else s.label))


def noisy(samples, d, seed):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(samples))
    return scored_from(samples, lambda i, s: s.label * d + noise[i])


def as_matches(samples, n):
    out = []
    for i in range(0, 2 * n, 2):
        for s in samples[i:i + 2]:
            out.append(replace(s, match_id=f"M{i // 2:04d}"))
    return out


# ---- bootstrap CIs -----------------------------------------------------------------------
def test_perfect_detector_has_degenerate_ci():
    s = make_samples(100, "test", seed=1)
    r = C.bootstrap_ci(s, perfect(s), 5.0, n_boot=200)
    for k in ("auroc", "accuracy", "f1", "pair_accuracy"):
        m = r["metrics"][k]
        assert m["estimate"] == m["lo"] == m["hi"] == 1.0
    assert r["cluster"] == "pair" and r["n_clusters"] == 100


def test_random_detector_ci_contains_chance_and_estimate_is_inside_ci():
    s = make_samples(500, "test", seed=2)
    r = C.bootstrap_ci(s, noisy(s, 0.0, 3), 0.0, n_boot=500)
    a = r["metrics"]["auroc"]
    assert a["lo"] < 0.5 < a["hi"] and a["lo"] <= a["estimate"] <= a["hi"]
    p = r["metrics"]["pair_accuracy"]
    assert p["lo"] < 0.5 < p["hi"]


def test_ci_shrinks_with_more_data():
    small, big = make_samples(60, "test", seed=4), make_samples(600, "test", seed=4)
    w = lambda s: (lambda r: r["metrics"]["auroc"]["hi"] - r["metrics"]["auroc"]["lo"])(
        C.bootstrap_ci(s, noisy(s, 1.0, 5), 0.5, n_boot=400))
    assert w(big) < 0.6 * w(small)


def test_reproducible_with_seed_and_changes_with_other_seed():
    s = make_samples(100, "test", seed=6)
    sc = noisy(s, 1.0, 7)
    a = C.bootstrap_ci(s, sc, 0.5, n_boot=200, seed=1)
    b = C.bootstrap_ci(s, sc, 0.5, n_boot=200, seed=1)
    c = C.bootstrap_ci(s, sc, 0.5, n_boot=200, seed=2)
    assert a == b and a != c


def test_resamples_pairs_not_samples_and_match_cluster_needs_match_ids():
    s = make_samples(50, "test", seed=8)
    assert C.bootstrap_ci(s, perfect(s), 5.0, n_boot=20)["n_clusters"] == 50           # 100 samples, 50 clusters
    with pytest.raises(ValueError, match="match id"):
        C.bootstrap_ci(s, perfect(s), 5.0, cluster="match", n_boot=20)
    m = as_matches(s, 30)
    r = C.bootstrap_ci(m, perfect(m), 5.0, cluster="match", n_boot=50)
    assert r["n_clusters"] == 30 and "match_accuracy" in r["metrics"]
    with pytest.raises(ValueError):
        C.bootstrap_ci(s[:-1], perfect(s[:-1]), 5.0, n_boot=20)                          # broken pair


def test_ci_coverage_is_close_to_nominal():
    """Simulate many datasets with a known true AUROC; ~95% of CIs should contain it."""
    d = 1.5
    truth = norm.cdf(d / np.sqrt(2))
    s = make_samples(150, "test", seed=9)
    hits, trials = 0, 60
    for t in range(trials):
        r = C.bootstrap_ci(s, noisy(s, d, 1000 + t), d / 2, n_boot=150, seed=t)["metrics"]["auroc"]
        hits += r["lo"] <= truth <= r["hi"]
    assert 0.85 <= hits / trials <= 1.0


# ---- paired comparison ------------------------------------------------------------------
def test_paired_difference_detects_a_better_detector():
    s = make_samples(300, "test", seed=10)
    good, bad = noisy(s, 3.0, 11), noisy(s, 0.3, 12)
    r = C.paired_bootstrap_diff(s, good, bad, 1.5, 0.15, n_boot=300)["metric_differences_a_minus_b"]
    assert r["auroc"]["estimate"] > 0.2 and r["auroc"]["ci_excludes_zero"] and r["auroc"]["frac_a_better"] > 0.99
    flipped = C.paired_bootstrap_diff(s, bad, good, 0.15, 1.5, n_boot=300)["metric_differences_a_minus_b"]
    assert flipped["auroc"]["estimate"] == pytest.approx(-r["auroc"]["estimate"])


def test_paired_difference_of_identical_detectors_is_zero():
    s = make_samples(100, "test", seed=13)
    sc = noisy(s, 1.0, 14)
    r = C.paired_bootstrap_diff(s, sc, sc, 0.5, 0.5, n_boot=100)["metric_differences_a_minus_b"]
    for k in ("auroc", "accuracy"):
        assert r[k]["estimate"] == r[k]["lo"] == r[k]["hi"] == 0 and not r[k]["ci_excludes_zero"]


# ---- error overlap ----------------------------------------------------------------------
def test_error_overlap_counts_mcnemar_and_oracle():
    s = make_samples(50, "test", seed=15)                       # 100 samples
    a = perfect(s, flip=range(0, 10))                           # A wrong on 0..9
    b = perfect(s, flip=range(5, 15))                           # B wrong on 5..14
    r = C.error_overlap(s, {"A": (a, 5.0), "B": (b, 5.0)})
    p = r["pairwise"]["A vs B"]
    assert (p["both_wrong"], p["only_A_right"], p["only_B_right"], p["both_right"]) == (5, 5, 5, 85)
    assert p["mcnemar_exact_p"] == 1.0
    assert r["oracle_any_right_accuracy"] == pytest.approx(0.95)
    assert r["all_wrong_sample_ids"] == [x.sample_id for x in s[5:10]]
    assert r["accuracy"] == {"A": 0.90, "B": 0.90}
    # one-sided dominance is significant
    c = perfect(s, flip=range(0, 20))
    r2 = C.error_overlap(s, {"good": (perfect(s), 5.0), "bad": (c, 5.0)})
    assert r2["pairwise"]["good vs bad"]["mcnemar_exact_p"] == pytest.approx(2 * 0.5 ** 20)


# ---- evaluator integration --------------------------------------------------------------
def test_evaluate_detector_reports_confidence_intervals_and_can_skip_them():
    f = lambda a: a.answer_length_chars
    val = make_samples(200, "validation", seed=16)
    test = make_samples(200, "test", seed=17, start=5000)
    ctrl = as_matches(test, 40)
    recs = lambda xs: [{"sample_id": x.sample_id, "score": f(x)} for x in xs]
    rep = E.evaluate_detector("len", val_samples=val, val_records=recs(val), test_samples=test,
                              test_records=recs(test), ctrl_samples=ctrl, n_boot=100)
    ci = rep["test"]["confidence_intervals"]
    assert ci["cluster"] == "pair" and ci["metrics"]["auroc"]["lo"] <= rep["test"]["metrics"]["auroc"]
    assert rep["length_controlled"]["confidence_intervals"]["cluster"] == "match"
    E.print_summary(rep)
    off = E.evaluate_detector("len", val_samples=val, val_records=recs(val), test_samples=test,
                              test_records=recs(test), n_boot=0)
    assert "confidence_intervals" not in off["test"]


def test_stratified_bins_flag_unreliable_auroc():
    s = make_samples(400, "test", seed=18)
    y, x = E.align(s, scored_from(s, lambda i, a: a.answer_length_chars))
    bins = [b for b in E.length_stratified(s, y, x, 29.5, n_bins=4) if b["n"]]
    for b in bins:
        assert b["n_min"] == min(b["n_pos"], b["n_neg"])
        assert b["auroc_reliable"] == (b["n_min"] >= E.MIN_CLASS_FOR_AUROC)
    assert any(not b["auroc_reliable"] for b in bins)


def test_error_analysis_shows_full_text_by_default():
    s = make_samples(5, "test", seed=19)
    s[0] = replace(s[0], question="Q " * 300, answer="a" * 450)
    y, x = E.align(s, scored_from(s, lambda i, a: 1.0 - a.label))        # everything wrong
    out = E.error_analysis(s, y, x, 0.5, top_k=20)
    row = next(r for r in out["top_false_positives"] + out["top_false_negatives"] if r["sample_id"] == s[0].sample_id)
    assert len(row["question"]) == 600 and len(row["answer"]) == 450
    short = E.error_analysis(s, y, x, 0.5, top_k=20, max_chars=100)
    row = next(r for r in short["top_false_positives"] + short["top_false_negatives"] if r["sample_id"] == s[0].sample_id)
    assert len(row["question"]) == 100
