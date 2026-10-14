"""Uncertainty and comparison tools shared by every detector (Person 1).

Why this exists
---------------
With ~1,000 test questions (and only 137 matched pairs in the length-controlled
set) many differences between detectors are within sampling noise. Every number
that is compared should carry a confidence interval, and every "A beats B"
claim should come from a PAIRED comparison on the same samples.

Resampling unit
---------------
The two answers of one question are not independent, so the bootstrap resamples
CLUSTERS, never single samples:
  * cluster="pair"   : the factual + hallucinated answer of one question (test / validation)
  * cluster="match"  : one length-matched factual/hallucinated couple (length-controlled set)
  * cluster="sample" : plain sample bootstrap (only when there is no pair structure)

Functions
---------
bootstrap_ci            percentile CIs for AUROC, accuracy, F1 and pair/match accuracy
paired_bootstrap_diff   CI for metric(A) - metric(B) using the SAME resampled clusters
error_overlap           who is right/wrong where, exact McNemar test, oracle-union accuracy
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.stats import binomtest

from src.common import evaluator as E
from src.common.data import Sample

CLUSTERS = ("pair", "match", "sample")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _cluster_index(samples: Sequence[Sample], cluster: str) -> np.ndarray:
    """(n_clusters, k) integer array of sample positions, k constant."""
    if cluster not in CLUSTERS:
        raise ValueError(f"cluster must be one of {CLUSTERS}")
    if cluster == "sample":
        return np.arange(len(samples)).reshape(-1, 1)
    key = (lambda s: s.pair_id) if cluster == "pair" else (lambda s: s.match_id)
    groups: dict = {}
    for i, s in enumerate(samples):
        k = key(s)
        if k is None:
            raise ValueError(f"sample {s.sample_id} has no {cluster} id")
        groups.setdefault(k, []).append(i)
    sizes = {len(v) for v in groups.values()}
    if sizes != {2}:
        raise ValueError(f"every {cluster} must contain exactly 2 samples; found sizes {sorted(sizes)}")
    return np.array(list(groups.values()), dtype=int)


def _win_vector_aligned(samples, scored, cluster, idx) -> Optional[np.ndarray]:
    """Win vector in the SAME cluster order as `idx` (clusters keyed by first member)."""
    if cluster == "sample":
        return None
    w = np.zeros(len(idx))
    for c, members in enumerate(idx):
        a, b = samples[members[0]], samples[members[1]]
        f, h = (a, b) if a.label == 0 else (b, a)
        sf, sh = scored[f.sample_id]["score"], scored[h.sample_id]["score"]
        w[c] = 1.0 if sh > sf else 0.5 if sh == sf else 0.0
    return w


def _f1(y: np.ndarray, pred: np.ndarray) -> Optional[float]:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    if tp == 0:
        return 0.0 if (fp + fn) > 0 else None
    return 2 * tp / (2 * tp + fp + fn)


def _point_and_boot(y, x, threshold, win, idx, n_boot, rng):
    n_c = len(idx)
    pred_all = (x >= threshold).astype(int)

    def metrics(sel_clusters):
        sel = idx[sel_clusters].ravel()
        ys, xs, ps = y[sel], x[sel], pred_all[sel]
        out = {"auroc": E.auroc(ys, xs), "accuracy": float((ps == ys).mean()), "f1": _f1(ys, ps)}
        if win is not None:
            out["pair_like_accuracy"] = float(win[sel_clusters].mean())
        return out

    point = metrics(np.arange(n_c))
    boots = {k: [] for k in point}
    for _ in range(n_boot):
        m = metrics(rng.integers(0, n_c, n_c))
        for k, v in m.items():
            if v is not None:
                boots[k].append(v)
    return point, {k: np.array(v) for k, v in boots.items()}


def _ci(arr: np.ndarray, alpha: float):
    if len(arr) == 0:
        return None, None
    return float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def bootstrap_ci(
    samples: Sequence[Sample], scored: dict, threshold: float, *,
    cluster: str = "pair", n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap CIs, resampling clusters (see module docstring).

    `scored` is the output of evaluator.records_to_scores. The threshold is
    applied as-is (it is never re-selected on the resample).
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    samples = list(samples)
    y, x = E.align(samples, scored, allow_extra=True)
    idx = _cluster_index(samples, cluster)
    win = _win_vector_aligned(samples, scored, cluster, idx)
    rng = np.random.default_rng(seed)
    point, boots = _point_and_boot(y, x, threshold, win, idx, n_boot, rng)
    names = {"pair": "pair_accuracy", "match": "match_accuracy"}
    out = {}
    for k, est in point.items():
        lo, hi = _ci(boots[k], alpha)
        key = names.get(cluster, k) if k == "pair_like_accuracy" else k
        out[key] = {"estimate": est, "lo": lo, "hi": hi}
    return {"cluster": cluster, "n_clusters": int(len(idx)), "n_boot": n_boot, "alpha": alpha, "seed": seed,
            "metrics": out}


def paired_bootstrap_diff(
    samples: Sequence[Sample], scored_a: dict, scored_b: dict, threshold_a: float, threshold_b: float, *,
    cluster: str = "pair", n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> dict:
    """CI for metric(A) - metric(B) on identical resampled clusters.

    Positive difference = A better. `ci_excludes_zero` is the usual reading of
    "significant at level alpha". Uses AUROC and accuracy (at each detector's own
    validation threshold).
    """
    samples = list(samples)
    y, xa = E.align(samples, scored_a, allow_extra=True)
    _, xb = E.align(samples, scored_b, allow_extra=True)
    idx = _cluster_index(samples, cluster)
    n_c = len(idx)
    pa, pb = (xa >= threshold_a).astype(int), (xb >= threshold_b).astype(int)

    def diffs(sel_clusters):
        sel = idx[sel_clusters].ravel()
        ys = y[sel]
        da = E.auroc(ys, xa[sel])
        db = E.auroc(ys, xb[sel])
        return (None if da is None or db is None else da - db,
                float((pa[sel] == ys).mean() - (pb[sel] == ys).mean()))

    point = diffs(np.arange(n_c))
    rng = np.random.default_rng(seed)
    boots = {"auroc": [], "accuracy": []}
    for _ in range(n_boot):
        da, dacc = diffs(rng.integers(0, n_c, n_c))
        if da is not None:
            boots["auroc"].append(da)
        boots["accuracy"].append(dacc)
    out = {}
    for name, est in zip(("auroc", "accuracy"), point):
        arr = np.array(boots[name])
        lo, hi = _ci(arr, alpha)
        out[name] = {"estimate": est, "lo": lo, "hi": hi,
                     "ci_excludes_zero": bool(lo is not None and (lo > 0 or hi < 0)),
                     "frac_a_better": float((arr > 0).mean()) if len(arr) else None}
    return {"cluster": cluster, "n_clusters": int(n_c), "n_boot": n_boot, "alpha": alpha, "seed": seed,
            "metric_differences_a_minus_b": out}


def error_overlap(samples: Sequence[Sample], detectors: dict) -> dict:
    """Where do detectors err, and do they err on the same samples?

    `detectors`: name -> (scored dict, threshold). Reports, for each pair of
    detectors, the 2x2 right/wrong table, the exact McNemar p-value for "different
    error rates", and the accuracy of an ORACLE that is right whenever any detector
    is right (an upper bound on what combining them could achieve).
    """
    samples = list(samples)
    correct = {}
    for name, (scored, thr) in detectors.items():
        y, x = E.align(samples, scored, allow_extra=True)
        correct[name] = ((x >= thr).astype(int) == y)
    names = list(correct)
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ca, cb = correct[a], correct[b]
            only_a_right = int((ca & ~cb).sum())
            only_b_right = int((~ca & cb).sum())
            disc = only_a_right + only_b_right
            pairs[f"{a} vs {b}"] = {
                "both_right": int((ca & cb).sum()), "both_wrong": int((~ca & ~cb).sum()),
                f"only_{a}_right": only_a_right, f"only_{b}_right": only_b_right,
                "mcnemar_exact_p": 1.0 if disc == 0 else float(binomtest(min(only_a_right, only_b_right), disc, 0.5).pvalue),
            }
    stacked = np.vstack([correct[n] for n in names])
    return {
        "n": len(samples),
        "accuracy": {n: float(correct[n].mean()) for n in names},
        "pairwise": pairs,
        "oracle_any_right_accuracy": float(stacked.any(axis=0).mean()),
        "all_wrong_sample_ids": [s.sample_id for s, w in zip(samples, ~stacked.any(axis=0)) if w],
        "wrong_by_detector": {n: [s.sample_id for s, c in zip(samples, correct[n]) if not c] for n in names},
    }
