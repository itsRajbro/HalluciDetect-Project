"""Length-controlled test-set feasibility analysis (deterministic one-to-one matching)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp

REQUIRED = ["sample_id", "pair_id", "language", "question", "knowledge", "answer", "label"]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_test(path, expected_n=2000, expected_per_class=1000) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(rows) != expected_n:
        raise ValueError(f"test sample count {len(rows)} != {expected_n}")
    for r in rows:
        for k in ("answer", "sample_id", "pair_id"):
            if r.get(k) is None or r.get(k) == "":
                raise ValueError(f"missing {k} in row {r.get('sample_id')}")
        if r.get("label") not in (0, 1):
            raise ValueError(f"bad label in {r['sample_id']}")
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample_id")
    for lab in (0, 1):
        n = sum(r["label"] == lab for r in rows)
        if n != expected_per_class:
            raise ValueError(f"label {lab} count {n} != {expected_per_class}")
    for r in rows:
        r["answer_length_chars"] = len(r["answer"])
        r["answer_length_words"] = len(r["answer"].split())
    return rows


def match_one_to_one(factual: list[dict], halluc: list[dict], delta: int) -> list[tuple[dict, dict]]:
    """Max-cardinality, then min-total-|length gap| matching. Deterministic.

    Implemented as a full assignment where invalid edges (gap > delta) carry a
    penalty BIG > (max possible total valid cost), so the optimum uses as many
    valid edges as possible, then minimises their total gap. Invalid assignments
    are dropped afterwards. Inputs are sorted by (length, sample_id) so ties
    resolve identically on every run.
    """
    F = sorted(factual, key=lambda r: (r["answer_length_chars"], r["sample_id"]))
    H = sorted(halluc, key=lambda r: (r["answer_length_chars"], r["sample_id"]))
    lf = np.array([r["answer_length_chars"] for r in F], dtype=np.int64)
    lh = np.array([r["answer_length_chars"] for r in H], dtype=np.int64)
    gap = np.abs(lf[:, None] - lh[None, :])
    big = int(min(len(F), len(H)) * (delta + 1) + 1)
    cost = np.where(gap <= delta, gap, big)
    ri, ci = linear_sum_assignment(cost)
    return [(F[i], H[j]) for i, j in zip(ri, ci) if gap[i, j] <= delta]


def _stats(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return dict(mean=np.nan, median=np.nan, std=np.nan, min=np.nan, max=np.nan, count=0)
    return dict(mean=x.mean(), median=float(np.median(x)), std=x.std(ddof=1) if len(x) > 1 else 0.0,
                min=x.min(), max=x.max(), count=len(x))


def verify_matching(matches, delta, all_ids: set):
    fs = [f["sample_id"] for f, _ in matches]
    hs = [h["sample_id"] for _, h in matches]
    allm = fs + hs
    assert len(set(allm)) == len(allm), "sample matched more than once"
    assert len(fs) == len(hs), "unequal matched counts"
    assert all(abs(f["answer_length_chars"] - h["answer_length_chars"]) <= delta for f, h in matches), \
        "edge exceeds tolerance"
    assert all(f["label"] == 0 and h["label"] == 1 for f, h in matches), "label mismatch"
    assert set(allm) <= all_ids, "sample not from test.jsonl"


def summarize(matches, delta, n_total=2000) -> dict:
    fc = [f["answer_length_chars"] for f, _ in matches]
    hc = [h["answer_length_chars"] for _, h in matches]
    fw = [f["answer_length_words"] for f, _ in matches]
    hw = [h["answer_length_words"] for _, h in matches]
    gaps = np.abs(np.array(fc) - np.array(hc)) if matches else np.array([])
    sf, sh = _stats(fc), _stats(hc)
    if matches:
        ks = ks_2samp(fc, hc)
        ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
    else:
        ks_stat = ks_p = np.nan
    return dict(
        tolerance_chars=delta, matched_factual=len(fc), matched_hallucinated=len(hc),
        total_matched=len(fc) + len(hc), coverage_percent=100 * (len(fc) + len(hc)) / n_total,
        factual_mean_chars=sf["mean"], hallucinated_mean_chars=sh["mean"],
        factual_median_chars=sf["median"], hallucinated_median_chars=sh["median"],
        factual_std_chars=sf["std"], hallucinated_std_chars=sh["std"],
        factual_min_chars=sf["min"], hallucinated_min_chars=sh["min"],
        factual_max_chars=sf["max"], hallucinated_max_chars=sh["max"],
        mean_abs_length_gap=float(gaps.mean()) if len(gaps) else np.nan,
        median_abs_length_gap=float(np.median(gaps)) if len(gaps) else np.nan,
        max_abs_length_gap=int(gaps.max()) if len(gaps) else np.nan,
        total_abs_length_gap=int(gaps.sum()),
        ks_statistic=ks_stat, ks_pvalue=ks_p,
        factual_mean_words=_stats(fw)["mean"], hallucinated_mean_words=_stats(hw)["mean"],
        factual_median_words=_stats(fw)["median"], hallucinated_median_words=_stats(hw)["median"],
    )


def assign_match_ids(matches, delta) -> list[dict]:
    """Return per-sample records with an added match_id; originals untouched."""
    out = []
    for k, (f, h) in enumerate(sorted(matches, key=lambda m: m[0]["sample_id"])):
        mid = f"d{delta}_m{k:04d}"
        for r in (f, h):
            rec = {c: r[c] for c in REQUIRED}
            rec["match_id"] = mid
            rec["answer_length_chars"] = r["answer_length_chars"]
            out.append(rec)
    return out
