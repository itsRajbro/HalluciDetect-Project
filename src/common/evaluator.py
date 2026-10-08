"""Common evaluator shared by every detector (Step 9).

Every detector (length-only, Transformer, SE, SEP, hybrid) is evaluated by the
SAME code here, so numbers are comparable.

Conventions (see config.py)
---------------------------
* label 1 = hallucinated = positive class.
* score: HIGHER = MORE LIKELY HALLUCINATED. No sign flipping is ever done.
* prediction rule: pred = 1 if score >= threshold else 0.
* The threshold is selected on VALIDATION only, then applied unchanged to
  test (and to the length-controlled set). `evaluate_detector` never selects a
  threshold on test.

Layers
------
1. Metric core (pure numpy; takes arrays): compute_metrics, auroc,
   select_threshold, threshold_analysis.
2. Sample-aware analyses: pair_level, length_stratified, error_analysis.
3. Orchestrator: evaluate_detector(...) -> JSON-serializable report.

Detector records are read through `records_to_scores`, which accepts dicts or
objects (e.g. schema.DetectorRecord) with `sample_id`, `score` and optionally
`pred_label`, `latency_s` (or `runtime_sec`), `n_generations`,
`n_generated_tokens`, `peak_gpu_mem_mb`.

`evaluate_run` is the entry point for files written with
schema.write_detector_output: it reads and validates them, checks that the
validation and test runs come from the same detector/variant/config/prompt,
and evaluates.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from scipy.stats import rankdata

from src.common.data import Sample, group_by_match, group_by_pair

CRITERIA = ("youden", "f1", "accuracy")


# ---------------------------------------------------------------------------
# 0. Records -> scores
# ---------------------------------------------------------------------------
def _get(rec: Any, key: str, default=None):
    if isinstance(rec, dict):
        return rec.get(key, default)
    return getattr(rec, key, default)


def records_to_scores(records: Iterable[Any]) -> dict[str, dict]:
    """sample_id -> {"score": float, "pred_label": int|None, "runtime_sec": float|None}."""
    out: dict[str, dict] = {}
    for r in records:
        sid = _get(r, "sample_id")
        if sid is None:
            raise ValueError("record without sample_id")
        if sid in out:
            raise ValueError(f"duplicate record for sample_id {sid}")
        score = _get(r, "score")
        try:
            score = float(score)
        except (TypeError, ValueError):
            raise ValueError(f"{sid}: score {score!r} is not numeric")
        if not math.isfinite(score):
            raise ValueError(f"{sid}: score is NaN/inf")
        pl = _get(r, "pred_label")
        if pl is not None and int(pl) not in (0, 1):
            raise ValueError(f"{sid}: pred_label {pl!r} not in {{0,1}}")
        rt = _get(r, "latency_s")            # Step 8 schema name
        if rt is None:
            rt = _get(r, "runtime_sec")      # legacy / ad-hoc name
        out[sid] = {"score": score, "pred_label": None if pl is None else int(pl),
                    "runtime_sec": rt,
                    "n_generations": _get(r, "n_generations"),
                    "n_generated_tokens": _get(r, "n_generated_tokens"),
                    "peak_gpu_mem_mb": _get(r, "peak_gpu_mem_mb")}
    return out


def align(samples: Sequence[Sample], scored: dict[str, dict], *, allow_extra: bool = False):
    """Return (labels, scores) arrays in `samples` order; require exact id coverage."""
    ids = [s.sample_id for s in samples]
    missing = [i for i in ids if i not in scored]
    if missing:
        raise ValueError(f"{len(missing)} samples have no detector score (e.g. {missing[:3]})")
    if not allow_extra:
        extra = set(scored) - set(ids)
        if extra:
            raise ValueError(f"{len(extra)} scored ids are not in this sample set (e.g. {sorted(extra)[:3]})")
    y = np.array([s.label for s in samples], dtype=int)
    x = np.array([scored[i]["score"] for i in ids], dtype=float)
    return y, x


# ---------------------------------------------------------------------------
# 1. Metric core
# ---------------------------------------------------------------------------
def auroc(y: np.ndarray, x: np.ndarray) -> Optional[float]:
    """AUROC with label 1 positive, ties counted half. None if one class only."""
    y = np.asarray(y)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(x)  # average ranks for ties
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _safe_div(a: float, b: float) -> Optional[float]:
    return None if b == 0 else a / b


def compute_metrics(y: np.ndarray, pred: np.ndarray, scores: Optional[np.ndarray] = None) -> dict:
    """Accuracy, precision, recall, F1, confusion matrix and (if scores) AUROC.

    Undefined values (e.g. precision with no positive predictions) are None,
    not silently 0.
    """
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    n = tp + fp + fn + tn
    prec, rec = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    f1 = None if (prec is None or rec is None or (prec + rec) == 0) else 2 * prec * rec / (prec + rec)
    m = {
        "n": n, "n_pos": tp + fn, "n_neg": tn + fp,
        "accuracy": _safe_div(tp + tn, n), "precision": prec, "recall": rec, "f1": f1,
        "specificity": _safe_div(tn, tn + fp),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "auroc": None if scores is None else auroc(y, np.asarray(scores, float)),
    }
    return m


def _criterion_value(m: dict, criterion: str) -> float:
    if criterion == "accuracy":
        v = m["accuracy"]
    elif criterion == "f1":
        v = m["f1"]
    elif criterion == "youden":
        v = None if (m["recall"] is None or m["specificity"] is None) else m["recall"] + m["specificity"] - 1
    else:
        raise ValueError(f"criterion must be one of {CRITERIA}")
    return -math.inf if v is None else float(v)


def candidate_thresholds(x: np.ndarray) -> np.ndarray:
    """Midpoints between consecutive distinct scores (generalise better than raw scores)."""
    u = np.unique(np.asarray(x, float))
    if len(u) == 1:
        return u
    return (u[:-1] + u[1:]) / 2.0


def select_threshold(y_val: np.ndarray, x_val: np.ndarray, criterion: str = "youden") -> dict:
    """Pick the threshold on VALIDATION data. pred = 1 if score >= threshold.

    Ties between equally good thresholds resolve to the lowest threshold
    (deterministic). Default criterion is Youden's J (recall + specificity - 1),
    which does not reward the degenerate 'predict everything positive' rule the
    way F1 can.
    """
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}")
    y_val, x_val = np.asarray(y_val, int), np.asarray(x_val, float)
    if len(np.unique(y_val)) < 2:
        raise ValueError("validation labels contain a single class; cannot select a threshold")
    best_t, best_v = None, -math.inf
    for t in candidate_thresholds(x_val):
        v = _criterion_value(compute_metrics(y_val, (x_val >= t).astype(int)), criterion)
        if v > best_v:
            best_t, best_v = float(t), v
    return {"value": best_t, "criterion": criterion, "criterion_value_on_validation": best_v,
            "selected_on": "validation"}


def threshold_analysis(y: np.ndarray, x: np.ndarray, n_points: int = 21) -> list[dict]:
    """Metrics at evenly spaced score quantiles (for threshold curves)."""
    y, x = np.asarray(y, int), np.asarray(x, float)
    qs = np.linspace(0, 1, n_points)
    rows, seen = [], set()
    for t in np.quantile(x, qs):
        t = float(t)
        if t in seen:
            continue
        seen.add(t)
        m = compute_metrics(y, (x >= t).astype(int))
        rows.append({"threshold": t, **{k: m[k] for k in ("accuracy", "precision", "recall", "f1", "specificity")}})
    return rows


# ---------------------------------------------------------------------------
# 2. Sample-aware analyses
# ---------------------------------------------------------------------------
def pair_level(samples: Sequence[Sample], scored: dict[str, dict]) -> dict:
    """Does the detector score the hallucinated answer above the factual answer
    to the SAME question? Ties count 0.5. Threshold-free."""
    pairs = group_by_pair(samples)
    wins = ties = 0
    for f, h in pairs.values():
        sf, sh = scored[f.sample_id]["score"], scored[h.sample_id]["score"]
        if sh > sf:
            wins += 1
        elif sh == sf:
            ties += 1
    n = len(pairs)
    return {"n_pairs": n, "pair_accuracy": (wins + 0.5 * ties) / n if n else None, "n_ties": ties}


def match_level(ctrl_samples: Sequence[Sample], scored: dict[str, dict]) -> dict:
    """Same idea for the length-controlled set (pairs matched on length)."""
    matches = group_by_match(ctrl_samples)
    wins = ties = 0
    for f, h in matches.values():
        sf, sh = scored[f.sample_id]["score"], scored[h.sample_id]["score"]
        wins += sh > sf
        ties += sh == sf
    n = len(matches)
    return {"n_matches": n, "match_accuracy": (wins + 0.5 * ties) / n if n else None, "n_ties": int(ties)}


def length_edges(lengths: np.ndarray, n_bins: int = 4) -> list[float]:
    """Quantile bin edges (de-duplicated). Compute once and reuse across detectors."""
    qs = np.linspace(0, 1, n_bins + 1)
    return [float(e) for e in np.unique(np.quantile(np.asarray(lengths, float), qs))]


def length_stratified(
    samples: Sequence[Sample], y: np.ndarray, x: np.ndarray, threshold: float,
    *, edges: Optional[Sequence[float]] = None, n_bins: int = 4,
) -> list[dict]:
    """Metrics per answer-length bin (characters).

    In HaluEval the label is strongly tied to length, so many bins are almost
    single-class: AUROC is then None and class counts are reported so the
    reader can see it. A detector that only wins in unbalanced bins is
    exploiting length.
    """
    lengths = np.array([s.answer_length_chars for s in samples], float)
    edges = list(edges) if edges is not None else length_edges(lengths, n_bins)
    if len(edges) < 2:
        edges = [edges[0], edges[0]]
    inner = np.array(edges[1:-1], float)
    idx = np.searchsorted(inner, lengths, side="right")
    out = []
    for b in range(len(edges) - 1):
        mask = idx == b
        last = b == len(edges) - 2
        label = f"[{edges[b]:g}, {edges[b + 1]:g}{']' if last else ')'}"
        if not mask.any():
            out.append({"bin": label, "n": 0})
            continue
        m = compute_metrics(y[mask], (x[mask] >= threshold).astype(int), x[mask])
        out.append({"bin": label, "n": m["n"], "n_pos": m["n_pos"], "n_neg": m["n_neg"],
                    "frac_hallucinated": m["n_pos"] / m["n"], "accuracy": m["accuracy"],
                    "recall": m["recall"], "specificity": m["specificity"], "auroc": m["auroc"]})
    return out


def error_analysis(samples: Sequence[Sample], y: np.ndarray, x: np.ndarray, threshold: float,
                   *, top_k: int = 10, max_chars: int = 200) -> dict:
    """Most confident false positives / false negatives, plus length by outcome."""
    pred = (x >= threshold).astype(int)
    lengths = np.array([s.answer_length_chars for s in samples], float)
    groups = {"TP": (pred == 1) & (y == 1), "TN": (pred == 0) & (y == 0),
              "FP": (pred == 1) & (y == 0), "FN": (pred == 0) & (y == 1)}
    length_by_outcome = {k: (None if not m.any() else {"n": int(m.sum()), "mean_len_chars": float(lengths[m].mean()),
                                                         "median_len_chars": float(np.median(lengths[m]))})
                         for k, m in groups.items()}

    def rows(mask, descending):
        order = np.argsort(-x if descending else x, kind="stable")
        picked = [i for i in order if mask[i]][:top_k]
        return [{"sample_id": samples[i].sample_id, "pair_id": samples[i].pair_id, "score": float(x[i]),
                 "answer_length_chars": samples[i].answer_length_chars,
                 "question": samples[i].question[:max_chars], "answer": samples[i].answer[:max_chars]}
                for i in picked]

    return {"length_by_outcome": length_by_outcome,
            "top_false_positives": rows(groups["FP"], True),    # factual answers scored most hallucinated
            "top_false_negatives": rows(groups["FN"], False)}   # hallucinations scored most factual


# ---------------------------------------------------------------------------
# 3. Orchestrator
# ---------------------------------------------------------------------------
def _split_report(samples, scored, threshold, *, n_bins, top_k, edges=None, native=True) -> dict:
    y, x = align(samples, scored)
    pred = (x >= threshold).astype(int)
    rep = {"metrics": compute_metrics(y, pred, x),
           "pair_level": pair_level(samples, scored),
           "length_stratified": length_stratified(samples, y, x, threshold, edges=edges, n_bins=n_bins),
           "threshold_analysis": threshold_analysis(y, x),
           "errors": error_analysis(samples, y, x, threshold, top_k=top_k)}
    if native and all(scored[s.sample_id]["pred_label"] is not None for s in samples):
        npred = np.array([scored[s.sample_id]["pred_label"] for s in samples], int)
        rep["metrics_native_pred_label"] = compute_metrics(y, npred, x)
    return rep


def evaluate_detector(
    detector_name: str,
    *,
    test_samples: Sequence[Sample],
    test_records: Iterable[Any],
    val_samples: Optional[Sequence[Sample]] = None,
    val_records: Optional[Iterable[Any]] = None,
    threshold: Optional[float] = None,
    criterion: str = "youden",
    ctrl_samples: Optional[Sequence[Sample]] = None,
    n_bins: int = 4,
    top_k: int = 10,
    config_hash: Optional[str] = None,
) -> dict:
    """Full evaluation of one detector. Returns a JSON-serializable report.

    Threshold: pass `threshold` (a frozen value) OR give val_samples/val_records
    so it is selected on validation. It is never selected on test.
    `ctrl_samples` (optional) is the length-controlled set; its records are
    read from `test_records` (it is a subset of test).
    """
    warnings: list[str] = []
    test_scored = records_to_scores(test_records)

    if threshold is None:
        if val_samples is None or val_records is None:
            raise ValueError("provide `threshold`, or validation samples+records to select it on validation")
        val_scored = records_to_scores(val_records)
        y_val, x_val = align(val_samples, val_scored)
        thr = select_threshold(y_val, x_val, criterion)
    else:
        thr = {"value": float(threshold), "criterion": "frozen_given", "selected_on": "provided"}
        val_scored = records_to_scores(val_records) if (val_samples is not None and val_records is not None) else None

    t = thr["value"]
    report: dict = {"detector": detector_name, "config_hash": config_hash, "threshold": thr,
                    "score_convention": "higher score = more likely hallucinated; pred = score >= threshold"}

    # shared length-bin edges, from VALIDATION lengths when available (never from test labels)
    ref = val_samples if val_samples is not None else test_samples
    edges = length_edges(np.array([s.answer_length_chars for s in ref], float), n_bins)
    report["length_bin_edges_chars"] = edges

    if val_samples is not None and val_scored is not None:
        report["validation"] = _split_report(val_samples, val_scored, t, n_bins=n_bins, top_k=top_k, edges=edges)
    report["test"] = _split_report(test_samples, test_scored, t, n_bins=n_bins, top_k=top_k, edges=edges)

    if ctrl_samples is not None:
        yc, xc = align(ctrl_samples, test_scored, allow_extra=True)
        report["length_controlled"] = {
            "metrics": compute_metrics(yc, (xc >= t).astype(int), xc),
            "match_level": match_level(ctrl_samples, test_scored)}

    runtimes = [v["runtime_sec"] for v in test_scored.values() if v["runtime_sec"] is not None]
    if runtimes:
        eff = {"test_total_runtime_sec": float(sum(runtimes)),
               "test_mean_runtime_sec_per_sample": float(sum(runtimes) / len(runtimes))}
        gens = [v["n_generations"] for v in test_scored.values() if v["n_generations"] is not None]
        toks = [v["n_generated_tokens"] for v in test_scored.values() if v["n_generated_tokens"] is not None]
        mems = [v["peak_gpu_mem_mb"] for v in test_scored.values() if v["peak_gpu_mem_mb"] is not None]
        if gens:
            eff["test_total_generations"] = int(sum(gens))
        if toks:
            eff["test_total_generated_tokens"] = int(sum(toks))
        if mems:
            eff["test_peak_gpu_mem_mb"] = float(max(mems))
        report["efficiency"] = eff
    else:
        warnings.append("no latency_s/runtime_sec in records; efficiency not reported")

    auc = report["test"]["metrics"]["auroc"]
    if auc is not None and auc < 0.45:
        warnings.append(f"test AUROC={auc:.3f} is well below 0.5: score direction may be inverted "
                        f"(convention: higher = more hallucinated)")
    report["warnings"] = warnings
    return report


def evaluate_run(
    *,
    test_path,
    val_path=None,
    val_samples: Optional[Sequence[Sample]] = None,
    test_samples: Optional[Sequence[Sample]] = None,
    ctrl_samples: Optional[Sequence[Sample]] = None,
    include_length_controlled: bool = True,
    threshold: Optional[float] = None,
    criterion: str = "youden",
    n_bins: int = 4,
    top_k: int = 10,
    save: bool = False,
    out_path=None,
) -> dict:
    """Evaluate detector runs written with schema.write_detector_output.

    Reads + validates the run files, checks that validation and test runs share
    detector / variant / config_hash / prompt_version (otherwise the numbers are
    not comparable), and requires exact sample coverage for each split.

    Threshold precedence: explicit `threshold` argument > threshold stored in the
    TEST run's meta (schema guarantees it came from validation) > selected here on
    the validation run. It is never selected on test.

    Samples default to the dataset interface (load_split / load_length_controlled).
    """
    from src.common.config import REPO_ROOT, get_config
    from src.common.data import load_length_controlled, load_split
    from src.common.schema import check_coverage, read_detector_output

    warnings: list[str] = []
    test_recs, test_meta = read_detector_output(test_path)
    if test_meta.split != "test":
        raise ValueError(f"{test_path} is a {test_meta.split!r} run, expected 'test'")
    test_samples = list(test_samples) if test_samples is not None else load_split("test")
    check_coverage(test_recs, [s.sample_id for s in test_samples], strict=True)

    val_recs = None
    if val_path is not None:
        val_recs, val_meta = read_detector_output(val_path)
        if val_meta.split != "validation":
            raise ValueError(f"{val_path} is a {val_meta.split!r} run, expected 'validation'")
        for key in ("detector", "variant", "config_hash", "prompt_version"):
            if getattr(val_meta, key) != getattr(test_meta, key):
                raise ValueError(f"validation and test runs differ in {key}: "
                                 f"{getattr(val_meta, key)!r} vs {getattr(test_meta, key)!r}")
        val_samples = list(val_samples) if val_samples is not None else load_split("validation")
        check_coverage(val_recs, [s.sample_id for s in val_samples], strict=True)
    else:
        val_samples = None

    threshold_from_meta = False
    if threshold is None and test_meta.threshold is not None:
        threshold = test_meta.threshold
        threshold_from_meta = True
    if threshold is None and val_recs is None:
        raise ValueError("no threshold: pass `threshold`, a test run written with a threshold, or `val_path`")

    if ctrl_samples is None and include_length_controlled:
        try:
            ctrl_samples = load_length_controlled()
        except FileNotFoundError:
            warnings.append("length-controlled set not found; skipped")

    name = f"{test_meta.detector}/{test_meta.variant}"
    report = evaluate_detector(
        name, test_samples=test_samples, test_records=test_recs,
        val_samples=val_samples, val_records=val_recs, threshold=threshold, criterion=criterion,
        ctrl_samples=ctrl_samples, n_bins=n_bins, top_k=top_k, config_hash=test_meta.config_hash)
    report["prompt_version"] = test_meta.prompt_version
    if threshold_from_meta:
        report["threshold"]["criterion"] = "stored_in_run_meta"
        report["threshold"]["selected_on"] = test_meta.threshold_source   # always "validation" (schema-enforced)
    report["warnings"] = warnings + report["warnings"]
    if save:
        cfg = get_config()
        path = Path(out_path) if out_path else (
            REPO_ROOT / cfg.paths.results_dir / "eval" / test_meta.detector / f"{test_meta.variant}.json")
        save_report(report, path)
    return report


def save_report(report: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def print_summary(report: dict) -> None:
    def f(v):
        return "n/a" if v is None else f"{v:.4f}"
    print(f"== {report['detector']}  (threshold {report['threshold']['value']:.4g}, "
          f"{report['threshold'].get('criterion')}) ==")
    for part in ("validation", "test"):
        if part in report:
            m = report[part]["metrics"]
            print(f"{part:<11} acc={f(m['accuracy'])} P={f(m['precision'])} R={f(m['recall'])} "
                  f"F1={f(m['f1'])} AUROC={f(m['auroc'])}  pair_acc={f(report[part]['pair_level']['pair_accuracy'])}")
    if "length_controlled" in report:
        m = report["length_controlled"]["metrics"]
        print(f"len-ctrl    acc={f(m['accuracy'])} AUROC={f(m['auroc'])}  "
              f"match_acc={f(report['length_controlled']['match_level']['match_accuracy'])}")
    for w in report["warnings"]:
        print("WARNING:", w)
