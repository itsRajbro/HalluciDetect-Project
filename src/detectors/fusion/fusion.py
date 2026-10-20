"""Late fusion of detector scores (Person 1). Implements docs/fusion_analysis_plan.md.

    S_hybrid = w0 + w1 * S_1 + w2 * S_2 [+ ...]      (inputs standardized, higher = more hallucinated)

What this module does
  1. reads standard schema runs (validation + test) of each input detector, and checks
     coverage, split names and provenance;
  2. optionally transforms an input (``logit`` for probability outputs such as SEP);
  3. standardizes every input with VALIDATION mean/sd (same constants applied to test);
  4. fits an L2 logistic regression on VALIDATION only; C is chosen by 5-fold CV grouped by
     ``pair_id`` (both answers of a question always fall in the same fold);
  5. selects the decision threshold on the OUT-OF-FOLD validation scores (Youden by default);
  6. writes the hybrid (fitted) and an unfitted equal-weight average as ordinary detector runs.

Test labels are never read by ``run_fusion``: test samples are used only for their ids and
answer lengths. (Labels are only used afterwards by ``analyze`` / the evaluator.)

Notes
  * Fusion needs validation AND test scores of every input on exactly the same samples.
  * A pair-constant input (e.g. global SE) cancels inside a pair, so it cannot change pair-level
    accuracy; it can only help by calibrating across questions (plan, section 2).
  * Validation scores written for the hybrid are the out-of-fold ones (honest); test scores
    come from the model fitted on all of validation.
"""
from __future__ import annotations

import re
import warnings
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.common import comparison as C
from src.common import evaluator as E
from src.common.config import Config, get_config
from src.common.data import Sample
from src.common.schema import DetectorRecord, check_coverage, read_detector_output, write_detector_output

DETECTOR = "hybrid"
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
N_FOLDS = 5
LOGIT_EPS = 1e-12      # keeps resolution for very confident probabilities (float64 resolves ~1e-16)
TRANSFORMS = ("none", "logit")
_NAME_RE = re.compile(r"^[A-Za-z0-9.]+$")
LENGTH_FEATURE = "len"


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InputSpec:
    name: str                 # short alias used in variant names, e.g. "se", "sep"
    val_path: Path
    test_path: Path
    transform: str = "none"   # "logit" for probability outputs (saturate near 0/1)

    def __post_init__(self):
        if not _NAME_RE.match(self.name):
            raise ValueError(f"input name {self.name!r} must match [A-Za-z0-9.]+ (no underscores: they are used as separators)")
        if self.name == LENGTH_FEATURE:
            raise ValueError(f"input name {LENGTH_FEATURE!r} is reserved for the length feature")
        if self.transform not in TRANSFORMS:
            raise ValueError(f"transform must be one of {TRANSFORMS}")


@dataclass
class LoadedInput:
    spec: InputSpec
    val_raw: np.ndarray
    test_raw: np.ndarray
    val_meta: object
    test_meta: object
    own_threshold: Optional[float]      # threshold the owner selected on validation (None if none)
    val_latency: Optional[np.ndarray]   # per-sample amortized latency (pair cost split across the pair)
    test_latency: Optional[np.ndarray]
    cost_test: dict                     # exact test-split cost totals of the WHOLE test run (pair-level counted once)
    val_has: Optional[np.ndarray] = None    # which validation samples this input covers (all True unless partial)
    test_has: Optional[np.ndarray] = None


def _amortized_latency(recs, meta, samples) -> Optional[np.ndarray]:
    """Per-sample latency; if the run reports pair-level cost, split it across the pair's samples."""
    lat = {r.sample_id: r.latency_s for r in recs}
    if any(v is None for v in lat.values()):
        return None
    nan = float("nan")
    if (meta.extra or {}).get("generation_statistics_unit") == "pair":
        pid = {r.sample_id: (r.extra or {}).get("pair_id") for r in recs}
        if any(v is None for v in pid.values()):
            raise ValueError(f"{meta.detector}/{meta.variant}: pair-level cost but records lack extra.pair_id")
        size: dict = {}
        for v in pid.values():
            size[v] = size.get(v, 0) + 1
        return np.array([lat[s.sample_id] / size[pid[s.sample_id]] if s.sample_id in lat else nan
                         for s in samples], dtype=float)
    return np.array([lat.get(s.sample_id, nan) for s in samples], dtype=float)


def _cost_totals(recs, meta) -> dict:
    items = list(recs)
    if (meta.extra or {}).get("generation_statistics_unit") == "pair":
        seen, uniq = set(), []
        for r in items:
            pid = (r.extra or {}).get("pair_id")
            if pid is None:
                raise ValueError(f"{meta.detector}/{meta.variant}: pair-level cost but records lack extra.pair_id")
            if pid not in seen:
                seen.add(pid)
                uniq.append(r)
        items = uniq
    tot: dict = {"unit": "pair" if len(items) != len(recs) else "sample"}
    for field, key in (("latency_s", "total_latency_s"), ("n_generations", "total_generations"),
                       ("n_generated_tokens", "total_generated_tokens")):
        vals = [getattr(r, field) for r in items if getattr(r, field) is not None]
        tot[key] = float(sum(vals)) if vals and field == "latency_s" else (int(sum(vals)) if vals else None)
    mems = [r.peak_gpu_mem_mb for r in items if r.peak_gpu_mem_mb is not None]
    tot["peak_gpu_mem_mb"] = float(max(mems)) if mems else None
    return tot


def load_input(spec: InputSpec, val_samples: Sequence[Sample], test_samples: Sequence[Sample],
               allow_partial: bool = False) -> LoadedInput:
    """Read and validate one input's validation and test runs.

    Strict by default: the runs must cover exactly the official splits. With ``allow_partial`` a run may
    cover a subset (e.g. the SE runner that scores only the length-controlled test samples); samples
    without a record get NaN and are marked in ``val_has`` / ``test_has``.
    """
    val_recs, val_meta = read_detector_output(spec.val_path)
    test_recs, test_meta = read_detector_output(spec.test_path)
    if val_meta.split != "validation" or test_meta.split != "test":
        raise ValueError(f"{spec.name}: expected a validation run and a test run, got "
                         f"{val_meta.split!r} and {test_meta.split!r}")
    for key in ("detector", "variant", "config_hash", "prompt_version"):
        if getattr(val_meta, key) != getattr(test_meta, key):
            raise ValueError(f"{spec.name}: validation and test runs differ in {key}: "
                             f"{getattr(val_meta, key)!r} vs {getattr(test_meta, key)!r}")
    for recs, samples, what in ((val_recs, val_samples, "validation"), (test_recs, test_samples, "test")):
        ids = [s.sample_id for s in samples]
        if allow_partial:
            extra = sorted(set(r.sample_id for r in recs) - set(ids))
            if extra:
                raise ValueError(f"{spec.name}: {len(extra)} {what} records are not in the {what} split (e.g. {extra[:3]})")
        else:
            check_coverage(recs, ids, strict=True)
    sv = E.records_to_scores(val_recs)
    st = E.records_to_scores(test_recs)
    nan = float("nan")
    return LoadedInput(
        spec=spec,
        val_raw=np.array([sv[s.sample_id]["score"] if s.sample_id in sv else nan for s in val_samples], dtype=float),
        test_raw=np.array([st[s.sample_id]["score"] if s.sample_id in st else nan for s in test_samples], dtype=float),
        val_has=np.array([s.sample_id in sv for s in val_samples]),
        test_has=np.array([s.sample_id in st for s in test_samples]),
        val_meta=val_meta, test_meta=test_meta, own_threshold=test_meta.threshold,
        val_latency=_amortized_latency(val_recs, val_meta, val_samples),
        test_latency=_amortized_latency(test_recs, test_meta, test_samples),
        cost_test=_cost_totals(test_recs, test_meta))


def _slice_input(li: LoadedInput, vmask: np.ndarray, tmask: np.ndarray) -> LoadedInput:
    cut = lambda a, m: None if a is None else a[m]
    return replace(li, val_raw=li.val_raw[vmask], test_raw=li.test_raw[tmask],
                   val_latency=cut(li.val_latency, vmask), test_latency=cut(li.test_latency, tmask),
                   val_has=li.val_has[vmask], test_has=li.test_has[tmask])


# ---------------------------------------------------------------------------
# transforms, standardization, folds
# ---------------------------------------------------------------------------
def apply_transform(x: np.ndarray, kind: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if kind == "none":
        return x
    if kind == "logit":
        if (x < 0).any() or (x > 1).any():
            raise ValueError("logit transform needs scores in [0, 1]")
        p = np.clip(x, LOGIT_EPS, 1 - LOGIT_EPS)
        return np.log(p / (1 - p))
    raise ValueError(f"unknown transform {kind!r}")


def standardize_fit(X: np.ndarray, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    mean, sd = X.mean(axis=0), X.std(axis=0)
    for j, n in enumerate(names):
        if not np.isfinite(sd[j]) or sd[j] == 0:
            raise ValueError(f"input {n!r} is constant on validation: nothing to fuse")
    return mean, sd


def grouped_folds(groups: Sequence[str], n_splits: int = N_FOLDS, seed: int = 0) -> np.ndarray:
    """Fold index per sample; all samples of a group (question pair) share a fold. Deterministic."""
    uniq = sorted(set(groups))
    if len(uniq) < n_splits:
        raise ValueError(f"need at least {n_splits} groups for {n_splits}-fold CV, got {len(uniq)}")
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {uniq[i]: k % n_splits for k, i in enumerate(perm)}
    return np.array([fold_of[g] for g in groups], dtype=int)


# ---------------------------------------------------------------------------
# the fitted model
# ---------------------------------------------------------------------------
@dataclass
class FusionFit:
    feature_names: list
    mean: np.ndarray
    sd: np.ndarray
    C: float
    cv_logloss: dict                 # C -> mean out-of-fold log-loss
    coef: np.ndarray                 # on standardized features
    intercept: float
    oof_val: np.ndarray              # out-of-fold decision values on validation (best C)
    threshold: dict                  # from evaluator.select_threshold on oof_val
    n_splits: int
    seed: int

    def decision(self, X_transformed: np.ndarray) -> np.ndarray:
        return ((X_transformed - self.mean) / self.sd) @ self.coef + self.intercept

    def raw_scale_weights(self) -> dict:
        """Weights expressed on the (transformed) input scale: S = w0 + sum_j w_j * x_j."""
        w = self.coef / self.sd
        w0 = float(self.intercept - (self.coef * self.mean / self.sd).sum())
        return {"w0": w0, **{n: float(v) for n, v in zip(self.feature_names, w)}}

    def to_dict(self) -> dict:
        return {"feature_names": self.feature_names, "C": self.C, "cv_logloss": {str(k): v for k, v in self.cv_logloss.items()},
                "weights_standardized": {n: float(v) for n, v in zip(self.feature_names, self.coef)},
                "intercept_standardized": float(self.intercept), "weights_raw_scale": self.raw_scale_weights(),
                "standardization_mean": {n: float(v) for n, v in zip(self.feature_names, self.mean)},
                "standardization_sd": {n: float(v) for n, v in zip(self.feature_names, self.sd)},
                "threshold": self.threshold, "n_splits": self.n_splits, "cv_seed": self.seed,
                "negative_weights": [n for n, v in zip(self.feature_names, self.coef) if v < 0]}


def _lr(C_value: float) -> LogisticRegression:
    return LogisticRegression(C=C_value, max_iter=2000)       # L2 (default), lbfgs


def fit_fusion(X_val: np.ndarray, y_val: np.ndarray, groups_val: Sequence[str], feature_names: Sequence[str], *,
               C_grid: Sequence[float] = C_GRID, n_splits: int = N_FOLDS, seed: int = 0,
               criterion: str = "youden") -> FusionFit:
    """Fit the fusion on VALIDATION data (X_val already transformed, not yet standardized)."""
    X_val, y_val = np.asarray(X_val, dtype=float), np.asarray(y_val, dtype=int)
    if len(np.unique(y_val)) < 2:
        raise ValueError("validation labels contain a single class")
    names = list(feature_names)
    mean, sd = standardize_fit(X_val, names)
    Z = (X_val - mean) / sd
    folds = grouped_folds(groups_val, n_splits, seed)

    best_C, best_ll, best_oof, table = None, np.inf, None, {}
    for c in sorted(C_grid):                                   # ascending: ties go to the smaller C
        oof = np.zeros(len(y_val))
        prob = np.zeros(len(y_val))
        for f in range(n_splits):
            tr, te = folds != f, folds == f
            m = _lr(c).fit(Z[tr], y_val[tr])
            oof[te] = m.decision_function(Z[te])
            prob[te] = m.predict_proba(Z[te])[:, 1]
        ll = float(log_loss(y_val, np.clip(prob, 1e-15, 1 - 1e-15)))
        table[float(c)] = ll
        if ll < best_ll:
            best_C, best_ll, best_oof = float(c), ll, oof
    final = _lr(best_C).fit(Z, y_val)
    thr = E.select_threshold(y_val, best_oof, criterion)
    return FusionFit(feature_names=names, mean=mean, sd=sd, C=best_C, cv_logloss=table,
                     coef=final.coef_[0].copy(), intercept=float(final.intercept_[0]), oof_val=best_oof,
                     threshold=thr, n_splits=n_splits, seed=seed)


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------
def _design(loaded: Sequence[LoadedInput], val_samples, test_samples, include_length: bool):
    names = [li.spec.name for li in loaded]
    Xv = np.column_stack([apply_transform(li.val_raw, li.spec.transform) for li in loaded])
    Xt = np.column_stack([apply_transform(li.test_raw, li.spec.transform) for li in loaded])
    if include_length:
        names = names + [LENGTH_FEATURE]
        lv = np.log(np.maximum([s.answer_length_chars for s in val_samples], 1).astype(float))
        lt = np.log(np.maximum([s.answer_length_chars for s in test_samples], 1).astype(float))
        Xv, Xt = np.column_stack([Xv, lv]), np.column_stack([Xt, lt])
    return names, Xv, Xt


def _write(records_by_split: dict, variant: str, thr_value: float, extra: dict, description: str,
           out_dir: Optional[Path], cfg: Config) -> dict:
    paths = {}
    for split, (samples, scores, latency, peak) in records_by_split.items():
        recs = []
        for i, (s, sc) in enumerate(zip(samples, scores)):
            recs.append(DetectorRecord(
                sample_id=s.sample_id, split=split, detector=DETECTOR, variant=variant, score=float(sc),
                config_hash=cfg.config_hash(), prompt_version=cfg.prompt.version,
                latency_s=None if latency is None else float(latency[i]), peak_gpu_mem_mb=peak))
        path = None if out_dir is None else Path(out_dir) / variant / f"{split}.jsonl"
        paths[split] = write_detector_output(recs, path=path, threshold=thr_value, description=description,
                                             extra=extra, cfg=cfg)
    return paths


def run_fusion(specs: Sequence[InputSpec], val_samples: Sequence[Sample], test_samples: Sequence[Sample], *,
               include_length: bool = False, criterion: str = "youden", seed: int = 0, n_splits: int = N_FOLDS,
               C_grid: Sequence[float] = C_GRID, out_dir: Optional[Path] = None, cfg: Optional[Config] = None,
               allow_mixed_provenance: bool = False, allow_partial: bool = False,
               min_val_pairs: int = 100, write: bool = True) -> dict:
    """Fit the fusion on validation and write the `hybrid` runs (fitted LR + unfitted average).

    Never reads test labels. Returns a dict with the loaded inputs, the fit, the written paths and
    the design matrices (for `analyze`).

    ``allow_partial=True`` (plan amendment 4): when an input covers only part of a split, fit on the
    validation samples covered by EVERY input (at least ``min_val_pairs`` question pairs) and score only
    the test samples covered by every input (e.g. the length-controlled set). The coverage is recorded
    in the run metadata and the analysis is restricted accordingly.

    ``write=False`` (dry run) fits and scores but writes no files; ``paths`` are then None.
    """
    if len(specs) < 2:
        raise ValueError("fusion needs at least two inputs")
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError("input names must be unique")
    cfg = cfg or get_config()
    val_samples, test_samples = list(val_samples), list(test_samples)
    loaded = [load_input(s, val_samples, test_samples, allow_partial) for s in specs]
    val_total, test_total = len(val_samples), len(test_samples)
    if allow_partial:
        vmask = np.logical_and.reduce([li.val_has for li in loaded])
        tmask = np.logical_and.reduce([li.test_has for li in loaded])
        n_pairs = len({s.pair_id for s, k in zip(val_samples, vmask) if k})
        if n_pairs < min_val_pairs:
            raise ValueError(f"only {n_pairs} validation question pairs are covered by every input "
                             f"(need at least {min_val_pairs}); run the inputs on more validation pairs")
        if not tmask.any():
            raise ValueError("no test sample is covered by every input")
        val_samples = [s for s, k in zip(val_samples, vmask) if k]
        test_samples = [s for s, k in zip(test_samples, tmask) if k]
        loaded = [_slice_input(li, vmask, tmask) for li in loaded]
    coverage = {"validation_used": len(val_samples), "validation_total": val_total,
                "test_used": len(test_samples), "test_total": test_total, "partial": allow_partial and (
                    len(val_samples) != val_total or len(test_samples) != test_total)}

    prov = {(li.val_meta.config_hash, li.val_meta.prompt_version) for li in loaded}
    if len(prov) > 1:
        msg = "inputs were produced under different config_hash/prompt_version: " + "; ".join(
            f"{li.spec.name}={li.val_meta.config_hash}/{li.val_meta.prompt_version}" for li in loaded)
        if not allow_mixed_provenance:
            raise ValueError(msg + " (pass allow_mixed_provenance=True to accept and record it)")
        warnings.warn(msg, UserWarning, stacklevel=2)

    y_val = np.array([s.label for s in val_samples], dtype=int)         # validation labels only
    groups = [s.pair_id for s in val_samples]
    feat_names, Xv, Xt = _design(loaded, val_samples, test_samples, include_length)
    fit = fit_fusion(Xv, y_val, groups, feat_names, C_grid=C_grid, n_splits=n_splits, seed=seed, criterion=criterion)

    test_scores = fit.decision(Xt)
    # unfitted equal-weight average of the standardized features
    avg_val = ((Xv - fit.mean) / fit.sd).mean(axis=1)
    avg_test = ((Xt - fit.mean) / fit.sd).mean(axis=1)
    avg_thr = E.select_threshold(y_val, avg_val, criterion)

    # cost of the hybrid = cost of its inputs (pair-level costs split across the pair)
    def sum_latency(attr):
        parts = [getattr(li, attr) for li in loaded]
        return None if any(p is None for p in parts) else np.sum(parts, axis=0)
    lat_v, lat_t = sum_latency("val_latency"), sum_latency("test_latency")
    peaks = [li.cost_test["peak_gpu_mem_mb"] for li in loaded if li.cost_test["peak_gpu_mem_mb"] is not None]
    peak = float(max(peaks)) if peaks else None
    base = "-".join(feat_names)

    common_extra = {
        "plan": "docs/fusion_analysis_plan.md",
        "inputs": [{"name": li.spec.name, "transform": li.spec.transform, "val_path": str(li.spec.val_path),
                    "test_path": str(li.spec.test_path), "detector": li.val_meta.detector,
                    "variant": li.val_meta.variant, "config_hash": li.val_meta.config_hash,
                    "own_threshold": li.own_threshold} for li in loaded],
        "include_length": include_length, "coverage": coverage,
        "cost_test_totals_by_input": {li.spec.name: li.cost_test for li in loaded},
        "fusion_fit_uses": "validation labels only",
    }
    lr_extra = {**common_extra, "method": "L2 logistic regression on standardized inputs",
                "fit": fit.to_dict(), "validation_scores_are_out_of_fold": True}
    avg_extra = {**common_extra, "method": "unfitted equal-weight average of standardized inputs",
                 "standardization": {"mean": dict(zip(feat_names, map(float, fit.mean))),
                                     "sd": dict(zip(feat_names, map(float, fit.sd)))},
                 "threshold_criterion": criterion}
    def write_or_none(*a, **k):
        return _write(*a, **k) if write else None

    lr_paths = write_or_none({"validation": (val_samples, fit.oof_val, lat_v, peak), "test": (test_samples, test_scores, lat_t, peak)},
                      f"{base}-lr", fit.threshold["value"], lr_extra,
                      f"hybrid late fusion ({', '.join(feat_names)}), L2 logistic regression fitted on validation",
                      out_dir, cfg)
    avg_paths = write_or_none({"validation": (val_samples, avg_val, lat_v, peak), "test": (test_samples, avg_test, lat_t, peak)},
                       f"{base}-avg", avg_thr["value"], avg_extra,
                       f"hybrid late fusion ({', '.join(feat_names)}), unfitted equal-weight average", out_dir, cfg)
    return {"loaded": loaded, "fit": fit, "feature_names": feat_names, "Xv": Xv, "Xt": Xt, "y_val": y_val,
            "val_samples": val_samples, "test_samples": test_samples, "coverage": coverage,
            "lr": {"paths": lr_paths, "variant": f"{base}-lr", "test_scores": test_scores, "val_scores": fit.oof_val,
                   "threshold": fit.threshold["value"]},
            "avg": {"paths": avg_paths, "variant": f"{base}-avg", "test_scores": avg_test, "val_scores": avg_val,
                    "threshold": avg_thr["value"]}}


# ---------------------------------------------------------------------------
# analysis (uses labels: run AFTER the fusion is frozen)
# ---------------------------------------------------------------------------
def _scored(samples, scores) -> dict:
    return E.records_to_scores([{"sample_id": s.sample_id, "score": float(v)} for s, v in zip(samples, scores)])


def diagnostics(result: dict, val_samples: Sequence[Sample]) -> dict:
    """Validation-only facts about the inputs: AUROC, distinct values, correlations, length correlation."""
    y = result["y_val"]
    X = result["Xv"]
    names = result["feature_names"]
    lengths = np.array([s.answer_length_chars for s in val_samples], dtype=float)
    out = {"inputs": {}, "pearson_between_inputs": {}, "spearman_with_length": {}}
    raw_by_name = {li.spec.name: li for li in result["loaded"]}
    for j, n in enumerate(names):
        out["inputs"][n] = {"val_auroc": E.auroc(y, X[:, j]), "n_distinct_values": int(len(np.unique(X[:, j])))}
        li = raw_by_name.get(n)
        if li is not None and li.spec.transform == "logit":      # how many scores hit the clip (information lost)
            out["inputs"][n]["clipped_fraction"] = float(np.mean((li.val_raw <= LOGIT_EPS) | (li.val_raw >= 1 - LOGIT_EPS)))
        out["spearman_with_length"][n] = float(spearmanr(X[:, j], lengths)[0])
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            out["pearson_between_inputs"][f"{names[a]} ~ {names[b]}"] = float(np.corrcoef(X[:, a], X[:, b])[0, 1])
    return out


def analyze(result: dict, val_samples: Sequence[Sample], test_samples: Sequence[Sample],
            ctrl_samples: Optional[Sequence[Sample]] = None, *, n_boot: int = 5000, seed: int = 0) -> dict:
    """Primary and secondary comparisons of the plan: hybrid vs each input, vs the unfitted average.

    Primary (plan section 5): the paired-bootstrap CI of AUROC(hybrid) - AUROC(input) on the
    length-controlled set must exclude 0 in the positive direction for EVERY input.
    """
    val_samples, test_samples = list(result["val_samples"]), list(result["test_samples"])   # samples actually used
    loaded, y_val = result["loaded"], result["y_val"]
    hyb = _scored(test_samples, result["lr"]["test_scores"])
    avg = _scored(test_samples, result["avg"]["test_scores"])
    thr_h, thr_a = result["lr"]["threshold"], result["avg"]["threshold"]

    inputs = {}
    for li in loaded:
        thr = li.own_threshold
        thr_src = "owner (validation)"
        if thr is None:                      # e.g. SE runs carry no threshold: Youden on validation scores
            thr = E.select_threshold(y_val, li.val_raw, "youden")["value"]
            thr_src = "youden on validation (run had none)"
        inputs[li.spec.name] = {"scored": _scored(test_samples, li.test_raw), "thr": thr, "thr_source": thr_src}

    sets = {}
    notes = []
    if all(c == 2 for c in Counter(s.pair_id for s in test_samples).values()):
        sets["test"] = (test_samples, "pair")
    else:
        notes.append("test metrics skipped: the test samples scored do not form complete question pairs (partial coverage)")
    if ctrl_samples:
        covered = {s.sample_id for s in test_samples}
        missing = [s.sample_id for s in ctrl_samples if s.sample_id not in covered]
        if missing:
            notes.append(f"length-controlled set skipped: {len(missing)} of {len(list(ctrl_samples))} control samples "
                         f"are not covered by every input")
        else:
            sets["length_controlled"] = (list(ctrl_samples), "match")

    out: dict = {"n_boot": n_boot, "notes": notes, "coverage": result["coverage"], "metrics": {}, "paired": {}, "thresholds": {
        n: {"value": v["thr"], "source": v["thr_source"]} for n, v in inputs.items()}}
    detectors = {"hybrid": (hyb, thr_h), "average": (avg, thr_a), **{n: (v["scored"], v["thr"]) for n, v in inputs.items()}}
    for label, (samples, cluster) in sets.items():
        out["metrics"][label] = {n: C.bootstrap_ci(samples, sc, thr, cluster=cluster, n_boot=n_boot, seed=seed)["metrics"]
                                 for n, (sc, thr) in detectors.items()}
        out["paired"][label] = {}
        for n, (sc, thr) in detectors.items():
            if n == "hybrid":
                continue
            out["paired"][label][f"hybrid_minus_{n}"] = C.paired_bootstrap_diff(
                samples, hyb, sc, thr_h, thr, cluster=cluster, n_boot=n_boot, seed=seed)["metric_differences_a_minus_b"]

    # plan section 5: primary verdict on the length-controlled set, against every input
    if "length_controlled" in out["paired"]:
        diffs = [out["paired"]["length_controlled"][f"hybrid_minus_{n}"]["auroc"] for n in inputs]
        if all(d["lo"] is not None and d["lo"] > 0 for d in diffs):
            verdict = "IMPROVES: hybrid beats every input (95% CI of the AUROC difference > 0)"
        elif any(d["hi"] is not None and d["hi"] < 0 for d in diffs):
            verdict = "WORSE: hybrid is below at least one input (95% CI < 0)"
        else:
            verdict = "INCONCLUSIVE: at least one CI includes 0 (not evidence of 'no benefit'; the control set is small)"
    else:
        verdict = "UNAVAILABLE: " + ("; ".join(n for n in notes if "length-controlled" in n)
                                     or "no length-controlled set was provided")
    out["primary_verdict"] = verdict
    out["error_overlap_test"] = C.error_overlap(test_samples, detectors)
    out["diagnostics"] = diagnostics(result, val_samples)
    return out
