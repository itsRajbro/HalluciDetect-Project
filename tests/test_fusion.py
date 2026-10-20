"""Tests for src/detectors/fusion/fusion.py and the pair-level cost handling in the evaluator.

SE / SEP inputs are SIMULATED here and written through the real schema, so the whole path
(read runs -> fit on validation -> write hybrid runs -> evaluate) is exercised without a GPU.
Run: python -m pytest tests -q
"""
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.common import evaluator as E  # noqa: E402
from src.common.config import get_config  # noqa: E402
from src.common.schema import DetectorRecord, read_detector_output, write_detector_output  # noqa: E402
from src.detectors.fusion import fusion as F  # noqa: E402
from tests.test_comparison import as_matches  # noqa: E402
from tests.test_evaluator import make_samples  # noqa: E402

CFG = get_config()


# ---- helpers -------------------------------------------------------------------------------
def sig(samples, d, seed):
    rng = np.random.default_rng(seed)
    return np.array([s.label * d for s in samples], float) + rng.standard_normal(len(samples))


def write_run(tmp, detector, samples, split, scores, *, threshold=None, cfg=CFG, meta_extra=None,
              rec_extra=None, latency=None):
    recs = [DetectorRecord(sample_id=s.sample_id, split=split, detector=detector, variant="v1", score=float(sc),
                           config_hash=cfg.config_hash(), prompt_version=cfg.prompt.version, latency_s=latency,
                           extra=(rec_extra(s) if rec_extra else {}))
            for s, sc in zip(samples, scores)]
    return write_detector_output(recs, path=Path(tmp) / f"{detector}_{split}.jsonl", threshold=threshold,
                                 extra=meta_extra, cfg=cfg)


def spec(tmp, detector, name, transform="none"):
    return F.InputSpec(name, Path(tmp) / f"{detector}_validation.jsonl", Path(tmp) / f"{detector}_test.jsonl", transform)


@pytest.fixture()
def world(tmp_path):
    return {"val": make_samples(400, "validation", seed=1, start=0),
            "test": make_samples(400, "test", seed=2, start=10_000), "tmp": tmp_path}


def two_inputs(world, d1=1.0, d2=1.0, corr=None, seed=0, **kw):
    """Write simulated 'se_candidate' and 'sep' runs; returns specs."""
    tmp, val, test = world["tmp"], world["val"], world["test"]
    s1v, s1t = sig(val, d1, seed + 1), sig(test, d1, seed + 2)
    if corr == "identical":
        s2v, s2t = s1v + 0.01 * sig(val, 0, seed + 9), s1t + 0.01 * sig(test, 0, seed + 8)
    else:
        s2v, s2t = sig(val, d2, seed + 3), sig(test, d2, seed + 4)
    write_run(tmp, "se_candidate", val, "validation", s1v)
    write_run(tmp, "se_candidate", test, "test", s1t)
    write_run(tmp, "sep", val, "validation", s2v, **kw)
    write_run(tmp, "sep", test, "test", s2t, **kw)
    return [spec(tmp, "se_candidate", "se"), spec(tmp, "sep", "sep")]


def fuse(world, specs, **kw):
    return F.run_fusion(specs, world["val"], world["test"], out_dir=world["tmp"] / "hybrid", cfg=CFG, **kw)


def auroc_of(samples, scores):
    return E.auroc(np.array([s.label for s in samples]), np.asarray(scores))


# ---- the model behaves like a fusion should -------------------------------------------------
def test_independent_errors_fusion_beats_both_inputs(world):
    r = fuse(world, two_inputs(world, 1.0, 1.0))
    test = world["test"]
    a_h = auroc_of(test, r["lr"]["test_scores"])
    a_in = [auroc_of(test, li.test_raw) for li in r["loaded"]]
    assert a_h > max(a_in) + 0.04                      # ~0.84 vs ~0.76
    assert all(w > 0 for w in r["fit"].coef)           # both oriented "higher = hallucinated"


def test_fully_correlated_inputs_add_nothing(world):
    r = fuse(world, two_inputs(world, 1.0, corr="identical"))
    test = world["test"]
    best = max(auroc_of(test, li.test_raw) for li in r["loaded"])
    assert abs(auroc_of(test, r["lr"]["test_scores"]) - best) < 0.02


def test_stronger_input_gets_larger_weight(world):
    r = fuse(world, two_inputs(world, 2.0, 0.5))
    se_w, sep_w = r["fit"].coef
    assert se_w > sep_w > 0


def test_unfitted_average_is_mean_of_standardized_inputs_and_variants_are_named(world):
    r = fuse(world, two_inputs(world))
    f = r["fit"]
    assert r["avg"]["variant"] == "se-sep-avg" and r["lr"]["variant"] == "se-sep-lr"
    np.testing.assert_allclose(r["avg"]["test_scores"], ((r["Xt"] - f.mean) / f.sd).mean(axis=1))
    r2 = fuse(world, two_inputs(world), include_length=True)
    assert r2["lr"]["variant"] == "se-sep-len-lr" and r2["fit"].feature_names == ["se", "sep", "len"]


# ---- protocol guarantees --------------------------------------------------------------------
def test_test_labels_never_influence_the_fit(world):
    specs = two_inputs(world)
    real = fuse(world, specs)
    flipped = [replace(s, label=1 - s.label) for s in world["test"]]
    other = F.run_fusion(specs, world["val"], flipped, out_dir=world["tmp"] / "hybrid2", cfg=CFG)
    np.testing.assert_array_equal(real["fit"].coef, other["fit"].coef)
    np.testing.assert_array_equal(real["lr"]["test_scores"], other["lr"]["test_scores"])
    assert real["lr"]["threshold"] == other["lr"]["threshold"]


def test_grouped_folds_keep_pairs_together_and_are_deterministic():
    groups = [f"P{i // 2}" for i in range(200)]                    # 100 pairs, 2 samples each
    f = F.grouped_folds(groups, 5, seed=0)
    for g in set(groups):
        assert len({f[i] for i, x in enumerate(groups) if x == g}) == 1
    assert sorted(set(f)) == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(f, F.grouped_folds(groups, 5, seed=0))
    assert not np.array_equal(f, F.grouped_folds(groups, 5, seed=1))
    with pytest.raises(ValueError):
        F.grouped_folds(["a", "a", "b", "b"], 5)


def test_hybrid_runs_are_valid_schema_runs_with_validation_threshold(world):
    r = fuse(world, two_inputs(world))
    for key in ("lr", "avg"):
        recs, meta = read_detector_output(r[key]["paths"]["test"])
        assert meta.detector == "hybrid" and meta.threshold_source == "validation" and len(recs) == 400 * 2
        assert meta.config_hash == CFG.config_hash()
    _, meta = read_detector_output(r["lr"]["paths"]["validation"])
    assert meta.extra["validation_scores_are_out_of_fold"] is True
    assert set(meta.extra["fit"]["weights_standardized"]) == {"se", "sep"} and meta.extra["fit"]["C"] in F.C_GRID
    rep = E.evaluate_run(val_path=r["lr"]["paths"]["validation"], test_path=r["lr"]["paths"]["test"],
                         val_samples=world["val"], test_samples=world["test"], include_length_controlled=False,
                         n_boot=50)
    assert rep["detector"] == "hybrid/se-sep-lr" and rep["test"]["metrics"]["auroc"] > 0.7


def test_validation_scores_are_out_of_fold_not_in_sample(world):
    r = fuse(world, two_inputs(world, 0.8, 0.8))
    in_sample = r["fit"].decision(r["Xv"])
    assert not np.allclose(in_sample, r["fit"].oof_val)


# ---- input validation -----------------------------------------------------------------------
def test_missing_samples_are_rejected(world):
    tmp, val, test = world["tmp"], world["val"], world["test"]
    write_run(tmp, "se_candidate", val, "validation", sig(val, 1, 1))
    write_run(tmp, "se_candidate", test[:-2], "test", sig(test[:-2], 1, 2))          # two samples missing
    write_run(tmp, "sep", val, "validation", sig(val, 1, 3))
    write_run(tmp, "sep", test, "test", sig(test, 1, 4))
    with pytest.raises(ValueError, match="coverage"):
        fuse(world, [spec(tmp, "se_candidate", "se"), spec(tmp, "sep", "sep")])


def test_provenance_mismatch_is_rejected_unless_explicitly_allowed(world):
    specs = two_inputs(world)
    other = replace(CFG, run=replace(CFG.run, seed=7))
    tmp, val, test = world["tmp"], world["val"], world["test"]
    write_run(tmp, "sep", val, "validation", sig(val, 1, 3), cfg=other)
    write_run(tmp, "sep", test, "test", sig(test, 1, 4), cfg=other)
    with pytest.raises(ValueError, match="config_hash"):
        fuse(world, specs)
    with pytest.warns(UserWarning, match="config_hash"):
        fuse(world, specs, allow_mixed_provenance=True)


def test_wrong_split_files_and_bad_arguments_are_rejected(world):
    specs = two_inputs(world)
    swapped = F.InputSpec("se", specs[0].test_path, specs[0].val_path)
    with pytest.raises(ValueError, match="expected a validation run"):
        fuse(world, [swapped, specs[1]])
    with pytest.raises(ValueError):
        fuse(world, specs[:1])
    with pytest.raises(ValueError):
        F.InputSpec("se_cand", Path("a"), Path("b"))              # underscore reserved as separator
    with pytest.raises(ValueError):
        F.InputSpec("len", Path("a"), Path("b"))
    with pytest.raises(ValueError):
        F.InputSpec("x", Path("a"), Path("b"), transform="sqrt")


def test_constant_input_is_rejected(world):
    tmp, val, test = world["tmp"], world["val"], world["test"]
    write_run(tmp, "se_candidate", val, "validation", np.ones(len(val)))
    write_run(tmp, "se_candidate", test, "test", np.ones(len(test)))
    write_run(tmp, "sep", val, "validation", sig(val, 1, 3))
    write_run(tmp, "sep", test, "test", sig(test, 1, 4))
    with pytest.raises(ValueError, match="constant"):
        fuse(world, [spec(tmp, "se_candidate", "se"), spec(tmp, "sep", "sep")])


# ---- logit transform for probability outputs (SEP) ------------------------------------------
def test_logit_transform_handles_saturated_probabilities():
    x = np.array([0.0, 1e-12, 0.5, 1 - 1e-12, 1.0])
    z = F.apply_transform(x, "logit")
    assert np.isfinite(z).all() and z[0] == z[1] and z[3] == z[4] and z[2] == 0.0
    assert z[0] == pytest.approx(-np.log((1 - F.LOGIT_EPS) / F.LOGIT_EPS))
    with pytest.raises(ValueError):
        F.apply_transform(np.array([0.2, 1.3]), "logit")


def test_fusion_with_saturated_sep_probabilities_uses_logit(world):
    tmp, val, test = world["tmp"], world["val"], world["test"]
    write_run(tmp, "se_candidate", val, "validation", sig(val, 1, 1))
    write_run(tmp, "se_candidate", test, "test", sig(test, 1, 2))
    to_prob = lambda x: 1 / (1 + np.exp(-12 * x))                  # very saturated, like a probe
    write_run(tmp, "sep", val, "validation", to_prob(sig(val, 1, 3)))
    write_run(tmp, "sep", test, "test", to_prob(sig(test, 1, 4)))
    s = [spec(tmp, "se_candidate", "se"), spec(tmp, "sep", "sep", transform="logit")]
    r = fuse(world, s)
    assert np.isfinite(r["lr"]["test_scores"]).all() and r["fit"].coef[1] > 0
    _, meta = read_detector_output(r["lr"]["paths"]["test"])
    assert meta.extra["inputs"][1]["transform"] == "logit"


# ---- the structural fact stated in the plan (section 2) --------------------------------------
def test_pair_constant_input_cannot_change_pair_accuracy(world):
    """A global-SE-like input (same value for both answers of a question) cancels inside a pair."""
    tmp, val, test = world["tmp"], world["val"], world["test"]

    def per_question(samples, seed):
        rng = np.random.default_rng(seed)
        q = {p: rng.standard_normal() for p in sorted({s.pair_id for s in samples})}
        return np.array([q[s.pair_id] for s in samples])

    write_run(tmp, "se_global", val, "validation", per_question(val, 1))
    write_run(tmp, "se_global", test, "test", per_question(test, 2))
    write_run(tmp, "sep", val, "validation", sig(val, 1.5, 3))
    write_run(tmp, "sep", test, "test", sig(test, 1.5, 4))
    r = fuse(world, [spec(tmp, "se_global", "seg"), spec(tmp, "sep", "sep")])
    se_scored = E.records_to_scores([{"sample_id": s.sample_id, "score": v} for s, v in zip(test, r["loaded"][0].test_raw)])
    assert E.auroc(np.array([s.label for s in test]), r["loaded"][0].test_raw) == 0.5     # exactly chance
    assert E.pair_level(test, se_scored)["pair_accuracy"] == 0.5
    hyb = E.records_to_scores([{"sample_id": s.sample_id, "score": v} for s, v in zip(test, r["lr"]["test_scores"])])
    sep = E.records_to_scores([{"sample_id": s.sample_id, "score": v} for s, v in zip(test, r["loaded"][1].test_raw)])
    assert E.pair_level(test, hyb)["pair_accuracy"] == E.pair_level(test, sep)["pair_accuracy"]


# ---- cost handling -------------------------------------------------------------------------
def test_pair_level_cost_is_counted_once_in_inputs_hybrid_and_evaluator(world):
    tmp, val, test = world["tmp"], world["val"], world["test"]
    n_pairs = len(test) // 2
    pair_meta = {"generation_statistics_unit": "pair"}
    pid = lambda s: {"pair_id": s.pair_id}
    write_run(tmp, "se_candidate", val, "validation", sig(val, 1, 1), latency=10.0, meta_extra=pair_meta, rec_extra=pid)
    write_run(tmp, "se_candidate", test, "test", sig(test, 1, 2), latency=10.0, meta_extra=pair_meta, rec_extra=pid)
    write_run(tmp, "sep", val, "validation", sig(val, 1, 3), latency=0.5)
    write_run(tmp, "sep", test, "test", sig(test, 1, 4), latency=0.5)
    r = fuse(world, [spec(tmp, "se_candidate", "se"), spec(tmp, "sep", "sep")])
    assert r["loaded"][0].cost_test["total_latency_s"] == pytest.approx(10.0 * n_pairs)    # not 2x
    assert r["loaded"][1].cost_test["total_latency_s"] == pytest.approx(0.5 * len(test))
    recs, _ = read_detector_output(r["lr"]["paths"]["test"])
    assert sum(x.latency_s for x in recs) == pytest.approx(10.0 * n_pairs + 0.5 * len(test))
    # the single-run evaluator also counts each pair once when the run says so
    rep = E.evaluate_run(val_path=tmp / "se_candidate_validation.jsonl", test_path=tmp / "se_candidate_test.jsonl",
                         threshold=0.0, val_samples=val, test_samples=test, include_length_controlled=False, n_boot=0)
    assert rep["efficiency"]["cost_unit"] == "pair"
    assert rep["efficiency"]["test_total_runtime_sec"] == pytest.approx(10.0 * n_pairs)
    assert rep["efficiency"]["test_mean_runtime_sec_per_sample"] == pytest.approx(5.0)


def test_sample_level_cost_is_unchanged(world):
    tmp, val, test = world["tmp"], world["val"], world["test"]
    write_run(tmp, "sep", val, "validation", sig(val, 1, 3), latency=0.5)
    write_run(tmp, "sep", test, "test", sig(test, 1, 4), latency=0.5)
    rep = E.evaluate_run(val_path=tmp / "sep_validation.jsonl", test_path=tmp / "sep_test.jsonl", threshold=0.5,
                         val_samples=val, test_samples=test, include_length_controlled=False, n_boot=0)
    assert rep["efficiency"]["cost_unit"] == "sample"
    assert rep["efficiency"]["test_total_runtime_sec"] == pytest.approx(0.5 * len(test))


# ---- analysis and the pre-registered decision rule -------------------------------------------
def big_world(tmp_path):
    return {"val": make_samples(500, "validation", seed=11, start=0),
            "test": make_samples(700, "test", seed=12, start=10_000), "tmp": tmp_path}


def test_primary_verdict_improves_when_inputs_are_independent(tmp_path):
    w = big_world(tmp_path)
    r = fuse(w, two_inputs(w, 1.5, 1.5, seed=20))
    ctrl = as_matches(w["test"], 300)
    out = F.analyze(r, w["val"], w["test"], ctrl, n_boot=300)
    assert out["primary_verdict"].startswith("IMPROVES")
    d = out["paired"]["length_controlled"]["hybrid_minus_se"]["auroc"]
    assert d["estimate"] > 0.03 and d["lo"] > 0
    assert set(out["paired"]["test"]) == {"hybrid_minus_average", "hybrid_minus_se", "hybrid_minus_sep"}


def test_primary_verdict_is_not_improves_when_inputs_are_duplicates(tmp_path):
    w = big_world(tmp_path)
    r = fuse(w, two_inputs(w, 1.5, corr="identical", seed=30))
    out = F.analyze(r, w["val"], w["test"], as_matches(w["test"], 300), n_boot=200)
    assert not out["primary_verdict"].startswith("IMPROVES")


def test_verdict_unavailable_without_a_control_set_and_thresholds_are_reported(tmp_path):
    w = big_world(tmp_path)
    r = fuse(w, two_inputs(w, 1.0, 1.0, seed=40))
    out = F.analyze(r, w["val"], w["test"], None, n_boot=100)
    assert out["primary_verdict"].startswith("UNAVAILABLE") and "length_controlled" not in out["metrics"]
    assert out["thresholds"]["se"]["source"].startswith("youden")          # SE runs carry no threshold
    assert out["error_overlap_test"]["oracle_any_right_accuracy"] >= max(out["error_overlap_test"]["accuracy"].values())


def test_diagnostics_report_inputs_correlations_and_length(tmp_path):
    w = big_world(tmp_path)
    r = fuse(w, two_inputs(w, 1.0, 1.0, seed=50), include_length=True)
    d = F.diagnostics(r, w["val"])
    assert set(d["inputs"]) == {"se", "sep", "len"} and d["inputs"]["len"]["val_auroc"] > 0.95
    assert "se ~ sep" in d["pearson_between_inputs"] and d["spearman_with_length"]["len"] > 0.9
    assert d["inputs"]["se"]["n_distinct_values"] > 100


def test_no_warning_noise_in_normal_use(world):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fuse(world, two_inputs(world))


# ---- partial coverage (SE scored only some validation pairs and only the control samples) -------------
def cross_pair_control(test, n):
    """n matches, each a factual answer from one question and a hallucinated answer from ANOTHER question."""
    by_pair = {}
    for s in test:
        by_pair.setdefault(s.pair_id, {})[s.label] = s
    ids = sorted(by_pair)
    out = []
    for k in range(n):
        out += [replace(by_pair[ids[k]][0], match_id=f"M{k:04d}"), replace(by_pair[ids[k + n]][1], match_id=f"M{k:04d}")]
    return out


def partial_world(tmp_path, val_pairs=150, n_ctrl=60):
    w = {"val": make_samples(400, "validation", seed=61, start=0), "test": make_samples(400, "test", seed=62, start=10_000),
         "tmp": tmp_path}
    ctrl = cross_pair_control(w["test"], n_ctrl)
    se_val = w["val"][: 2 * val_pairs]                          # first val_pairs question pairs
    write_run(tmp_path, "se_candidate", se_val, "validation", sig(se_val, 1.2, 1))
    write_run(tmp_path, "se_candidate", ctrl, "test", sig(ctrl, 1.2, 2))
    write_run(tmp_path, "sep", w["val"], "validation", sig(w["val"], 1.2, 3))
    write_run(tmp_path, "sep", w["test"], "test", sig(w["test"], 1.2, 4))
    return w, ctrl, [spec(tmp_path, "se_candidate", "se"), spec(tmp_path, "sep", "sep")]


def test_partial_inputs_are_rejected_by_default(tmp_path):
    w, ctrl, specs = partial_world(tmp_path)
    with pytest.raises(ValueError, match="coverage"):
        fuse(w, specs)


def test_partial_mode_fits_on_covered_validation_and_scores_only_covered_test(tmp_path):
    w, ctrl, specs = partial_world(tmp_path)
    r = fuse(w, specs, allow_partial=True)
    assert r["coverage"] == {"validation_used": 300, "validation_total": 800, "test_used": 120, "test_total": 800,
                             "partial": True}
    assert [s.sample_id for s in r["val_samples"]] == [s.sample_id for s in w["val"][:300]]
    assert {s.sample_id for s in r["test_samples"]} == {s.sample_id for s in ctrl}
    recs, meta = read_detector_output(r["lr"]["paths"]["test"])
    assert len(recs) == 120 and meta.extra["coverage"]["partial"] is True and meta.threshold_source == "validation"
    out = F.analyze(r, w["val"], w["test"], ctrl, n_boot=100)
    assert "test" not in out["metrics"] and "length_controlled" in out["metrics"]        # pairs incomplete -> test block skipped
    assert any("complete question pairs" in n for n in out["notes"])
    assert not out["primary_verdict"].startswith("UNAVAILABLE")
    assert out["coverage"]["validation_used"] == 300


def test_partial_mode_enforces_a_minimum_of_validation_pairs(tmp_path):
    w, ctrl, specs = partial_world(tmp_path, val_pairs=50)
    with pytest.raises(ValueError, match="only 50 validation question pairs"):
        fuse(w, specs, allow_partial=True)
    assert fuse(w, specs, allow_partial=True, min_val_pairs=40)["coverage"]["validation_used"] == 100


def test_partial_mode_still_rejects_records_outside_the_split_and_empty_test_overlap(tmp_path):
    w, ctrl, specs = partial_world(tmp_path)
    stranger = make_samples(5, "test", seed=99, start=50_000)                     # ids not in the test split
    write_run(tmp_path, "se_candidate", ctrl + stranger[:2], "test", np.arange(len(ctrl) + 2, dtype=float))
    with pytest.raises(ValueError, match="not in the test split"):
        fuse(w, specs, allow_partial=True)
    write_run(tmp_path, "se_candidate", ctrl[:2], "test", sig(ctrl[:2], 1, 5))      # SEP test overlap is fine; SE covers 2
    other = [s for s in w["test"] if s.sample_id not in {c.sample_id for c in ctrl[:2]}][:4]
    write_run(tmp_path, "se_candidate", other, "test", sig(other, 1, 6))
    write_run(tmp_path, "sep", ctrl[:2], "test", sig(ctrl[:2], 1, 7))                # no common test sample at all
    with pytest.raises(ValueError, match="no test sample is covered"):
        fuse(w, specs, allow_partial=True)


def test_verdict_is_unavailable_when_the_control_set_is_not_fully_covered(tmp_path):
    w, ctrl, specs = partial_world(tmp_path, n_ctrl=60)
    bigger = cross_pair_control(w["test"], 70)                    # asks for 140 control samples; SE covers only 120
    r = fuse(w, specs, allow_partial=True)
    out = F.analyze(r, w["val"], w["test"], bigger, n_boot=50)
    assert out["primary_verdict"].startswith("UNAVAILABLE") and "not covered" in out["primary_verdict"]


# ---- dry run and command-line wiring ---------------------------------------------------------------
def run_cli(monkeypatch, capsys, w, specs, extra_args, ctrl=None):
    from experiments import run_fusion as R
    monkeypatch.setattr(R, "load_split", lambda name: w["val"] if name == "validation" else w["test"])
    if ctrl is None:
        def no_ctrl():
            raise FileNotFoundError()
        monkeypatch.setattr(R, "load_length_controlled", no_ctrl)
    else:
        monkeypatch.setattr(R, "load_length_controlled", lambda: ctrl)
    argv = []
    for sp in specs:
        argv += ["--input", sp.name, str(sp.val_path), str(sp.test_path)]
    code = R.main(argv + extra_args)
    return code, capsys.readouterr().out


def test_dry_run_writes_nothing_and_uses_no_test_labels(world, monkeypatch, capsys, tmp_path):
    specs = two_inputs(world)
    out_dir = tmp_path / "hybrid_out"
    code, text = run_cli(monkeypatch, capsys, world, specs, ["--dry-run", "--out-dir", str(out_dir)])
    assert code == 0 and "DRY RUN" in text and "validation diagnostics" in text
    assert not out_dir.exists()                                         # nothing written
    assert "Paired differences" not in text and "PRIMARY DECISION" not in text
    # the fit in a dry run is identical to the real one
    real = fuse(world, specs, write=True)
    dry = F.run_fusion(specs, world["val"], world["test"], cfg=CFG, write=False)
    np.testing.assert_array_equal(real["fit"].coef, dry["fit"].coef)
    assert dry["lr"]["paths"] is None


def test_full_cli_run_prints_the_plan_sections(world, monkeypatch, capsys, tmp_path):
    specs = two_inputs(world, 1.5, 1.5, seed=70)
    ctrl = as_matches(world["test"], 100)
    code, text = run_cli(monkeypatch, capsys, world, specs,
                         ["--n-boot", "100", "--out-dir", str(tmp_path / "h"), "--out", str(tmp_path / "r.json")], ctrl=ctrl)
    assert code == 0
    for heading in ("1. Fitted fusion", "2. Metrics with 95% bootstrap CIs", "3. Paired differences",
                    "4. PRIMARY DECISION", "5. Error overlap", "6. Cost of the hybrid"):
        assert heading in text
    import json
    saved = json.loads((tmp_path / "r.json").read_text())
    assert saved["analysis"]["primary_verdict"].split(":")[0] in {"IMPROVES", "INCONCLUSIVE", "WORSE"}


def test_cli_partial_mode_reports_coverage(tmp_path, monkeypatch, capsys):
    w, ctrl, specs = partial_world(tmp_path)
    code, text = run_cli(monkeypatch, capsys, w, specs,
                         ["--allow-partial", "--n-boot", "50", "--out-dir", str(tmp_path / "h")], ctrl=ctrl)
    assert code == 0 and "PARTIAL" in text and "validation 300 of 800" in text and "test 120 of 800" in text
    assert "--scope controlled" in text


def test_diagnostics_report_how_many_logit_scores_are_clipped(world):
    tmp, val, test = world["tmp"], world["val"], world["test"]
    rng = np.random.default_rng(5)
    p = np.clip(1 / (1 + np.exp(-(np.array([s.label for s in val]) * 3 + rng.standard_normal(len(val))))), 0, 1)
    p[:40] = 1.0                                         # exactly saturated scores -> must be counted as clipped
    write_run(tmp, "se_candidate", val, "validation", sig(val, 1, 1))
    write_run(tmp, "se_candidate", test, "test", sig(test, 1, 2))
    write_run(tmp, "sep", val, "validation", p)
    write_run(tmp, "sep", test, "test", np.clip(1 / (1 + np.exp(-sig(test, 3, 4))), 0, 1))
    r = fuse(world, [spec(tmp, "se_candidate", "se"), spec(tmp, "sep", "sep", transform="logit")])
    d = F.diagnostics(r, world["val"])
    assert d["inputs"]["sep"]["clipped_fraction"] == pytest.approx(40 / len(val), abs=0.02)
    assert "clipped_fraction" not in d["inputs"]["se"]
