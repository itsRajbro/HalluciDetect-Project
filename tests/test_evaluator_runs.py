"""Tests for evaluator.evaluate_run + the Step 8 schema. Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.common import evaluator as E  # noqa: E402
from src.common.schema import DetectorRecord, write_detector_output  # noqa: E402
from tests.test_evaluator import make_samples  # noqa: E402


def to_records(samples, split, fn, *, detector="length_only", variant="chars",
               config_hash="cfg1", prompt_version="v1", **eff):
    return [DetectorRecord(sample_id=s.sample_id, split=split, detector=detector, variant=variant,
                           score=fn(s), config_hash=config_hash, prompt_version=prompt_version, **eff)
            for s in samples]


LEN = lambda s: s.answer_length_chars


@pytest.fixture()
def data():
    return {"val": make_samples(200, "validation", seed=21),
            "test": make_samples(200, "test", seed=22, start=9000)}


def write_pair(tmp, data, *, thr=None, test_kwargs=None, val_kwargs=None):
    vp = write_detector_output(to_records(data["val"], "validation", LEN, **(val_kwargs or {})),
                               path=tmp / "val.jsonl")
    tp = write_detector_output(to_records(data["test"], "test", LEN, **(test_kwargs or {})),
                               path=tmp / "test.jsonl", threshold=thr)
    return vp, tp


def run(vp, tp, data, **kw):
    return E.evaluate_run(test_path=tp, val_path=vp, val_samples=data["val"], test_samples=data["test"],
                          include_length_controlled=False, **kw)


def test_end_to_end_selects_threshold_on_validation(tmp_path, data):
    vp, tp = write_pair(tmp_path, data)
    rep = run(vp, tp, data)
    assert rep["detector"] == "length_only/chars" and rep["config_hash"] == "cfg1"
    assert rep["prompt_version"] == "v1"
    assert rep["threshold"]["selected_on"] == "validation"
    assert rep["test"]["metrics"]["auroc"] > 0.98


def test_threshold_stored_in_test_meta_is_used(tmp_path, data):
    vp, tp = write_pair(tmp_path, data, thr=29.5)
    rep = run(vp, tp, data)
    assert rep["threshold"]["value"] == 29.5 and rep["threshold"]["criterion"] == "stored_in_run_meta"
    assert rep["threshold"]["selected_on"] == "validation"
    # the detector's own pred_label agrees with the evaluator's decision rule
    assert rep["test"]["metrics_native_pred_label"]["accuracy"] == rep["test"]["metrics"]["accuracy"]


def test_test_run_only_with_stored_threshold(tmp_path, data):
    _, tp = write_pair(tmp_path, data, thr=29.5)
    rep = E.evaluate_run(test_path=tp, test_samples=data["test"], include_length_controlled=False)
    assert "validation" not in rep and rep["threshold"]["value"] == 29.5


def test_test_run_without_any_threshold_is_rejected(tmp_path, data):
    _, tp = write_pair(tmp_path, data)
    with pytest.raises(ValueError):
        E.evaluate_run(test_path=tp, test_samples=data["test"], include_length_controlled=False)


def test_provenance_mismatch_is_rejected(tmp_path, data):
    for key, val in (("config_hash", "OTHER"), ("prompt_version", "v2"), ("variant", "tokens"), ("detector", "sep")):
        vp, tp = write_pair(tmp_path, data, val_kwargs={key: val})
        with pytest.raises(ValueError, match=key):
            run(vp, tp, data)


def test_wrong_split_files_are_rejected(tmp_path, data):
    vp, tp = write_pair(tmp_path, data)
    with pytest.raises(ValueError):
        E.evaluate_run(test_path=vp, val_path=vp, val_samples=data["val"], test_samples=data["val"],
                       include_length_controlled=False)
    with pytest.raises(ValueError):
        E.evaluate_run(test_path=tp, val_path=tp, val_samples=data["val"], test_samples=data["test"],
                       include_length_controlled=False)


def test_missing_samples_are_rejected(tmp_path, data):
    vp = write_detector_output(to_records(data["val"], "validation", LEN), path=tmp_path / "val.jsonl")
    tp = write_detector_output(to_records(data["test"][:-2], "test", LEN), path=tmp_path / "test.jsonl")
    with pytest.raises(ValueError, match="coverage"):
        run(vp, tp, data)


def test_efficiency_fields_flow_into_report(tmp_path, data):
    vp, tp = write_pair(tmp_path, data, test_kwargs=dict(latency_s=0.5, n_generations=10,
                                                         n_generated_tokens=100, peak_gpu_mem_mb=2200.0))
    rep = run(vp, tp, data)
    n = len(data["test"])
    eff = rep["efficiency"]
    assert eff["test_total_runtime_sec"] == pytest.approx(0.5 * n)
    assert eff["test_total_generations"] == 10 * n and eff["test_total_generated_tokens"] == 100 * n
    assert eff["test_peak_gpu_mem_mb"] == 2200.0


def test_missing_efficiency_gives_warning(tmp_path, data):
    vp, tp = write_pair(tmp_path, data)
    rep = run(vp, tp, data)
    assert any("efficiency" in w for w in rep["warnings"])


def test_save_report(tmp_path, data):
    vp, tp = write_pair(tmp_path, data)
    out = tmp_path / "eval" / "r.json"
    run(vp, tp, data, save=True, out_path=out)
    assert out.is_file()
