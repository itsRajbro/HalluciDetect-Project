"""Step 8 tests: src/common/schema.py (pure Python, no GPU)."""
import json
import math

import pytest

from src.common import schema as S

KW = dict(split="validation", detector="sep", variant="layer20_last_answer",
          config_hash="abc123", prompt_version="v1")


def rec(sid, score=0.3, **over):
    kw = dict(KW, sample_id=sid, score=score)
    kw.update(over)
    return S.DetectorRecord(**kw)


def recs():
    return [rec("1", 0.1), rec("2", 0.5), rec("3", 0.9)]


def test_predict_label_rule_and_tie():
    assert S.predict_label(0.5, 0.5) == 1
    assert S.predict_label(0.4999, 0.5) == 0
    assert S.predict_label(0.9, 0.5) == 1


@pytest.mark.parametrize("over", [
    dict(score=float("nan")),
    dict(score=float("inf")),
    dict(score="abc"),
    dict(score=True),
    dict(split="dev"),
    dict(detector="SEP"),
    dict(detector="se global"),
    dict(variant="a/b"),
    dict(sample_id=""),
    dict(pred_label=2),
    dict(pred_label=True),
    dict(config_hash=""),
    dict(prompt_version=""),
    dict(latency_s=-1.0),
    dict(n_generations=-1),
    dict(extra={"x": float("nan")}),
    dict(extra={"x": object()}),
])
def test_invalid_records(over):
    with pytest.raises(ValueError):
        rec("1", **over)


def test_sample_id_and_score_coercion():
    r = rec(7, 1)
    assert r.sample_id == "7" and r.score == 1.0 and isinstance(r.score, float)


def test_roundtrip_scores_only(tmp_path):
    p = tmp_path / "validation.jsonl"
    out = S.write_detector_output(recs(), path=p, description="d")
    assert out == p
    back, meta = S.read_detector_output(p)
    assert [r.sample_id for r in back] == ["1", "2", "3"]
    assert [r.score for r in back] == [0.1, 0.5, 0.9]
    assert meta.threshold is None and all(r.pred_label is None for r in back)
    assert meta.n_records == 3 and meta.detector == "sep" and meta.description == "d"
    assert (tmp_path / "validation.meta.json").exists()


def test_roundtrip_with_threshold(tmp_path):
    p = tmp_path / "validation.jsonl"
    S.write_detector_output(recs(), path=p, threshold=0.5)
    back, meta = S.read_detector_output(p)
    assert [r.pred_label for r in back] == [0, 1, 1]
    assert meta.threshold == 0.5 and meta.threshold_source == "validation"


def test_threshold_source_must_be_validation(tmp_path):
    with pytest.raises(ValueError):
        S.write_detector_output(recs(), path=tmp_path / "x.jsonl", threshold=0.5,
                                threshold_source="test")


def test_validate_run_catches_inconsistencies():
    meta = S.DetectorRunMeta(detector="sep", variant="layer20_last_answer",
                             split="validation", config_hash="abc123",
                             prompt_version="v1", n_records=3)
    S.validate_run(recs(), meta)  # ok
    # duplicates
    with pytest.raises(ValueError):
        S.validate_run([rec("1", 0.1), rec("1", 0.2), rec("3", 0.3)], meta)
    # mixed detectors
    with pytest.raises(ValueError):
        S.validate_run([rec("1", 0.1), rec("2", 0.2, detector="se_global"), rec("3", 0.3)], meta)
    # mixed config hashes
    with pytest.raises(ValueError):
        S.validate_run([rec("1", 0.1), rec("2", 0.2, config_hash="zzz"), rec("3", 0.3)], meta)
    # pred_label without threshold
    with pytest.raises(ValueError):
        S.validate_run([rec("1", 0.1, pred_label=0), rec("2", 0.2), rec("3", 0.3)], meta)
    # wrong n_records
    bad = S.DetectorRunMeta(**{**meta.__dict__, "n_records": 2})
    with pytest.raises(ValueError):
        S.validate_run(recs(), bad)
    # empty
    with pytest.raises(ValueError):
        S.validate_run([], meta)


def test_pred_label_must_match_threshold():
    meta = S.DetectorRunMeta(detector="sep", variant="layer20_last_answer",
                             split="validation", config_hash="abc123",
                             prompt_version="v1", n_records=3,
                             threshold=0.5, threshold_source="validation")
    good = S.with_threshold(recs(), 0.5)
    S.validate_run(good, meta)
    wrong = [rec("1", 0.1, pred_label=1), good[1], good[2]]
    with pytest.raises(ValueError):
        S.validate_run(wrong, meta)
    missing = [rec("1", 0.1), good[1], good[2]]
    with pytest.raises(ValueError):
        S.validate_run(missing, meta)


def test_read_rejects_unknown_fields_and_tampering(tmp_path):
    p = tmp_path / "validation.jsonl"
    S.write_detector_output(recs(), path=p, threshold=0.5)
    lines = p.read_text().splitlines()
    d = json.loads(lines[0]); d["oops"] = 1
    p.write_text("\n".join([json.dumps(d)] + lines[1:]) + "\n")
    with pytest.raises(ValueError):
        S.read_detector_output(p)
    # tamper pred_label
    S.write_detector_output(recs(), path=p, threshold=0.5)
    lines = p.read_text().splitlines()
    d = json.loads(lines[0]); d["pred_label"] = 1
    p.write_text("\n".join([json.dumps(d)] + lines[1:]) + "\n")
    with pytest.raises(ValueError):
        S.read_detector_output(p)


def test_read_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        S.read_detector_output(tmp_path / "nope.jsonl")


def test_check_coverage():
    r = recs()
    assert S.check_coverage(r, [1, 2, 3]) == {"missing": [], "extra": []}
    with pytest.raises(ValueError):
        S.check_coverage(r, [1, 2, 3, 4])
    info = S.check_coverage(r, [2, 3, 4], strict=False)
    assert info == {"missing": ["4"], "extra": ["1"]}


def test_efficiency_from_gen_stats():
    class St:
        wall_time_s = 1.5
        num_sequences = 10
        new_tokens_total = 120
        peak_gpu_mem_mb = None
    e = S.efficiency_from_gen_stats(St())
    assert e == dict(latency_s=1.5, n_generations=10, n_generated_tokens=120, peak_gpu_mem_mb=None)
    r = rec("1", 0.3, **e)
    assert r.n_generations == 10


def test_output_path_layout():
    p = S.output_path("sep", "layer20_last_answer", "test")
    assert p.parts[-3:] == ("sep", "layer20_last_answer", "test.jsonl")
    assert "detectors" in p.parts
    with pytest.raises(ValueError):
        S.output_path("sep", "v", "dev")


def test_with_threshold_does_not_mutate():
    r = recs()
    out = S.with_threshold(r, 0.5)
    assert all(x.pred_label is None for x in r)
    assert [x.pred_label for x in out] == [0, 1, 1]
    with pytest.raises(ValueError):
        S.with_threshold(r, math.nan)
