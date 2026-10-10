"""Protocol tests for src/detectors/baselines/transformer.py (no torch needed).

The real TorchBackend is NOT exercised here (run `python -m experiments.run_transformer smoke`
on a GPU for that). These tests check the part that makes the baseline scientifically valid:
selection on validation only, frozen config, test locked until freeze, re-selection blocked.
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.common.config import get_config  # noqa: E402
from src.common.schema import read_detector_output  # noqa: E402
from src.detectors.baselines import transformer as T  # noqa: E402
from tests.test_evaluator import make_samples  # noqa: E402


class FakeBackend:
    """Scores = label + noise; the noise level per (lr, epoch) is set by `quality`."""

    def __init__(self, quality):
        self.quality = quality
        self.calls = []

    def train(self, cfg, lr, train_samples, val_samples, workdir):
        self.calls.append(("train", lr, len(train_samples)))
        workdir.mkdir(parents=True, exist_ok=True)
        y = np.array([s.label for s in val_samples], float)
        out = []
        for epoch in range(1, cfg.max_epochs + 1):
            noise = self.quality[(lr, epoch)]
            rng = np.random.default_rng(int(lr * 1e6) + epoch)
            ckpt = workdir / f"epoch{epoch}.pt"
            ckpt.write_text(f"{noise}\nlr={lr}\nepoch={epoch}\n")
            out.append(T.EpochResult(epoch, y + noise * rng.standard_normal(len(y)), ckpt, 0.5, 1.0))
        return out

    def score(self, cfg, checkpoint, samples):
        self.calls.append(("score", len(samples)))
        noise = float(Path(checkpoint).read_text().splitlines()[0])
        y = np.array([s.label for s in samples], float)
        rng = np.random.default_rng(999)
        return T.ScoreResult(y + noise * rng.standard_normal(len(y)), 0.002, 1234.0)


QUALITY = {(2e-5, 1): 1.5, (2e-5, 2): 1.0, (2e-5, 3): 0.9,
           (5e-5, 1): 1.2, (5e-5, 2): 0.3, (5e-5, 3): 0.6}     # best: lr=5e-5, epoch 2


@pytest.fixture()
def world(tmp_path):
    return {"train": make_samples(100, "train", seed=1),
            "val": make_samples(150, "validation", seed=2, start=1000),
            "test": make_samples(150, "test", seed=3, start=5000),
            "out": tmp_path / "artifacts", "runs": tmp_path / "runs"}


def freeze(world, backend=None, cfg=None, project_cfg=None, force=False):
    backend = backend or FakeBackend(QUALITY)
    cfg = cfg or T.TransformerBaselineConfig()
    return backend, T.select_and_freeze(
        cfg, world["out"], project_cfg=project_cfg or get_config(), backend=backend,
        train_samples=world["train"], val_samples=world["val"], runs_dir=world["runs"], force=force)


def test_config_variant_and_validation():
    c = T.TransformerBaselineConfig()
    assert c.variant() == "distilroberta-base_qak"
    assert T.TransformerBaselineConfig(include_knowledge=False).variant() == "distilroberta-base_qa"
    assert T.TransformerBaselineConfig(model_name="org/My Model", train_subset=400).variant() == "My-Model_qak_smoke"
    assert T.TransformerBaselineConfig.from_dict(json.loads(json.dumps(c.to_dict()))) == c
    with pytest.raises(ValueError):
        T.TransformerBaselineConfig(train_subset=401)          # would split a question pair


def test_build_inputs_formats():
    s = make_samples(1, "train")[0]
    a, b = T.build_inputs(s, True)
    assert a.startswith("Question: ") and "\nAnswer: " in a and b.startswith("Knowledge: ")
    assert T.build_inputs(s, False)[1] is None


def test_pick_best_prefers_highest_and_handles_missing():
    h = [{"val_auroc": 0.7}, {"val_auroc": None}, {"val_auroc": 0.9}, {"val_auroc": 0.9}, {"val_auroc": float("nan")}]
    assert T.pick_best(h) == 2                                  # tie -> first
    assert T.pick_best([{"val_auroc": None}, {"val_auroc": 0.1}]) == 1
    with pytest.raises(ValueError):
        T.pick_best([])


def test_select_and_freeze_chooses_best_on_validation_and_never_scores_test(world):
    backend, frozen = freeze(world)
    assert (frozen["selected"]["lr"], frozen["selected"]["epoch"]) == (5e-5, 2)
    assert all(c[0] == "train" for c in backend.calls)          # no scoring => test untouched
    assert [c[2] for c in backend.calls] == [len(world["train"])] * 2
    assert (world["out"] / T.BEST_CKPT).read_text().startswith("0.3")   # the epoch-2, lr=5e-5 weights
    assert not (world["out"] / "epochs").exists()               # other epochs' weights removed
    assert not (world["out"] / T.TEST_MARKER).exists()
    recs, meta = read_detector_output(world["runs"] / "validation.jsonl")
    assert meta.detector == "transformer" and meta.threshold_source == "validation"
    assert len(recs) == len(world["val"]) and meta.extra["uses_llama_prompt"] is False
    log = json.loads((world["out"] / T.LOG_FILE).read_text())["history"]
    assert len(log) == 6 and max(r["val_auroc"] for r in log) == frozen["selected"]["val_auroc"]


def test_test_phase_requires_freeze(world):
    with pytest.raises(FileNotFoundError):
        T.predict_test_and_evaluate(world["out"], backend=FakeBackend(QUALITY), test_samples=world["test"],
                                    val_samples=world["val"], include_length_controlled=False)


def test_full_protocol_then_reselection_is_blocked(world):
    backend, frozen = freeze(world)
    rep = T.predict_test_and_evaluate(world["out"], backend=backend, test_samples=world["test"],
                                      val_samples=world["val"], include_length_controlled=False, save=False)
    assert rep["detector"] == "transformer/distilroberta-base_qak"
    assert rep["threshold"]["value"] == frozen["threshold"]["value"]            # frozen threshold, not re-tuned
    assert rep["threshold"]["selected_on"] == "validation"
    assert rep["test"]["metrics"]["auroc"] > 0.9
    assert rep["efficiency"]["test_peak_gpu_mem_mb"] == 1234.0
    assert backend.calls.count(("score", len(world["test"]))) == 1              # test scored exactly once
    assert (world["out"] / T.TEST_MARKER).exists()
    # test run on disk uses the frozen threshold
    _, meta = read_detector_output(world["runs"] / "test.jsonl")
    assert meta.threshold == frozen["threshold"]["value"] and meta.threshold_source == "validation"
    # re-selection after test: blocked, unless forced (and then it is recorded)
    with pytest.raises(RuntimeError, match="already been evaluated"):
        freeze(world)
    _, again = freeze(world, force=True)
    assert again["reselected_after_test"] is True


def test_tampered_checkpoint_is_rejected(world):
    backend, _ = freeze(world)
    (world["out"] / T.BEST_CKPT).write_text("0.0\nswapped weights\n")
    with pytest.raises(ValueError, match="checkpoint does not match"):
        T.predict_test_and_evaluate(world["out"], backend=backend, test_samples=world["test"],
                                    val_samples=world["val"], include_length_controlled=False)
    assert not any(c[0] == "score" for c in backend.calls)


def test_changed_project_config_is_rejected(world):
    backend, _ = freeze(world)
    base = get_config()
    other = replace(base, run=replace(base.run, seed=7))
    assert other.config_hash() != base.config_hash()
    with pytest.raises(ValueError, match="config_hash changed"):
        T.predict_test_and_evaluate(world["out"], project_cfg=other, backend=backend,
                                    test_samples=world["test"], val_samples=world["val"],
                                    include_length_controlled=False)


def test_smoke_subset_is_applied_and_labelled(world):
    cfg = T.TransformerBaselineConfig(train_subset=40, lr_grid=(5e-5,), max_epochs=1)
    quality = {(5e-5, 1): 0.5}
    backend, frozen = freeze(world, backend=FakeBackend(quality), cfg=cfg)
    assert backend.calls == [("train", 5e-5, 40)]
    assert frozen["variant"].endswith("_smoke")


def test_invalid_backend_scores_are_rejected(world):
    class Bad(FakeBackend):
        def train(self, cfg, lr, train_samples, val_samples, workdir):
            res = super().train(cfg, lr, train_samples, val_samples, workdir)
            res[0].val_scores = res[0].val_scores[:-1]
            return res
    with pytest.raises(ValueError, match="invalid validation scores"):
        freeze(world, backend=Bad(QUALITY))
