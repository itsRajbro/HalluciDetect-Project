"""Length-only baseline (Person 1).

Predicts hallucination from answer length and nothing else.

Protocol (all selection on VALIDATION only; test is only scored afterwards):
  1. direction: +1 if longer answers are more often hallucinated (validation
     AUROC of the raw length >= 0.5), else -1.
  2. score = direction * length, so the project convention holds
     (HIGHER score = MORE LIKELY HALLUCINATED) whatever the direction is.
  3. threshold: chosen on validation scores with the shared rule from
     src/common/evaluator.py (default Youden's J). pred = 1 iff score >= threshold.

Features
  "chars"  : answer_length_chars = len(answer). OFFICIAL feature (same metric
             that defines the length-controlled test set).
  "tokens" : Llama-tokenizer length of the answer alone (secondary; needs
             data.attach_token_lengths first).

Outputs are written with the shared schema as detector="length_only",
variant=<feature>, so src/common/evaluator.evaluate_run reads them like any
other detector.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.common import evaluator as E
from src.common.config import Config, get_config
from src.common.data import Sample
from src.common.schema import DetectorRecord, write_detector_output

DETECTOR = "length_only"
FEATURES = {"chars": "answer_length_chars", "tokens": "answer_length_tokens"}


@dataclass(frozen=True)
class LengthOnlyModel:
    feature: str            # "chars" | "tokens"
    direction: int          # +1: longer = more hallucinated, -1: shorter = more hallucinated
    threshold: float        # on the directed score (direction * length)
    criterion: str          # threshold-selection criterion
    val_auroc_raw: float    # validation AUROC of the RAW length (before applying direction)
    n_val: int

    def to_dict(self) -> dict:
        return asdict(self)


def lengths(samples: Sequence[Sample], feature: str = "chars") -> np.ndarray:
    if feature not in FEATURES:
        raise ValueError(f"feature must be one of {sorted(FEATURES)}")
    attr = FEATURES[feature]
    vals = [getattr(s, attr) for s in samples]
    if any(v is None for v in vals):
        raise ValueError(f"{attr} is missing; call data.attach_token_lengths(...) first")
    return np.asarray(vals, dtype=float)


def fit(val_samples: Sequence[Sample], *, feature: str = "chars", criterion: str = "youden") -> LengthOnlyModel:
    """Select direction and threshold on VALIDATION samples."""
    y = np.array([s.label for s in val_samples], dtype=int)
    x = lengths(val_samples, feature)
    auc = E.auroc(y, x)
    if auc is None:
        raise ValueError("validation labels contain a single class")
    direction = 1 if auc >= 0.5 else -1
    thr = E.select_threshold(y, direction * x, criterion)
    return LengthOnlyModel(feature=feature, direction=direction, threshold=thr["value"],
                           criterion=criterion, val_auroc_raw=float(auc), n_val=len(y))


def score(samples: Sequence[Sample], model: LengthOnlyModel) -> np.ndarray:
    """Directed score: higher = more likely hallucinated."""
    return model.direction * lengths(samples, model.feature)


def to_records(samples: Sequence[Sample], split: str, model: LengthOnlyModel, cfg: Config,
               latency_s: float = 0.0) -> list[DetectorRecord]:
    scores = score(samples, model)
    return [DetectorRecord(sample_id=s.sample_id, split=split, detector=DETECTOR, variant=model.feature,
                           score=float(sc), config_hash=cfg.config_hash(), prompt_version=cfg.prompt.version,
                           latency_s=latency_s, n_generations=0)
            for s, sc in zip(samples, scores)]


def run(val_samples: Sequence[Sample], test_samples: Sequence[Sample], *, feature: str = "chars",
        criterion: str = "youden", out_dir: Optional[Path] = None, cfg: Optional[Config] = None) -> dict:
    """Fit on validation, score validation and test, write both runs with the validation threshold.

    Returns {"model": LengthOnlyModel, "val_path": Path, "test_path": Path}.
    `out_dir` (optional) replaces the default outputs/detectors/length_only/<feature>/.
    """
    cfg = cfg or get_config()
    model = fit(val_samples, feature=feature, criterion=criterion)       # validation only
    t0 = time.perf_counter()
    score(list(val_samples) + list(test_samples), model)
    each = (time.perf_counter() - t0) / (len(val_samples) + len(test_samples))
    extra = {"model": model.to_dict(), "feature_definition":
             "len(answer) in characters" if feature == "chars" else "Llama tokens of the answer alone"}
    paths = {}
    for split, samples in (("validation", val_samples), ("test", test_samples)):
        path = None if out_dir is None else Path(out_dir) / f"{split}.jsonl"
        paths[split] = write_detector_output(
            to_records(samples, split, model, cfg, each), path=path, threshold=model.threshold,
            description=f"length-only baseline ({feature}); direction {model.direction:+d}", extra=extra, cfg=cfg)
    return {"model": model, "val_path": paths["validation"], "test_path": paths["test"]}
