"""Standard detector output schema (Step 8). Shared by Person 1, 2 and 3.

Every detector (length_only, transformer, se_global, se_candidate, sep, hybrid)
writes its results in this one format, and the common evaluator (Step 9) reads
only this format.

    from src.common.schema import DetectorRecord, write_detector_output, read_detector_output

    recs = [DetectorRecord(sample_id=s.sample_id, split="validation",
                           detector="sep", variant="layer20_last_answer",
                           score=p_hallucinated,
                           config_hash=cfg.config_hash(),
                           prompt_version=cfg.prompt.version,
                           latency_s=0.02)
            for s in samples]
    write_detector_output(recs)                        # scores only, no threshold yet
    write_detector_output(recs, threshold=0.37)        # + predicted labels (threshold from VALIDATION)

One run = one detector variant on one split = two files:
    outputs/detectors/<detector>/<variant>/<split>.jsonl       one DetectorRecord per line
    outputs/detectors/<detector>/<variant>/<split>.meta.json   run-level metadata

Conventions (from config.py, enforced here)
-------------------------------------------
* score: finite float, HIGHER = MORE LIKELY HALLUCINATED (label 1). No NaN/inf:
  decide how a failed sample is scored before writing (do not drop silently).
* pred_label: 1 iff score >= threshold, else 0 (`predict_label`). A run has a
  threshold if and only if every record has a pred_label.
* threshold_source must be "validation". Test-set tuning is rejected.
* Ground-truth labels, pair_id, match_id and answer_length_chars are NOT stored
  here. The evaluator joins on (split, sample_id) with the dataset interface,
  so there is a single source of truth for them.
* Provenance: config_hash and prompt_version are mandatory on every record.
* Efficiency (optional but please fill in): latency_s, n_generations,
  n_generated_tokens, peak_gpu_mem_mb. For SE these come from GenStats
  (`efficiency_from_gen_stats`); for SEP/baselines n_generations = 0.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.common.config import REPO_ROOT, SPLITS, Config, get_config

SCHEMA_VERSION = "1"
THRESHOLD_SOURCE_VALIDATION = "validation"

_DETECTOR_RE = re.compile(r"^[a-z0-9_]+$")
_VARIANT_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def predict_label(score: float, threshold: float) -> int:
    """The one project-wide decision rule: hallucinated (1) iff score >= threshold."""
    return int(score >= threshold)


def _is_label(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x in (0, 1)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class DetectorRecord:
    sample_id: str
    split: str
    detector: str
    variant: str
    score: float
    config_hash: str
    prompt_version: str
    pred_label: Optional[int] = None
    latency_s: Optional[float] = None
    n_generations: Optional[int] = None
    n_generated_tokens: Optional[int] = None
    peak_gpu_mem_mb: Optional[float] = None
    extra: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sample_id = str(self.sample_id)
        if not self.sample_id:
            raise ValueError("empty sample_id")
        if self.split not in SPLITS:
            raise ValueError(f"unknown split {self.split!r}; expected one of {SPLITS}")
        if not _DETECTOR_RE.match(str(self.detector)):
            raise ValueError(f"detector must match [a-z0-9_]+, got {self.detector!r}")
        if not _VARIANT_RE.match(str(self.variant)):
            raise ValueError(f"variant must match [A-Za-z0-9_.-]+, got {self.variant!r}")
        if isinstance(self.score, bool):
            raise ValueError("score must be a number, not bool")
        try:
            self.score = float(self.score)
        except (TypeError, ValueError):
            raise ValueError(f"score must be a number, got {self.score!r}") from None
        if not math.isfinite(self.score):
            raise ValueError(f"non-finite score for sample {self.sample_id}: {self.score}")
        if self.pred_label is not None and not _is_label(self.pred_label):
            raise ValueError(f"pred_label must be 0, 1 or None, got {self.pred_label!r}")
        if not self.config_hash or not self.prompt_version:
            raise ValueError("config_hash and prompt_version are required")
        for name in ("latency_s", "peak_gpu_mem_mb"):
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or not math.isfinite(float(v)) or v < 0):
                raise ValueError(f"{name} must be a finite number >= 0 or None, got {v!r}")
        for name in ("n_generations", "n_generated_tokens"):
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or int(v) != v or v < 0):
                raise ValueError(f"{name} must be an int >= 0 or None, got {v!r}")
        try:
            json.dumps(self.extra, allow_nan=False)
        except (TypeError, ValueError) as e:
            raise ValueError(f"extra must be strictly JSON-serializable: {e}") from None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DetectorRecord":
        allowed = {f.name for f in fields(cls)}
        unknown = set(d) - allowed
        if unknown:
            raise ValueError(f"unknown record fields: {sorted(unknown)}")
        return cls(**d)


@dataclass
class DetectorRunMeta:
    detector: str
    variant: str
    split: str
    config_hash: str
    prompt_version: str
    n_records: int
    threshold: Optional[float] = None
    threshold_source: Optional[str] = None
    description: str = ""
    created_utc: str = ""
    schema_version: str = SCHEMA_VERSION
    extra: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "DetectorRunMeta":
        allowed = {f.name for f in fields(cls)}
        unknown = set(d) - allowed
        if unknown:
            raise ValueError(f"unknown meta fields: {sorted(unknown)}")
        return cls(**d)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def efficiency_from_gen_stats(stats) -> dict:
    """Map Step 6 `GenStats` (duck-typed) to DetectorRecord efficiency fields.

    Pass `stats` of the whole per-sample SE computation; for per-sample cost
    spread over several calls, sum latency/tokens yourself.
    """
    return dict(
        latency_s=float(stats.wall_time_s),
        n_generations=int(stats.num_sequences),
        n_generated_tokens=int(stats.new_tokens_total),
        peak_gpu_mem_mb=None if stats.peak_gpu_mem_mb is None else float(stats.peak_gpu_mem_mb),
    )


def output_path(detector: str, variant: str, split: str, cfg: Optional[Config] = None) -> Path:
    cfg = cfg or get_config()
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return REPO_ROOT / cfg.paths.outputs_dir / "detectors" / detector / variant / f"{split}.jsonl"


def meta_path(jsonl_path) -> Path:
    p = Path(jsonl_path)
    return p.parent / (p.stem + ".meta.json")


def with_threshold(records: Sequence[DetectorRecord], threshold: float) -> List[DetectorRecord]:
    """Copy of records with pred_label = predict_label(score, threshold)."""
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    return [replace(r, pred_label=predict_label(r.score, threshold)) for r in records]


def check_coverage(
    records: Iterable[DetectorRecord], expected_sample_ids: Iterable, strict: bool = True
) -> dict:
    """Compare record ids with the dataset's ids for that split (str-normalised)."""
    got = {r.sample_id for r in records}
    exp = {str(i) for i in expected_sample_ids}
    missing, extra = sorted(exp - got), sorted(got - exp)
    if strict and (missing or extra):
        raise ValueError(
            f"coverage mismatch: {len(missing)} missing (e.g. {missing[:3]}), "
            f"{len(extra)} unexpected (e.g. {extra[:3]})"
        )
    return {"missing": missing, "extra": extra}


def validate_run(records: Sequence[DetectorRecord], meta: DetectorRunMeta) -> None:
    """Cross-record and record-vs-meta consistency. Raises ValueError."""
    if not records:
        raise ValueError("no records")
    if meta.schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version {meta.schema_version!r} != {SCHEMA_VERSION!r}")
    if meta.n_records != len(records):
        raise ValueError(f"meta.n_records={meta.n_records} but {len(records)} records")
    ids = [r.sample_id for r in records]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})[:3]
        raise ValueError(f"duplicate sample_ids, e.g. {dup}")
    for key in ("detector", "variant", "split", "config_hash", "prompt_version"):
        vals = {getattr(r, key) for r in records} | {getattr(meta, key)}
        if len(vals) != 1:
            raise ValueError(f"inconsistent {key} across records/meta: {sorted(vals)}")
    if meta.threshold is None:
        if meta.threshold_source is not None:
            raise ValueError("threshold_source set but threshold is None")
        if any(r.pred_label is not None for r in records):
            raise ValueError("pred_label present but run has no threshold")
    else:
        if not math.isfinite(meta.threshold):
            raise ValueError("threshold must be finite")
        if meta.threshold_source != THRESHOLD_SOURCE_VALIDATION:
            raise ValueError(
                f"threshold_source must be {THRESHOLD_SOURCE_VALIDATION!r} "
                f"(thresholds are never tuned on test); got {meta.threshold_source!r}"
            )
        for r in records:
            if r.pred_label is None or r.pred_label != predict_label(r.score, meta.threshold):
                raise ValueError(
                    f"pred_label inconsistent with threshold {meta.threshold} "
                    f"for sample {r.sample_id}"
                )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_detector_output(
    records: Sequence[DetectorRecord],
    path=None,
    threshold: Optional[float] = None,
    threshold_source: Optional[str] = None,
    description: str = "",
    extra: Optional[dict] = None,
    cfg: Optional[Config] = None,
) -> Path:
    """Validate and write one run. Returns the .jsonl path.

    If `threshold` is given (must come from validation), pred_label is
    recomputed for every record, and threshold_source defaults to "validation".
    `path` defaults to outputs/detectors/<detector>/<variant>/<split>.jsonl.
    """
    records = list(records)
    if not records:
        raise ValueError("no records")
    first = records[0]
    if threshold is not None:
        records = with_threshold(records, threshold)
        threshold_source = threshold_source or THRESHOLD_SOURCE_VALIDATION
    meta = DetectorRunMeta(
        detector=first.detector,
        variant=first.variant,
        split=first.split,
        config_hash=first.config_hash,
        prompt_version=first.prompt_version,
        n_records=len(records),
        threshold=None if threshold is None else float(threshold),
        threshold_source=threshold_source,
        description=description,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        extra=dict(extra or {}),
    )
    validate_run(records, meta)
    path = Path(path) if path is not None else output_path(
        first.detector, first.variant, first.split, cfg
    )
    lines = [json.dumps(r.to_dict(), sort_keys=True, allow_nan=False) for r in records]
    _atomic_write_text(path, "\n".join(lines) + "\n")
    _atomic_write_text(
        meta_path(path), json.dumps(asdict(meta), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return path


def read_detector_output(path) -> Tuple[List[DetectorRecord], DetectorRunMeta]:
    """Read and fully validate one run (records + meta)."""
    path = Path(path)
    mp = meta_path(path)
    if not path.exists() or not mp.exists():
        raise FileNotFoundError(f"need both {path} and {mp}")
    meta = DetectorRunMeta.from_dict(json.loads(mp.read_text(encoding="utf-8")))
    records = [
        DetectorRecord.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_run(records, meta)
    return records, meta
