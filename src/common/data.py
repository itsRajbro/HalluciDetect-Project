"""Dataset interface shared by every detector (Person 1, 2 and 3).

Everybody loads data through this module so that all methods see exactly the
same samples, fields, splits and length features.

Typical use
-----------
    from src.common.data import load_split, load_length_controlled

    train = load_split("train")
    val   = load_split("validation")
    test  = load_split("test")
    ctrl  = load_length_controlled()      # secondary, length-matched test set

Guarantees
----------
* Fields are exactly those written by split.py (sample_id, pair_id, language,
  question, knowledge, answer, label) plus derived length features.
* By default every file is checked against data/split_manifest.json (content
  hash + sample ids), so a stale or edited local copy fails loudly instead of
  silently changing results.
* `answer_length_chars` is len(answer) on the stored answer, with no
  stripping. This is the same definition used to build the length-controlled
  test set, and it is the OFFICIAL length feature (see config.py).
* `answer_length_tokens` is optional (needs the Llama tokenizer): the number
  of tokens of the answer text alone, add_special_tokens=False, no chat
  template. Use `attach_token_lengths`.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from src.common.config import (
    LABEL_FACTUAL, LABEL_HALLUCINATED, SPLITS, Config, get_config,
)

# Tolerance (characters) used when the length-controlled set was built.
# Frozen in experiments/run_length_controlled_test.py (DELTA = 1).
LENGTH_MATCH_TOLERANCE_CHARS = 1

REQUIRED_FIELDS = ("sample_id", "pair_id", "language", "question", "knowledge", "answer", "label")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    pair_id: str
    split: str
    language: str
    question: str
    knowledge: str
    answer: str                      # the candidate answer being judged
    label: int                       # 0 = factual, 1 = hallucinated
    answer_length_chars: int         # OFFICIAL length feature
    answer_length_tokens: Optional[int] = None   # filled by attach_token_lengths
    match_id: Optional[str] = None   # only for the length-controlled set


class DataIntegrityError(RuntimeError):
    """Raised when local data does not match the committed split manifest."""


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------
def _sha256_normalized(path: Path) -> str:
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} not found. Get data/processed from Person 1 or re-run the data pipeline.")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@lru_cache(maxsize=4)
def _load_manifest(path_str: str) -> dict:
    p = Path(path_str)
    if not p.is_file():
        raise FileNotFoundError(
            f"split manifest not found at {p}. Pull the latest repo, or on Person 1's "
            f"machine run: python -m src.data.make_split_manifest build")
    return json.loads(p.read_text(encoding="utf-8"))


def _to_sample(r: dict, split: str) -> Sample:
    for k in REQUIRED_FIELDS:
        if k not in r or r[k] is None:
            raise ValueError(f"row {r.get('sample_id')} missing field '{k}'")
    label = int(r["label"])
    if label not in (LABEL_FACTUAL, LABEL_HALLUCINATED):
        raise ValueError(f"row {r['sample_id']} has invalid label {r['label']!r}")
    ans = r["answer"]
    n_chars = len(ans)
    if "answer_length_chars" in r and r["answer_length_chars"] != n_chars:
        raise ValueError(f"row {r['sample_id']}: stored answer_length_chars={r['answer_length_chars']} "
                         f"but len(answer)={n_chars}")
    return Sample(
        sample_id=r["sample_id"], pair_id=r["pair_id"], split=split, language=r["language"],
        question=r["question"], knowledge=r["knowledge"], answer=ans, label=label,
        answer_length_chars=n_chars, match_id=r.get("match_id"),
    )


def _check_unique_ids(samples: list[Sample], what: str) -> None:
    ids = [s.sample_id for s in samples]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{what}: duplicate sample_id values")


# ---------------------------------------------------------------------------
# public loaders
# ---------------------------------------------------------------------------
def load_split(
    split: str,
    *,
    verify: bool = True,
    processed_dir: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    cfg: Optional[Config] = None,
) -> list[Sample]:
    """Load one of 'train' | 'validation' | 'test' as a list of Sample."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    cfg = cfg or get_config()
    pdir = Path(processed_dir) if processed_dir else cfg.paths.resolve(cfg.paths.processed_dir)
    path = pdir / f"{split}.jsonl"
    rows = _read_jsonl(path)
    samples = [_to_sample(r, split) for r in rows]
    _check_unique_ids(samples, split)

    if verify:
        mpath = Path(manifest_path) if manifest_path else cfg.paths.resolve(cfg.paths.split_manifest)
        manifest = _load_manifest(str(mpath))
        expected_hash = manifest["file_sha256"][split]
        if _sha256_normalized(path) != expected_hash:
            raise DataIntegrityError(
                f"{path} does not match the committed manifest (content hash differs). "
                f"Run: python -m src.data.make_split_manifest verify")
        expected_ids = {f"{stem}_{lab}" for _, stem in manifest["splits"][split] for lab in (0, 1)}
        if {s.sample_id for s in samples} != expected_ids:
            raise DataIntegrityError(f"{split}: sample_ids differ from manifest")
    return samples


def load_all(**kwargs) -> dict[str, list[Sample]]:
    """Load train, validation and test."""
    return {s: load_split(s, **kwargs) for s in SPLITS}


def load_length_controlled(
    *,
    verify: bool = True,
    path: Optional[Path] = None,
    **split_kwargs,
) -> list[Sample]:
    """Load the secondary length-matched test set (a subset of test.jsonl).

    Checks that every row is identical to its row in test.jsonl, that each
    match_id has exactly one factual and one hallucinated sample, and that the
    two answers differ in length by at most LENGTH_MATCH_TOLERANCE_CHARS.
    """
    cfg = split_kwargs.get("cfg") or get_config()
    p = Path(path) if path else cfg.paths.resolve(cfg.paths.length_controlled_test)
    rows = _read_jsonl(p)
    samples = [_to_sample(r, "test") for r in rows]
    _check_unique_ids(samples, "length-controlled")
    for s in samples:
        if not s.match_id:
            raise ValueError(f"{s.sample_id}: missing match_id")

    if verify:
        test = {s.sample_id: s for s in load_split("test", **split_kwargs)}
        for s in samples:
            t = test.get(s.sample_id)
            if t is None:
                raise DataIntegrityError(f"{s.sample_id} is not in test.jsonl")
            if (t.answer, t.label, t.pair_id, t.question) != (s.answer, s.label, s.pair_id, s.question):
                raise DataIntegrityError(f"{s.sample_id} differs from its row in test.jsonl")
        for mid, grp in group_by_match(samples).items():
            if abs(grp[0].answer_length_chars - grp[1].answer_length_chars) > LENGTH_MATCH_TOLERANCE_CHARS:
                raise DataIntegrityError(
                    f"{mid}: length gap exceeds {LENGTH_MATCH_TOLERANCE_CHARS} chars")
    return samples


# ---------------------------------------------------------------------------
# grouping helpers (pair-level and match-level evaluation)
# ---------------------------------------------------------------------------
def group_by_pair(samples: Iterable[Sample]) -> dict[str, tuple[Sample, Sample]]:
    """pair_id -> (factual_sample, hallucinated_sample).

    Used for PAIR-LEVEL evaluation: does the detector score the hallucinated
    answer above the factual answer to the SAME question? Raises if any pair
    is incomplete.
    """
    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        groups[s.pair_id].append(s)
    out = {}
    for pid, grp in groups.items():
        labs = sorted(g.label for g in grp)
        if labs != [LABEL_FACTUAL, LABEL_HALLUCINATED]:
            raise ValueError(f"pair {pid} is incomplete: labels {labs}")
        grp.sort(key=lambda g: g.label)
        out[pid] = (grp[0], grp[1])
    return out


def group_by_match(samples: Iterable[Sample]) -> dict[str, tuple[Sample, Sample]]:
    """match_id -> (factual_sample, hallucinated_sample) for the length-controlled set.

    NOTE: a match pairs a factual and a hallucinated answer of similar length;
    they usually come from DIFFERENT questions (different pair_id).
    """
    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        if s.match_id is None:
            raise ValueError(f"{s.sample_id} has no match_id")
        groups[s.match_id].append(s)
    out = {}
    for mid, grp in groups.items():
        labs = sorted(g.label for g in grp)
        if labs != [LABEL_FACTUAL, LABEL_HALLUCINATED]:
            raise ValueError(f"match {mid} is incomplete: labels {labs}")
        grp.sort(key=lambda g: g.label)
        out[mid] = (grp[0], grp[1])
    return out


# ---------------------------------------------------------------------------
# token-length feature (needs the Llama tokenizer)
# ---------------------------------------------------------------------------
def _tokenizer_fingerprint(tokenizer) -> str:
    name = getattr(tokenizer, "name_or_path", "unknown")
    try:
        size = len(tokenizer)
    except TypeError:
        size = -1
    return f"{name}|{size}"


def attach_token_lengths(
    samples: list[Sample],
    tokenizer,
    *,
    cache_path: Optional[Path] = None,
    batch_size: int = 512,
) -> list[Sample]:
    """Return new Samples with `answer_length_tokens` filled in.

    Token length = len(tokenizer(answer, add_special_tokens=False).input_ids).
    Results are cached by sample_id in `cache_path` (JSON) and reused only if
    the tokenizer fingerprint matches.
    """
    fp = _tokenizer_fingerprint(tokenizer)
    cache: dict[str, int] = {}
    if cache_path and Path(cache_path).is_file():
        blob = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        if blob.get("tokenizer") == fp:
            cache = blob["lengths"]

    missing = [s for s in samples if s.sample_id not in cache]
    for i in range(0, len(missing), batch_size):
        chunk = missing[i:i + batch_size]
        enc = tokenizer([s.answer for s in chunk], add_special_tokens=False)
        for s, ids in zip(chunk, enc["input_ids"]):
            cache[s.sample_id] = len(ids)

    if cache_path and missing:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps({"tokenizer": fp, "lengths": cache}), encoding="utf-8")
    return [replace(s, answer_length_tokens=cache[s.sample_id]) for s in samples]
