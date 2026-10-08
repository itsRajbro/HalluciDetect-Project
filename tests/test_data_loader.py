"""Tests for src/common/data.py (Step 3). Run: python -m pytest tests -q

Uses a synthetic HaluEval-shaped dataset pushed through the project's real
preprocess.py and split.py, so no real data / GPU / tokenizer download needed.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.common import data as D  # noqa: E402
from src.data import make_split_manifest as mm  # noqa: E402
from src.data import preprocess, split  # noqa: E402


def _write_raw(path: Path, n: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            k = 3 + (i % 25)
            f.write(json.dumps({
                "knowledge": f"Background passage {i} about topic {i}.",
                "question": f"What is the answer to question {i}?",
                "right_answer": "x" * k + f"{i}",
                "hallucinated_answer": "y" * (k + 1) + f"{i}",
            }) + "\n")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("data_env")
    raw, interim, proc = root / "raw.jsonl", root / "interim.jsonl", root / "processed"
    _write_raw(raw, 600)
    preprocess.preprocess_pipeline(input_path=raw, output_path=interim)
    split.split_pipeline(input_path=interim, output_dir=proc, random_seed=42)
    manifest = root / "split_manifest.json"
    mm.write_manifest(mm.build_manifest(proc), manifest)

    # synthetic length-controlled set: pair each test pair's two answers
    # (their lengths differ by 1 by construction) under one match_id.
    rows = mm.read_jsonl(proc / "test.jsonl")
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r)
    ctrl = proc / "test_length_controlled.jsonl"
    with open(ctrl, "w", encoding="utf-8") as f:
        for j, grp in enumerate(by_pair.values()):
            for r in grp:
                out = {k: r[k] for k in D.REQUIRED_FIELDS}
                out["match_id"] = f"M{j:04d}"
                out["answer_length_chars"] = len(r["answer"])
                f.write(json.dumps(out) + "\n")
    return {"proc": proc, "manifest": manifest, "ctrl": ctrl, "root": root}


def kw(env):
    return {"processed_dir": env["proc"], "manifest_path": env["manifest"]}


def test_load_all_splits_and_fields(env):
    allsets = D.load_all(**kw(env))
    assert set(allsets) == {"train", "validation", "test"}
    for split_name, samples in allsets.items():
        assert samples and all(s.split == split_name for s in samples)
        assert all(s.label in (0, 1) for s in samples)
        assert all(s.answer_length_chars == len(s.answer) for s in samples)
        assert all(s.answer_length_tokens is None and s.match_id is None for s in samples)
    ids = [s.sample_id for v in allsets.values() for s in v]
    assert len(ids) == len(set(ids)) == 1200


def test_no_pair_overlap_between_splits(env):
    a = D.load_all(**kw(env))
    pairs = {k: {s.pair_id for s in v} for k, v in a.items()}
    assert not (pairs["train"] & pairs["validation"])
    assert not (pairs["train"] & pairs["test"])
    assert not (pairs["validation"] & pairs["test"])


def test_edited_file_is_rejected(env, tmp_path):
    import shutil
    bad = tmp_path / "proc_bad"
    shutil.copytree(env["proc"], bad)
    rows = mm.read_jsonl(bad / "validation.jsonl")
    rows[0]["answer"] += " tampered"
    (bad / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(D.DataIntegrityError):
        D.load_split("validation", processed_dir=bad, manifest_path=env["manifest"])
    # verify=False bypasses the manifest (explicit opt-out)
    assert D.load_split("validation", processed_dir=bad, manifest_path=env["manifest"], verify=False)


def test_unknown_split_and_missing_manifest(env, tmp_path):
    with pytest.raises(ValueError):
        D.load_split("dev", **kw(env))
    with pytest.raises(FileNotFoundError):
        D.load_split("test", processed_dir=env["proc"], manifest_path=tmp_path / "nope.json")


def test_group_by_pair_gives_factual_then_hallucinated(env):
    test = D.load_split("test", **kw(env))
    pairs = D.group_by_pair(test)
    assert len(pairs) == len(test) // 2
    for pid, (f, h) in pairs.items():
        assert (f.label, h.label) == (0, 1) and f.pair_id == h.pair_id == pid
    with pytest.raises(ValueError):
        D.group_by_pair(test[:-1])   # one pair left incomplete


def test_length_controlled_loads_and_groups(env):
    ctrl = D.load_length_controlled(path=env["ctrl"], **kw(env))
    assert ctrl and all(s.match_id for s in ctrl)
    m = D.group_by_match(ctrl)
    for mid, (f, h) in m.items():
        assert (f.label, h.label) == (0, 1)
        assert abs(f.answer_length_chars - h.answer_length_chars) <= D.LENGTH_MATCH_TOLERANCE_CHARS


def test_length_controlled_rejects_forged_row(env, tmp_path):
    rows = mm.read_jsonl(env["ctrl"])
    rows[0]["answer"] = rows[0]["answer"][:-1] + "Z"       # same length, different text
    rows[0]["answer_length_chars"] = len(rows[0]["answer"])
    p = tmp_path / "forged.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(D.DataIntegrityError):
        D.load_length_controlled(path=p, **kw(env))


def test_length_controlled_rejects_large_gap(env, tmp_path):
    rows = mm.read_jsonl(env["ctrl"])
    # move one sample into a different test sample's match so the gap grows
    a = next(r for r in rows if r["label"] == 0)
    far = max((r for r in rows if r["label"] == 1), key=lambda r: abs(len(r["answer"]) - len(a["answer"])))
    assert abs(len(far["answer"]) - len(a["answer"])) > D.LENGTH_MATCH_TOLERANCE_CHARS
    for r in rows:
        if r["sample_id"] == far["sample_id"]:
            r["match_id"] = a["match_id"]
        elif r["match_id"] == a["match_id"] and r["label"] == 1:
            r["match_id"] = far["match_id"]
    p = tmp_path / "gap.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(D.DataIntegrityError):
        D.load_length_controlled(path=p, **kw(env))


class FakeTokenizer:
    """Whitespace/char tokenizer stand-in with the HF call signature we use."""
    name_or_path = "fake/tok"
    calls = 0

    def __len__(self):
        return 1000

    def __call__(self, texts, add_special_tokens=True):
        FakeTokenizer.calls += 1
        assert add_special_tokens is False
        return {"input_ids": [[1] * (len(t) // 2 + 1) for t in texts]}


def test_token_lengths_and_cache(env, tmp_path):
    test = D.load_split("test", **kw(env))
    cache = tmp_path / "tok_cache.json"
    FakeTokenizer.calls = 0
    out = D.attach_token_lengths(test, FakeTokenizer(), cache_path=cache, batch_size=64)
    assert all(s.answer_length_tokens == len(s.answer) // 2 + 1 for s in out)
    assert out[0].sample_id == test[0].sample_id and test[0].answer_length_tokens is None  # originals untouched
    first_calls = FakeTokenizer.calls
    assert first_calls > 0
    D.attach_token_lengths(test, FakeTokenizer(), cache_path=cache)   # fully cached
    assert FakeTokenizer.calls == first_calls

    class Other(FakeTokenizer):
        name_or_path = "another/tok"
    D.attach_token_lengths(test, Other(), cache_path=cache)           # fingerprint changed -> recompute
    assert FakeTokenizer.calls > first_calls
