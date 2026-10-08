"""Tests for Step 0 (split manifest) and Steps 1-2 (config).

Run from the repo root:  python -m pytest tests -q

They use a synthetic HaluEval-shaped dataset, run the project's real
preprocess.py and split.py on it, and then exercise the manifest tooling, so
no real data or GPU is needed.
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.common.config import (  # noqa: E402
    LABEL_FACTUAL, LABEL_HALLUCINATED, SPLITS, get_config,
)
from src.data import make_split_manifest as mm  # noqa: E402
from src.data import preprocess, split  # noqa: E402


def _make_raw(path: Path, n: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "knowledge": f"Background passage number {i} about topic {i}.",
                "question": f"What is the answer to question {i}?",
                "right_answer": f"Ans{i}",
                "hallucinated_answer": f"A longer made up answer for item {i} that is wrong",
            }) + "\n")


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    raw = root / "raw.jsonl"
    interim = root / "interim.jsonl"
    out = root / "processed"
    _make_raw(raw, 1000)
    preprocess.preprocess_pipeline(input_path=raw, output_path=interim)
    split.split_pipeline(input_path=interim, output_dir=out, random_seed=42)
    return {"root": root, "interim": interim, "processed": out}


def test_manifest_build_and_verify_roundtrip(processed, tmp_path):
    mpath = tmp_path / "manifest.json"
    manifest = mm.build_manifest(processed["processed"])
    mm.write_manifest(manifest, mpath)
    assert json.loads(mpath.read_text(encoding="utf-8")) == manifest
    assert mm.main(["verify", "--processed-dir", str(processed["processed"]), "--manifest", str(mpath)]) == 0


def test_manifest_counts_and_disjointness(processed):
    m = mm.build_manifest(processed["processed"])
    assert sum(m["counts"][s]["samples"] for s in SPLITS) == 2000
    pairs = [set(e[0] for e in m["splits"][s]) for s in SPLITS]
    assert not (pairs[0] & pairs[1]) and not (pairs[0] & pairs[2]) and not (pairs[1] & pairs[2])
    for s in SPLITS:
        assert m["counts"][s]["samples"] == 2 * m["counts"][s]["pairs"]


def test_verify_detects_tampering(processed, tmp_path):
    mpath = tmp_path / "manifest.json"
    mm.write_manifest(mm.build_manifest(processed["processed"]), mpath)
    bad = tmp_path / "processed_bad"
    shutil.copytree(processed["processed"], bad)
    rows = mm.read_jsonl(bad / "test.jsonl")
    rows[0]["answer"] = rows[0]["answer"] + " edited"
    (bad / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert mm.main(["verify", "--processed-dir", str(bad), "--manifest", str(mpath)]) == 1


def test_verify_detects_moved_pair(processed, tmp_path):
    """A pair moved from test to train (the leakage case) must be caught."""
    mpath = tmp_path / "manifest.json"
    mm.write_manifest(mm.build_manifest(processed["processed"]), mpath)
    bad = tmp_path / "processed_leak"
    shutil.copytree(processed["processed"], bad)
    test_rows = mm.read_jsonl(bad / "test.jsonl")
    train_rows = mm.read_jsonl(bad / "train.jsonl")
    moved_pid = test_rows[0]["pair_id"]
    moved = [r for r in test_rows if r["pair_id"] == moved_pid]
    test_rows = [r for r in test_rows if r["pair_id"] != moved_pid]
    train_rows += moved
    (bad / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in test_rows), encoding="utf-8")
    (bad / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train_rows), encoding="utf-8")
    assert mm.main(["verify", "--processed-dir", str(bad), "--manifest", str(mpath)]) == 1


def test_build_rejects_broken_pair(processed, tmp_path):
    bad = tmp_path / "processed_broken"
    shutil.copytree(processed["processed"], bad)
    rows = mm.read_jsonl(bad / "validation.jsonl")
    (bad / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]), encoding="utf-8")
    with pytest.raises(ValueError):
        mm.build_manifest(bad)


def test_hash_is_line_ending_independent(processed, tmp_path):
    src = processed["processed"] / "test.jsonl"
    crlf = tmp_path / "crlf.jsonl"
    crlf.write_bytes(src.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert mm.sha256_normalized(src) == mm.sha256_normalized(crlf)


def test_reproduce_from_interim(processed, tmp_path):
    mpath = tmp_path / "manifest.json"
    mm.write_manifest(mm.build_manifest(processed["processed"]), mpath)
    assert mm.main(["reproduce", "--interim", str(processed["interim"]), "--manifest", str(mpath)]) == 0


def test_config_conventions_and_hash():
    cfg = get_config()
    assert (LABEL_FACTUAL, LABEL_HALLUCINATED) == (0, 1)
    assert cfg.model.model_id == "meta-llama/Llama-3.2-3B-Instruct"
    assert cfg.model.bnb_4bit_quant_type == "nf4"
    assert cfg.config_hash() == get_config().config_hash()
    # paths must not influence the hash (they differ across machines)
    from dataclasses import replace
    other = replace(cfg, paths=replace(cfg.paths, outputs_dir="somewhere_else"))
    assert other.config_hash() == cfg.config_hash()
    # but a real setting must
    changed = replace(cfg, sampled=replace(cfg.sampled, temperature=0.5))
    assert changed.config_hash() != cfg.config_hash()
