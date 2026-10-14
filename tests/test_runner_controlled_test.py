import json

import pytest

from src.detectors.semantic_entropy.runner_controlled_test import (
    atomic_jsonl,
    read_checkpoint,
)


def test_atomic_jsonl_writes_valid_jsonl(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    records = [
        {"sample_id": "S001", "score": 0.25},
        {"sample_id": "S002", "score": 0.75},
    ]

    atomic_jsonl(path, records)

    loaded = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]

    assert loaded == records
    assert not path.with_name(path.name + ".tmp").exists()


def test_atomic_jsonl_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "checkpoint.jsonl"

    atomic_jsonl(path, [{"sample_id": "S001"}])

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["sample_id"] == "S001"


def test_atomic_jsonl_replaces_existing_checkpoint(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    atomic_jsonl(path, [{"sample_id": "OLD"}])

    atomic_jsonl(path, [{"sample_id": "NEW"}])

    assert read_checkpoint(path) == {"NEW": {"sample_id": "NEW"}}


def test_read_checkpoint_returns_empty_for_missing_file(tmp_path):
    assert read_checkpoint(tmp_path / "missing.jsonl") == {}


def test_read_checkpoint_loads_unique_sample_ids(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    path.write_text(
        '{"sample_id":"S001","score":0.2}\n'
        '{"sample_id":"S002","score":0.8}\n',
        encoding="utf-8",
    )

    result = read_checkpoint(path)

    assert set(result) == {"S001", "S002"}
    assert result["S002"]["score"] == 0.8


def test_read_checkpoint_rejects_duplicate_sample_ids(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    path.write_text(
        '{"sample_id":"S001","score":0.2}\n'
        '{"sample_id":"S001","score":0.8}\n',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Duplicate sample_id"):
        read_checkpoint(path)


@pytest.mark.parametrize(
    "row",
    [
        {"score": 0.2},
        {"sample_id": ""},
        {"sample_id": None},
        {"sample_id": 123},
    ],
)
def test_read_checkpoint_rejects_invalid_sample_id(tmp_path, row):
    path = tmp_path / "checkpoint.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Missing sample_id"):
        read_checkpoint(path)


def test_read_checkpoint_rejects_malformed_json(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    path.write_text('{"sample_id":\n', encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        read_checkpoint(path)


def test_read_checkpoint_ignores_blank_lines(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    path.write_text(
        '\n{"sample_id":"S001"}\n\n',
        encoding="utf-8",
    )

    assert read_checkpoint(path) == {"S001": {"sample_id": "S001"}}

def test_checkpoint_id_sets_must_match():
    global_ids = {"S001", "S002"}
    candidate_ids = {"S001"}

    assert global_ids != candidate_ids
    assert global_ids - candidate_ids == {"S002"}
    assert candidate_ids - global_ids == set()