import pytest

from src.detectors.semantic_entropy.runner_checkpoint import read_checkpoint


def test_read_checkpoint_returns_empty_for_missing_file(tmp_path):
    assert read_checkpoint(tmp_path / "missing.jsonl") == {}


def test_read_checkpoint_loads_records_by_sample_id(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    path.write_text(
        '{"sample_id": "sample_1", "config_hash": "abc"}\n',
        encoding="utf-8",
    )

    result = read_checkpoint(path)

    assert result["sample_1"]["config_hash"] == "abc"


def test_read_checkpoint_rejects_invalid_json(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    path.write_text("{invalid json}\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_checkpoint(path)

import pytest

from src.detectors.semantic_entropy.runner_checkpoint import (
    validate_checkpoint_records,
)


def valid_record():
    return {
        "sample_id": "sample_1",
        "detector": "se_global",
        "score": 0.5,
        "split": "validation",
        "variant": "nli",
        "config_hash": "abc",
        "prompt_version": "v1",
        "extra": {},
    }


def test_validate_checkpoint_accepts_matching_record():
    record = valid_record()

    validate_checkpoint_records(
        {"sample_1": record},
        expected_ids={"sample_1"},
        detector="se_global",
        config_hash="abc",
        prompt_version="v1",
    )


def test_validate_checkpoint_rejects_wrong_config():
    record = valid_record()

    with pytest.raises(RuntimeError, match="config_hash"):
        validate_checkpoint_records(
            {"sample_1": record},
            expected_ids={"sample_1"},
            detector="se_global",
            config_hash="different",
            prompt_version="v1",
        )


def test_validate_checkpoint_rejects_wrong_detector():
    record = valid_record()

    with pytest.raises(RuntimeError, match="detector"):
        validate_checkpoint_records(
            {"sample_1": record},
            expected_ids={"sample_1"},
            detector="se_candidate",
            config_hash="abc",
            prompt_version="v1",
        )