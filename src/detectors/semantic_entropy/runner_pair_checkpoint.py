
"""Pair-aware, resumable validation runner for semantic entropy detectors.

Generates responses once per pair and scores both candidate answers separately.
Uses a single atomic checkpoint file to keep each pair's records together.
Existing runners and result files are not modified.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from src.common.config import get_config
from src.common.data import group_by_pair, load_split
from src.common.generation import generate_samples
from src.common.llama_loader import load_llama
from src.common.schema import (
    DetectorRecord,
    efficiency_from_gen_stats,
    write_detector_output,
)
from src.detectors.semantic_entropy.candidate_aware import score_candidate
from src.detectors.semantic_entropy.clustering import cluster_answers
from src.detectors.semantic_entropy.entropy import semantic_entropy
from src.detectors.semantic_entropy.nli_equivalence import (
    NLISemanticEquivalence,
)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    """Atomically replace a JSONL checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            for row in rows:
                file.write(
                    json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                )
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_checkpoint(path: Path) -> dict[str, dict]:
    """Read completed pair transactions, rejecting malformed checkpoints."""
    if not path.exists():
        return {}

    completed = {}

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid checkpoint JSON at {path}:{line_number}"
            ) from exc

        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise RuntimeError(
                f"Missing pair_id in checkpoint {path}:{line_number}"
            )

        if pair_id in completed:
            raise RuntimeError(
                f"Duplicate pair_id {pair_id!r} in checkpoint {path}"
            )

        completed[pair_id] = row

    return completed


def validate_checkpoint_pair(
    pair_id: str,
    row: dict,
    expected_ids: set[str],
    config_hash: str,
    prompt_version: str,
) -> None:
    """Validate both detector outputs before trusting a checkpointed pair."""
    if row.get("config_hash") != config_hash:
        raise RuntimeError(
            f"Checkpoint config hash mismatch for pair {pair_id!r}. "
            "Use a new tag."
        )

    if row.get("prompt_version") != prompt_version:
        raise RuntimeError(
            f"Checkpoint prompt version mismatch for pair {pair_id!r}. "
            "Use a new tag."
        )

    for key in ("global_records", "candidate_records"):
        records = row.get(key)

        if not isinstance(records, list) or len(records) != 2:
            raise RuntimeError(
                f"Checkpoint pair {pair_id!r} must contain two {key}."
            )

        ids = {record.get("sample_id") for record in records}

        if ids != expected_ids:
            raise RuntimeError(
                f"Checkpoint {key} have incorrect sample IDs for pair "
                f"{pair_id!r}."
            )

        for record in records:
            # Reuse the schema's own validation.
            DetectorRecord(**record)


def run(pair_limit: int, tag: str) -> None:
    if pair_limit < 1:
        raise ValueError("--pairs must be >= 1")

    if not tag or any(char not in
                      "abcdefghijklmnopqrstuvwxyz"
                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                      "0123456789_.-" for char in tag):
        raise ValueError("--tag may contain only letters, numbers, _, . and -")

    cfg = get_config()
    samples = load_split("validation", verify=True)
    pairs = group_by_pair(samples)

    selected_pairs = list(pairs.items())[:pair_limit]

    if not selected_pairs:
        raise ValueError("No complete validation pairs selected")

    expected_pair_ids = {pair_id for pair_id, _ in selected_pairs}
    expected_sample_ids = {
        sample.sample_id
        for _, pair in selected_pairs
        for sample in pair
    }

    output_dir = Path(cfg.paths.outputs_dir) / "detectors"
    checkpoint_dir = output_dir / "_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # This new checkpoint format stores both detectors in one transaction.
    checkpoint_path = checkpoint_dir / f"se_pair_{tag}.jsonl"

    global_final = (
        output_dir / "se_global" / "nli" / f"validation_{tag}.jsonl"
    )
    candidate_final = (
        output_dir / "se_candidate" / "nli" / f"validation_{tag}.jsonl"
    )

    # Never overwrite existing final outputs or metadata.
    for path in (global_final, candidate_final):
        metadata_path = path.with_name(path.stem + ".meta.json")
        if path.exists() or metadata_path.exists():
            raise FileExistsError(
                f"{path} or its metadata already exists. "
                "Choose a fresh --tag."
            )

    checkpoint = read_checkpoint(checkpoint_path)

    unexpected_pairs = set(checkpoint) - expected_pair_ids
    if unexpected_pairs:
        raise RuntimeError(
            f"Checkpoint contains unexpected pair IDs: "
            f"{sorted(unexpected_pairs)}. Use a new tag."
        )

    config_hash = cfg.config_hash()
    prompt_version = cfg.prompt.version

    # Validate all checkpoint entries before loading the model.
    for pair_id, row in checkpoint.items():
        factual, hallucinated = pairs[pair_id]
        validate_checkpoint_pair(
            pair_id=pair_id,
            row=row,
            expected_ids={factual.sample_id, hallucinated.sample_id},
            config_hash=config_hash,
            prompt_version=prompt_version,
        )

    print(f"Validation pairs selected: {len(selected_pairs)}")
    print(f"Validation samples selected: {len(expected_sample_ids)}")
    print(f"Completed pairs in checkpoint: {len(checkpoint)}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Loading shared model: {cfg.model.model_id}", flush=True)

    model, tok = load_llama()
    equivalent = NLISemanticEquivalence()

    start_all = time.perf_counter()

    for index, (pair_id, (factual, hallucinated)) in enumerate(
        selected_pairs, start=1
    ):
        if pair_id in checkpoint:
            print(
                f"[{index}/{len(selected_pairs)}] {pair_id} — resumed",
                flush=True,
            )
            continue

        print(
            f"[{index}/{len(selected_pairs)}] Generating pair {pair_id}",
            flush=True,
        )

        pair_start = time.perf_counter()

        # One generation call supplies responses for both candidate answers.
        # The pair_id seed key is intentionally different from sample-level
        # generation; this change must be documented in the experiment.
        generated = generate_samples(
            model,
            tok,
            question=factual.question,
            knowledge=factual.knowledge,
            sample_id=pair_id,
            cfg=cfg,
        )

        answers = [item.text for item in generated.generations]

        if not answers:
            raise RuntimeError(
                f"No generations returned for pair {pair_id!r}"
            )

        clusters = cluster_answers(answers, equivalent=equivalent)
        global_score = float(semantic_entropy(clusters))

        efficiency = efficiency_from_gen_stats(generated.stats)

        global_pair_records = []
        candidate_pair_records = []

        for sample in (factual, hallucinated):
            candidate = score_candidate(
                answers,
                sample.answer,
                equivalent=equivalent,
            )

            common_extra = {
                "pair_id": pair_id,
                "pair_generation_shared": True,
                "generation_seed_key": pair_id,
                "candidate_matched": candidate["candidate_matched"],
                "candidate_cluster_probability": candidate[
                    "candidate_cluster_probability"
                ],
                "candidate_cluster_id": candidate["candidate_cluster_id"],
                "num_clusters": candidate["num_clusters"],
            }

            global_record = DetectorRecord(
                sample_id=sample.sample_id,
                split="validation",
                detector="se_global",
                variant="nli",
                score=global_score,
                config_hash=config_hash,
                prompt_version=prompt_version,
                latency_s=efficiency["latency_s"],
                n_generations=efficiency["n_generations"],
                n_generated_tokens=efficiency["n_generated_tokens"],
                peak_gpu_mem_mb=efficiency["peak_gpu_mem_mb"],
                extra={
                    **common_extra,
                    "cluster_ids": clusters,
                },
            )

            candidate_record = DetectorRecord(
                sample_id=sample.sample_id,
                split="validation",
                detector="se_candidate",
                variant="nli",
                score=float(candidate["candidate_score"]),
                config_hash=config_hash,
                prompt_version=prompt_version,
                latency_s=efficiency["latency_s"],
                n_generations=efficiency["n_generations"],
                n_generated_tokens=efficiency["n_generated_tokens"],
                peak_gpu_mem_mb=efficiency["peak_gpu_mem_mb"],
                extra=common_extra,
            )

            global_pair_records.append(asdict(global_record))
            candidate_pair_records.append(asdict(candidate_record))

        # One checkpoint row contains both detectors for both samples.
        # A crash before the atomic replace leaves the prior checkpoint intact.
        
        checkpoint[pair_id] = {
            "pair_id": pair_id,
            "config_hash": config_hash,
            "prompt_version": prompt_version,
            "generations": [
                item.to_dict() for item in generated.generations
            ],
            "generation_stats": generated.stats.to_dict(),
            "global_records": global_pair_records,
            "candidate_records": candidate_pair_records,
        }


        atomic_jsonl(checkpoint_path, list(checkpoint.values()))

        elapsed = time.perf_counter() - pair_start
        print(
            f"  Completed pair in {elapsed:.1f}s; "
            f"{len(answers)} generations shared by two candidates.",
            flush=True,
        )

    # Reconstruct records in the selected validation-pair order.
    global_by_id = {}
    candidate_by_id = {}

    for pair_id, _ in selected_pairs:
        row = checkpoint[pair_id]

        for record in row["global_records"]:
            global_by_id[record["sample_id"]] = record

        for record in row["candidate_records"]:
            candidate_by_id[record["sample_id"]] = record

    global_output = [
        DetectorRecord(**global_by_id[sample.sample_id])
        for _, pair in selected_pairs
        for sample in pair
    ]

    candidate_output = [
        DetectorRecord(**candidate_by_id[sample.sample_id])
        for _, pair in selected_pairs
        for sample in pair
    ]

    common_metadata = {
        "run_tag": tag,
        "threshold_selected": False,
        "pair_generation": True,
        "generation_seed_key": "pair_id",
        "n_pairs": len(selected_pairs),
        "generation_statistics_unit": "pair",
    }

    write_detector_output(
        global_output,
        path=global_final,
        description=(
            f"Pair-aware validation run {tag}; global semantic entropy. "
            "Responses are generated once per pair. Generation statistics "
            "are repeated on the two sample records and represent pair-level "
            "cost, not two independent generation calls."
        ),
        extra=common_metadata,
        cfg=cfg,
    )

    write_detector_output(
        candidate_output,
        path=candidate_final,
        description=(
            f"Pair-aware validation run {tag}; candidate-aware score. "
            "Responses are generated once per pair and each candidate is "
            "scored separately. Generation statistics represent pair-level "
            "cost, not two independent generation calls."
        ),
        extra=common_metadata,
        cfg=cfg,
    )

    print("\nPair-aware validation run completed.")
    print(f"Global SE:       {global_final}")
    print(f"Candidate-aware: {candidate_final}")
    print(f"Checkpoint:      {checkpoint_path}")
    print(f"Elapsed:         {time.perf_counter() - start_all:.1f}s")
    print("No threshold selected; test data was not loaded.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pair-aware resumable validation SE runner."
    )
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--tag", default="pair_smoke_v2")
    args = parser.parse_args()
    run(pair_limit=args.pairs, tag=args.tag)


if __name__ == "__main__":
    main()
