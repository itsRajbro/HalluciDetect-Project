
"""Resumable validation-only runner for semantic entropy detectors."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from src.common.config import get_config
from src.common.data import load_split
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
from src.detectors.semantic_entropy.nli_equivalence import NLISemanticEquivalence


def atomic_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def read_checkpoint(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    results = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            results[row["sample_id"]] = row
    return results


def run(limit: int | None, tag: str) -> None:
    cfg = get_config()
    samples = load_split("validation", verify=True)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        samples = samples[:limit]
    if not samples:
        raise ValueError("No validation samples selected")

    output_dir = Path(cfg.paths.outputs_dir) / "detectors"
    checkpoint_dir = output_dir / "_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_checkpoint = checkpoint_dir / f"se_global_{tag}.jsonl"
    candidate_checkpoint = checkpoint_dir / f"se_candidate_{tag}.jsonl"
    global_final = output_dir / "se_global" / "nli" / f"validation_{tag}.jsonl"
    candidate_final = output_dir / "se_candidate" / "nli" / f"validation_{tag}.jsonl"

    for path in (global_final, candidate_final):
        if path.exists() or path.with_name(path.stem + ".meta.json").exists():
            raise FileExistsError(
                f"{path} already exists. Use a new --tag to avoid overwriting results."
            )

    global_done = read_checkpoint(global_checkpoint)
    candidate_done = read_checkpoint(candidate_checkpoint)
    expected_ids = {sample.sample_id for sample in samples}

    if not set(global_done).issubset(expected_ids):
        raise RuntimeError("Global checkpoint contains unexpected sample IDs.")
    if not set(candidate_done).issubset(expected_ids):
        raise RuntimeError("Candidate checkpoint contains unexpected sample IDs.")

    common_ids = set(global_done) & set(candidate_done)
    # Ignore unmatched checkpoint entries and recompute those samples.
    global_done = {key: value for key, value in global_done.items() if key in common_ids}
    candidate_done = {
        key: value for key, value in candidate_done.items() if key in common_ids
    }

    print(f"Validation samples selected: {len(samples)}")
    print(f"Already checkpointed: {len(common_ids)}")
    print(f"Loading shared Llama model: {cfg.model.model_id}", flush=True)
    model, tok = load_llama()
    equivalent = NLISemanticEquivalence()

    for index, sample in enumerate(samples, start=1):
        if sample.sample_id in common_ids:
            print(f"[{index}/{len(samples)}] {sample.sample_id} — resumed", flush=True)
            continue

        print(f"[{index}/{len(samples)}] {sample.sample_id}", flush=True)
        started = time.perf_counter()
        generated = generate_samples(
            model,
            tok,
            question=sample.question,
            knowledge=sample.knowledge,
            sample_id=sample.sample_id,
            cfg=cfg,
        )
        answers = [item.text for item in generated.generations]
        if not answers or any(not answer.strip() for answer in answers):
            raise RuntimeError(f"{sample.sample_id}: empty generation; stopping safely.")

        clusters = cluster_answers(answers, equivalent=equivalent)
        global_score = semantic_entropy(clusters)
        candidate = score_candidate(answers, sample.answer, equivalent=equivalent)
        efficiency = efficiency_from_gen_stats(generated.stats)
        elapsed = time.perf_counter() - started

        shared = {
            "sample_id": sample.sample_id,
            "split": "validation",
            "variant": "nli",
            "config_hash": cfg.config_hash(),
            "prompt_version": cfg.prompt.version,
            "latency_s": elapsed,
            "n_generations": len(generated.generations),
            "n_generated_tokens": efficiency["n_generated_tokens"],
            "peak_gpu_mem_mb": efficiency["peak_gpu_mem_mb"],
        }

        global_record = DetectorRecord(
            detector="se_global",
            score=global_score,
            extra={
                "num_clusters": len(set(clusters)),
                "cluster_ids": clusters,
                "generation_config_hash": generated.stats.config_hash,
            },
            **shared,
        ).to_dict()

        candidate_record = DetectorRecord(
            detector="se_candidate",
            score=candidate["candidate_score"],
            extra={
                "global_entropy": candidate["global_entropy"],
                "candidate_cluster_probability": candidate[
                    "candidate_cluster_probability"
                ],
                "candidate_matched": candidate["candidate_matched"],
                "num_clusters": candidate["num_clusters"],
                "generation_config_hash": generated.stats.config_hash,
            },
            **shared,
        ).to_dict()

        # Write each checkpoint atomically after a sample completes.
        global_done[sample.sample_id] = global_record
        candidate_done[sample.sample_id] = candidate_record
        ordered_global = [
            global_done[s.sample_id] for s in samples if s.sample_id in global_done
        ]
        ordered_candidate = [
            candidate_done[s.sample_id] for s in samples if s.sample_id in candidate_done
        ]
        atomic_jsonl(global_checkpoint, ordered_global)
        atomic_jsonl(candidate_checkpoint, ordered_candidate)

    if set(global_done) != expected_ids or set(candidate_done) != expected_ids:
        raise RuntimeError("Checkpoint coverage incomplete; final files were not written.")

    global_records = [
        DetectorRecord.from_dict(global_done[s.sample_id]) for s in samples
    ]
    candidate_records = [
        DetectorRecord.from_dict(candidate_done[s.sample_id]) for s in samples
    ]

    write_detector_output(
        global_records,
        path=global_final,
        description=f"Validation resumable run {tag}; global semantic entropy in nats.",
        extra={"run_tag": tag, "threshold_selected": False},
        cfg=cfg,
    )
    write_detector_output(
        candidate_records,
        path=candidate_final,
        description=f"Validation resumable run {tag}; candidate-aware score.",
        extra={"run_tag": tag, "threshold_selected": False},
        cfg=cfg,
    )

    print("\nValidation run completed.")
    print(f"Global SE:       {global_final}")
    print(f"Candidate-aware: {candidate_final}")
    print(f"Checkpoints retained in: {checkpoint_dir}")
    print("No threshold selected; test data was not loaded.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--tag", default="resume_test")
    args = parser.parse_args()
    run(limit=None if args.all else args.limit, tag=args.tag)


if __name__ == "__main__":
    main()
