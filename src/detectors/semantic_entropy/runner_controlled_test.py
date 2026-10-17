"""Resumable runner for the length-controlled semantic entropy test set."""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from src.common.config import get_config
from src.common.data import load_length_controlled
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
    """Atomically save a checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            for record in records:
                file.write(
                    json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
                )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
def read_checkpoint(path: Path) -> dict[str, dict]:
    """Load a checkpoint and reject duplicate sample IDs."""
    if not path.exists():
        return {}
    results = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise RuntimeError(
                f"Missing sample_id in checkpoint {path}:{line_number}"
            )
        if sample_id in results:
            raise RuntimeError(
                f"Duplicate sample_id {sample_id!r} in {path}"
            )
        results[sample_id] = row
    return results
def validate_checkpoint_pair_ids(
    global_done: dict[str, dict],
    candidate_done: dict[str, dict],
) -> set[str]:
    """Reject inconsistent global and candidate checkpoint coverage."""
    global_ids = set(global_done)
    candidate_ids = set(candidate_done)

    if global_ids != candidate_ids:
        global_only = sorted(global_ids - candidate_ids)
        candidate_only = sorted(candidate_ids - global_ids)
        raise RuntimeError(
            "Global and candidate checkpoints are inconsistent. "
            f"Global-only IDs: {global_only[:10]}; "
            f"candidate-only IDs: {candidate_only[:10]}. "
            "Repair the checkpoints or use a fresh --tag."
        )

    return global_ids


def run(limit: int | None, tag: str) -> None:
    if not tag or any(
        char not in
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_.-"
        for char in tag
    ):
        raise ValueError(
            "--tag may contain only letters, numbers, _, . and -"
        )
    cfg = get_config()
    samples = load_length_controlled(verify=True)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        samples = samples[:limit]
    if not samples:
        raise ValueError("No controlled-test samples selected")
    output_dir = Path(cfg.paths.outputs_dir) / "detectors"
    checkpoint_dir = output_dir / "_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Dedicated checkpoints: never reuse validation checkpoint names.
    global_checkpoint = (
        checkpoint_dir / f"se_global_length_controlled_{tag}.jsonl"
    )
    candidate_checkpoint = (
        checkpoint_dir / f"se_candidate_length_controlled_{tag}.jsonl"
    )
    global_final = (
        output_dir / "se_global" / "nli"
        / f"test_length_controlled_{tag}.jsonl"
    )
    candidate_final = (
        output_dir / "se_candidate" / "nli"
        / f"test_length_controlled_{tag}.jsonl"
    )
    # Refuse to overwrite either final output or its metadata.
    for path in (global_final, candidate_final):
        metadata_path = path.with_name(path.stem + ".meta.json")
        if path.exists() or metadata_path.exists():
            raise FileExistsError(
                f"{path} or its metadata already exists. "
                "Choose a fresh --tag."
            )
    global_done = read_checkpoint(global_checkpoint)
    candidate_done = read_checkpoint(candidate_checkpoint)
    expected_ids = {sample.sample_id for sample in samples}
    # Validate checkpoint IDs and run compatibility before loading the model.
    for name, checkpoint in (
        ("global", global_done),
        ("candidate", candidate_done),
    ):
        unexpected = set(checkpoint) - expected_ids
        if unexpected:
            raise RuntimeError(
                f"{name} checkpoint contains unexpected IDs: "
                f"{sorted(unexpected)[:10]}. Use a fresh --tag."
            )
        for sample_id, record in checkpoint.items():
            if record.get("config_hash") != cfg.config_hash():
                raise RuntimeError(
                    f"{name} checkpoint config mismatch for {sample_id}. "
                    "Use a fresh --tag."
                )
            if record.get("prompt_version") != cfg.prompt.version:
                raise RuntimeError(
                    f"{name} checkpoint prompt mismatch for {sample_id}. "
                    "Use a fresh --tag."
                )
            if record.get("split") != "test":
                raise RuntimeError(
                    f"{name} checkpoint has wrong split for {sample_id}."
                )
    # Both detector checkpoints must have matching completed IDs.
    common_ids = validate_checkpoint_pair_ids(global_done, candidate_done)

    print(f"Controlled-test samples selected: {len(samples)}")
    print(f"Already checkpointed: {len(common_ids)}")
    print(f"Config hash: {cfg.config_hash()}")
    print(f"Prompt version: {cfg.prompt.version}")
    print(f"Loading shared model: {cfg.model.model_id}", flush=True)
    model, tok = load_llama()
    equivalent = NLISemanticEquivalence()
    start_all = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        sample_id = sample.sample_id
        if sample_id in common_ids:
            print(
                f"[{index}/{len(samples)}] {sample_id} — resumed",
                flush=True,
            )
            continue
        print(
            f"[{index}/{len(samples)}] Generating {sample_id}",
            flush=True,
        )
        started = time.perf_counter()
        generated = generate_samples(
            model,
            tok,
            question=sample.question,
            knowledge=sample.knowledge,
            sample_id=sample_id,
            cfg=cfg,
        )
        answers = [item.text for item in generated.generations]
        if not answers or any(not answer.strip() for answer in answers):
            raise RuntimeError(
                f"{sample_id}: empty generation; stopping safely."
            )
        clusters = cluster_answers(answers, equivalent=equivalent)
        global_score = float(semantic_entropy(clusters))
        candidate = score_candidate(
            answers,
            sample.answer,
            equivalent=equivalent,
        )
        efficiency = efficiency_from_gen_stats(generated.stats)
        elapsed = time.perf_counter() - started
        shared = {
            "sample_id": sample_id,
            "split": "test",
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
                "length_controlled": True,
                "match_id": sample.match_id,
                "pair_id": sample.pair_id,
            },
            **shared,
        ).to_dict()
        candidate_record = DetectorRecord(
            detector="se_candidate",
            score=float(candidate["candidate_score"]),
            extra={
                "global_entropy": candidate["global_entropy"],
                "candidate_cluster_probability": candidate[
                    "candidate_cluster_probability"
                ],
                "candidate_matched": candidate["candidate_matched"],
                "num_clusters": candidate["num_clusters"],
                "generation_config_hash": generated.stats.config_hash,
                "length_controlled": True,
                "match_id": sample.match_id,
                "pair_id": sample.pair_id,
            },
            **shared,
        ).to_dict()
        global_done[sample_id] = global_record
        candidate_done[sample_id] = candidate_record
        ordered_global = [
            global_done[s.sample_id]
            for s in samples
            if s.sample_id in global_done
        ]
        ordered_candidate = [
            candidate_done[s.sample_id]
            for s in samples
            if s.sample_id in candidate_done
        ]
        # Persist both checkpoints after every completed sample.
        atomic_jsonl(global_checkpoint, ordered_global)
        atomic_jsonl(candidate_checkpoint, ordered_candidate)
        print(f"  Completed in {elapsed:.1f}s", flush=True)
    if (
        set(global_done) != expected_ids
        or set(candidate_done) != expected_ids
    ):
        raise RuntimeError(
            "Checkpoint coverage incomplete; final files were not written."
        )
    global_records = [
        DetectorRecord.from_dict(global_done[s.sample_id])
        for s in samples
    ]
    candidate_records = [
        DetectorRecord.from_dict(candidate_done[s.sample_id])
        for s in samples
    ]
    metadata = {
        "run_tag": tag,
        "length_controlled": True,
        "n_samples": len(samples),
        "threshold_selected": False,
        "threshold": None,
        "generation_seed_key": "sample_id",
        "match_id_used_for_generation": False,
    }
    write_detector_output(
        global_records,
        path=global_final,
        description=(
            f"Length-controlled test run {tag}; global semantic entropy "
            "in nats. No threshold selected on test."
        ),
        extra={**metadata, "detector": "se_global"},
        cfg=cfg,
    )
    write_detector_output(
        candidate_records,
        path=candidate_final,
        description=(
            f"Length-controlled test run {tag}; candidate-aware score. "
            "No threshold selected on test."
        ),
        extra={**metadata, "detector": "se_candidate"},
        cfg=cfg,
    )
    print("\nControlled-test run completed.")
    print(f"Global SE:       {global_final}")
    print(f"Candidate-aware: {candidate_final}")
    print(f"Checkpoints:     {checkpoint_dir}")
    print(f"Elapsed:         {time.perf_counter() - start_all:.1f}s")
    print("No threshold selected; validation outputs were not loaded.")
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resumable runner for the length-controlled test set."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Number of samples to process (default: 2).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all 274 controlled-test samples.",
    )
    parser.add_argument(
        "--tag",
        default="lc_v1",
        help="Unique run tag; use the same tag to resume.",
    )
    args = parser.parse_args()
    run(limit=None if args.all else args.limit, tag=args.tag)
if __name__ == "__main__":
    main()