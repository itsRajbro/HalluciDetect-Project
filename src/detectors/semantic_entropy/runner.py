
"""Validation-only runner for semantic entropy detectors.

Uses shared dataset, generation, NLI clustering, and output schema.
Never loads the test split or selects a threshold.
"""
from __future__ import annotations

import argparse
import json
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


def run_validation(limit: int | None = 2, output_tag: str = "smoke") -> None:
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1, or None for all validation samples")

    cfg = get_config()
    samples = load_split("validation", verify=True)

    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError("No validation samples selected")

    # Dedicated paths ensure smoke results are never overwritten.
    base = Path(cfg.paths.outputs_dir) / "detectors"
    global_path = base / "se_global" / "nli" / f"validation_{output_tag}.jsonl"
    candidate_path = base / "se_candidate" / "nli" / f"validation_{output_tag}.jsonl"

    for path in (global_path, candidate_path):
        if path.exists() or path.with_name(path.stem + ".meta.json").exists():
            raise FileExistsError(
                f"Output already exists: {path}. Choose a new --tag; "
                "existing results will not be overwritten."
            )

    print(f"Split: validation | Samples: {len(samples)}")
    print(f"Loading shared Llama model: {cfg.model.model_id}")
    model, tok = load_llama()
    equivalent = NLISemanticEquivalence()

    global_records = []
    candidate_records = []
    raw_generations_path = base / "se_generations" / "nli" / f"validation_{output_tag}.jsonl"
    raw_generations_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_generations_path.exists():
        raise FileExistsError(f"Raw generations already exist: {raw_generations_path}")
    raw_generation_rows = []

    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] {sample.sample_id}", flush=True)
        started = time.perf_counter()

        output = generate_samples(
            model,
            tok,
            question=sample.question,
            knowledge=sample.knowledge,
            sample_id=sample.sample_id,
            cfg=cfg,
        )
        answers = [g.text for g in output.generations]
        for generation in output.generations:
            raw_generation_rows.append({
                **generation.to_dict(),
                "split": "validation",
                "config_hash": cfg.config_hash(),
                "generation_config_hash": output.stats.config_hash,
                "prompt_version": cfg.prompt.version,
                "wall_time_s": output.stats.wall_time_s,
                "prompt_num_tokens": output.stats.prompt_num_tokens,
            })

        if not answers or any(not answer.strip() for answer in answers):
            raise RuntimeError(
                f"{sample.sample_id}: empty generated answer; refusing to score it"
            )

        cluster_ids = cluster_answers(answers, equivalent=equivalent)
        global_score = semantic_entropy(cluster_ids)
        candidate = score_candidate(
            answers, sample.answer, equivalent=equivalent
        )

        efficiency = efficiency_from_gen_stats(output.stats)
        # Includes generation and detector scoring/clustering.
        elapsed = time.perf_counter() - started
        common = dict(
            sample_id=sample.sample_id,
            split="validation",
            variant="nli",
            config_hash=cfg.config_hash(),
            prompt_version=cfg.prompt.version,
            latency_s=elapsed,
            n_generations=len(output.generations),
            n_generated_tokens=efficiency["n_generated_tokens"],
            peak_gpu_mem_mb=efficiency["peak_gpu_mem_mb"],
        )

        global_records.append(
            DetectorRecord(
                detector="se_global",
                score=global_score,
                extra={
                    "num_clusters": len(set(cluster_ids)),
                    "cluster_ids": cluster_ids,
                    "generation_config_hash": output.stats.config_hash,
                },
                **common,
            )
        )
        candidate_records.append(
            DetectorRecord(
                detector="se_candidate",
                score=candidate["candidate_score"],
                extra={
                    "global_entropy": candidate["global_entropy"],
                    "candidate_cluster_probability":
                        candidate["candidate_cluster_probability"],
                    "candidate_matched": candidate["candidate_matched"],
                    "num_clusters": candidate["num_clusters"],
                    "generation_config_hash": output.stats.config_hash,
                },
                **common,
            )
        )

    with raw_generations_path.open("x", encoding="utf-8", newline="\n") as raw_file:
        for row in raw_generation_rows:
            raw_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Raw generations: {raw_generations_path}")

    description = (
        f"Validation {output_tag} run; scores only; no threshold selected."
    )
    global_result = write_detector_output(
        global_records,
        path=global_path,
        description=description + " Global semantic entropy in nats.",
        extra={"run_tag": output_tag, "threshold_selected": False},
        cfg=cfg,
    )
    candidate_result = write_detector_output(
        candidate_records,
        path=candidate_path,
        description=(
            description + " Candidate score is 1 minus generated "
            "semantic-cluster probability."
        ),
        extra={"run_tag": output_tag, "threshold_selected": False},
        cfg=cfg,
    )

    print("\nValidation run completed.")
    print(f"Global SE:       {global_result}")
    print(f"Candidate-aware: {candidate_result}")
    print("No threshold selected; test data was not loaded.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=2,
        help="Number of validation samples; omit for all samples using --all",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process the entire validation split",
    )
    parser.add_argument("--tag", default="smoke", help="Unique output filename tag")
    args = parser.parse_args()

    run_validation(
        limit=None if args.all else args.limit,
        output_tag=args.tag,
    )


if __name__ == "__main__":
    main()
