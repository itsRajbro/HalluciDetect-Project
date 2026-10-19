
"""Resumable candidate-split validation runner for semantic entropy."""
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
from src.common.schema import DetectorRecord, write_detector_output
from src.detectors.semantic_entropy.candidate_aware import score_candidate
from src.detectors.semantic_entropy.clustering import cluster_answers
from src.detectors.semantic_entropy.entropy import semantic_entropy
from src.detectors.semantic_entropy.nli_equivalence import NLISemanticEquivalence


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ROOT = ROOT / "results" / "leakage_audit" / "proposed_split"
CANDIDATE_DATA = CANDIDATE_ROOT / "candidate_data"
CANDIDATE_MANIFEST = CANDIDATE_ROOT / "candidate_manifest.json"
OUTPUT_ROOT = CANDIDATE_ROOT / "outputs"


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_checkpoint(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row.get("sample_id")
        if not sid or sid in rows:
            raise RuntimeError(f"Invalid/duplicate checkpoint ID at line {line_no}")
        rows[sid] = row
    return rows


def run_validation(limit: int | None, tag: str) -> None:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be >= 1")
    if not tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in tag):
        raise ValueError("Invalid --tag")

    if not CANDIDATE_DATA.is_dir() or not CANDIDATE_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Candidate data/manifest missing:\n{CANDIDATE_DATA}\n{CANDIDATE_MANIFEST}"
        )

    cfg = get_config()
    samples = load_split(
        "validation",
        verify=True,
        processed_dir=CANDIDATE_DATA,
        manifest_path=CANDIDATE_MANIFEST,
    )
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise RuntimeError("No candidate validation samples")

    run_dir = OUTPUT_ROOT / tag
    checkpoint_path = run_dir / "checkpoint.jsonl"
    raw_path = run_dir / "raw_generations.jsonl"
    global_path = run_dir / "se_global_validation.jsonl"
    candidate_path = run_dir / "se_candidate_validation.jsonl"

    # Final results are never overwritten.
    for path in (raw_path, global_path, candidate_path):
        if path.exists() or path.with_name(path.stem + ".meta.json").exists():
            raise FileExistsError(f"Output exists: {path}. Choose another --tag.")

    expected = {s.sample_id for s in samples}
    checkpoint = read_checkpoint(checkpoint_path)
    if not set(checkpoint).issubset(expected):
        raise RuntimeError("Checkpoint IDs do not match this run; choose a new --tag.")

    for sid, row in checkpoint.items():
        if (
            row.get("config_hash") != cfg.config_hash()
            or row.get("prompt_version") != cfg.prompt.version
            or row.get("split") != "validation"
            or row.get("status") != "generated"
            or len(row.get("generations", [])) != cfg.sampled.num_samples
        ):
            raise RuntimeError(f"Invalid checkpoint for {sid}; use a new --tag.")

    missing = [s for s in samples if s.sample_id not in checkpoint]
    if missing:
        print(f"Samples: {len(samples)} | Already generated: {len(checkpoint)}")
        print(f"Loading model: {cfg.model.model_id}")
        model, tok = load_llama()

        for i, sample in enumerate(missing, 1):
            print(f"Generating {i}/{len(missing)}: {sample.sample_id}", flush=True)
            started = time.perf_counter()
            output = generate_samples(
                model, tok,
                question=sample.question,
                knowledge=sample.knowledge,
                sample_id=sample.sample_id,
                cfg=cfg,
            )
            generations = [g.to_dict() for g in output.generations]

            if len(generations) != cfg.sampled.num_samples:
                raise RuntimeError(
                    f"{sample.sample_id}: expected {cfg.sampled.num_samples} generations, "
                    f"got {len(generations)}"
                )
            if any(not str(g.get("text", "")).strip() for g in generations):
                raise RuntimeError(f"{sample.sample_id}: empty generation")

            checkpoint[sample.sample_id] = {
                "sample_id": sample.sample_id,
                "pair_id": sample.pair_id,
                "split": "validation",
                "status": "generated",
                "config_hash": cfg.config_hash(),
                "generation_config_hash": output.stats.config_hash,
                "prompt_version": cfg.prompt.version,
                "generations": generations,
                "stats": output.stats.to_dict(),
                "generation_wall_time_s": time.perf_counter() - started,
            }
            atomic_jsonl(
                checkpoint_path,
                [checkpoint[s.sample_id] for s in samples if s.sample_id in checkpoint],
            )

    print("Scoring saved generations with NLI...")
    equivalent = NLISemanticEquivalence()
    global_records, candidate_records, raw_rows = [], [], []

    for i, sample in enumerate(samples, 1):
        row = checkpoint[sample.sample_id]
        generations = row["generations"]
        answers = [g["text"] for g in generations]

        cluster_ids = cluster_answers(answers, equivalent=equivalent)
        global_score = semantic_entropy(cluster_ids)
        candidate = score_candidate(answers, sample.answer, equivalent=equivalent)
        stats = row.get("stats", {})

        common = dict(
            sample_id=sample.sample_id,
            split="validation",
            variant="nli",
            config_hash=cfg.config_hash(),
            prompt_version=cfg.prompt.version,
            latency_s=row.get("generation_wall_time_s"),
            n_generations=len(generations),
            n_generated_tokens=stats.get("n_generated_tokens"),
            peak_gpu_mem_mb=stats.get("peak_gpu_mem_mb"),
        )

        global_records.append(DetectorRecord(
            detector="se_global",
            score=global_score,
            extra={
                "num_clusters": len(set(cluster_ids)),
                "cluster_ids": cluster_ids,
                "generation_config_hash": row["generation_config_hash"],
            },
            **common,
        ))

        candidate_records.append(DetectorRecord(
            detector="se_candidate",
            score=candidate["candidate_score"],
            extra={
                "global_entropy": candidate["global_entropy"],
                "candidate_cluster_probability": candidate["candidate_cluster_probability"],
                "candidate_matched": candidate["candidate_matched"],
                "num_clusters": candidate["num_clusters"],
                "generation_config_hash": row["generation_config_hash"],
            },
            **common,
        ))

        for generation in generations:
            raw_rows.append({
                **generation,
                "sample_id": sample.sample_id,
                "pair_id": sample.pair_id,
                "split": "validation",
                "config_hash": cfg.config_hash(),
                "generation_config_hash": row["generation_config_hash"],
                "prompt_version": cfg.prompt.version,
            })

        if i % 100 == 0 or i == len(samples):
            print(f"Scored {i}/{len(samples)}", flush=True)

    atomic_jsonl(raw_path, raw_rows)
    description = f"Candidate validation {tag}; scores only; no threshold selected."

    global_result = write_detector_output(
        global_records,
        path=global_path,
        description=description + " Global semantic entropy in nats.",
        extra={"run_tag": tag, "candidate_split": True, "threshold_selected": False},
        cfg=cfg,
    )
    candidate_result = write_detector_output(
        candidate_records,
        path=candidate_path,
        description=description + " Candidate-aware score.",
        extra={"run_tag": tag, "candidate_split": True, "threshold_selected": False},
        cfg=cfg,
    )

    print("\nCANDIDATE VALIDATION COMPLETE")
    print(f"Samples: {len(samples)}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Raw generations: {raw_path}")
    print(f"Global scores: {global_result}")
    print(f"Candidate scores: {candidate_result}")
    print("Threshold not selected. Test data was not loaded.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--tag", default="candidate_validation_v1")
    args = parser.parse_args()
    run_validation(None if args.all else args.limit, args.tag)


if __name__ == "__main__":
    main()
