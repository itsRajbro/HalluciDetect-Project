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
from src.common.evaluator import evaluate_detector
from src.detectors.semantic_entropy.candidate_aware import score_candidate
from src.detectors.semantic_entropy.clustering import cluster_answers
from src.detectors.semantic_entropy.entropy import semantic_entropy
from src.detectors.semantic_entropy.nli_equivalence import NLISemanticEquivalence


def atomic_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_rows(path):
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            sid = row["sample_id"]
            if sid in result:
                raise RuntimeError(f"Duplicate checkpoint sample: {sid}")
            result[sid] = row
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", default="full_anchor_v1")
    args = parser.parse_args()

    if not args.tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in args.tag):
        raise ValueError("Invalid tag")

    cfg = get_config()
    samples = load_split("test", verify=True)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        samples = samples[:args.limit]
    if not samples:
        raise RuntimeError("No test samples")

    base = Path(cfg.paths.outputs_dir) / "detectors"
    ckpt = base / "_checkpoints" / f"se_test_sample_{args.tag}.jsonl"
    global_path = base / "se_global" / "nli" / f"test_{args.tag}.jsonl"
    candidate_path = base / "se_candidate" / "nli" / f"test_{args.tag}.jsonl"

    for p in (global_path, candidate_path):
        if p.exists() or p.with_name(p.stem + ".meta.json").exists():
            raise FileExistsError(f"Output already exists: {p}; choose another --tag")

    expected = {s.sample_id for s in samples}
    rows = read_rows(ckpt)
    if not set(rows).issubset(expected):
        raise RuntimeError("Checkpoint has unexpected sample IDs; use a new tag")

    for sid, row in rows.items():
        if row.get("config_hash") != cfg.config_hash() or row.get("prompt_version") != cfg.prompt.version:
            raise RuntimeError(f"Checkpoint config mismatch for {sid}")
        if row.get("split") != "test":
            raise RuntimeError(f"Invalid split in checkpoint: {sid}")
        if row.get("status") not in ("generated", "scored"):
            raise RuntimeError(f"Invalid checkpoint status for {sid}")
        if len(row.get("generations", [])) != cfg.sampled.num_samples:
            raise RuntimeError(f"Wrong generation count for {sid}")

    print(f"Test samples: {len(samples)}")
    print(f"Checkpointed samples: {len(rows)}")
    print(f"Checkpoint: {ckpt}")
    print("Loading canonical Llama model...", flush=True)
    model, tok = load_llama()
    equivalent = NLISemanticEquivalence()
    global_records, candidate_records = [], []
    start_all = time.perf_counter()

    for i, sample in enumerate(samples, 1):
        row = rows.get(sample.sample_id)
        if row and row["status"] == "scored":
            print(f"[{i}/{len(samples)}] {sample.sample_id} — resumed", flush=True)
            global_records.append(DetectorRecord(**row["global_record"]))
            candidate_records.append(DetectorRecord(**row["candidate_record"]))
            continue

        if row is None:
            print(f"[{i}/{len(samples)}] Generating {sample.sample_id}", flush=True)
            gen = generate_samples(
                model, tok, question=sample.question,
                knowledge=sample.knowledge, sample_id=sample.sample_id, cfg=cfg
            )
            if len(gen.generations) != cfg.sampled.num_samples:
                raise RuntimeError(f"Wrong generation count: {sample.sample_id}")
            if any(not g.text.strip() for g in gen.generations):
                raise RuntimeError(f"Empty generation: {sample.sample_id}")

            row = {
                "sample_id": sample.sample_id,
                "split": "test",
                "config_hash": cfg.config_hash(),
                "prompt_version": cfg.prompt.version,
                "status": "generated",
                "generations": [g.to_dict() for g in gen.generations],
                "stats": gen.stats.to_dict(),
            }
            rows[sample.sample_id] = row
            atomic_write(ckpt, [rows[s.sample_id] for s in samples if s.sample_id in rows])
        else:
            print(f"[{i}/{len(samples)}] Rescoring saved generations for {sample.sample_id}", flush=True)

        answers = [g["text"] for g in row["generations"]]
        clusters = cluster_answers(answers, equivalent=equivalent)
        global_score = float(semantic_entropy(clusters))
        candidate = score_candidate(answers, sample.answer, equivalent=equivalent)

        stats = row["stats"]
        common = dict(
            sample_id=sample.sample_id, split="test", variant="nli",
            config_hash=cfg.config_hash(), prompt_version=cfg.prompt.version,
            latency_s=float(stats["wall_time_s"]),
            n_generations=len(answers),
            n_generated_tokens=int(stats["new_tokens_total"]),
            peak_gpu_mem_mb=stats["peak_gpu_mem_mb"],
        )
        gr = DetectorRecord(
            detector="se_global", score=global_score,
            extra={"num_clusters": len(set(clusters)), "cluster_ids": clusters,
                   "generation_config_hash": cfg.config_hash(),
                   "experimental_matcher": "nli_short_answer_anchor_experimental"},
            **common,
        ).to_dict()
        cr = DetectorRecord(
            detector="se_candidate", score=float(candidate["candidate_score"]),
            extra={"global_entropy": candidate["global_entropy"],
                   "candidate_cluster_probability": candidate["candidate_cluster_probability"],
                   "candidate_matched": candidate["candidate_matched"],
                   "num_clusters": candidate["num_clusters"],
                   "experimental_matcher": "nli_short_answer_anchor_experimental"},
            **common,
        ).to_dict()

        row["global_record"] = gr
        row["candidate_record"] = cr
        row["status"] = "scored"
        rows[sample.sample_id] = row
        atomic_write(ckpt, [rows[s.sample_id] for s in samples if s.sample_id in rows])
        global_records.append(DetectorRecord(**gr))
        candidate_records.append(DetectorRecord(**cr))
        print(f"  Scored and checkpointed; {len(answers)} generations", flush=True)

    if len(rows) != len(samples) or any(rows[s.sample_id]["status"] != "scored" for s in samples):
        raise RuntimeError("Checkpoint coverage incomplete; final outputs not written")

    # Thresholds were selected on validation, never on test.
    thresholds = {
        "se_global": 0.1625414866957241,
        "se_candidate": 0.95,
    }
    for records, detector, path in (
        (global_records, "se_global", global_path),
        (candidate_records, "se_candidate", candidate_path),
    ):
        records_by_id = {r.sample_id: r for r in records}
        ordered = [records_by_id[s.sample_id] for s in samples]
        result = write_detector_output(
            ordered, path=path, threshold=thresholds[detector],
            threshold_source="validation",
            description=f"Full test evaluation {args.tag}; threshold fixed from validation.",
            extra={"run_tag": args.tag, "threshold_selected": False,
                   "threshold_source": "validation",
                   "experimental_matcher": "nli_short_answer_anchor_experimental"},
            cfg=cfg,
        )
        print(f"{detector}: {result}")

    print(f"Checkpoint: {ckpt}")
    print(f"Elapsed hours: {(time.perf_counter() - start_all) / 3600:.2f}")
    print("Full test run complete. Thresholds were not tuned on test.")


if __name__ == "__main__":
    main()
