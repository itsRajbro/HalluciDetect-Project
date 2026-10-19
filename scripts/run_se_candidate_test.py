
"""EXPLORATORY ONLY: evaluate semantic entropy on the candidate test split.

This proposed split is not the official evaluation split and must not be used
for final results. The official split remains frozen. Thresholds are loaded
from the validation-only selection artifact. Original test runners and
outputs are not modified.
"""
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
MANIFEST = CANDIDATE_ROOT / "candidate_manifest.json"
VALIDATION_DIR = CANDIDATE_ROOT / "outputs" / "candidate_validation_full"


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


def read_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row.get("sample_id")
        if not sid or sid in rows:
            raise RuntimeError(f"Missing/duplicate sample ID at line {line_no}: {path}")
        rows[sid] = row
    return rows


def load_frozen_thresholds(cfg) -> dict[str, float]:
    path = VALIDATION_DIR / "thresholds_validation.json"
    if not path.is_file():
        raise FileNotFoundError(f"Validation threshold artifact missing: {path}")

    artifact = json.loads(path.read_text(encoding="utf-8"))

    if artifact.get("split") != "validation":
        raise RuntimeError("Threshold artifact was not selected on validation")
    if artifact.get("test_data_loaded") is not False:
        raise RuntimeError("Threshold artifact does not confirm test data was untouched")
    if artifact.get("criterion") != "youden":
        raise RuntimeError("Unexpected threshold-selection criterion")
    if artifact.get("config_hash") != cfg.config_hash():
        raise RuntimeError("Threshold config hash differs from current config")
    if artifact.get("prompt_version") != cfg.prompt.version:
        raise RuntimeError("Threshold prompt version differs from current config")

    threshold_block = artifact.get("thresholds", {})
    thresholds = {}

    for detector in ("se_global", "se_candidate"):
        result = threshold_block.get(detector, {})
        if result.get("selected_on") != "validation":
            raise RuntimeError(f"{detector}: threshold was not selected on validation")

        value = result.get("threshold", result.get("value"))
        if not isinstance(value, (int, float)):
            raise RuntimeError(f"{detector}: missing numeric threshold")
        if not __import__("math").isfinite(float(value)):
            raise RuntimeError(f"{detector}: non-finite threshold")
        thresholds[detector] = float(value)

    print("Frozen validation thresholds:")
    for detector, value in thresholds.items():
        print(f"  {detector}: {value:.12g}")
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--tag", default="candidate_test_smoke")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if not args.tag or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for c in args.tag
    ):
        raise ValueError("Invalid tag")

    if not CANDIDATE_DATA.is_dir() or not MANIFEST.is_file():
        raise FileNotFoundError("Candidate test data or manifest is missing")

    cfg = get_config()
    thresholds = load_frozen_thresholds(cfg)

    samples = load_split(
        "test",
        verify=True,
        processed_dir=CANDIDATE_DATA,
        manifest_path=MANIFEST,
        cfg=cfg,
    )

    if args.all:
        selected_samples = samples
    else:
        selected_samples = samples[:args.limit]

    if not selected_samples:
        raise RuntimeError("No candidate test samples loaded")

    expected_ids = {s.sample_id for s in selected_samples}
    run_dir = CANDIDATE_ROOT / "outputs" / "candidate_test" / args.tag
    checkpoint_path = run_dir / "checkpoint.jsonl"
    global_path = run_dir / "se_global_test.jsonl"
    candidate_path = run_dir / "se_candidate_test.jsonl"

    # Never overwrite final outputs from an earlier run.
    for path in (global_path, candidate_path):
        meta_path = path.with_name(path.stem + ".meta.json")
        if path.exists() or meta_path.exists():
            raise FileExistsError(
                f"Output already exists: {path}. Use a new --tag."
            )

    rows = read_jsonl(checkpoint_path)
    if not set(rows).issubset(expected_ids):
        raise RuntimeError("Checkpoint IDs do not match this run; use a new tag")

    for sid, row in rows.items():
        if row.get("config_hash") != cfg.config_hash():
            raise RuntimeError(f"Checkpoint config mismatch: {sid}")
        if row.get("prompt_version") != cfg.prompt.version:
            raise RuntimeError(f"Checkpoint prompt mismatch: {sid}")
        if row.get("split") != "test":
            raise RuntimeError(f"Invalid checkpoint split: {sid}")
        if row.get("status") not in ("generated", "scored"):
            raise RuntimeError(f"Invalid checkpoint status: {sid}")
        if len(row.get("generations", [])) != cfg.sampled.num_samples:
            raise RuntimeError(f"Wrong generation count: {sid}")

    print(f"Candidate test samples: {len(selected_samples)}")
    print(f"Already checkpointed: {len(rows)}")
    print(f"Checkpoint: {checkpoint_path}")
    print("Loading canonical Llama model...", flush=True)

    model, tok = load_llama()
    equivalent = NLISemanticEquivalence()
    global_records = []
    candidate_records = []
    raw_rows = []
    start_all = time.perf_counter()

    for i, sample in enumerate(selected_samples, 1):
        row = rows.get(sample.sample_id)

        if row is None or row["status"] != "scored":
            if row is None:
                print(
                    f"[{i}/{len(selected_samples)}] Generating {sample.sample_id}",
                    flush=True,
                )
                gen = generate_samples(
                    model,
                    tok,
                    question=sample.question,
                    knowledge=sample.knowledge,
                    sample_id=sample.sample_id,
                    cfg=cfg,
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
                atomic_jsonl(
                    checkpoint_path,
                    [rows[s.sample_id] for s in selected_samples
                     if s.sample_id in rows],
                )
            else:
                print(
                    f"[{i}/{len(selected_samples)}] Rescoring saved "
                    f"generations for {sample.sample_id}",
                    flush=True,
                )

            generations = row["generations"]
            answers = [g["text"] for g in generations]
            clusters = cluster_answers(answers, equivalent=equivalent)
            global_score = float(semantic_entropy(clusters))
            candidate = score_candidate(
                answers, sample.answer, equivalent=equivalent
            )

            stats = row["stats"]
            common = dict(
                sample_id=sample.sample_id,
                split="test",
                variant="nli",
                config_hash=cfg.config_hash(),
                prompt_version=cfg.prompt.version,
                latency_s=float(stats["wall_time_s"]),
                n_generations=len(answers),
                n_generated_tokens=int(stats["new_tokens_total"]),
                peak_gpu_mem_mb=stats.get("peak_gpu_mem_mb"),
            )

            gr = DetectorRecord(
                detector="se_global",
                score=global_score,
                extra={
                    "num_clusters": len(set(clusters)),
                    "cluster_ids": clusters,
                    "generation_config_hash": cfg.config_hash(),
                    "candidate_split": True,
                },
                **common,
            ).to_dict()

            cr = DetectorRecord(
                detector="se_candidate",
                score=float(candidate["candidate_score"]),
                extra={
                    "global_entropy": candidate["global_entropy"],
                    "candidate_cluster_probability":
                        candidate["candidate_cluster_probability"],
                    "candidate_matched": candidate["candidate_matched"],
                    "num_clusters": candidate["num_clusters"],
                    "generation_config_hash": cfg.config_hash(),
                    "candidate_split": True,
                },
                **common,
            ).to_dict()

            row["global_record"] = gr
            row["candidate_record"] = cr
            row["status"] = "scored"
            rows[sample.sample_id] = row

            atomic_jsonl(
                checkpoint_path,
                [rows[s.sample_id] for s in selected_samples
                 if s.sample_id in rows],
            )

        global_records.append(DetectorRecord(**row["global_record"]))
        candidate_records.append(DetectorRecord(**row["candidate_record"]))

        for generation in row["generations"]:
            raw_rows.append({
                **generation,
                "sample_id": sample.sample_id,
                "pair_id": sample.pair_id,
                "split": "test",
                "config_hash": cfg.config_hash(),
                "prompt_version": cfg.prompt.version,
            })

        if i % 100 == 0 or i == len(selected_samples):
            print(f"Processed {i}/{len(selected_samples)}", flush=True)

    if any(s.sample_id not in rows or rows[s.sample_id]["status"] != "scored"
           for s in selected_samples):
        raise RuntimeError("Checkpoint coverage incomplete; no final outputs written")

    atomic_jsonl(run_dir / "raw_generations.jsonl", raw_rows)

    description = (
        f"Candidate test evaluation {args.tag}; thresholds fixed on validation; "
        "no threshold tuning on test."
    )

    for records, detector, path in (
        (global_records, "se_global", global_path),
        (candidate_records, "se_candidate", candidate_path),
    ):
        by_id = {r.sample_id: r for r in records}
        ordered = [by_id[s.sample_id] for s in selected_samples]

        result = write_detector_output(
            ordered,
            path=path,
            threshold=thresholds[detector],
            threshold_source="validation",
            description=description,
            extra={
                "run_tag": args.tag,
                "candidate_split": True,
                "threshold_selected": False,
                "threshold_source": "validation",
                "frozen_threshold": thresholds[detector],
            },
            cfg=cfg,
        )
        print(f"{detector}: {result}")

    print("\nCANDIDATE TEST RUN COMPLETE")
    print(f"Samples: {len(selected_samples)}")
    print(f"Output directory: {run_dir}")
    print(f"Elapsed hours: {(time.perf_counter() - start_all) / 3600:.2f}")
    print("Thresholds came from validation; test data was not used for selection.")


if __name__ == "__main__":
    main()
