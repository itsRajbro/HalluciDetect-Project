
"""Rescore saved SE validation generations with the updated NLI matcher.

Does not generate new answers or overwrite the original experiment outputs.
"""

import json
from pathlib import Path

from src.common.config import get_config
from src.common.data import group_by_pair, load_split
from src.detectors.semantic_entropy.candidate_aware import score_candidate
from src.detectors.semantic_entropy.clustering import cluster_answers
from src.detectors.semantic_entropy.entropy import semantic_entropy
from src.detectors.semantic_entropy.nli_equivalence import (
    NLISemanticEquivalence,
)


ROOT = Path(__file__).resolve().parents[1]
TAG = "validation_anchor_v1"

CHECKPOINT = (
    ROOT / "outputs" / "detectors" / "_checkpoints"
    / "se_pair_validation_full_v1.jsonl"
)

GLOBAL_OUT = (
    ROOT / "outputs" / "detectors" / "se_global" / "nli"
    / f"validation_{TAG}.jsonl"
)

CANDIDATE_OUT = (
    ROOT / "outputs" / "detectors" / "se_candidate" / "nli"
    / f"validation_{TAG}.jsonl"
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, allow_nan=False) + "\n")


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    # Refuse to overwrite any previous output.
    for path in (GLOBAL_OUT, CANDIDATE_OUT):
        if path.exists() or path.with_suffix(".meta.json").exists():
            raise FileExistsError(
                f"Output already exists: {path}. Choose a new TAG."
            )

    cfg = get_config()
    samples = load_split("validation", verify=True)
    pairs = group_by_pair(samples)

    checkpoint_rows = read_jsonl(CHECKPOINT)
    checkpoint = {row["pair_id"]: row for row in checkpoint_rows}

    if len(checkpoint) != 1001:
        raise RuntimeError(
            f"Expected 1001 checkpoint pairs, found {len(checkpoint)}"
        )

    if set(checkpoint) != set(pairs):
        raise RuntimeError(
            "Checkpoint pair IDs do not match the current validation split."
        )

    equivalent = NLISemanticEquivalence()
    global_rows = []
    candidate_rows = []

    for index, (pair_id, pair) in enumerate(pairs.items(), start=1):
        saved = checkpoint[pair_id]

        if (
            saved.get("config_hash") != cfg.config_hash()
            or saved.get("prompt_version") != cfg.prompt.version
        ):
            raise RuntimeError(
                f"Metadata mismatch for pair {pair_id}; stopping."
            )

        generations = saved.get("generations", [])
        answers = [item.get("text", "") for item in generations]

        if not answers or any(not answer for answer in answers):
            raise RuntimeError(
                f"Missing saved generation text for pair {pair_id}"
            )

        clusters = cluster_answers(answers, equivalent=equivalent)
        global_score = float(semantic_entropy(clusters))

        # Use the original records as templates, preserving their IDs,
        # labels, latency, generation counts, and other metadata.
        old_global = {
            row["sample_id"]: row
            for row in saved["global_records"]
        }
        old_candidate = {
            row["sample_id"]: row
            for row in saved["candidate_records"]
        }

        for sample in pair:
            if sample.sample_id not in old_global:
                raise RuntimeError(
                    f"Missing global record for {sample.sample_id}"
                )
            if sample.sample_id not in old_candidate:
                raise RuntimeError(
                    f"Missing candidate record for {sample.sample_id}"
                )

            candidate = score_candidate(
                answers,
                sample.answer,
                equivalent=equivalent,
            )

            common_extra = {
                "pair_id": pair_id,
                "pair_generation_shared": True,
                "generation_seed_key": pair_id,
                "candidate_label": sample.label,
                "candidate_matched": candidate["candidate_matched"],
                "candidate_cluster_probability": candidate[
                    "candidate_cluster_probability"
                ],
                "candidate_cluster_id": candidate["candidate_cluster_id"],
                "num_clusters": candidate["num_clusters"],
                "rescoring_tag": TAG,
                "matching_rule": "nli_short_answer_anchor_experimental",
            }

            global_record = dict(old_global[sample.sample_id])
            global_record["score"] = global_score
            global_record["pred_label"] = None
            global_record["extra"] = {
                **common_extra,
                "cluster_ids": clusters,
            }
            global_rows.append(global_record)

            candidate_record = dict(old_candidate[sample.sample_id])
            candidate_record["score"] = float(candidate["candidate_score"])
            candidate_record["pred_label"] = None
            candidate_record["extra"] = common_extra
            candidate_rows.append(candidate_record)

        if index % 100 == 0 or index == len(pairs):
            print(f"Rescored {index}/{len(pairs)} pairs", flush=True)

    # Exclusive file creation prevents accidental overwrite.
    write_jsonl(GLOBAL_OUT, global_rows)
    write_jsonl(CANDIDATE_OUT, candidate_rows)

    print("\nRescoring complete.")
    print(f"Global SE:       {GLOBAL_OUT}")
    print(f"Candidate-aware: {CANDIDATE_OUT}")
    print("No generations were run.")
    print("No thresholds were selected.")
    print("Original experiment files were not modified.")


if __name__ == "__main__":
    main()
