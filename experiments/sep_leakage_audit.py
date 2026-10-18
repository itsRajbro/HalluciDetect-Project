
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
PRED_PATH = (
    ROOT / "results" / "detector_runs" / "sep"
    / "mean_answer_layer_16" / "test.jsonl"
)
OUT_PATH = ROOT / "results" / "sanity_checks" / "leakage_audit"
OUT_PATH.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.9854559515186279


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.casefold()
    text = " ".join(text.split())
    return text


def overlap(a, b, field):
    a_values = {normalize(r[field]) for r in a}
    b_values = {normalize(r[field]) for r in b}
    return a_values & b_values


def pair_keys(records):
    return {r["pair_id"] for r in records}


def summarize_metrics(rows):
    y = np.array([int(r["label"]) for r in rows])
    scores = np.array([float(r["score"]) for r in rows])
    preds = (scores >= THRESHOLD).astype(int)

    return {
        "n_rows": int(len(rows)),
        "n_pairs": len({r["pair_id"] for r in rows}),
        "label_0": int((y == 0).sum()),
        "label_1": int((y == 1).sum()),
        "auroc": float(roc_auc_score(y, scores))
        if len(np.unique(y)) == 2 else None,
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "threshold": THRESHOLD,
    }


def main():
    train = load_jsonl(DATA_DIR / "train.jsonl")
    val = load_jsonl(DATA_DIR / "validation.jsonl")
    test = load_jsonl(DATA_DIR / "test.jsonl")
    predictions = load_jsonl(PRED_PATH)

    # Confirm unique sample IDs and exact prediction/test correspondence.
    for name, rows in [
        ("train", train), ("validation", val), ("test", test)
    ]:
        ids = [r["sample_id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sample IDs in {name}")

    test_by_id = {r["sample_id"]: r for r in test}
    pred_by_id = {r["sample_id"]: r for r in predictions}

    if set(test_by_id) != set(pred_by_id):
        raise ValueError(
            "Prediction IDs do not exactly match test IDs. "
            "No results were written."
        )

    # Attach scores to official test rows without changing source files.
    scored_test = []
    for row in test:
        scored = dict(row)
        scored["score"] = float(pred_by_id[row["sample_id"]]["score"])
        scored_test.append(scored)

    report = {
        "dataset_counts": {
            "train_rows": len(train),
            "validation_rows": len(val),
            "test_rows": len(test),
        },
        "overlaps": {},
        "metrics": {},
        "excluded_test_pairs": {},
        "normalization": (
            "Unicode NFKC, casefold, whitespace normalization"
        ),
        "threshold": THRESHOLD,
    }

    splits = {"train": train, "validation": val, "test": test}
    comparisons = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    for left_name, right_name in comparisons:
        left = splits[left_name]
        right = splits[right_name]
        key = f"{left_name}_vs_{right_name}"

        pair_overlap = pair_keys(left) & pair_keys(right)
        report["overlaps"][key] = {
            "pair_id_unique_overlap": len(pair_overlap),
            "question_unique_overlap": len(
                overlap(left, right, "question")
            ),
            "knowledge_unique_overlap": len(
                overlap(left, right, "knowledge")
            ),
            "answer_unique_overlap": len(
                overlap(left, right, "answer")
            ),
            "question_knowledge_unique_overlap": len(
                {
                    (normalize(r["question"]), normalize(r["knowledge"]))
                    for r in left
                }
                &
                {
                    (normalize(r["question"]), normalize(r["knowledge"]))
                    for r in right
                }
            ),
        }

    train_answers_by_label = defaultdict(set)
    for row in train:
        train_answers_by_label[normalize(row["answer"])].add(
            int(row["label"])
        )

    test_answer_seen = []
    test_answer_opposite_label = []

    for row in test:
        labels = train_answers_by_label.get(normalize(row["answer"]), set())
        test_answer_seen.append(bool(labels))
        test_answer_opposite_label.append(
            any(label != int(row["label"]) for label in labels)
        )

    report["test_answer_overlap"] = {
        "rows_with_answer_seen_in_train": int(sum(test_answer_seen)),
        "rows_with_opposite_label_answer_in_train": int(
            sum(test_answer_opposite_label)
        ),
    }

    # Baseline metrics on all test rows.
    report["metrics"]["all_test"] = summarize_metrics(scored_test)

    # Exclude an entire test pair if its normalized knowledge occurs in train.
    train_knowledge = {normalize(r["knowledge"]) for r in train}
    excluded_pairs = {
        r["pair_id"] for r in test
        if normalize(r["knowledge"]) in train_knowledge
    }

    clean_test = [
        r for r in scored_test if r["pair_id"] not in excluded_pairs
    ]

    report["excluded_test_pairs"]["knowledge_seen_in_train"] = {
        "excluded_pair_count": len(excluded_pairs),
        "excluded_row_count": len(scored_test) - len(clean_test),
        "remaining_pair_count": len({r["pair_id"] for r in clean_test}),
        "remaining_row_count": len(clean_test),
    }

    if clean_test and len({r["label"] for r in clean_test}) == 2:
        report["metrics"]["test_without_train_knowledge_overlap"] = (
            summarize_metrics(clean_test)
        )
    else:
        report["metrics"]["test_without_train_knowledge_overlap"] = None

    # Exclude pairs with either knowledge or answer overlap.
    train_answers = {normalize(r["answer"]) for r in train}
    strict_excluded_pairs = {
        r["pair_id"] for r in test
        if (
            normalize(r["knowledge"]) in train_knowledge
            or normalize(r["answer"]) in train_answers
        )
    }

    strict_test = [
        r for r in scored_test if r["pair_id"] not in strict_excluded_pairs
    ]

    report["excluded_test_pairs"]["knowledge_or_answer_seen_in_train"] = {
        "excluded_pair_count": len(strict_excluded_pairs),
        "excluded_row_count": len(scored_test) - len(strict_test),
        "remaining_pair_count": len({r["pair_id"] for r in strict_test}),
        "remaining_row_count": len(strict_test),
    }

    if strict_test and len({r["label"] for r in strict_test}) == 2:
        report["metrics"]["test_without_train_knowledge_or_answer_overlap"] = (
            summarize_metrics(strict_test)
        )
    else:
        report["metrics"]["test_without_train_knowledge_or_answer_overlap"] = None

    output_path = OUT_PATH / "report.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()
