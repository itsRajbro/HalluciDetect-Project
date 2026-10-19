from pathlib import Path
import json
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

DATA = Path("data/processed")
PRED = Path("results/detector_runs/sep/mean_answer_layer_16/test.jsonl")
OUT = Path("results/sanity_checks/length_controlled")
OUT.mkdir(parents=True, exist_ok=True)

def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

test = load_jsonl(DATA / "test.jsonl")
predictions = load_jsonl(PRED)

pred_by_id = {}
for p in predictions:
    sid = p["sample_id"]
    if sid in pred_by_id:
        raise ValueError(f"Duplicate prediction ID: {sid}")
    pred_by_id[sid] = float(p["score"])

rows = []
for r in test:
    sid = r["sample_id"]
    if sid not in pred_by_id:
        raise ValueError(f"Missing prediction for {sid}")
    answer = str(r["answer"])
    rows.append({
        "sample_id": sid,
        "pair_id": r["pair_id"],
        "label": int(r["label"]),
        "char_len": len(answer),
        "word_len": len(answer.split()),
        "score": pred_by_id[sid],
    })

if len(rows) != len(predictions):
    raise ValueError("Prediction count does not match official test split")

# Match label-0 and label-1 answers with the smallest absolute
# character-length difference, without reusing either sample.
zeros = sorted((r for r in rows if r["label"] == 0),
               key=lambda r: (r["char_len"], r["sample_id"]))
ones = sorted((r for r in rows if r["label"] == 1),
              key=lambda r: (r["char_len"], r["sample_id"]))

available = set(r["sample_id"] for r in ones)
one_by_id = {r["sample_id"]: r for r in ones}
pairs = []

# For each factual answer, find the closest remaining hallucinated answer.
for z in zeros:
    if not available:
        break
    candidate_id = min(
        available,
        key=lambda sid: (
            abs(z["char_len"] - one_by_id[sid]["char_len"]),
            sid
        )
    )
    h = one_by_id[candidate_id]
    available.remove(candidate_id)
    pairs.append((z, h))

if not pairs:
    raise ValueError("No matched pairs found")

matched = [r for pair in pairs for r in pair]
y = np.asarray([r["label"] for r in matched])
scores = np.asarray([r["score"] for r in matched])
preds = (scores >= 0.9854559515186279).astype(int)

char_diffs = [abs(z["char_len"] - h["char_len"]) for z, h in pairs]
word_diffs = [abs(z["word_len"] - h["word_len"]) for z, h in pairs]

result = {
    "experiment": "greedy_one_to_one_char_length_matching",
    "matching_rule": "closest absolute character length, no replacement",
    "test_rows": len(rows),
    "matched_pairs": len(pairs),
    "matched_rows": len(matched),
    "unmatched_label_0": len(zeros) - len(pairs),
    "unmatched_label_1": len(ones) - len(pairs),
    "mean_absolute_character_length_difference": float(np.mean(char_diffs)),
    "median_absolute_character_length_difference": float(np.median(char_diffs)),
    "mean_absolute_word_count_difference": float(np.mean(word_diffs)),
    "median_absolute_word_count_difference": float(np.median(word_diffs)),
    "matched_auroc": float(roc_auc_score(y, scores)),
    "matched_accuracy_at_original_threshold": float(accuracy_score(y, preds)),
    "threshold": 0.9854559515186279,
    "note": "Greedy matching is order-dependent and is not guaranteed to find the globally optimal set of matches."
}

out_path = OUT / "metrics.json"
out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
print(f"\nSaved to: {out_path}")
