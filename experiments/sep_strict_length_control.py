from pathlib import Path
import json
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DATA = Path("data/processed")
PRED = Path("results/detector_runs/sep/mean_answer_layer_16/test.jsonl")
OUT = Path("results/sanity_checks/strict_length_control")
OUT.mkdir(parents=True, exist_ok=True)

MAX_CHAR_DIFF = 5
THRESHOLD = 0.9854559515186279

def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

train = load_jsonl(DATA / "train.jsonl")
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
        "label": int(r["label"]),
        "char_len": len(answer),
        "word_len": len(answer.split()),
        "score": pred_by_id[sid],
    })

if len(rows) != len(predictions):
    raise ValueError("Prediction count differs from official test split")

zeros = [r for r in rows if r["label"] == 0]
ones = [r for r in rows if r["label"] == 1]

# Build all eligible pairs within the character-length caliper.
edges = []
for z in zeros:
    for h in ones:
        diff = abs(z["char_len"] - h["char_len"])
        if diff <= MAX_CHAR_DIFF:
            edges.append((diff, z["sample_id"], h["sample_id"]))

# Greedily select the closest eligible pairs, without reusing samples.
edges.sort()
zero_by_id = {r["sample_id"]: r for r in zeros}
one_by_id = {r["sample_id"]: r for r in ones}
used_zero, used_one, pairs = set(), set(), []

for diff, zid, hid in edges:
    if zid in used_zero or hid in used_one:
        continue
    used_zero.add(zid)
    used_one.add(hid)
    pairs.append((zero_by_id[zid], one_by_id[hid]))

if len(pairs) < 20:
    raise RuntimeError(
        f"Only {len(pairs)} pairs matched within {MAX_CHAR_DIFF} characters. "
        "Do not interpret the metrics as reliable; consider a wider caliper."
    )

matched = [r for pair in pairs for r in pair]
y = np.asarray([r["label"] for r in matched])
sep_scores = np.asarray([r["score"] for r in matched])
sep_preds = (sep_scores >= THRESHOLD).astype(int)

# Fit length-only baseline on training data and score the exact same matched rows.
def length_features(answer):
    return [len(str(answer)), len(str(answer).split())]

X_train = np.asarray(
    [length_features(r["answer"]) for r in train], dtype=float
)
y_train = np.asarray([int(r["label"]) for r in train])

X_matched = np.asarray([
    [r["char_len"], r["word_len"]] for r in matched
], dtype=float)

length_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)
)
length_model.fit(X_train, y_train)
length_scores = length_model.predict_proba(X_matched)[:, 1]

char_diffs = [abs(z["char_len"] - h["char_len"]) for z, h in pairs]
word_diffs = [abs(z["word_len"] - h["word_len"]) for z, h in pairs]

result = {
    "caliper_characters": MAX_CHAR_DIFF,
    "matching_method": "greedy closest eligible pairs without replacement",
    "test_rows": len(rows),
    "matched_pairs": len(pairs),
    "matched_rows": len(matched),
    "test_coverage": len(matched) / len(rows),
    "mean_absolute_char_difference": float(np.mean(char_diffs)),
    "max_absolute_char_difference": int(max(char_diffs)),
    "mean_absolute_word_difference": float(np.mean(word_diffs)),
    "sep_matched_auroc": float(roc_auc_score(y, sep_scores)),
    "sep_matched_accuracy_at_original_threshold": float(
        accuracy_score(y, sep_preds)
    ),
    "length_only_matched_auroc": float(roc_auc_score(y, length_scores)),
    "threshold": THRESHOLD,
    "warning": "Matching is greedy, not globally optimal. Interpret results alongside matched sample coverage."
}

path = OUT / "metrics.json"
path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
print(f"\nSaved to: {path}")
