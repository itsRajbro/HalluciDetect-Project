from pathlib import Path
import json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

DATA = Path("data/processed")
OUT = Path("results/sanity_checks/length_only")
OUT.mkdir(parents=True, exist_ok=True)

def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def features(row):
    answer = str(row["answer"])
    return [len(answer), len(answer.split())]

train = load_jsonl(DATA / "train.jsonl")
test = load_jsonl(DATA / "test.jsonl")

X_train = np.asarray([features(r) for r in train], dtype=np.float64)
y_train = np.asarray([int(r["label"]) for r in train])
X_test = np.asarray([features(r) for r in test], dtype=np.float64)
y_test = np.asarray([int(r["label"]) for r in test])

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)
)
model.fit(X_train, y_train)

scores = model.predict_proba(X_test)[:, 1]
preds = (scores >= 0.5).astype(int)

metrics = {
    "experiment": "length_only_char_and_word_count",
    "train_rows": len(y_train),
    "test_rows": len(y_test),
    "features": ["answer_character_count", "answer_word_count"],
    "test_auroc": float(roc_auc_score(y_test, scores)),
    "test_accuracy": float(accuracy_score(y_test, preds)),
    "test_precision": float(precision_score(y_test, preds, zero_division=0)),
    "test_recall": float(recall_score(y_test, preds, zero_division=0)),
    "test_f1": float(f1_score(y_test, preds, zero_division=0)),
}

out_path = OUT / "metrics.json"
out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
print(f"\nSaved to: {out_path}")
