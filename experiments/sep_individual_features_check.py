from pathlib import Path
import json
import numpy as np
import string

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

DATA = Path("data/processed")
OUT = Path("results/sanity_checks/individual_features")
OUT.mkdir(parents=True, exist_ok=True)

def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def extract(answer):
    s = str(answer)
    n = max(len(s), 1)
    words = s.split()
    letters = [c for c in s if c.isalpha()]
    letter_n = max(len(letters), 1)
    word_n = max(len(words), 1)

    return {
        "char_count": len(s),
        "word_count": len(words),
        "avg_word_length": len(s.replace(" ", "")) / word_n,
        "uppercase_ratio": sum(c.isupper() for c in letters) / letter_n,
        "lowercase_ratio": sum(c.islower() for c in letters) / letter_n,
        "digit_ratio": sum(c.isdigit() for c in s) / n,
        "punctuation_ratio": sum(c in string.punctuation for c in s) / n,
        "whitespace_ratio": sum(c.isspace() for c in s) / n,
        "unique_char_ratio": len(set(s)) / n,
        "starts_uppercase": int(bool(s) and s[0].isupper()),
        "ends_punctuation": int(bool(s) and s[-1] in string.punctuation),
        "comma_ratio": s.count(",") / n,
        "period_ratio": s.count(".") / n,
        "question_mark_ratio": s.count("?") / n,
        "exclamation_ratio": s.count("!") / n,
        "parenthesis_ratio": (s.count("(") + s.count(")")) / n,
        "non_ascii_ratio": sum(ord(c) > 127 for c in s) / n,
    }

train = load_jsonl(DATA / "train.jsonl")
test = load_jsonl(DATA / "test.jsonl")

train_ids = [r["sample_id"] for r in train]
test_ids = [r["sample_id"] for r in test]
if len(train_ids) != len(set(train_ids)) or len(test_ids) != len(set(test_ids)):
    raise ValueError("Duplicate sample IDs detected")
if set(train_ids) & set(test_ids):
    raise ValueError("Train/test sample ID overlap detected")

train_f = [extract(r["answer"]) for r in train]
test_f = [extract(r["answer"]) for r in test]
y_train = np.asarray([int(r["label"]) for r in train])
y_test = np.asarray([int(r["label"]) for r in test])

names = list(train_f[0])
results = []

for name in names:
    X_train = np.asarray([[r[name]] for r in train_f], dtype=float)
    X_test = np.asarray([[r[name]] for r in test_f], dtype=float)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)
    )
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    preds = (scores >= 0.5).astype(int)

    results.append({
        "feature": name,
        "test_auroc": float(roc_auc_score(y_test, scores)),
        "test_accuracy": float(accuracy_score(y_test, preds)),
    })

results.sort(key=lambda r: r["test_auroc"], reverse=True)

payload = {
    "train_rows": len(train),
    "test_rows": len(test),
    "results_sorted_by_auroc": results,
    "note": "Single-feature diagnostic. High AUROC indicates association, not causal leakage."
}

path = OUT / "metrics.json"
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
print(f"\nSaved to: {path}")
