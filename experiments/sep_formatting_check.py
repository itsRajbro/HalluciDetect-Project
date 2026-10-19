from pathlib import Path
import json
import string
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score
)

DATA = Path("data/processed")
OUT = Path("results/sanity_checks/formatting_only")
OUT.mkdir(parents=True, exist_ok=True)

def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def extract_features(answer):
    s = str(answer)
    n = max(len(s), 1)
    words = s.split()
    word_count = max(len(words), 1)
    letters = [c for c in s if c.isalpha()]
    letter_count = max(len(letters), 1)

    return {
        "char_count": len(s),
        "word_count": len(words),
        "avg_word_length": len(s.replace(" ", "")) / word_count,
        "uppercase_ratio": sum(c.isupper() for c in letters) / letter_count,
        "lowercase_ratio": sum(c.islower() for c in letters) / letter_count,
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

if len(train_ids) != len(set(train_ids)):
    raise ValueError("Duplicate training sample IDs")
if len(test_ids) != len(set(test_ids)):
    raise ValueError("Duplicate test sample IDs")
if set(train_ids) & set(test_ids):
    raise ValueError("Train/test sample ID overlap detected")

train_features = [extract_features(r["answer"]) for r in train]
test_features = [extract_features(r["answer"]) for r in test]

all_names = list(train_features[0].keys())
surface_names = [n for n in all_names if n not in ("char_count", "word_count")]

y_train = np.asarray([int(r["label"]) for r in train])
y_test = np.asarray([int(r["label"]) for r in test])

def matrix(rows, names):
    return np.asarray([[r[n] for n in names] for r in rows], dtype=float)

def evaluate(name, feature_names):
    X_train = matrix(train_features, feature_names)
    X_test = matrix(test_features, feature_names)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="liblinear", max_iter=1000, random_state=42
        )
    )
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    preds = (scores >= 0.5).astype(int)

    return {
        "model": name,
        "features": feature_names,
        "test_auroc": float(roc_auc_score(y_test, scores)),
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "test_precision": float(precision_score(y_test, preds, zero_division=0)),
        "test_recall": float(recall_score(y_test, preds, zero_division=0)),
        "test_f1": float(f1_score(y_test, preds, zero_division=0)),
    }

results = {
    "train_rows": len(train),
    "test_rows": len(test),
    "baseline_with_length": evaluate("formatting_plus_length", all_names),
    "baseline_without_char_and_word_counts": evaluate(
        "surface_features_without_explicit_length", surface_names
    ),
    "caution": (
        "The second model excludes raw character and word counts, but some "
        "ratios and formatting features may still correlate with answer length. "
        "These are diagnostic baselines, not proof of semantic generalization."
    )
}

out = OUT / "metrics.json"
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
print(f"\nSaved to: {out}")
