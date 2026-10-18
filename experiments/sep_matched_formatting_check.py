
from pathlib import Path
import json
import string
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

DATA = Path("data/processed")
PRED = Path("results/detector_runs/sep/mean_answer_layer_16/test.jsonl")
OUT = Path("results/sanity_checks/matched_formatting")
OUT.mkdir(parents=True, exist_ok=True)

CALIPER = 5
THRESHOLD = 0.9854559515186279
SEED = 42


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def features(answer):
    s = str(answer)
    n = max(len(s), 1)
    words = s.split()
    letters = [c for c in s if c.isalpha()]
    nl = max(len(letters), 1)

    return {
        "char_count": len(s),
        "word_count": len(words),
        "avg_word_length": len(s.replace(" ", "")) / max(len(words), 1),
        "uppercase_ratio": sum(c.isupper() for c in letters) / nl,
        "lowercase_ratio": sum(c.islower() for c in letters) / nl,
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
predictions = load_jsonl(PRED)

pred_by_id = {p["sample_id"]: float(p["score"]) for p in predictions}
if len(pred_by_id) != len(predictions):
    raise ValueError("Duplicate SEP prediction IDs")

test_ids = {r["sample_id"] for r in test}
if set(pred_by_id) != test_ids:
    raise ValueError("SEP prediction IDs do not exactly match test IDs")

train_ids = {r["sample_id"] for r in train}
if train_ids & test_ids:
    raise ValueError("Train/test ID overlap")

train_f = [features(r["answer"]) for r in train]
test_f = [features(r["answer"]) for r in test]

# Greedy one-to-one matching on character length.
zeros = [i for i, r in enumerate(test) if int(r["label"]) == 0]
ones = [i for i, r in enumerate(test) if int(r["label"]) == 1]

edges = []
for i in zeros:
    for j in ones:
        diff = abs(test_f[i]["char_count"] - test_f[j]["char_count"])
        if diff <= CALIPER:
            edges.append((diff, i, j))

edges.sort()
used0, used1, matches = set(), set(), []
for diff, i, j in edges:
    if i not in used0 and j not in used1:
        used0.add(i)
        used1.add(j)
        matches.append((i, j))

if len(matches) < 20:
    raise RuntimeError(f"Too few matched pairs: {len(matches)}")

matched_idx = [i for pair in matches for i in pair]
matched = [test[i] for i in matched_idx]
y = np.array([int(r["label"]) for r in matched])
sep_scores = np.array([pred_by_id[r["sample_id"]] for r in matched])

names = list(train_f[0].keys())
surface_names = [
    n for n in names if n not in ("char_count", "word_count")
]


def matrix(rows, cols):
    return np.asarray([[r[c] for c in cols] for r in rows], dtype=float)


def fit_score(cols):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="liblinear", max_iter=1000, random_state=SEED
        ),
    )
    model.fit(matrix(train_f, cols), np.array([int(r["label"]) for r in train]))
    return model.predict_proba(matrix([test_f[i] for i in matched_idx], cols))[:, 1]


length_scores = fit_score(["char_count", "word_count"])
format_scores = fit_score(surface_names)
sep_preds = (sep_scores >= THRESHOLD).astype(int)

# Pair-cluster bootstrap: resample matched pairs, not individual rows.
rng = np.random.default_rng(SEED)
groups = np.arange(len(matches))
metrics = {
    "SEP": (sep_scores, sep_preds),
    "length_only": (length_scores, (length_scores >= 0.5).astype(int)),
    "formatting_only": (format_scores, (format_scores >= 0.5).astype(int)),
}


def evaluate(y_true, scores, preds):
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, preds)),
    }


results = {
    "caliper_characters": CALIPER,
    "matching_method": "greedy one-to-one matching without replacement",
    "matched_pairs": len(matches),
    "matched_rows": len(matched),
    "test_coverage": len(matched) / len(test),
    "mean_char_difference": float(np.mean([
        abs(test_f[i]["char_count"] - test_f[j]["char_count"])
        for i, j in matches
    ])),
    "models": {},
    "bootstrap": {"replicates": 2000, "seed": SEED},
    "warning": (
        "Matching controls character length only, not punctuation or style. "
        "Greedy matching is not globally optimal. Matched-subset results may "
        "not generalize to the full test set."
    ),
}

for name, (scores, preds) in metrics.items():
    point = evaluate(y, scores, preds)
    boot_auc, boot_acc = [], []

    for _ in range(2000):
        selected = rng.choice(groups, size=len(groups), replace=True)
        row_idx = np.array([k for g in selected for k in (2*g, 2*g+1)])
        yb = y[row_idx]
        sb = scores[row_idx]
        pb = preds[row_idx]

        if len(np.unique(yb)) == 2:
            boot_auc.append(roc_auc_score(yb, sb))
        boot_acc.append(accuracy_score(yb, pb))

    results["models"][name] = {
        **point,
        "auroc_95_ci": [
            float(np.percentile(boot_auc, 2.5)),
            float(np.percentile(boot_auc, 97.5)),
        ],
        "accuracy_95_ci": [
            float(np.percentile(boot_acc, 2.5)),
            float(np.percentile(boot_acc, 97.5)),
        ],
    }

out = OUT / "metrics.json"
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
print(f"\nSaved to: {out}")
