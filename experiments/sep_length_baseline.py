
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common.config import get_config
from src.common.data import load_split


def main():
    cfg = get_config()

    train = load_split("train", verify=True, cfg=cfg)
    val = load_split("validation", verify=True, cfg=cfg)
    test = load_split("test", verify=True, cfg=cfg)

    def make_xy(samples):
        X = np.array(
            [[s.answer_length_chars] for s in samples],
            dtype=np.float64,
        )
        y = np.array([s.label for s in samples], dtype=np.int64)
        return X, y

    X_train, y_train = make_xy(train)
    X_val, y_val = make_xy(val)
    X_test, y_test = make_xy(test)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42, max_iter=1000),
    )
    model.fit(X_train, y_train)

    val_scores = model.predict_proba(X_val)[:, 1]
    thresholds = np.unique(
        np.concatenate(([0.0, 1.0], val_scores))
    )

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        preds = (val_scores >= threshold).astype(int)
        score = f1_score(y_val, preds, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    test_scores = model.predict_proba(X_test)[:, 1]
    test_preds = (test_scores >= best_threshold).astype(int)

    results = {
        "feature": "answer_length_chars_only",
        "threshold_source": "validation",
        "threshold": best_threshold,
        "validation_f1": float(best_f1),
        "test": {
            "accuracy": float(accuracy_score(y_test, test_preds)),
            "precision": float(
                precision_score(y_test, test_preds, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, test_preds, zero_division=0)
            ),
            "f1": float(f1_score(y_test, test_preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, test_scores)),
        },
    }

    out = Path("outputs/sep_length_baseline.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"\nSaved results to: {out}")


if __name__ == "__main__":
    main()
