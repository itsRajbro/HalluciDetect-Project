"""Train/select an SEP probe using train/validation data and evaluate test once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.common.config import get_config
from src.common.data import load_split
from src.common.schema import DetectorRecord, write_detector_output
from src.detectors.sep.probe import fit_layer_probe, select_threshold


def read_feature_dir(path: Path):
    """Load cached feature chunks from a split directory."""
    files = sorted(path.glob("chunk_*.npz"))

    if not files:
        raise FileNotFoundError(
            f"No feature chunks found in {path}. "
            "Run sep_extract_features.py first."
        )

    xs, ys, ids = [], [], []
    layers = None
    config_hash = prompt_version = None

    for file in files:
        with np.load(file, allow_pickle=False) as data:
            xs.append(data["features"].astype(np.float32))
            ys.append(data["labels"].astype(np.int64))
            ids.extend(data["sample_ids"].astype(str).tolist())

            this_layers = data["layers"].astype(np.int32)

            if layers is None:
                layers = this_layers
            elif not np.array_equal(layers, this_layers):
                raise ValueError(f"Layer list mismatch in {file}")

            this_hash = str(data["config_hash"].item())
            this_prompt = str(data["prompt_version"].item())

            if config_hash is None:
                config_hash = this_hash
                prompt_version = this_prompt
            elif (
                config_hash != this_hash
                or prompt_version != this_prompt
            ):
                raise ValueError(f"Config/prompt mismatch in {file}")

    return (
        np.concatenate(xs),
        np.concatenate(ys),
        ids,
        layers,
        config_hash,
        prompt_version,
    )


def compute_metrics(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    """Calculate evaluation metrics using a fixed threshold."""
    pred = (scores >= threshold).astype(np.int64)

    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y, pred, zero_division=0)
        ),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }

    if len(np.unique(y)) == 2:
        result["roc_auc"] = float(roc_auc_score(y, scores))

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Semantic Entropy Probe."
    )
    parser.add_argument(
        "--pooling",
        default="mean_answer",
        choices=["tbg", "last_answer", "mean_answer", "eot"],
    )
    parser.add_argument("--variant", default="mean_answer")
    args = parser.parse_args()

    cfg = get_config()
    cache = Path("outputs/sep_features") / args.pooling

    # Load cached features. Cache directories use "val", while the
    # official dataset schema uses "validation".
    Xtr, ytr, ids_tr, layers, htr, ptr = read_feature_dir(
        cache / "train"
    )
    Xv, yv, ids_v, layers_v, hv, pv = read_feature_dir(
        cache / "val"
    )
    Xt, yt, ids_t, layers_t, ht, pt = read_feature_dir(
        cache / "test"
    )

    expected_hash = cfg.config_hash()
    expected_prompt = cfg.prompt.version

    # Validate that all feature caches match the current configuration.
    for name, digest, prompt in [
        ("train", htr, ptr),
        ("val", hv, pv),
        ("test", ht, pt),
    ]:
        if digest != expected_hash or prompt != expected_prompt:
            raise ValueError(
                f"{name} feature cache does not match "
                "the current config/prompt."
            )

    if (
        not np.array_equal(layers, layers_v)
        or not np.array_equal(layers, layers_t)
    ):
        raise ValueError("Layer list differs between splits.")

    # Verify cached IDs and labels against official dataset splits.
    for name, ids, labels in [
        ("train", ids_tr, ytr),
        ("val", ids_v, yv),
        ("test", ids_t, yt),
    ]:
        official_split = (
            "validation" if name == "val" else name
        )
        official = load_split(
            official_split,
            verify=True,
            cfg=cfg,
        )

        lookup = {
            str(sample.sample_id): int(sample.label)
            for sample in official
        }

        if len(ids) != len(set(ids)) or set(ids) != set(lookup):
            raise ValueError(
                f"Feature sample IDs do not match official "
                f"{official_split} split."
            )

        if any(
            lookup[sid] != int(label)
            for sid, label in zip(ids, labels)
        ):
            raise ValueError(
                f"Feature labels do not match official "
                f"{official_split} labels."
            )

    # Select the best layer using validation ROC AUC.
    # Fit each candidate on training data only.
    layer_aucs = []

    for i, layer_id in enumerate(layers):
        print(
            f"Evaluating layer {int(layer_id)} "
            f"({i + 1}/{len(layers)})...",
            flush=True,
        )

        candidate = fit_layer_probe(Xtr[:, i, :], ytr)
        candidate_scores = candidate.predict_proba(
            Xv[:, i, :]
        )[:, 1]

        auc = float(roc_auc_score(yv, candidate_scores))
        layer_aucs.append((auc, i, int(layer_id)))

    _, best_i, best_layer = max(
        layer_aucs,
        key=lambda item: item[0],
    )

    print(f"Selected layer: {best_layer}", flush=True)

    # Refit the selected layer's probe on training data.
    model = fit_layer_probe(Xtr[:, best_i, :], ytr)

    # Select the threshold using validation data only.
    val_scores = model.predict_proba(Xv[:, best_i, :])[:, 1]
    threshold, val_f1 = select_threshold(yv, val_scores)

    # Evaluate test data once using the frozen threshold.
    test_scores = model.predict_proba(Xt[:, best_i, :])[:, 1]
    test_metrics = compute_metrics(yt, test_scores, threshold)

    variant = f"{args.variant}_layer_{best_layer}"
    model_dir = Path("outputs/detector_models/sep") / variant
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "probe.joblib")

    metadata = {
        "detector": "sep",
        "variant": variant,
        "pooling": args.pooling,
        "selected_layer": best_layer,
        "threshold": float(threshold),
        "threshold_source": "validation",
        "validation_f1": float(val_f1),
        "validation_layer_roc_auc": [
            {"layer": layer, "roc_auc": auc}
            for auc, _, layer in layer_aucs
        ],
        "test_metrics": test_metrics,
        "config_hash": expected_hash,
        "prompt_version": expected_prompt,
    }

    (model_dir / "frozen_config.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Save validation and test detector records.
    # DetectorRecord accepts labels/threshold metadata through "extra".
    for split_name, ids, labels, scores in [
        ("val", ids_v, yv, val_scores),
        ("test", ids_t, yt, test_scores),
    ]:
        output_split = (
            "validation" if split_name == "val" else split_name
        )

        records = [
            DetectorRecord(
                sample_id=str(sample_id),
                split=output_split,
                detector="sep",
                variant=variant,
                score=float(score),
                config_hash=expected_hash,
                prompt_version=expected_prompt,
                pred_label=int(score >= threshold),
                extra={
                    "threshold": float(threshold),
                    "threshold_source": "validation",
                    "label": int(label),
                },
            )
            for sample_id, label, score in zip(
                ids, labels, scores
            )
        ]
        write_detector_output(
            records,
            threshold=float(threshold),
            threshold_source="validation",
            cfg=cfg,
        )


    print(
        json.dumps(
            {
                "selected_layer": best_layer,
                "threshold": float(threshold),
                "validation_f1": float(val_f1),
                "test_metrics": test_metrics,
                "model_dir": str(model_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
