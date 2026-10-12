"""Train/select an SEP probe using train/validation data and evaluate test once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.common.config import get_config
from src.common.data import load_split
from src.common.schema import DetectorRecord, write_detector_output
from src.detectors.sep.probe import fit_layer_probe, select_threshold


def read_feature_dir(path: Path):
    files = sorted(path.glob("chunk_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No feature chunks found in {path}. Run sep_extract_features.py first."
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
                config_hash, prompt_version = this_hash, this_prompt
            elif config_hash != this_hash or prompt_version != this_prompt:
                raise ValueError(f"Config/prompt mismatch in {file}")
    return np.concatenate(xs), np.concatenate(ys), ids, layers, config_hash, prompt_version


def compute_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(np.int64)
    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    if len(np.unique(y)) == 2:
        result["roc_auc"] = float(roc_auc_score(y, scores))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooling", default="mean_answer",
                        choices=["tbg", "last_answer", "mean_answer", "eot"])
    parser.add_argument("--variant", default="mean_answer")
    args = parser.parse_args()

    cfg = get_config()
    cache = Path("outputs/sep_features") / args.pooling
    Xtr, ytr, ids_tr, layers, htr, ptr = read_feature_dir(cache / "train")
    Xv, yv, ids_v, layers_v, hv, pv = read_feature_dir(cache / "val")
    Xt, yt, ids_t, layers_t, ht, pt = read_feature_dir(cache / "test")
    expected_hash, expected_prompt = cfg.config_hash(), cfg.prompt.version
    for name, digest, prompt in [
        ("train", htr, ptr), ("val", hv, pv), ("test", ht, pt)
    ]:
        if digest != expected_hash or prompt != expected_prompt:
            raise ValueError(f"{name} feature cache does not match current config/prompt")
    if not np.array_equal(layers, layers_v) or not np.array_equal(layers, layers_t):
        raise ValueError("Layer list differs between splits")

    # Validate cached IDs/labels against the official, manifest-verified splits.
    for name, ids, labels in [
        ("train", ids_tr, ytr), ("val", ids_v, yv), ("test", ids_t, yt)
    ]:
        official = load_split(name, verify=True, cfg=cfg)
        lookup = {str(sample.sample_id): int(sample.label) for sample in official}
        if len(ids) != len(set(ids)) or set(ids) != set(lookup):
            raise ValueError(f"Feature sample IDs do not match official {name} split")
        if any(lookup[sid] != int(label) for sid, label in zip(ids, labels)):
            raise ValueError(f"Feature labels do not match official {name} labels")

    # Select layer by validation ROC AUC; fit the selected probe on train only.
    layer_aucs = []
    for i, layer_id in enumerate(layers):
        candidate = fit_layer_probe(Xtr[:, i, :], ytr)
        val_scores = candidate.predict_proba(Xv[:, i, :])[:, 1]
        layer_aucs.append((float(roc_auc_score(yv, val_scores)), i, int(layer_id)))
    _, best_i, best_layer = max(layer_aucs, key=lambda item: item[0])

    model = fit_layer_probe(Xtr[:, best_i, :], ytr)
    val_scores = model.predict_proba(Xv[:, best_i, :])[:, 1]
    threshold, val_f1 = select_threshold(yv, val_scores)
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
        "threshold": threshold,
        "threshold_source": "validation",
        "validation_f1": val_f1,
        "validation_layer_roc_auc": [
            {"layer": layer, "roc_auc": auc} for auc, _, layer in layer_aucs
        ],
        "test_metrics": test_metrics,
        "config_hash": expected_hash,
        "prompt_version": expected_prompt,
    }
    (model_dir / "frozen_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    for split_name, ids, labels, scores in [
        ("val", ids_v, yv, val_scores), ("test", ids_t, yt, test_scores)
    ]:
        records = [
            DetectorRecord(
                sample_id=str(sample_id),
                split=split_name,
                detector="sep",
                variant=variant,
                score=float(score),
                config_hash=expected_hash,
                prompt_version=expected_prompt,
                threshold=float(threshold),
                threshold_source="validation",
                pred_label=int(score >= threshold),
                label=int(label),
            )
            for sample_id, label, score in zip(ids, labels, scores)
        ]
        write_detector_output(
            records, detector="sep", variant=variant, split=split_name, cfg=cfg
        )
    print(json.dumps({
        "selected_layer": best_layer,
        "threshold": threshold,
        "validation_f1": val_f1,
        "test_metrics": test_metrics,
        "model_dir": str(model_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
