
"""Select SE thresholds using candidate validation data only."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from src.common.config import get_config
from src.common.data import load_split
from src.common.evaluator import select_threshold


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ROOT = ROOT / "results" / "leakage_audit" / "proposed_split"
DATA_DIR = CANDIDATE_ROOT / "candidate_data"
MANIFEST = CANDIDATE_ROOT / "candidate_manifest.json"
RUN_DIR = CANDIDATE_ROOT / "outputs" / "candidate_validation_full"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    seen = set()

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row.get("sample_id")

            if not sid or sid in seen:
                raise ValueError(f"{path.name}: missing/duplicate ID at line {line_no}")
            seen.add(sid)
            rows.append(row)

    return rows


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def main() -> None:
    cfg = get_config()

    samples = load_split(
        "validation",
        verify=True,
        processed_dir=DATA_DIR,
        manifest_path=MANIFEST,
        cfg=cfg,
    )
    labels = {s.sample_id: int(s.label) for s in samples}

    if len(labels) != 2022:
        raise RuntimeError(f"Expected 2022 validation samples, got {len(labels)}")

    threshold_results = {}

    for detector in ("se_global", "se_candidate"):
        source = RUN_DIR / f"{detector}_validation.jsonl"
        metadata_path = RUN_DIR / f"{detector}_validation.meta.json"

        rows = read_jsonl(source)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        if metadata.get("split") != "validation":
            raise RuntimeError(f"{detector}: metadata split is not validation")
        if metadata.get("detector") != detector:
            raise RuntimeError(f"{detector}: metadata detector mismatch")
        if metadata.get("threshold_selected") is True:
            raise RuntimeError(f"{detector}: threshold already marked selected")

        row_ids = {r["sample_id"] for r in rows}
        if row_ids != set(labels):
            raise RuntimeError(
                f"{detector}: score IDs do not exactly match validation IDs"
            )

        if any(r.get("split") != "validation" for r in rows):
            raise RuntimeError(f"{detector}: non-validation score found")

        if any(r.get("detector") != detector for r in rows):
            raise RuntimeError(f"{detector}: detector mismatch in score rows")

        ordered = sorted(rows, key=lambda r: r["sample_id"])
        y = np.asarray([labels[r["sample_id"]] for r in ordered], dtype=int)
        x = np.asarray([float(r["score"]) for r in ordered], dtype=float)

        if not np.isfinite(x).all():
            raise RuntimeError(f"{detector}: non-finite score found")

        result = select_threshold(y, x, criterion="youden")
        threshold = float(result["value"])

        predictions = (x >= threshold).astype(int)
        accuracy = float(np.mean(predictions == y))

        threshold_results[detector] = {
            **result,
            "threshold": threshold,
            "n_validation_samples": len(y),
            "n_factual": int(np.sum(y == 0)),
            "n_hallucinated": int(np.sum(y == 1)),
            "validation_accuracy_at_threshold": accuracy,
            "comparison": "pred_label = 1 if score >= threshold",
        }

        # Write separate outputs; never overwrite the original score files.
        output_rows = []
        prediction_by_id = {
            r["sample_id"]: int(pred)
            for r, pred in zip(ordered, predictions)
        }

        for row in rows:
            updated = dict(row)
            updated["pred_label"] = prediction_by_id[row["sample_id"]]
            output_rows.append(updated)

        out_path = RUN_DIR / f"{detector}_validation_thresholded.jsonl"
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
            for row in output_rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        tmp_path.replace(out_path)

        updated_metadata = dict(metadata)
        updated_metadata["threshold"] = threshold
        updated_metadata["threshold_source"] = "validation"
        updated_metadata["description"] = (
            f"Candidate validation scores; Youden threshold selected on validation."
        )
        extra = dict(updated_metadata.get("extra") or {})
        extra["threshold_selected"] = True
        extra["threshold_criterion"] = "youden"
        updated_metadata["extra"] = extra

        write_json(
            RUN_DIR / f"{detector}_validation_thresholded.meta.json",
            updated_metadata,
        )

        print(f"\n{detector}")
        print(f"  Threshold: {threshold:.12g}")
        print(f"  Validation accuracy: {accuracy:.4f}")
        print(f"  Thresholded output: {out_path.name}")

    write_json(RUN_DIR / "thresholds_validation.json", {
        "split": "validation",
        "criterion": "youden",
        "config_hash": cfg.config_hash(),
        "prompt_version": cfg.prompt.version,
        "thresholds": threshold_results,
        "test_data_loaded": False,
    })

    print("\nTHRESHOLD SELECTION COMPLETE")
    print(f"Thresholds: {RUN_DIR / 'thresholds_validation.json'}")
    print("Original score files were not modified.")
    print("No test data was loaded.")


if __name__ == "__main__":
    main()
