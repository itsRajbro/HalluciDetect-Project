"""Extract hidden-state features for SEP using the repository's shared APIs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.common.config import get_config
from src.common.data import load_split
from src.common.hidden_states import extract_hidden_states
from src.common.llama_loader import load_llama


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--pooling", default="mean_answer",
                        choices=["tbg", "last_answer", "mean_answer", "eot"])
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None, help="Small smoke-test limit")
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be >= 1")
    cfg = get_config()
    split_name = "validation" if args.split == "val" else args.split
    samples = load_split(split_name, verify=True, cfg=cfg)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be >= 1")
        samples = samples[:args.limit]

    out_root = Path("outputs/sep_features_smoke" if args.limit is not None
                    else "outputs/sep_features")
    out_dir = out_root / args.pooling / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_llama(cfg=cfg)
    metadata = {
        "config_hash": cfg.config_hash(),
        "prompt_version": cfg.prompt.version,
        "split": args.split,
        "pooling": args.pooling,
        "count": len(samples),
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    total_chunks = (len(samples) + args.chunk_size - 1) // args.chunk_size
    for chunk_idx, start in enumerate(range(0, len(samples), args.chunk_size)):
        path = out_dir / f"chunk_{chunk_idx:05d}.npz"
        if path.exists():
            print(f"Skipping existing {path}")
            continue
        chunk = samples[start:start + args.chunk_size]
        features, labels, ids = [], [], []
        layer_ids = None
        for sample in chunk:
            hidden = extract_hidden_states(
                model, tok, sample.question, sample.knowledge, sample.answer,
                sample_id=sample.sample_id, cfg=cfg, with_logprobs=False,
                poolings=[args.pooling], clear_cache=True,
            )
            arr = hidden.features[args.pooling].detach().cpu().numpy()
            features.append(arr.astype(np.float16))
            labels.append(int(sample.label))
            ids.append(str(sample.sample_id))
            if layer_ids is None:
                layer_ids = np.arange(arr.shape[0], dtype=np.int32)
        np.savez_compressed(
            path,
            features=np.stack(features),
            labels=np.asarray(labels, dtype=np.int64),
            sample_ids=np.asarray(ids),
            layers=np.asarray(layer_ids, dtype=np.int32),
            config_hash=np.asarray(cfg.config_hash()),
            prompt_version=np.asarray(cfg.prompt.version),
            pooling=np.asarray(args.pooling),
            split=np.asarray(args.split),
        )
        print(f"Saved chunk {chunk_idx + 1}/{total_chunks}: {path}")
    print(f"Extraction complete: {len(samples)} samples in {out_dir}")


if __name__ == "__main__":
    main()
