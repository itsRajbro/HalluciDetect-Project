"""Transformer baseline runner (Person 1). Needs a GPU for sensible speed.

    python -m experiments.run_transformer smoke     # 1) quick check of the training loop (never touches test)
    python -m experiments.run_transformer select    # 2) full grid on TRAIN, choose on VALIDATION, freeze
    python -m experiments.run_transformer test      # 3) ONLY after `select`: score test once, evaluate

Common options (use the same ones for `select` and `test`):
    --model distilroberta-base     any Hugging Face encoder for sequence classification
    --no-knowledge                 drop the knowledge passage from the input
    --lrs 2e-5,5e-5                learning-rate grid
    --epochs 3  --batch-size 16  --max-length 320

`select` refuses to run again after `test` has been run (use --force only if you
accept breaking the protocol and will say so in the report).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.common import evaluator as E  # noqa: E402
from src.common.config import REPO_ROOT, get_config  # noqa: E402
from src.detectors.baselines import transformer as T  # noqa: E402


def _cfg_from_args(args, project_cfg) -> T.TransformerBaselineConfig:
    kw = dict(model_name=args.model,
              include_knowledge=project_cfg.prompt.include_knowledge and not args.no_knowledge,
              lr_grid=tuple(float(x) for x in args.lrs.split(",")),
              max_epochs=args.epochs, batch_size=args.batch_size, max_length=args.max_length,
              seed=project_cfg.run.seed)
    if args.cmd == "smoke":
        kw.update(train_subset=args.smoke_samples, lr_grid=kw["lr_grid"][-1:], max_epochs=1)
    return T.TransformerBaselineConfig(**kw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["smoke", "select", "test"])
    ap.add_argument("--model", default="distilroberta-base")
    ap.add_argument("--no-knowledge", action="store_true")
    ap.add_argument("--lrs", default="2e-5,5e-5")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=320)
    ap.add_argument("--smoke-samples", type=int, default=400, help="training samples used by `smoke` (even number)")
    ap.add_argument("--out-dir", default=None, help="artifact folder (default: outputs/transformer/<variant>)")
    ap.add_argument("--force", action="store_true", help="select: allow re-selection after test was evaluated")
    args = ap.parse_args(argv)

    project_cfg = get_config()
    cfg = _cfg_from_args(args, project_cfg)
    out_dir = Path(args.out_dir) if args.out_dir else T.default_out_dir(project_cfg, cfg)
    print(f"variant: {cfg.variant()}   artifacts: {out_dir}\n")

    if args.cmd in ("smoke", "select"):
        frozen = T.select_and_freeze(cfg, out_dir, project_cfg=project_cfg, force=args.force)
        print("\nvalidation AUROC per (lr, epoch):")
        import json
        for row in json.loads((out_dir / T.LOG_FILE).read_text(encoding="utf-8"))["history"]:
            loss = "n/a" if row["train_loss"] is None else f"{row['train_loss']:.4f}"
            sec = "n/a" if row["seconds"] is None else f"{row['seconds']:.0f}s"
            if row.get("val_seconds") is not None:
                sec += f" (validation scoring {row['val_seconds']:.0f}s)"
            print(f"  lr={row['lr']:<8g} epoch={row['epoch']}  val_auroc={row['val_auroc']:.4f}  "
                  f"train_loss={loss}  time={sec}")
        s, t = frozen["selected"], frozen["threshold"]
        print(f"\nselected lr={s['lr']:g}, epoch={s['epoch']} (val AUROC {s['val_auroc']:.4f}); "
              f"threshold {t['value']:.4f} ({t['criterion']}, validation)")
        print(f"frozen config: {out_dir / T.FROZEN_FILE}")
        if args.cmd == "smoke":
            print("\nSMOKE OK. This was a tiny run (variant ends in _smoke); do not report it. "
                  "Next: python -m experiments.run_transformer select")
        else:
            print("\nNext, once you are happy with the configuration: python -m experiments.run_transformer test")
        return 0

    report = T.predict_test_and_evaluate(out_dir, project_cfg=project_cfg)
    E.print_summary(report)
    print("\nlength-stratified (test):")
    for b in report["test"]["length_stratified"]:
        if b["n"]:
            auc = "n/a" if b["auroc"] is None else f"{b['auroc']:.4f}"
            print(f"  {b['bin']:<16} n={b['n']:>4}  hallucinated={b['frac_hallucinated']:.0%}  "
                  f"acc={b['accuracy']:.4f}  AUROC={auc}")
    eff = report.get("efficiency")
    if eff:
        print(f"\nefficiency: {eff['test_mean_runtime_sec_per_sample'] * 1000:.1f} ms/sample inference, "
              f"peak GPU {eff.get('test_peak_gpu_mem_mb', float('nan')):.0f} MB")
    print(f"\nreport: {REPO_ROOT / project_cfg.paths.results_dir / 'eval' / 'transformer' / (cfg.variant() + '.json')}")
    print(f"The test set is now used: {out_dir / T.TEST_MARKER} blocks re-selection (see docstring).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
