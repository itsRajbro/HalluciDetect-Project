"""Run the length-only baseline end to end (Person 1).

    python -m experiments.run_length_only                    # official: answer length in characters
    python -m experiments.run_length_only --plot             # + figure in results/figures/
    python -m experiments.run_length_only --feature tokens   # secondary: Llama tokens (needs HF_TOKEN)

Steps: load validation/test -> fit direction + threshold on VALIDATION ->
write schema runs (outputs/detectors/length_only/<feature>/) -> evaluate with
the common evaluator (results/eval/length_only/<feature>.json) -> print the
length analysis.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.common import evaluator as E  # noqa: E402
from src.common.config import REPO_ROOT, get_config  # noqa: E402
from src.common.data import attach_token_lengths, load_length_controlled, load_split  # noqa: E402
from src.detectors.baselines import length_only as LO  # noqa: E402


def _f(v):
    return "n/a" if v is None else f"{v:.4f}"


def make_plot(report: dict, test_samples, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
    lens0 = [s.answer_length_chars for s in test_samples if s.label == 0]
    lens1 = [s.answer_length_chars for s in test_samples if s.label == 1]
    bins = np.unique(np.round(np.geomspace(1, max(lens0 + lens1) + 1, 30)))
    ax[0].hist(lens0, bins=bins, alpha=0.6, label="factual")
    ax[0].hist(lens1, bins=bins, alpha=0.6, label="hallucinated")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("answer length (characters)")
    ax[0].set_ylabel("test samples")
    ax[0].set_title("Answer length by label (test)")
    ax[0].legend()

    bs = [b for b in report["test"]["length_stratified"] if b["n"]]
    ax[1].bar(range(len(bs)), [b["accuracy"] for b in bs])
    ax[1].set_xticks(range(len(bs)))
    ax[1].set_xticklabels([f"{b['bin']}\nn={b['n']}\n{b['frac_hallucinated']:.0%} hall." for b in bs], fontsize=6.5)
    ax[1].set_ylim(0, 1)
    ax[1].axhline(0.5, color="grey", ls="--", lw=1)
    ax[1].set_ylabel("accuracy")
    ax[1].set_title("Length-only accuracy per length bin (test)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature", choices=sorted(LO.FEATURES), default="chars")
    ap.add_argument("--criterion", choices=E.CRITERIA, default="youden")
    ap.add_argument("--n-bins", type=int, default=8, help="quantile length bins for the stratified analysis")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    cfg = get_config()
    val, test = load_split("validation"), load_split("test")
    try:
        ctrl = load_length_controlled()
    except FileNotFoundError:
        ctrl = None
        print("note: length-controlled set not found; skipping it")

    if args.feature == "tokens":
        from src.common.llama_loader import load_tokenizer
        tok = load_tokenizer(cfg, False)
        cache = REPO_ROOT / cfg.paths.outputs_dir / "cache" / "answer_token_lengths.json"
        val = attach_token_lengths(val, tok, cache_path=cache)
        test = attach_token_lengths(test, tok, cache_path=cache)

    out = LO.run(val, test, feature=args.feature, criterion=args.criterion, cfg=cfg)
    m = out["model"]
    print(f"direction {m.direction:+d} (validation AUROC of raw length = {m.val_auroc_raw:.4f}); "
          f"threshold {m.threshold:.1f} on the directed score ({m.criterion}, validation)\n")

    report = E.evaluate_run(val_path=out["val_path"], test_path=out["test_path"], val_samples=val,
                            test_samples=test, ctrl_samples=ctrl, include_length_controlled=False,
                            n_bins=args.n_bins, save=True)
    E.print_summary(report)
    print("\nlength-stratified (test; quantile bins from validation lengths):")
    print(f"  {'bin (chars)':<16}{'n':>5}{'%hall.':>8}{'acc':>9}{'recall':>9}{'spec.':>9}{'AUROC':>9}")
    for b in report["test"]["length_stratified"]:
        if b["n"]:
            print(f"  {b['bin']:<16}{b['n']:>5}{b['frac_hallucinated']:>8.0%}{_f(b['accuracy']):>9}"
                  f"{_f(b['recall']):>9}{_f(b['specificity']):>9}{_f(b['auroc']):>9}")
    print("\nA bin that is almost one class (very low or very high %hall.) has an unreliable AUROC; "
          "the length-controlled set is the cleaner test.")
    if args.plot:
        fig_path = REPO_ROOT / cfg.paths.results_dir / "figures" / f"length_only_{args.feature}.png"
        make_plot(report, test, fig_path)
        print(f"figure: {fig_path}")
    print(f"\nruns:   {out['val_path'].parent}\nreport: results/eval/length_only/{args.feature}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
