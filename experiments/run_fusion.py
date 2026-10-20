"""Fit and evaluate the SE + SEP hybrid (late fusion). Person 1. Implements docs/fusion_analysis_plan.md.

    python -m experiments.run_fusion \\
        --input se  <se_validation.jsonl>  <se_test.jsonl> \\
        --input sep outputs/detectors/sep/mean_answer_layer_16/validation.jsonl \\
                    outputs/detectors/sep/mean_answer_layer_16/test.jsonl \\
        --logit sep

Every --input is `NAME VALIDATION_RUN TEST_RUN` (any two schema run files; the SE runner's
`validation_<tag>.jsonl` names are fine). `--logit NAME ...` fuses those inputs in logit space (use it
for probability outputs such as SEP, whose scores saturate near 0 and 1).

Secondary analyses (plan section 7):  --add-length   adds log(answer length) as an extra input.

Writes (outputs/detectors/hybrid/<variant>/{validation,test}.jsonl):
    <names>-lr   fitted logistic-regression fusion (validation scores are out-of-fold)
    <names>-avg  unfitted equal-weight average of the standardized inputs
and prints the plan's analysis. Test labels are used only AFTER the fusion is fitted and written.
Then compare against the baselines with experiments/compare_detectors.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.common.config import get_config  # noqa: E402
from src.common.data import load_length_controlled, load_split  # noqa: E402
from src.detectors.fusion import fusion as F  # noqa: E402


def _fmt(m, nd=4):
    if m is None or m["estimate"] is None:
        return "n/a"
    return f"{m['estimate']:.{nd}f} [{m['lo']:.{nd}f}, {m['hi']:.{nd}f}]" if m["lo"] is not None else f"{m['estimate']:.{nd}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs=3, action="append", metavar=("NAME", "VAL_RUN", "TEST_RUN"), required=True)
    ap.add_argument("--logit", nargs="*", default=[], metavar="NAME", help="inputs fused in logit space")
    ap.add_argument("--add-length", action="store_true", help="secondary analysis: add log(answer length)")
    ap.add_argument("--criterion", default="youden", choices=["youden", "f1", "accuracy"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out-dir", default=None, help="folder for the hybrid runs (default outputs/detectors/hybrid)")
    ap.add_argument("--out", default=None, help="save the full analysis as JSON")
    ap.add_argument("--allow-mixed-provenance", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="fit on validation and show weights + validation diagnostics only: writes no files and "
                         "reads no test labels (use it to check the pipeline on real files)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="inputs may cover only part of a split (e.g. SE scored only the length-controlled test samples): "
                         "fit on the validation samples every input covers, score the test samples every input covers")
    ap.add_argument("--min-val-pairs", type=int, default=100,
                    help="with --allow-partial: minimum validation question pairs covered by every input")
    args = ap.parse_args(argv)

    names = [n for n, _, _ in args.input]
    unknown = set(args.logit) - set(names)
    if unknown:
        raise SystemExit(f"--logit names not among the inputs: {sorted(unknown)}")
    specs = [F.InputSpec(n, Path(v), Path(t), "logit" if n in args.logit else "none") for n, v, t in args.input]

    cfg = get_config()
    val, test = load_split("validation"), load_split("test")
    try:
        ctrl = load_length_controlled()
    except FileNotFoundError:
        ctrl = None
        print("note: length-controlled set not found; the primary verdict will be UNAVAILABLE\n")

    result = F.run_fusion(specs, val, test, include_length=args.add_length, criterion=args.criterion, seed=args.seed,
                          out_dir=Path(args.out_dir) if args.out_dir else None, cfg=cfg,
                          allow_mixed_provenance=args.allow_mixed_provenance, allow_partial=args.allow_partial,
                          min_val_pairs=args.min_val_pairs, write=not args.dry_run)
    fit = result["fit"]
    cov = result["coverage"]
    d = fit.to_dict()

    print("=" * 100)
    print("1. Fitted fusion (validation only; test labels not used)")
    print("=" * 100)
    print(f"coverage: validation {cov['validation_used']} of {cov['validation_total']} samples, "
          f"test {cov['test_used']} of {cov['test_total']} samples"
          + ("   <- PARTIAL (plan amendment 4: fit on covered validation samples; claims limited to what is covered)"
             if cov["partial"] else ""))
    print(f"inputs: {', '.join(f'{s.name} ({s.transform})' for s in specs)}"
          + ("  + log(answer length)" if args.add_length else ""))
    print(f"C = {fit.C:g}  (5-fold CV grouped by question pair; mean out-of-fold log-loss per C: "
          + ", ".join(f"{k}: {v:.4f}" for k, v in sorted(fit.cv_logloss.items())) + ")")
    print("standardized weights: " + "   ".join(f"{k} {v:+.3f}" for k, v in d["weights_standardized"].items())
          + f"   intercept {d['intercept_standardized']:+.3f}")
    print("raw-scale weights:    " + "   ".join(f"{k} {v:+.4g}" for k, v in d["weights_raw_scale"].items()))
    if d["negative_weights"]:
        print(f"NOTE: negative weight(s) for {d['negative_weights']} (reported as fitted, not clipped; see plan section 3)")
    print(f"threshold {fit.threshold['value']:.4f} ({fit.threshold['criterion']}, on out-of-fold validation scores)")
    if not args.dry_run:
        for k in ("lr", "avg"):
            print(f"  wrote {result[k]['paths']['test'].parent}")

    def print_diag(diag):
        print("\nvalidation diagnostics:")
        for n, v in diag["inputs"].items():
            clip = f"   clipped {v['clipped_fraction']:.1%}" if "clipped_fraction" in v else ""
            print(f"  {n:<6} AUROC {v['val_auroc']:.4f}   distinct values {v['n_distinct_values']:>5}   "
                  f"Spearman with answer length {diag['spearman_with_length'][n]:+.3f}{clip}")
        for k, v in diag["pearson_between_inputs"].items():
            print(f"  correlation {k}: {v:+.3f}")

    if args.dry_run:
        print_diag(F.diagnostics(result, result["val_samples"]))
        print("\nDRY RUN: nothing was written and no test label was used. The pipeline reads these inputs correctly.")
        return 0

    out = F.analyze(result, val, test, ctrl, n_boot=args.n_boot, seed=args.seed)
    for note in out["notes"]:
        print(f"NOTE: {note}")
    print_diag(out["diagnostics"])

    print("\n" + "=" * 100)
    print(f"2. Metrics with 95% bootstrap CIs (n_boot={args.n_boot}; resampling question pairs / matches)")
    print("=" * 100)
    for label, block in out["metrics"].items():
        print(f"\n[{label}]")
        key2 = "pair_accuracy" if label == "test" else "match_accuracy"
        for n, m in block.items():
            print(f"  {n:<8} AUROC {_fmt(m['auroc'])}   acc {_fmt(m['accuracy'])}   {key2.split('_')[0]}_acc {_fmt(m.get(key2))}")
    print("\ninput thresholds used for accuracy: " + "; ".join(
        f"{n}: {v['value']:.4g} ({v['source']})" for n, v in out["thresholds"].items()))

    print("\n" + "=" * 100)
    print("3. Paired differences, hybrid minus comparator (positive = hybrid better)")
    print("=" * 100)
    for label, block in out["paired"].items():
        print(f"\n[{label}]")
        for k, v in block.items():
            a = v["auroc"]
            tag = "CI excludes 0" if a["ci_excludes_zero"] else "CI includes 0"
            print(f"  {k:<24} dAUROC {_fmt(a)}   {tag}")

    print("\n" + "=" * 100)
    print("4. PRIMARY DECISION (plan section 5: length-controlled set, against every input)")
    print("=" * 100)
    print(out["primary_verdict"])

    ov = out["error_overlap_test"]
    print("\n" + "=" * 100)
    print("5. Error overlap on the test set")
    print("=" * 100)
    print("accuracy: " + ", ".join(f"{n} {v:.4f}" for n, v in ov["accuracy"].items()))
    print(f"oracle (right whenever ANY detector is right): {ov['oracle_any_right_accuracy']:.4f}; "
          f"wrong for ALL: {len(ov['all_wrong_sample_ids'])} samples")
    for pair, v in ov["pairwise"].items():
        if pair.startswith("hybrid vs"):
            keys = [k for k in v if k.startswith("only_")]
            print(f"  {pair}: both right {v['both_right']}, both wrong {v['both_wrong']}, "
                  + ", ".join(f"only {k[5:-6]} right {v[k]}" for k in keys) + f"   McNemar p = {v['mcnemar_exact_p']:.3g}")

    print("\n" + "=" * 100)
    print("6. Cost of the hybrid (= cost of its inputs; pair-level costs counted once)")
    print("=" * 100)
    for li in result["loaded"]:
        c = li.cost_test
        lat = "n/a" if c["total_latency_s"] is None else f"{c['total_latency_s']:.0f}s"
        print(f"  {li.spec.name:<6} test total {lat}, generations {c['total_generations']}, "
              f"tokens {c['total_generated_tokens']}, peak GPU {c['peak_gpu_mem_mb']} MB  (unit: {c['unit']})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        payload = {"fit": d, "analysis": out, "inputs": [vars(s) | {"val_path": str(s.val_path), "test_path": str(s.test_path)}
                                                         for s in specs]}
        Path(args.out).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        print(f"\nsaved {args.out}")
    variants = " ".join(f"hybrid/{result[k]['variant']}" for k in ("lr", "avg"))
    scope = " --scope controlled" if cov["partial"] else ""
    print(f"\nNext: python -m experiments.compare_detectors length_only/chars transformer/distilroberta-base_a "
          f"transformer/distilroberta-base_qak sep/mean_answer_layer_16 {variants} --n-boot 5000{scope}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
