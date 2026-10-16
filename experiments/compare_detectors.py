"""Compare detectors on the same samples with proper uncertainty (Person 1).

    python -m experiments.compare_detectors length_only/chars transformer/distilroberta-base_qak
    python -m experiments.compare_detectors se_candidate/n10 sep/layer20_last_answer --ref transformer/distilroberta-base_qak ...

Each argument is `<detector>/<variant>`; the run's test file is looked up in
results/detector_runs/ first, then outputs/detectors/ (override with --runs-dir).
Every run must have been written with a validation-selected threshold.

Output
  1. per detector: test and length-controlled metrics with 95% bootstrap CIs
     (resampling question pairs / length matches, not single samples)
  2. paired differences vs a reference detector (first one, or --ref), with CIs
  3. error overlap on the test set: who is right where, McNemar test, and the
     accuracy of an oracle that is right whenever any detector is
Use --out to save everything as JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.common import comparison as C  # noqa: E402
from src.common import evaluator as E  # noqa: E402
from src.common.config import REPO_ROOT, get_config  # noqa: E402
from src.common.data import load_length_controlled, load_split  # noqa: E402
from src.common.schema import check_coverage, read_detector_output  # noqa: E402


def _fmt(m, nd=4):
    if m is None or m["estimate"] is None:
        return "n/a"
    if m["lo"] is None:
        return f"{m['estimate']:.{nd}f}"
    return f"{m['estimate']:.{nd}f} [{m['lo']:.{nd}f}, {m['hi']:.{nd}f}]"


def find_run(spec: str, dirs: list[Path]) -> Path:
    if "/" not in spec:
        raise SystemExit(f"'{spec}' must look like <detector>/<variant>")
    detector, variant = spec.split("/", 1)
    for d in dirs:
        p = d / detector / variant / "test.jsonl"
        if p.is_file():
            return p
    raise SystemExit(f"no test run for {spec} under: " + ", ".join(str(d) for d in dirs))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("detectors", nargs="+", help="<detector>/<variant> (at least two for comparisons)")
    ap.add_argument("--ref", default=None, help="reference detector for paired differences (default: first)")
    ap.add_argument("--runs-dir", default=None, help="folder holding <detector>/<variant>/test.jsonl")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="save the full results as JSON")
    args = ap.parse_args(argv)

    cfg = get_config()
    dirs = [Path(args.runs_dir)] if args.runs_dir else [REPO_ROOT / "results" / "detector_runs",
                                                       REPO_ROOT / cfg.paths.outputs_dir / "detectors"]
    ref = args.ref or args.detectors[0]
    names = list(dict.fromkeys(args.detectors + [ref]))

    test = load_split("test")
    try:
        ctrl = load_length_controlled()
    except FileNotFoundError:
        ctrl = None
        print("note: length-controlled set not found; skipping it\n")

    runs: dict = {}
    for spec in names:
        recs, meta = read_detector_output(find_run(spec, dirs))
        if meta.split != "test":
            raise SystemExit(f"{spec}: not a test run")
        if meta.threshold is None:
            raise SystemExit(f"{spec}: run has no validation threshold (write it with threshold=...)")
        check_coverage(recs, [s.sample_id for s in test], strict=True)
        runs[spec] = {"scored": E.records_to_scores(recs), "thr": meta.threshold, "meta": meta}
    hashes = {(r["meta"].config_hash, r["meta"].prompt_version) for r in runs.values()}
    if len(hashes) > 1:
        print("WARNING: runs were produced with different config_hash / prompt_version:")
        for n, r in runs.items():
            print(f"  {n}: {r['meta'].config_hash} / {r['meta'].prompt_version}")
        print()

    results: dict = {"detectors": {}, "paired_vs_reference": {}, "reference": ref, "n_boot": args.n_boot}

    print("=" * 100)
    print(f"1. Metrics with 95% bootstrap CIs (n_boot={args.n_boot}; resampling question pairs / matches)")
    print("=" * 100)
    for n, r in runs.items():
        ci_t = C.bootstrap_ci(test, r["scored"], r["thr"], cluster="pair", n_boot=args.n_boot, seed=args.seed)
        entry = {"threshold": r["thr"], "test": ci_t}
        m = ci_t["metrics"]
        print(f"\n{n}   (threshold {r['thr']:.4g}, selected on validation)")
        print(f"  test      AUROC {_fmt(m['auroc'])}   acc {_fmt(m['accuracy'])}   pair_acc {_fmt(m['pair_accuracy'])}")
        if ctrl is not None:
            ci_c = C.bootstrap_ci(ctrl, r["scored"], r["thr"], cluster="match", n_boot=args.n_boot, seed=args.seed)
            entry["length_controlled"] = ci_c
            c = ci_c["metrics"]
            print(f"  len-ctrl  AUROC {_fmt(c['auroc'])}   acc {_fmt(c['accuracy'])}   match_acc {_fmt(c['match_accuracy'])}")
        results["detectors"][n] = entry

    others = [n for n in runs if n != ref]
    if others:
        print("\n" + "=" * 100)
        print(f"2. Paired differences vs reference {ref}   (positive = detector better than reference)")
        print("=" * 100)
        for n in others:
            res = {}
            for label, samples, cluster in (("test", test, "pair"), ("length_controlled", ctrl, "match")):
                if samples is None:
                    continue
                d = C.paired_bootstrap_diff(samples, runs[n]["scored"], runs[ref]["scored"], runs[n]["thr"],
                                            runs[ref]["thr"], cluster=cluster, n_boot=args.n_boot, seed=args.seed)
                res[label] = d
            results["paired_vs_reference"][n] = res
            print(f"\n{n}")
            for label, d in res.items():
                a = d["metric_differences_a_minus_b"]["auroc"]
                verdict = "CI excludes 0" if a["ci_excludes_zero"] else "CI includes 0 (no clear difference)"
                print(f"  {label:<18} dAUROC {_fmt(a)}   {verdict}")

    if len(runs) > 1:
        print("\n" + "=" * 100)
        print("3. Error overlap on the test set")
        print("=" * 100)
        ov = C.error_overlap(test, {n: (r["scored"], r["thr"]) for n, r in runs.items()})
        results["error_overlap_test"] = ov
        for pair, v in ov["pairwise"].items():
            keys = [k for k in v if k.startswith("only_")]
            print(f"\n{pair}:  both right {v['both_right']}, both wrong {v['both_wrong']}, "
                  + ", ".join(f"only {k[5:-6]} right {v[k]}" for k in keys)
                  + f"   McNemar exact p = {v['mcnemar_exact_p']:.4g}")
        print(f"\noracle (right whenever ANY detector is right): accuracy {ov['oracle_any_right_accuracy']:.4f} "
              f"(best single: {max(ov['accuracy'].values()):.4f}); wrong for ALL detectors: {len(ov['all_wrong_sample_ids'])} samples")
        print("An oracle gain over the best single detector is the most any fusion (e.g. SE + SEP) could add.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
