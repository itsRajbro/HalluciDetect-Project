"""Build the frozen length-controlled secondary test set (delta = 1 char).

Thin runner: all matching / validation / statistics logic lives in
src/evaluation/length_matching.py. No random sampling, no timestamps in the artifact.
"""
import argparse
import json
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.evaluation.length_matching import (  # noqa: E402
    assign_match_ids, load_test, match_one_to_one, sha256, summarize, verify_matching,
)

DELTA = 1  # frozen tolerance (chars)
EXPECTED_TOTAL = 274  # acceptance check only; never used to construct the set
FIELDS = ["sample_id", "pair_id", "language", "question", "knowledge", "answer",
          "label", "match_id", "answer_length_chars"]


def fail(msg):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def atomic_write(path: Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def validate_artifact(path: Path, original: dict, expected_total: int, n_orig: int):
    """Independently re-read the serialized JSONL and check every required property."""
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    checks = []

    def add(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    n0 = sum(r["label"] == 0 for r in rows); n1 = sum(r["label"] == 1 for r in rows)
    add("row count", len(rows) == expected_total, f"{len(rows)} (expected {expected_total})")
    add("class balance", n0 == n1 == expected_total // 2, f"label0={n0}, label1={n1}")
    ids = [r["sample_id"] for r in rows]
    add("sample_id uniqueness", len(set(ids)) == len(ids), f"{len(set(ids))} unique / {len(ids)}")

    by_match = defaultdict(list)
    for r in rows:
        by_match[r["match_id"]].append(r)
    struct_ok = len(by_match) == expected_total // 2 and all(len(v) == 2 for v in by_match.values())
    add("match_id structure", struct_ok, f"{len(by_match)} match_ids, 2 rows each")
    add("one factual + one hallucinated per match",
        all({x["label"] for x in v} == {0, 1} and len(v) == 2 for v in by_match.values()))
    gaps = []
    for v in by_match.values():
        lens = [len(x["answer"]) for x in v]
        gaps.append(abs(lens[0] - lens[1]))
    add("length tolerance", max(gaps, default=0) <= DELTA and all(
        r["answer_length_chars"] == len(r["answer"]) for r in rows),
        f"max gap = {max(gaps, default=0)}; stored length == len(answer)")
    add("source membership", all(i in original for i in ids))
    add("no sample reuse", len(set(ids)) == len(ids))
    pair_ok = all(r["sample_id"] in original and r["pair_id"] == original[r["sample_id"]]["pair_id"]
                  for r in rows)
    content_ok = all(all(r[k] == original[r["sample_id"]][k]
                         for k in ("language", "question", "knowledge", "answer", "label"))
                     for r in rows if r["sample_id"] in original)
    add("pair_id preservation", pair_ok and content_ok,
        "pair_id and question/knowledge/answer/label identical to original")
    return rows, checks


def build_report(rows, checks, meta) -> str:
    # Rebuild matches from the on-disk artifact so every statistic derives from it.
    by_match = defaultdict(list)
    for r in rows:
        r = dict(r); r["answer_length_words"] = len(r["answer"].split())
        by_match[r["match_id"]].append(r)
    matches = []
    for k in sorted(by_match):
        v = by_match[k]
        f = next(x for x in v if x["label"] == 0); h = next(x for x in v if x["label"] == 1)
        matches.append((f, h))
    s = summarize(matches, DELTA, meta["n_orig"])
    r2 = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
    L = []
    L += ["# Length-Controlled Test Set Report", "",
          "Generated programmatically from the written artifact. Secondary stress-test benchmark; "
          "the original `test.jsonl` remains the primary benchmark. `match_id` is a constructed "
          "length-control relationship, not a HaluEval pair; original `pair_id` values are unchanged.", "",
          "## A. Provenance", "",
          f"- Input: `{meta['input']}` (SHA-256 `{meta['in_hash']}`)",
          f"- Output: `{meta['output']}` (SHA-256 `{meta['out_hash']}`)",
          f"- Tolerance: {DELTA} character", "- Length metric: `len(answer)`",
          "- Algorithm: max-cardinality then min total absolute gap "
          "(`scipy.optimize.linear_sum_assignment`), inputs sorted by (length, sample_id)",
          f"- Python {platform.python_version()}, numpy {np.__version__}, "
          f"pandas {pd.__version__}, scipy {scipy.__version__}", "- Random seed: none", "",
          "## B. Counts", "",
          f"- Original test size: {meta['n_orig']} ({meta['n_f']} factual, {meta['n_h']} hallucinated)",
          f"- Controlled total: {len(rows)}",
          f"- Controlled factual: {s['matched_factual']}", f"- Controlled hallucinated: {s['matched_hallucinated']}",
          f"- Coverage: {s['coverage_percent']:.2f}%", "",
          "## C. Character-length statistics", "",
          "| class | mean | median | std | min | max |", "|---|---|---|---|---|---|",
          f"| factual | {r2(s['factual_mean_chars'])} | {r2(s['factual_median_chars'])} | "
          f"{r2(s['factual_std_chars'])} | {s['factual_min_chars']:.0f} | {s['factual_max_chars']:.0f} |",
          f"| hallucinated | {r2(s['hallucinated_mean_chars'])} | {r2(s['hallucinated_median_chars'])} | "
          f"{r2(s['hallucinated_std_chars'])} | {s['hallucinated_min_chars']:.0f} | {s['hallucinated_max_chars']:.0f} |", "",
          "## D. Gap statistics", "",
          f"- Mean absolute gap: {r2(s['mean_abs_length_gap'])}",
          f"- Median absolute gap: {r2(s['median_abs_length_gap'])}",
          f"- Maximum absolute gap: {s['max_abs_length_gap']}",
          f"- Total absolute gap: {s['total_abs_length_gap']}", "",
          "## E. Word-length diagnostics (not used for matching)", "",
          f"- Factual mean / median words: {r2(s['factual_mean_words'])} / {r2(s['factual_median_words'])}",
          f"- Hallucinated mean / median words: {r2(s['hallucinated_mean_words'])} / {r2(s['hallucinated_median_words'])}", "",
          "## F. KS diagnostic", "",
          f"- KS statistic: {r2(s['ks_statistic'])}", f"- KS p-value: {r2(s['ks_pvalue'])}",
          "- Note: a distribution diagnostic only, not the sole criterion for equivalence.", "",
          "## G. Validation results", "", "| check | result | detail |", "|---|---|---|"]
    for name, ok, detail in checks:
        L.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    L.append(f"| original test.jsonl unchanged | {'PASS' if meta['unchanged'] else 'FAIL'} | hash before == after |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--expected-total", type=int, default=EXPECTED_TOTAL)
    a = ap.parse_args()
    root = Path(a.root)
    test_path = root / "data/processed/test.jsonl"
    out_path = root / "data/processed/test_length_controlled.jsonl"
    report_path = root / "results/length_controlled_test_report.md"
    if not test_path.is_file():
        fail(f"missing input {test_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    in_hash = sha256(test_path)
    try:
        rows = load_test(test_path)
        factual = [r for r in rows if r["label"] == 0]
        halluc = [r for r in rows if r["label"] == 1]
        matches = match_one_to_one(factual, halluc, delta=DELTA)
        verify_matching(matches, delta=DELTA, all_ids={r["sample_id"] for r in rows})
        controlled = assign_match_ids(matches, delta=DELTA)
    except (ValueError, AssertionError) as e:
        fail(f"{type(e).__name__}: {e}")

    text = "".join(json.dumps({k: r[k] for k in FIELDS}, ensure_ascii=False) + "\n" for r in controlled)
    tmp_out = out_path.with_name(out_path.name + ".tmp")
    try:
        atomic_write(tmp_out, text)  # staged; promoted only after validation
    except OSError as e:
        fail(f"write failure: {e}")

    original = {r["sample_id"]: r for r in rows}
    staged_rows, checks = validate_artifact(tmp_out, original, a.expected_total, len(rows))
    unchanged = sha256(test_path) == in_hash
    checks_all = checks + [("original test.jsonl unchanged", unchanged, "")]
    for n, ok, d in checks_all:
        print(f"[{'PASS' if ok else 'FAIL'}] {n} {d}")
    if not all(ok for _, ok, _ in checks_all):
        tmp_out.unlink(missing_ok=True)
        fail("validation failed; controlled set NOT written")

    os.replace(tmp_out, out_path)
    # Final read-back of the promoted file drives the report.
    final_rows, final_checks = validate_artifact(out_path, original, a.expected_total, len(rows))
    if not all(ok for _, ok, _ in final_checks):
        fail("post-promotion validation failed")
    meta = dict(input=str(test_path.relative_to(root)), output=str(out_path.relative_to(root)),
                in_hash=in_hash, out_hash=sha256(out_path), n_orig=len(rows),
                n_f=len(factual), n_h=len(halluc), unchanged=unchanged)
    atomic_write(report_path, build_report(final_rows, final_checks, meta))
    print(f"wrote {out_path} ({len(final_rows)} rows) and {report_path}")


if __name__ == "__main__":
    main()
