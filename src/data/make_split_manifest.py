"""Split manifest: the committed source of truth for train/validation/test.

Why this exists
---------------
`data/processed/*` is gitignored, so teammates cannot get the splits from git.
The manifest lists, for every split, the `pair_id` and `sample_id`s it contains
plus a content hash of each processed file. It is small enough to commit, and
lets anyone prove that their local copy is identical to everyone else's.

Commands (run from the repo root)
---------------------------------
  # Person 1, once, on the machine that holds the ORIGINAL processed files:
  python -m src.data.make_split_manifest build

  # Everyone, after obtaining data/processed (copy or re-run the pipeline):
  python -m src.data.make_split_manifest verify

  # Optional: check that re-running split.py from the interim file reproduces
  # the manifest exactly (needs data/interim/halueval_qa_cleaned.jsonl):
  python -m src.data.make_split_manifest reproduce

`build` reads the existing processed files; it never re-splits, so the
original 80/10/10 split (seed 42) is preserved exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.common.config import (  # noqa: E402
    LABEL_FACTUAL, LABEL_HALLUCINATED, SPLITS, get_config,
)

MANIFEST_VERSION = 1
REQUIRED_FIELDS = ["sample_id", "pair_id", "language", "question", "knowledge", "answer", "label"]


def sha256_normalized(path: Path) -> str:
    """SHA-256 of a text file with CRLF folded to LF.

    split.py writes with Python's default text mode, so the same data has
    different raw bytes on Windows (CRLF) and Linux/macOS (LF). Normalizing
    makes hashes comparable across teammates' machines.
    """
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize_split(rows: list[dict], split: str) -> list[list[str]]:
    """Validate one split's rows; return ordered [pair_id, sample_stem] entries."""
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        for k in REQUIRED_FIELDS:
            if k not in r or r[k] in (None, ""):
                raise ValueError(f"[{split}] row {r.get('sample_id')} missing/empty '{k}'")
        by_pair[r["pair_id"]].append(r)

    entries: list[list[str]] = []
    seen_pairs: set[str] = set()
    for r in rows:  # preserve file order
        pid = r["pair_id"]
        if pid in seen_pairs:
            continue
        seen_pairs.add(pid)
        grp = by_pair[pid]
        labels = sorted(x["label"] for x in grp)
        if labels != [LABEL_FACTUAL, LABEL_HALLUCINATED]:
            raise ValueError(f"[{split}] pair {pid} has labels {labels}, expected one 0 and one 1")
        stems = {x["sample_id"].rsplit("_", 1)[0] for x in grp}
        if len(stems) != 1:
            raise ValueError(f"[{split}] pair {pid} sample_ids do not share a stem: {stems}")
        for x in grp:
            if x["sample_id"] != f"{next(iter(stems))}_{x['label']}":
                raise ValueError(f"[{split}] sample_id/label mismatch: {x['sample_id']} label={x['label']}")
        entries.append([pid, next(iter(stems))])
    return entries


def build_manifest(processed_dir: Path, seed_hint: int | None = None) -> dict:
    splits: dict[str, list[list[str]]] = {}
    hashes: dict[str, str] = {}
    n_samples: dict[str, int] = {}
    for s in SPLITS:
        p = processed_dir / f"{s}.jsonl"
        if not p.is_file():
            raise FileNotFoundError(f"missing {p}")
        rows = read_jsonl(p)
        splits[s] = summarize_split(rows, s)
        hashes[s] = sha256_normalized(p)
        n_samples[s] = len(rows)

    # Cross-split checks.
    pair_sets = {s: {e[0] for e in v} for s, v in splits.items()}
    stem_sets = {s: {e[1] for e in v} for s, v in splits.items()}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            if pair_sets[a] & pair_sets[b]:
                raise ValueError(f"pair_id overlap between {a} and {b}")
            if stem_sets[a] & stem_sets[b]:
                raise ValueError(f"sample_id overlap between {a} and {b}")

    info_path = processed_dir / "split_info.json"
    split_info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    seed = split_info.get("random_seed", seed_hint)
    if split_info:
        exp = {"train": split_info.get("train_pairs"), "validation": split_info.get("val_pairs"),
               "test": split_info.get("test_pairs")}
        for s in SPLITS:
            if exp[s] is not None and exp[s] != len(splits[s]):
                raise ValueError(f"{s}: {len(splits[s])} pairs but split_info.json says {exp[s]}")

    return {
        "manifest_version": MANIFEST_VERSION,
        "generated_by": "src/data/make_split_manifest.py build",
        "split_source": "src/data/split.py (pair-level, sklearn train_test_split, two-stage)",
        "random_seed": seed,
        "label_convention": {str(LABEL_FACTUAL): "factual (right_answer)",
                             str(LABEL_HALLUCINATED): "hallucinated (hallucinated_answer)"},
        "hash_note": "sha256 of each processed file with CRLF normalized to LF",
        "counts": {s: {"pairs": len(splits[s]), "samples": n_samples[s]} for s in SPLITS},
        "file_sha256": hashes,
        # each entry: [pair_id, sample_id_stem]; samples are f"{stem}_0" and f"{stem}_1"
        "splits": splits,
    }


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    head = {k: v for k, v in manifest.items() if k != "splits"}
    lines = ["{"]
    for k, v in head.items():
        lines.append(f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)},")
    lines.append('  "splits": {')
    items = list(manifest["splits"].items())
    for i, (s, entries) in enumerate(items):
        body = ",\n".join("      " + json.dumps(e) for e in entries)
        lines.append(f'    {json.dumps(s)}: [\n{body}\n    ]' + ("," if i < len(items) - 1 else ""))
    lines.append("  }")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def compare(manifest: dict, other: dict) -> list[str]:
    """Return a list of human-readable differences (empty = identical)."""
    diffs = []
    for s in SPLITS:
        if manifest["counts"][s] != other["counts"][s]:
            diffs.append(f"{s}: counts {other['counts'][s]} != manifest {manifest['counts'][s]}")
        if manifest["splits"][s] != other["splits"][s]:
            a = {tuple(e) for e in manifest["splits"][s]}
            b = {tuple(e) for e in other["splits"][s]}
            diffs.append(f"{s}: membership differs (only in manifest: {len(a - b)}, only local: {len(b - a)})"
                         if a != b else f"{s}: same members but different order/sample numbering")
        if manifest["file_sha256"][s] != other["file_sha256"][s]:
            diffs.append(f"{s}: file hash differs")
    return diffs


def cmd_build(args) -> int:
    cfg = get_config()
    pdir = Path(args.processed_dir) if args.processed_dir else cfg.paths.resolve(cfg.paths.processed_dir)
    out = Path(args.output) if args.output else cfg.paths.resolve(cfg.paths.split_manifest)
    manifest = build_manifest(pdir)
    write_manifest(manifest, out)
    print(f"wrote {out}")
    for s in SPLITS:
        print(f"  {s:<11} pairs={manifest['counts'][s]['pairs']:>5} samples={manifest['counts'][s]['samples']:>5} "
              f"sha256={manifest['file_sha256'][s][:16]}...")
    return 0


def cmd_verify(args) -> int:
    cfg = get_config()
    pdir = Path(args.processed_dir) if args.processed_dir else cfg.paths.resolve(cfg.paths.processed_dir)
    mpath = Path(args.manifest) if args.manifest else cfg.paths.resolve(cfg.paths.split_manifest)
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    try:
        local = build_manifest(pdir)
    except (FileNotFoundError, ValueError) as e:
        print(f"FAIL: {e}")
        return 1
    diffs = compare(manifest, local)
    if diffs:
        print("FAIL: local data does not match the committed manifest:")
        for d in diffs:
            print("  -", d)
        return 1
    print(f"OK: {pdir} matches {mpath.name} (train/validation/test identical).")
    return 0


def cmd_reproduce(args) -> int:
    from src.data import split as split_mod  # noqa: E402

    cfg = get_config()
    mpath = Path(args.manifest) if args.manifest else cfg.paths.resolve(cfg.paths.split_manifest)
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    interim = Path(args.interim) if args.interim else split_mod.INTERIM_DATA_PATH
    with tempfile.TemporaryDirectory() as tmp:
        split_mod.split_pipeline(input_path=interim, output_dir=Path(tmp), random_seed=manifest["random_seed"])
        local = build_manifest(Path(tmp))
    diffs = compare(manifest, local)
    if diffs:
        print("FAIL: re-running split.py does NOT reproduce the manifest:")
        for d in diffs:
            print("  -", d)
        return 1
    print("OK: re-running split.py from the interim file reproduces the manifest exactly.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--processed-dir"); b.add_argument("--output")
    v = sub.add_parser("verify"); v.add_argument("--processed-dir"); v.add_argument("--manifest")
    r = sub.add_parser("reproduce"); r.add_argument("--interim"); r.add_argument("--manifest")
    args = ap.parse_args(argv)
    return {"build": cmd_build, "verify": cmd_verify, "reproduce": cmd_reproduce}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
