from pathlib import Path
import json
import pandas as pd

DATA = Path("data/processed")
OUT = Path("results/sanity_checks/formatting_audit")
OUT.mkdir(parents=True, exist_ok=True)

def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

rows = load_jsonl(DATA / "train.jsonl") + load_jsonl(DATA / "test.jsonl")
df = pd.DataFrame(rows)

df["answer"] = df["answer"].fillna("").astype(str)
df["ends_punctuation"] = df["answer"].str[-1:].isin(list(".,!?;:"))
df["char_count"] = df["answer"].str.len()
df["word_count"] = df["answer"].str.split().str.len()

summary = df.groupby("label").agg(
    samples=("sample_id", "count"),
    punctuation_ending_rate=("ends_punctuation", "mean"),
    mean_char_count=("char_count", "mean"),
    median_char_count=("char_count", "median"),
    mean_word_count=("word_count", "mean"),
    median_word_count=("word_count", "median"),
).reset_index()

summary["punctuation_ending_rate"] *= 100

summary.to_csv(OUT / "label_formatting_summary.csv", index=False)
print(summary.to_string(index=False))
print(f"\nSaved to: {OUT / 'label_formatting_summary.csv'}")
