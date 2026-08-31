from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "halueval_qa.jsonl"

dataset = load_dataset(
    "pminervini/HaluEval",
    "qa"
)

qa_data = dataset["data"]

print(dataset)
print(qa_data.column_names)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
qa_data.to_json(OUTPUT_PATH)

print(f"Dataset downloaded successfully to {OUTPUT_PATH}.")
