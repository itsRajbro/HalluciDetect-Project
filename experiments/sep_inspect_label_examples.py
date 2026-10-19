from pathlib import Path
import json

for split in ("train", "test"):
    path = Path("data/processed") / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

    print(f"\n{'=' * 25} {split.upper()} {'=' * 25}")

    for label in (0, 1):
        examples = [r for r in rows if int(r["label"]) == label]
        print(f"\n--- LABEL {label}: 5 examples ---")

        for r in examples[:5]:
            answer = str(r["answer"]).replace("\n", "\\n")
            print("ID:", r["sample_id"])
            print("Question:", str(r["question"])[:180])
            print("Answer:", repr(answer[:300]))
            print("Character count:", len(answer))
            print("Word count:", len(answer.split()))
            print()
