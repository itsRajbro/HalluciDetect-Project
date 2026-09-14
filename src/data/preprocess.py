"""
Data Preprocessing Pipeline for HaluEval QA Dataset

This module implements the preprocessing pipeline for the HaluEval QA dataset.
Preprocessing includes:
- Loading raw data
- Validation
- Cleaning (whitespace normalization, Unicode normalization)
- ID generation (pair_id)
- Metadata addition (language)
- Saving interim data for splitting

The pipeline preserves the original pair structure (right_answer + hallucinated_answer)
for proper train/validation/test splitting.
"""

import json
import unicodedata
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "halueval_qa.jsonl"
INTERIM_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "halueval_qa_cleaned.jsonl"


def load_raw_data(path: Path) -> List[Dict]:
    """Load raw JSONL data."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def validate_record(record: Dict, index: int) -> bool:
    """
    Validate a single record.

    A valid record must have:
    - All required fields: knowledge, question, right_answer, hallucinated_answer
    - Non-empty strings for all fields

    Returns True if valid, False otherwise.
    """
    required_fields = ['knowledge', 'question', 'right_answer', 'hallucinated_answer']

    # Check all fields exist
    for field in required_fields:
        if field not in record:
            print(f"Warning: Record {index} missing field '{field}'")
            return False

    # Check all fields are non-empty after stripping
    for field in required_fields:
        if not isinstance(record[field], str) or record[field].strip() == '':
            print(f"Warning: Record {index} has empty '{field}'")
            return False

    return True


def clean_text(text: str) -> str:
    """
    Clean text with safe transformations that preserve meaning.

    - Strip leading/trailing whitespace
    - Normalize internal whitespace (multiple spaces -> single space)
    - Unicode NFC normalization (canonical composition)
    """
    # Strip leading/trailing whitespace
    text = text.strip()

    # Normalize repeated whitespace
    text = ' '.join(text.split())

    # Unicode NFC normalization
    text = unicodedata.normalize('NFC', text)

    return text


def preprocess_record(record: Dict, pair_id: int) -> Dict:
    """
    Preprocess a single record.

    - Clean all text fields
    - Add pair_id
    - Add language metadata
    """
    cleaned = {
        'pair_id': f"P{pair_id:05d}",  # P00001, P00002, etc.
        'language': 'en',  # HaluEval QA is English
        'knowledge': clean_text(record['knowledge']),
        'question': clean_text(record['question']),
        'right_answer': clean_text(record['right_answer']),
        'hallucinated_answer': clean_text(record['hallucinated_answer'])
    }

    return cleaned


def preprocess_pipeline(
    input_path: Path = RAW_DATA_PATH,
    output_path: Path = INTERIM_DATA_PATH
) -> None:
    """
    Run the complete preprocessing pipeline.

    Steps:
    1. Load raw data
    2. Validate records
    3. Clean and preprocess valid records
    4. Save interim data
    """
    print("="*80)
    print("HaluEval QA Preprocessing Pipeline")
    print("="*80)

    # Load raw data
    print(f"\n1. Loading raw data from: {input_path}")
    raw_records = load_raw_data(input_path)
    print(f"   Loaded {len(raw_records)} records")

    # Validate records
    print("\n2. Validating records...")
    valid_records = []
    invalid_count = 0

    for idx, record in enumerate(raw_records):
        if validate_record(record, idx):
            valid_records.append(record)
        else:
            invalid_count += 1

    print(f"   Valid records: {len(valid_records)}")
    print(f"   Invalid records: {invalid_count}")

    if len(valid_records) == 0:
        raise ValueError("No valid records found!")

    # Preprocess valid records
    print("\n3. Preprocessing records...")
    preprocessed_records = []

    for idx, record in enumerate(valid_records, start=1):
        preprocessed = preprocess_record(record, pair_id=idx)
        preprocessed_records.append(preprocessed)

    print(f"   Preprocessed {len(preprocessed_records)} records")

    # Check for duplicate pair issues
    print("\n4. Quality checks...")
    df = pd.DataFrame(preprocessed_records)

    # Check if right_answer == hallucinated_answer (should not happen)
    same_answers = (df['right_answer'] == df['hallucinated_answer']).sum()
    if same_answers > 0:
        print(f"   Warning: {same_answers} records have identical right and hallucinated answers")

    # Check duplicate questions
    dup_questions = df['question'].duplicated().sum()
    print(f"   Duplicate questions: {dup_questions}")

    # Check duplicate question-knowledge pairs
    dup_pairs = df[['question', 'knowledge']].duplicated().sum()
    print(f"   Duplicate question-knowledge pairs: {dup_pairs}")

    # Save interim data
    print(f"\n5. Saving preprocessed data to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in preprocessed_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"   Saved {len(preprocessed_records)} records")

    # Summary statistics
    print("\n" + "="*80)
    print("Preprocessing Summary")
    print("="*80)
    print(f"Input records:       {len(raw_records)}")
    print(f"Invalid records:     {invalid_count}")
    print(f"Valid records:       {len(valid_records)}")
    print(f"Preprocessed records: {len(preprocessed_records)}")
    print(f"Output file:         {output_path}")
    print("="*80)


if __name__ == "__main__":
    preprocess_pipeline()
