"""
Pair-Level Dataset Splitting for HaluEval QA

This module implements pair-level splitting to prevent data leakage.

Critical Rule:
- Split at the PAIR level (each original record with right + hallucinated answer)
- Then expand to binary labels
- Never split after expansion

Workflow:
1. Load preprocessed interim data (still in pair format)
2. Split pairs into train/validation/test (80/10/10)
3. Expand each pair into two binary-labeled records
4. Save processed datasets
5. Validate no leakage

Binary Label Mapping:
- right_answer → label = 0 (Non-Hallucination)
- hallucinated_answer → label = 1 (Hallucination)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "halueval_qa_cleaned.jsonl"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Fixed random seed for reproducibility
RANDOM_SEED = 42

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def load_interim_data(path: Path) -> List[Dict]:
    """Load preprocessed interim data."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def split_pairs(
    pairs: List[Dict],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split pairs into train/validation/test sets.

    Uses stratified splitting if possible, but since each pair will generate
    one positive and one negative example, the class balance is automatically
    maintained.
    """
    # Verify ratios
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    # First split: separate test set
    train_val_pairs, test_pairs = train_test_split(
        pairs,
        test_size=test_ratio,
        random_state=random_seed,
        shuffle=True
    )

    # Second split: separate validation from train
    # Calculate validation ratio relative to train_val set
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)

    train_pairs, val_pairs = train_test_split(
        train_val_pairs,
        test_size=val_ratio_adjusted,
        random_state=random_seed,
        shuffle=True
    )

    return train_pairs, val_pairs, test_pairs


def expand_pair_to_binary(pair: Dict, sample_index: int) -> Tuple[Dict, Dict]:
    """
    Expand a single pair into two binary-labeled records.

    Returns:
        (non_hallucination_record, hallucination_record)
    """
    # Non-hallucination record (label = 0)
    non_halluc = {
        'sample_id': f"S{sample_index:06d}_0",
        'pair_id': pair['pair_id'],
        'language': pair['language'],
        'question': pair['question'],
        'knowledge': pair['knowledge'],
        'answer': pair['right_answer'],
        'label': 0
    }

    # Hallucination record (label = 1)
    halluc = {
        'sample_id': f"S{sample_index:06d}_1",
        'pair_id': pair['pair_id'],
        'language': pair['language'],
        'question': pair['question'],
        'knowledge': pair['knowledge'],
        'answer': pair['hallucinated_answer'],
        'label': 1
    }

    return non_halluc, halluc


def expand_pairs_to_binary(pairs: List[Dict], start_index: int = 0) -> List[Dict]:
    """
    Expand a list of pairs into binary-labeled records.

    Each pair generates 2 records (label 0 and label 1).
    """
    binary_records = []
    sample_index = start_index

    for pair in pairs:
        non_halluc, halluc = expand_pair_to_binary(pair, sample_index)
        binary_records.append(non_halluc)
        binary_records.append(halluc)
        sample_index += 1

    return binary_records


def validate_no_leakage(
    train_pairs: List[Dict],
    val_pairs: List[Dict],
    test_pairs: List[Dict]
) -> bool:
    """
    Validate that there is no data leakage across splits.

    Checks:
    1. No pair_id overlap
    2. No question overlap
    3. No question-knowledge pair overlap
    """
    print("\nData Leakage Validation:")
    print("="*80)

    train_df = pd.DataFrame(train_pairs)
    val_df = pd.DataFrame(val_pairs)
    test_df = pd.DataFrame(test_pairs)

    leakage_detected = False

    # Check 1: pair_id overlap
    train_pairs_set = set(train_df['pair_id'])
    val_pairs_set = set(val_df['pair_id'])
    test_pairs_set = set(test_df['pair_id'])

    train_val_overlap = train_pairs_set & val_pairs_set
    train_test_overlap = train_pairs_set & test_pairs_set
    val_test_overlap = val_pairs_set & test_pairs_set

    print(f"\n1. Pair ID Overlap:")
    print(f"   Train-Val overlap: {len(train_val_overlap)}")
    print(f"   Train-Test overlap: {len(train_test_overlap)}")
    print(f"   Val-Test overlap: {len(val_test_overlap)}")

    if len(train_val_overlap) > 0 or len(train_test_overlap) > 0 or len(val_test_overlap) > 0:
        leakage_detected = True
        print("   WARNING: LEAKAGE DETECTED: Pair IDs overlap!")

    # Check 2: Question overlap
    train_questions = set(train_df['question'])
    val_questions = set(val_df['question'])
    test_questions = set(test_df['question'])

    q_train_val = train_questions & val_questions
    q_train_test = train_questions & test_questions
    q_val_test = val_questions & test_questions

    print(f"\n2. Question Overlap:")
    print(f"   Train-Val overlap: {len(q_train_val)}")
    print(f"   Train-Test overlap: {len(q_train_test)}")
    print(f"   Val-Test overlap: {len(q_val_test)}")

    if len(q_train_val) > 0 or len(q_train_test) > 0 or len(q_val_test) > 0:
        leakage_detected = True
        print("   WARNING: LEAKAGE DETECTED: Questions overlap!")

    # Check 3: Question-Knowledge pair overlap
    train_qk = set(zip(train_df['question'], train_df['knowledge']))
    val_qk = set(zip(val_df['question'], val_df['knowledge']))
    test_qk = set(zip(test_df['question'], test_df['knowledge']))

    qk_train_val = train_qk & val_qk
    qk_train_test = train_qk & test_qk
    qk_val_test = val_qk & test_qk

    print(f"\n3. Question-Knowledge Pair Overlap:")
    print(f"   Train-Val overlap: {len(qk_train_val)}")
    print(f"   Train-Test overlap: {len(qk_train_test)}")
    print(f"   Val-Test overlap: {len(qk_val_test)}")

    if len(qk_train_val) > 0 or len(qk_train_test) > 0 or len(qk_val_test) > 0:
        leakage_detected = True
        print("   WARNING: LEAKAGE DETECTED: Question-Knowledge pairs overlap!")

    print("\n" + "="*80)
    if leakage_detected:
        print("DATA LEAKAGE CHECK: FAIL")
        return False
    else:
        print("DATA LEAKAGE CHECK: PASS")
        return True


def split_pipeline(
    input_path: Path = INTERIM_DATA_PATH,
    output_dir: Path = PROCESSED_DIR,
    random_seed: int = RANDOM_SEED
) -> None:
    """
    Run the complete splitting pipeline.

    Steps:
    1. Load interim data (pair format)
    2. Split pairs into train/val/test
    3. Validate no leakage
    4. Expand pairs to binary labels
    5. Save processed datasets
    """
    print("="*80)
    print("HaluEval QA Pair-Level Dataset Splitting")
    print("="*80)
    print(f"Random Seed: {random_seed}")
    print(f"Train Ratio: {TRAIN_RATIO}")
    print(f"Val Ratio: {VAL_RATIO}")
    print(f"Test Ratio: {TEST_RATIO}")

    # Load interim data
    print(f"\n1. Loading interim data from: {input_path}")
    pairs = load_interim_data(input_path)
    print(f"   Loaded {len(pairs)} pairs")

    # Split pairs
    print("\n2. Splitting pairs...")
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_seed=random_seed
    )

    print(f"   Train pairs: {len(train_pairs)}")
    print(f"   Val pairs: {len(val_pairs)}")
    print(f"   Test pairs: {len(test_pairs)}")

    # Validate no leakage
    print("\n3. Validating data integrity...")
    leakage_check_passed = validate_no_leakage(train_pairs, val_pairs, test_pairs)

    if not leakage_check_passed:
        raise ValueError("Data leakage detected! Cannot proceed.")

    # Expand to binary labels
    print("\n4. Expanding pairs to binary labels...")
    train_binary = expand_pairs_to_binary(train_pairs, start_index=0)
    val_binary = expand_pairs_to_binary(val_pairs, start_index=len(train_pairs))
    test_binary = expand_pairs_to_binary(test_pairs, start_index=len(train_pairs) + len(val_pairs))

    print(f"   Train samples: {len(train_binary)} (from {len(train_pairs)} pairs)")
    print(f"   Val samples: {len(val_binary)} (from {len(val_pairs)} pairs)")
    print(f"   Test samples: {len(test_binary)} (from {len(test_pairs)} pairs)")

    # Verify class balance
    train_df = pd.DataFrame(train_binary)
    val_df = pd.DataFrame(val_binary)
    test_df = pd.DataFrame(test_binary)

    print("\n5. Verifying class balance...")
    print(f"   Train - Label 0: {(train_df['label'] == 0).sum()}, Label 1: {(train_df['label'] == 1).sum()}")
    print(f"   Val   - Label 0: {(val_df['label'] == 0).sum()}, Label 1: {(val_df['label'] == 1).sum()}")
    print(f"   Test  - Label 0: {(test_df['label'] == 0).sum()}, Label 1: {(test_df['label'] == 1).sum()}")

    # Save processed datasets
    print(f"\n6. Saving processed datasets to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"
    test_path = output_dir / "test.jsonl"

    with open(train_path, 'w', encoding='utf-8') as f:
        for record in train_binary:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    with open(val_path, 'w', encoding='utf-8') as f:
        for record in val_binary:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    with open(test_path, 'w', encoding='utf-8') as f:
        for record in test_binary:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"   Saved train: {train_path}")
    print(f"   Saved validation: {val_path}")
    print(f"   Saved test: {test_path}")

    # Save split info
    split_info = {
        'random_seed': random_seed,
        'train_ratio': TRAIN_RATIO,
        'val_ratio': VAL_RATIO,
        'test_ratio': TEST_RATIO,
        'total_pairs': len(pairs),
        'train_pairs': len(train_pairs),
        'val_pairs': len(val_pairs),
        'test_pairs': len(test_pairs),
        'train_samples': len(train_binary),
        'val_samples': len(val_binary),
        'test_samples': len(test_binary)
    }

    split_info_path = output_dir / "split_info.json"
    with open(split_info_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2)

    print(f"   Saved split info: {split_info_path}")

    # Summary
    print("\n" + "="*80)
    print("Splitting Summary")
    print("="*80)
    print(f"Total pairs:      {len(pairs)}")
    print(f"Train pairs:      {len(train_pairs)} ({len(train_pairs)/len(pairs)*100:.1f}%)")
    print(f"Val pairs:        {len(val_pairs)} ({len(val_pairs)/len(pairs)*100:.1f}%)")
    print(f"Test pairs:       {len(test_pairs)} ({len(test_pairs)/len(pairs)*100:.1f}%)")
    print(f"\nTotal samples:    {len(train_binary) + len(val_binary) + len(test_binary)}")
    print(f"Train samples:    {len(train_binary)}")
    print(f"Val samples:      {len(val_binary)}")
    print(f"Test samples:     {len(test_binary)}")
    print(f"\nLeakage check:    PASSED")
    print(f"Random seed:      {random_seed}")
    print("="*80)


if __name__ == "__main__":
    split_pipeline()
