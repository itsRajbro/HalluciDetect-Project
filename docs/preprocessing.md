# Data Preprocessing Documentation

**Project:** HalluciDetect  
**Component:** Data Preprocessing Pipeline  
**Date:** 2026-09-02

---

## Overview

This document describes the data preprocessing pipeline applied to the HaluEval QA dataset for the HalluciDetect hallucination detection research project.

## Raw Dataset

- **Source:** HaluEval QA (pminervini/HaluEval)
- **Format:** JSONL
- **Records:** 10,000 question-answer pairs
- **Location:** `data/raw/halueval_qa.jsonl`

### Raw Dataset Structure

Each raw record contains:
- `knowledge`: Background information/context
- `question`: The question to be answered
- `right_answer`: Factual correct answer
- `hallucinated_answer`: Incorrect/hallucinated answer

---

## Preprocessing Pipeline

### Stage 1: Data Loading

The raw JSONL file is loaded and parsed into structured records.

**Implementation:** `src/data/preprocess.py`

### Stage 2: Data Validation

Each record is validated to ensure:
- All required fields are present
- All fields contain non-empty strings
- No null or undefined values

**Results:**
- Total records loaded: 10,000
- Valid records: 10,000
- Invalid records: 0

### Stage 3: Text Cleaning

Safe text transformations are applied that preserve factual meaning:

1. **Whitespace Normalization**
   - Strip leading/trailing whitespace
   - Normalize repeated whitespace to single space

2. **Unicode Normalization**
   - Apply NFC (Canonical Composition) normalization
   - Ensures consistent character representation

**Rationale:** These transformations improve consistency without altering semantic content.

### Stage 4: ID Generation

Two types of IDs are generated:

1. **pair_id**: Unique identifier for each original question-answer pair
   - Format: `P00001`, `P00002`, etc.
   - Critical for preventing data leakage during splitting

2. **sample_id**: Unique identifier for each binary-labeled sample (generated during splitting)
   - Format: `S000001_0`, `S000001_1`, etc.

### Stage 5: Metadata Addition

- **language**: Set to `en` (English) for all HaluEval QA records

### Stage 6: Quality Checks

The preprocessing pipeline performs quality checks:

- **Duplicate questions:** 0 found
- **Duplicate question-knowledge pairs:** 0 found
- **Identical right and hallucinated answers:** 0 found

---

## Processed Dataset Structure (Interim)

After preprocessing, each record contains:

```json
{
  "pair_id": "P00001",
  "language": "en",
  "knowledge": "Cleaned background information",
  "question": "Cleaned question text",
  "right_answer": "Cleaned factual answer",
  "hallucinated_answer": "Cleaned hallucinated answer"
}
```

**Location:** `data/interim/halueval_qa_cleaned.jsonl`

---

## Dataset Splitting

### Critical Design Decision: Pair-Level Splitting

To prevent data leakage, splitting occurs at the **pair level** before expanding to binary labels.

#### Correct Workflow:

```text
10,000 pairs
     ↓
Split pairs (80/10/10)
     ↓
Train: 7,999 pairs | Val: 1,001 pairs | Test: 1,000 pairs
     ↓
Expand to binary labels
     ↓
Train: 15,998 samples | Val: 2,002 samples | Test: 2,000 samples
```

#### Why This Matters:

Each pair contains the same `question` and `knowledge` with two different answers. If we split after expansion, the same question-knowledge context could appear in both training and test sets, causing severe data leakage.

**Implementation:** `src/data/split.py`

### Split Ratios

- **Train:** 80% (7,999 pairs → 15,998 samples)
- **Validation:** 10% (1,001 pairs → 2,002 samples)
- **Test:** 10% (1,000 pairs → 2,000 samples)

### Random Seed

- **Fixed seed:** 42
- Ensures reproducibility across runs

---

## Binary Label Expansion

After splitting, each pair is expanded into two binary-labeled samples:

### Label Mapping:

- **Label 0 (Non-Hallucination):** `right_answer`
- **Label 1 (Hallucination):** `hallucinated_answer`

### Final Sample Structure:

```json
{
  "sample_id": "S000001_0",
  "pair_id": "P00001",
  "language": "en",
  "question": "Which magazine was started first?",
  "knowledge": "Arthur's Magazine (1844-1846) was...",
  "answer": "Arthur's Magazine",
  "label": 0
}
```

---

## Data Leakage Prevention

### Validation Checks Performed:

1. **Pair ID Overlap:** 
   - Train-Val: 0
   - Train-Test: 0
   - Val-Test: 0

2. **Question Overlap:**
   - Train-Val: 0
   - Train-Test: 0
   - Val-Test: 0

3. **Question-Knowledge Pair Overlap:**
   - Train-Val: 0
   - Train-Test: 0
   - Val-Test: 0

**Result:** ✓ DATA LEAKAGE CHECK: PASS

---

## Processed Dataset Files

### Output Files:

```text
data/processed/
├── train.jsonl          (15,998 samples)
├── validation.jsonl     (2,002 samples)
├── test.jsonl           (2,000 samples)
└── split_info.json      (metadata)
```

### Class Balance:

All splits maintain perfect class balance:
- Train: 7,999 label 0, 7,999 label 1
- Validation: 1,001 label 0, 1,001 label 1
- Test: 1,000 label 0, 1,000 label 1

---

## Preprocessing Summary

| Metric | Value |
|--------|-------|
| Input records | 10,000 |
| Invalid records removed | 0 |
| Valid records | 10,000 |
| Preprocessed pairs | 10,000 |
| Final binary samples | 20,000 |
| Train samples | 15,998 (80.0%) |
| Validation samples | 2,002 (10.0%) |
| Test samples | 2,000 (10.0%) |
| Data leakage | None detected |
| Class balance | Perfect (50/50) |

---

## Code Reproducibility

### Run Preprocessing:

```bash
python src/data/preprocess.py
```

### Run Splitting:

```bash
python src/data/split.py
```

### Expected Output:

- Cleaned interim data in `data/interim/`
- Train/validation/test splits in `data/processed/`
- No errors or warnings
- Leakage check passes

---

## Design Rationale

### Why This Approach?

1. **Minimal Transformation:** Preserve original meaning by applying only safe, necessary transformations
2. **Pair-Level Integrity:** Maintain the original paired structure to prevent leakage
3. **Reproducibility:** Fixed random seed ensures consistent splits
4. **Perfect Balance:** Binary expansion guarantees class balance
5. **Validation:** Automated leakage checks ensure data integrity

### What We Did NOT Do:

- ❌ Aggressive text normalization (lowercasing, stemming, lemmatization)
- ❌ Removal of punctuation or special characters
- ❌ Language-specific preprocessing (suitable for future multilingual work)
- ❌ Random splitting after binary expansion (would cause leakage)

---

## Future Work

1. Extension to multilingual datasets (Hindi, Hinglish)
2. Integration with additional hallucination benchmarks
3. Automated data quality monitoring
4. Pipeline optimization for larger datasets

---

## Files

- **Preprocessing script:** `src/data/preprocess.py`
- **Splitting script:** `src/data/split.py`
- **Exploration notebook:** `notebooks/01_dataset_exploration.ipynb`
- **Raw data:** `data/raw/halueval_qa.jsonl`
- **Interim data:** `data/interim/halueval_qa_cleaned.jsonl`
- **Processed data:** `data/processed/{train,validation,test}.jsonl`

---

**Document Status:** Complete  
**Last Updated:** 2026-09-02
