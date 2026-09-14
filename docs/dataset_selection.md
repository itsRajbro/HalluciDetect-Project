# Dataset Selection Documentation

**Project:** HalluciDetect  
**Component:** Dataset Selection and Justification  
**Date:** 2026-09-02

---

## Dataset Overview

### Selected Dataset: HaluEval QA

- **Name:** HaluEval (QA portion)
- **Source:** [pminervini/HaluEval](https://huggingface.co/datasets/pminervini/HaluEval) via Hugging Face Datasets
- **Task Type:** Question Answering with Hallucination Labels
- **Language:** English
- **Records:** 10,000 question-answer pairs
- **License:** Research use (check original source for commercial use)

---

## Dataset Purpose

HaluEval is a benchmark dataset designed for evaluating hallucination detection in large language models. The QA portion specifically focuses on factual question-answering scenarios where models can generate either correct or hallucinated responses.

### Relevance to HalluciDetect Research

This dataset is ideal for our research because:

1. **Clear Task Definition:** Binary classification (hallucinated vs non-hallucinated answers)
2. **Ground Truth Labels:** Expert-annotated correct and hallucinated answers
3. **Factual Grounding:** Each sample includes knowledge/context for verification
4. **Substantial Size:** 10,000 pairs → 20,000 binary samples
5. **Research Standard:** Widely used in hallucination detection literature

---

## Dataset Structure

### Raw Format

Each record in the raw HaluEval QA dataset contains:

```json
{
  "knowledge": "Background information providing factual context",
  "question": "The question to be answered",
  "right_answer": "The factual correct answer",
  "hallucinated_answer": "A hallucinated/plausible but incorrect answer"
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `knowledge` | string | Contextual information providing factual grounding |
| `question` | string | The question to be answered |
| `right_answer` | string | The correct factual answer (label 0) |
| `hallucinated_answer` | string | A hallucinated incorrect answer (label 1) |

---

## Processed Dataset Structure

After preprocessing and splitting, the canonical format is:

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

### Canonical Fields

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | string | Unique identifier for each binary sample |
| `pair_id` | string | Links the two answers (right and hallucinated) from the same question |
| `language` | string | Language code (currently 'en' for English) |
| `question` | string | The question text |
| `knowledge` | string | Background/context information |
| `answer` | string | The answer text (from right_answer or hallucinated_answer) |
| `label` | int | 0 = Non-Hallucination, 1 = Hallucination |

---

## Dataset Statistics

### Size and Splits

| Split | Pairs | Samples | Percentage |
|-------|-------|---------|------------|
| Train | 7,999 | 15,998 | 80.0% |
| Validation | 1,001 | 2,002 | 10.0% |
| Test | 1,000 | 2,000 | 10.0% |
| **Total** | **10,000** | **20,000** | **100%** |

### Class Distribution

- **Perfect Balance:** 50% Non-Hallucination (label 0), 50% Hallucination (label 1)
- **Balance maintained across all splits**

### Text Statistics

| Field | Mean (chars) | Mean (words) | Min | Max |
|-------|--------------|--------------|-----|-----|
| Question | 106.31 | 17.94 | 20 | 630 |
| Answer - Non-Hallucination | 13.65 | 2.22 | 1 | 529 |
| Answer - Hallucination | 66.22 | 10.99 | 3 | 380 |
| Knowledge | 343.85 | 55.44 | 75 | 1557 |

---

## Data Quality

### Quality Checks Performed

✅ **Missing Values:** None detected  
✅ **Invalid Records:** None detected  
✅ **Data Leakage:** None (verified via pair-level splitting)  
✅ **Label Integrity:** All labels are valid (0 or 1)  
✅ **Duplicate Records:** No exact duplicates  
✅ **Pair Integrity:** Each pair_id appears exactly twice (once per label)

### Issues Identified

⚠️ **Answer Length Correlation:** Hallucinated answers are systematically longer than factual answers (66.22 chars vs 13.65 chars mean). This represents a potential confounding factor that requires monitoring during model evaluation.

---

## Research Suitability

### Why HaluEval QA is Appropriate

1. **Clear Problem Alignment:** Directly supports binary hallucination classification research
2. **Ground Truth Available:** Expert annotations provide reliable labels
3. **Context Provided:** Knowledge field enables verification and reasoning
4. **Substantial Size:** Sufficient for fine-tuning experiments
5. **Balanced Classes:** No class imbalance issues
6. **Research Standard:** Facilitates comparison with existing literature

### Limitations

1. **Language:** English-only (future multilingual work requires additional data for Hindi/Hinglish)
2. **Task Scope:** QA only (hallucination patterns in summarization/dialogue may differ)
3. **Length Correlation:** Hallucinated answers tend to be longer (potential confounding factor)
4. **Domain:** General factual QA (domain-specific applications may require fine-tuning)

---

## Data Flow

```text
Raw HaluEval QA (10,000 pairs)
        ↓
    Download
        ↓
    Validation
        ↓
    Preprocessing
        ↓
Interim Cleaned Data (10,000 pairs)
        ↓
    Pair-Level Split (80/10/10)
        ↓
Binary Label Expansion
        ↓
Processed Dataset (20,000 samples)
```

---

## Files and Locations

### Raw Data
- `data/raw/halueval_qa.jsonl` - Original downloaded dataset

### Interim Data
- `data/interim/halueval_qa_cleaned.jsonl` - Preprocessed pairs

### Processed Data
- `data/processed/train.jsonl` - Training set (15,998 samples)
- `data/processed/validation.jsonl` - Validation set (2,002 samples)
- `data/processed/test.jsonl` - Test set (2,000 samples)
- `data/processed/split_info.json` - Split metadata

---

## Reproducibility

### Downloading the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("pminervini/HaluEval", "qa")
```

### Running Preprocessing

```bash
python src/data/download_halueval.py
python src/data/preprocess.py
python src/data/split.py
```

### Verification

All splits can be verified by checking:
- File sizes match expected counts
- Class balance is maintained
- No pair_id overlaps across splits
- All required fields are present

---

## Research Usage

### Training Input

The processed dataset provides:
- **Input:** (question, knowledge, answer) → Label
- **Task:** Binary classification
- **Output:** 0 (Non-Hallucination) or 1 (Hallucination)

### Future Extensions

For continued research:
1. **Base LLM Fine-tuning:** Train/fine-tune on the training set
2. **Hyperparameter Optimization:** Use validation set
3. **Final Evaluation:** Test on held-out test set
4. **Semantic Entropy Experiments:** Use knowledge and question for generation
5. **SEP Development:** Extract hidden states from the selected base LLM

---

## Attribution

**Dataset Source:**  
Li, J., Cheng, X., Zhao, W. X., et al. (2023). HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models.  

**Hugging Face Dataset:**  
https://huggingface.co/datasets/pminervini/HaluEval

---

## Research Roadmap

1. ✅ Dataset selected and documented
2. ✅ Dataset downloaded and inspected
3. ✅ Preprocessing completed
4. ✅ EDA completed
5. 🔄 Base LLM selection (in progress)
6. ⏳ Model training and evaluation (planned)

---

**Document Status:** Complete  
**Last Updated:** 2026-09-02
