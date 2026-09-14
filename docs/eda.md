# Exploratory Data Analysis (EDA) Documentation

**Project:** HalluciDetect  
**Component:** Exploratory Data Analysis  
**Date:** 2026-09-02

---

## Overview

This document presents the exploratory data analysis conducted on the processed HaluEval QA dataset for hallucination detection research.

## Dataset Overview

- **Total Samples:** 20,000 (10,000 question-answer pairs × 2)
- **Train Samples:** 15,998 (79.99%)
- **Validation Samples:** 2,002 (10.01%)
- **Test Samples:** 2,000 (10.00%)
- **Labels:** Binary (0 = Non-Hallucination, 1 = Hallucination)

---

## 1. Class Distribution Analysis

### Overall Distribution

The dataset exhibits **perfect class balance**:

- **Label 0 (Non-Hallucination):** 10,000 samples (50.0%)
- **Label 1 (Hallucination):** 10,000 samples (50.0%)

### Per-Split Distribution

All splits maintain the same perfect balance:

| Split | Label 0 | Label 1 | Total |
|-------|---------|---------|-------|
| Train | 7,999 (50.0%) | 7,999 (50.0%) | 15,998 |
| Validation | 1,001 (50.0%) | 1,001 (50.0%) | 2,002 |
| Test | 1,000 (50.0%) | 1,000 (50.0%) | 2,000 |

### Research Observation

> **Perfect class balance eliminates the need for class-weighted training or sampling strategies.** Standard accuracy metrics will be meaningful without requiring weighted adjustments. This balance is maintained by design through the pair-level splitting approach, where each pair contributes exactly one positive and one negative example.

**Visualization:** `results/figures/class_distribution.png`

---

## 2. Text Length Analysis

### 2.1 Question Length

Questions in the dataset show moderate length variation:

**Character Length Statistics (Unique Questions: 10,000):**
- Mean: 106.31 characters
- Median: 90 characters
- Min: 20 characters
- Max: 630 characters

**Word Count Statistics:**
- Mean: 17.94 words
- Median: 16 words
- Min: 4 words
- Max: 100 words

### Research Observation

> Questions are primarily medium-length, suitable for transformer-based models. The distribution shows no extreme outliers that would indicate data quality issues.

**Visualization:** `results/figures/question_length_distribution.png`

---

### 2.2 Answer Length Comparison (Factual vs Hallucinated)

This is a **critical analysis** for detecting potential dataset biases.

**Non-Hallucinated Answers (Label 0):**
- Character length mean: 13.65 characters
- Word count mean: 2.22 words
- Generally concise and direct

**Hallucinated Answers (Label 1):**
- Character length mean: 66.22 characters
- Word count mean: 10.99 words
- Tend to be significantly longer

### Critical Research Finding

> **Hallucinated answers show a systematic tendency to be longer than factual answers (4.7x difference).** This represents a potential confounding factor: a simple length-based classifier could achieve non-trivial performance without learning genuine hallucination patterns. 
> 
> **Research Implication:** When evaluating trained models, it is essential to analyze whether the model relies primarily on length features or learns deeper semantic patterns. Models should be tested on length-controlled subsets to verify they are not exploiting this bias.

**Visualization:** `results/figures/answer_length_comparison.png`

---

### 2.3 Knowledge/Context Length

**Character Length Statistics (Unique Knowledge Contexts: ~10,000):**
- Mean: 343.85 characters
- Median: 321 characters
- Min: 75 characters
- Max: 1,557 characters

**Word Count Statistics:**
- Mean: 55.44 words
- Median: 52 words

### Research Observation

> Knowledge contexts provide substantial grounding information for each question. The length distribution is reasonably consistent, indicating that most questions have adequate contextual support for verification.

**Visualization:** `results/figures/knowledge_length_distribution.png`

---

## 3. Data Quality Analysis

### 3.1 Missing Values

**Result:** Zero missing values detected across all fields.

| Field | Missing Count | Missing % |
|-------|---------------|-----------|
| sample_id | 0 | 0.00% |
| pair_id | 0 | 0.00% |
| language | 0 | 0.00% |
| question | 0 | 0.00% |
| knowledge | 0 | 0.00% |
| answer | 0 | 0.00% |
| label | 0 | 0.00% |

**Empty Strings:** 0 across all text fields

### Research Observation

> The dataset demonstrates **excellent data quality** with complete information for all records. No imputation or special handling for missing data is required.

---

### 3.2 Duplicate Analysis

**Exact duplicate rows:** 0  
**Duplicate sample_ids:** 0  
**Pairs with ≠ 2 samples:** 0

Each `pair_id` appears exactly twice in the combined dataset (once with label 0, once with label 1), as designed.

### Research Observation

> The dataset structure is consistent and correct. No unexpected duplicates or malformed pairs exist.

---

## 4. Outlier Analysis

### Question Outliers

- **Shortest question:** 20 characters (simple factual questions)
- **Longest question:** 630 characters (complex multi-part questions)

### Answer Outliers

**Non-Hallucinated Answers:**
- Shortest: 1 character (single word)
- Longest: 529 characters (detailed factual responses)

**Hallucinated Answers:**
- Shortest: 3 characters
- Longest: 380 characters (elaborate incorrect explanations)

### Research Observation

> Outliers represent natural variation in question and answer complexity rather than data quality issues. No records require removal based on length.

---

## 5. Dataset Statistics Summary

Comprehensive statistics saved in: `results/tables/dataset_statistics.csv`

### Key Metrics:

| Field | Count | Mean Length | Std Dev | Min | Max |
|-------|-------|-------------|---------|-----|-----|
| Question (chars) | 10,000 | 106.31 | 60.51 | 20 | 630 |
| Question (words) | 10,000 | 17.94 | 9.74 | 4 | 100 |
| Answer - Non-Halluc (chars) | 10,000 | 13.65 | 12.03 | 1 | 529 |
| Answer - Non-Halluc (words) | 10,000 | 2.22 | 1.84 | 1 | 81 |
| Answer - Halluc (chars) | 10,000 | 66.22 | 36.28 | 3 | 380 |
| Answer - Halluc (words) | 10,000 | 10.99 | 6.15 | 1 | 61 |
| Knowledge (chars) | 9,936 | 343.85 | 134.91 | 75 | 1,557 |
| Knowledge (words) | 9,936 | 55.44 | 21.62 | 11 | 256 |

---

## 6. Research-Oriented Observations

### 6.1 Class Balance

✅ **Perfect 50/50 balance maintained across all splits**

**Research Significance:** This eliminates class imbalance as a confounding factor. Standard accuracy, precision, recall, and F1-score metrics will be meaningful without weighted adjustments.

### 6.2 Answer Length Correlation

⚠️ **Hallucinated answers are systematically longer than factual answers**

**Research Significance:** This represents a **potential confounding factor**. A naive length-based classifier could achieve better-than-random performance (potentially 55-65% accuracy) without learning semantic hallucination patterns.

**Mitigation Strategy:**
- Analyze model attention patterns to verify semantic understanding
- Test on length-controlled subsets
- Compare performance to a length-only baseline
- Investigate whether the model learns beyond length features

### 6.3 Data Quality

✅ **High-quality dataset: no missing values, no leakage, no structural issues**

**Research Significance:** The dataset is production-ready for model training. No additional cleaning or preprocessing is required.

### 6.4 Dataset Size

✅ **20,000 samples (15,998 train, 2,002 val, 2,000 test)**

**Research Significance:** 
- Training set is substantial for fine-tuning pre-trained models
- Validation and test sets (2,000+ samples each) provide statistically robust evaluation
- May be limited for training large models from scratch

### 6.5 Language and Domain Scope

⚠️ **English-only, factual QA domain**

**Research Significance:**
- Multilingual capabilities (Hindi, Hinglish) will require additional data or transfer learning
- Hallucination patterns in other tasks (summarization, dialogue) may differ
- Domain-specific fine-tuning may be needed for production applications

---

## 7. Potential Biases and Limitations

### Identified Biases:

1. **Length Bias:** Hallucinated answers are systematically longer
   - **Research Risk:** Models may exploit this shortcut
   - **Mitigation:** Length-aware evaluation and baseline comparisons

2. **Language Bias:** Dataset is English-only
   - **Research Risk:** No direct evidence for Hindi/Hinglish performance
   - **Mitigation:** Require additional multilingual data

3. **Task Bias:** Focus on factual QA
   - **Research Risk:** May not generalize to other generation tasks
   - **Mitigation:** Future evaluation on diverse tasks

### Dataset Strengths:

1. ✅ Perfect class balance
2. ✅ No data leakage
3. ✅ High data quality
4. ✅ Adequate dataset size
5. ✅ Clear factual grounding (knowledge field)

---

## 8. Implications for Research

### For Model Development:

1. **Baseline Models:**
   - Implement a length-only baseline to quantify the bias
   - Compare learned models against this baseline

2. **Model Selection:**
   - Choose models capable of learning semantic patterns (transformers)
   - Verify the selected base LLM provides hidden-state access for SEP

3. **Training Strategy:**
   - Perfect class balance allows standard training procedures
   - No need for class weighting or oversampling

4. **Evaluation Strategy:**
   - Standard metrics are appropriate
   - Include length-controlled test subsets
   - Analyze model predictions by answer length quartiles

5. **Future Enhancements:**
   - Collect multilingual data (Hindi, Hinglish)
   - Expand to other generation tasks
   - Create length-balanced variants if bias is problematic

---

## 9. Visualizations Generated

All visualizations saved in: `results/figures/`

1. **class_distribution.png** - Overall and per-split class balance
2. **question_length_distribution.png** - Question character and word count distributions
3. **answer_length_comparison.png** - Factual vs hallucinated answer lengths (histograms and box plots)
4. **knowledge_length_distribution.png** - Knowledge/context length distributions

---

## 10. Summary Tables Generated

All tables saved in: `results/tables/`

1. **dataset_size_summary.csv** - Split sizes and percentages
2. **dataset_statistics.csv** - Comprehensive length statistics
3. **missing_values_analysis.csv** - Missing value report
4. **duplicate_analysis.csv** - Duplicate detection results

---

## Conclusion

The HaluEval QA dataset has been thoroughly analyzed and demonstrates:

✅ **High quality** - No missing values, no structural issues  
✅ **Proper splits** - No data leakage, good size distribution  
✅ **Perfect balance** - Equal class distribution  
⚠️ **Length correlation** - Requires monitoring during model evaluation  
✅ **Research-ready** - Suitable for rigorous experimentation

The dataset provides a solid foundation for hallucination detection research.

---

## Files and Notebooks

- **EDA Notebook:** `notebooks/03_eda.ipynb`
- **Executed Notebook:** `notebooks/03_eda_executed.ipynb`
- **Analysis Script:** `src/data/run_eda.py`
- **Figures:** `results/figures/`
- **Tables:** `results/tables/`
- **This Document:** `docs/eda.md`

---

**Document Status:** Complete  
**Last Updated:** 2026-09-02
