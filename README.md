# HalluciDetect - Hallucination Detection in Large Language Models

**A Research Project on Automated Hallucination Detection**

---

## Project Overview

HalluciDetect is a research initiative aimed at developing automated methods for detecting hallucinations in Large Language Model (LLM) generated responses. The project investigates uncertainty-based detection approaches, specifically Semantic Entropy and Semantic Entropy Probes (SEP), to identify when LLMs generate factually incorrect information with high confidence.

## Research Objectives

1. Develop automated hallucination detection methods for LLM-generated text
2. Investigate uncertainty-based detection using Semantic Entropy
3. Explore lightweight detection alternatives through Semantic Entropy Probes (SEP)
4. Establish a foundation for multilingual hallucination detection (English, Hindi, Hinglish)

---

## Current Phase: Data Foundation

### Completed Work

#### ✅ Dataset Selection and Acquisition
- Selected HaluEval QA benchmark dataset
- Acquired 10,000 question-answer pairs with hallucination labels
- Validated dataset quality and suitability

#### ✅ Data Preprocessing Pipeline
- Implemented robust preprocessing with quality checks
- Preserved original semantic meaning
- Generated stable identifiers for tracking
- Achieved zero data quality issues

#### ✅ Rigorous Dataset Splitting
- Designed pair-level splitting to prevent data leakage
- Implemented automated leakage validation
- Split: 80% train, 10% validation, 10% test
- Maintained perfect class balance across all splits

#### ✅ Comprehensive Exploratory Analysis
- Statistical analysis of text characteristics
- Class distribution verification
- Identified potential dataset biases (length correlation)
- Generated publication-ready visualizations

---

## Dataset

**Primary Dataset:** HaluEval QA

- **Source:** [pminervini/HaluEval](https://huggingface.co/datasets/pminervini/HaluEval)
- **Task:** Binary hallucination classification
- **Language:** English (with plans for multilingual extension)
- **Size:** 10,000 question-answer pairs → 20,000 binary samples
- **Splits:** Train (15,998), Validation (2,002), Test (2,000)
- **Class Balance:** Perfect 50/50 distribution

### Dataset Structure

Each sample contains:
- `question`: The question to be answered
- `knowledge`: Contextual factual information
- `answer`: The generated answer
- `label`: 0 (Non-Hallucination) or 1 (Hallucination)

---

## Repository Structure

```
HalluciDetect-Project/
│
├── data/
│   ├── raw/                          # Original datasets
│   ├── interim/                      # Intermediate processing
│   └── processed/                    # Train/val/test splits
│
├── src/
│   ├── data/                         # Data processing utilities
│   ├── models/                       # Model implementations
│   ├── semantic_entropy/             # Semantic Entropy implementation
│   └── sep/                          # SEP implementation
│
├── notebooks/                         # Research notebooks
├── results/                          # Experimental results
├── experiments/                      # Experiment scripts
├── docs/                            # Project documentation
└── paper/                           # Research paper drafts
```

---

## Setup and Installation

### Prerequisites
- Python 3.10+
- Virtual environment recommended

### Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

### Data Processing

```bash
# Preprocess raw data
python src/data/preprocess.py

# Create train/val/test splits
python src/data/split.py

# Run exploratory analysis
python src/data/run_eda.py
```

---

## Research Findings

### Data Quality Assessment

✅ **High-Quality Dataset**
- Zero missing values across all fields
- No data leakage (rigorously verified)
- Perfect class balance maintained
- Suitable for reliable model training and evaluation

### Dataset Characteristics

| Field | Mean Length (chars) | Mean Length (words) |
|-------|---------------------|---------------------|
| Question | 106 | 18 |
| Answer (Factual) | 14 | 2 |
| Answer (Hallucinated) | 66 | 11 |
| Knowledge Context | 344 | 55 |

### Critical Observation: Length Correlation

**Finding:** Hallucinated answers exhibit systematic length differences from factual answers (mean 66 vs 14 characters).

**Research Implication:** This correlation represents a potential confounding factor. Detection models must demonstrate semantic understanding beyond simple length-based heuristics. Evaluation protocols should include:
- Length-controlled test subsets
- Attention mechanism analysis
- Comparison against length-only baseline

---

## Technical Stack

- **Language:** Python 3.10+
- **ML Framework:** PyTorch
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **ML Utilities:** scikit-learn
- **Dataset Management:** Hugging Face Datasets
- **Analysis:** Jupyter Notebooks

---

## Research Roadmap

### Current Phase: Foundation ✓
- [x] Dataset selection and acquisition
- [x] Data preprocessing pipeline
- [x] Exploratory data analysis
- [x] Quality validation

### Next Phase: Model Development
- [ ] Base LLM selection and evaluation
- [ ] Baseline model implementation
- [ ] Fine-tuning experiments
- [ ] Performance benchmarking

### Future Work
- [ ] Semantic Entropy implementation
- [ ] Semantic Entropy Probes development
- [ ] Hybrid detection system
- [ ] Multilingual extension (Hindi, Hinglish)
- [ ] Production deployment pipeline

---

## Documentation

- **Dataset Selection:** `docs/dataset_selection.md`
- **Preprocessing Pipeline:** `docs/preprocessing.md`
- **Exploratory Analysis:** `docs/eda.md`
- **Complete Summary:** `PREPROCESSING_EDA_SUMMARY.md`

---

## Research Contributions

This project aims to contribute:

1. **Methodological:** Rigorous evaluation of uncertainty-based hallucination detection
2. **Technical:** Efficient detection through Semantic Entropy Probes
3. **Practical:** Production-ready hallucination detection system
4. **Multilingual:** Extension to low-resource languages (Hindi, Hinglish)

---

## Team

Research team of 5 members working collaboratively on:
- Dataset engineering and preprocessing
- Model architecture and infrastructure
- Semantic Entropy research
- SEP development
- Evaluation and analysis

---

## License

Research project for academic and research purposes.

---

## Citation

If you use this work, please cite:

```
HalluciDetect: Automated Hallucination Detection in Large Language Models
[Team Names], 2026
```

---

**Last Updated:** 2026-09-02  
**Status:** Foundation phase complete, entering model development phase
