# HalluciDetect Dataset Selection

## Objective

Select datasets suitable for developing and evaluating the HalluciDetect
hallucination detection system.

## Requirements

The dataset strategy should support:

- Hallucination detection
- LLM-generated responses
- Ground-truth/reference information where available
- Binary or usable hallucination labels
- Semantic Entropy experiments
- Semantic Entropy Probe experiments
- English evaluation
- Hindi evaluation
- Hinglish evaluation

## Candidate Datasets

| Dataset | Task | Size | Language | Labels | Role | Status |
|---|---|---:|---|---|---|---|
| HaluEval | QA / Dialogue / Summarization / General | 35K | English | Yes | Candidate primary | Under review |
| TruthfulQA | Truthfulness QA | 817 | English | Truthfulness-oriented | Candidate external | Under review |
| HaluEval 2.0 | Factual hallucination | TBD | English | TBD | Candidate external | Under review |
| HaluBench | Hallucination evaluation | TBD | English | Yes | Candidate external | Under review |
| Custom Hindi | QA | TBD | Hindi | To be defined | Candidate | Under review |
| Custom Hinglish | QA | TBD | Hinglish | To be defined | Candidate | Under review |

## Selection Criteria

1. Relevance to hallucination detection
2. Dataset quality
3. Availability of labels
4. Ground-truth/reference information
5. Compatibility with selected base LLM
6. Suitability for Semantic Entropy experiments
7. Suitability for SEP experiments
8. Language coverage
9. Dataset license
10. Computational feasibility

## Decision

To be finalized after dataset comparison.