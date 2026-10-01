# Length-Controlled Test Set Report

Generated programmatically from the written artifact. Secondary stress-test benchmark; the original `test.jsonl` remains the primary benchmark. `match_id` is a constructed length-control relationship, not a HaluEval pair; original `pair_id` values are unchanged.

## A. Provenance

- Input: `data\processed\test.jsonl` (SHA-256 `528d0d91678c8f98e6b26368ef55c75a09ffc04e6ea16d88928bb9e9965ec246`)
- Output: `data\processed\test_length_controlled.jsonl` (SHA-256 `41d9ac7e0ba3231ef0d35b67d40a3d6e2c008ced12012beab5e41c500826e661`)
- Tolerance: 1 character
- Length metric: `len(answer)`
- Algorithm: max-cardinality then min total absolute gap (`scipy.optimize.linear_sum_assignment`), inputs sorted by (length, sample_id)
- Python 3.11.9, numpy 2.4.6, pandas 3.0.5, scipy 1.17.1
- Random seed: none

## B. Counts

- Original test size: 2000 (1000 factual, 1000 hallucinated)
- Controlled total: 274
- Controlled factual: 137
- Controlled hallucinated: 137
- Coverage: 13.70%

## C. Character-length statistics

| class | mean | median | std | min | max |
|---|---|---|---|---|---|
| factual | 28.686 | 27.000 | 18.633 | 3 | 120 |
| hallucinated | 28.861 | 28.000 | 18.621 | 3 | 120 |

## D. Gap statistics

- Mean absolute gap: 0.175
- Median absolute gap: 0.000
- Maximum absolute gap: 1
- Total absolute gap: 24

## E. Word-length diagnostics (not used for matching)

- Factual mean / median words: 4.365 / 4.000
- Hallucinated mean / median words: 4.847 / 5.000

## F. KS diagnostic

- KS statistic: 0.073
- KS p-value: 0.861
- Note: a distribution diagnostic only, not the sole criterion for equivalence.

## G. Validation results

| check | result | detail |
|---|---|---|
| row count | PASS | 274 (expected 274) |
| class balance | PASS | label0=137, label1=137 |
| sample_id uniqueness | PASS | 274 unique / 274 |
| match_id structure | PASS | 137 match_ids, 2 rows each |
| one factual + one hallucinated per match | PASS |  |
| length tolerance | PASS | max gap = 1; stored length == len(answer) |
| source membership | PASS |  |
| no sample reuse | PASS |  |
| pair_id preservation | PASS | pair_id and question/knowledge/answer/label identical to original |
| original test.jsonl unchanged | PASS | hash before == after |
