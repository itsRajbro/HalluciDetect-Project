# Length-Matching Feasibility Report

## Measurements (auto-generated)

Original test: factual mean 13.80, hallucinated mean 65.02 chars.

|   tolerance_chars |   total_matched |   coverage_percent |   factual_mean_chars |   hallucinated_mean_chars |   mean_abs_length_gap |   max_abs_length_gap |   ks_statistic |   ks_pvalue |
|------------------:|----------------:|-------------------:|---------------------:|--------------------------:|----------------------:|---------------------:|---------------:|------------:|
|                 1 |             274 |               13.7 |               28.686 |                    28.861 |                 0.175 |                    1 |          0.073 |       0.861 |
|                 3 |             304 |               15.2 |               28.25  |                    29.092 |                 0.842 |                    3 |          0.164 |       0.033 |
|                 5 |             348 |               17.4 |               27.477 |                    29.592 |                 2.115 |                    5 |          0.27  |       0     |
|                10 |             472 |               23.6 |               25.419 |                    31.322 |                 5.903 |                   10 |          0.462 |       0     |
|                15 |             634 |               31.7 |               23.331 |                    33.64  |                10.309 |                   15 |          0.599 |       0     |

Full table: `results/tables/length_matching_feasibility.csv`. Figures in `results/figures/`.

## Decision (fill in after reviewing the results above)

Consider coverage, residual mean/median imbalance, mean & max gap, and distribution similarity (KS as diagnostic only). If no tolerance is acceptable, state that explicitly.

**Answer:** _TODO_

## Provenance

```json
{
  "dataset": "data/processed/test.jsonl",
  "sha256": "528d0d91678c8f98e6b26368ef55c75a09ffc04e6ea16d88928bb9e9965ec246",
  "tolerances": [
    1,
    3,
    5,
    10,
    15
  ],
  "length_metric": "len(answer)",
  "algorithm": "scipy.optimize.linear_sum_assignment, max-cardinality then min total gap; inputs sorted by (length, sample_id)",
  "python": "3.11.9",
  "scipy": "1.17.1",
  "numpy": "2.4.6",
  "pandas": "3.0.5",
  "random_seed": null
}
```
