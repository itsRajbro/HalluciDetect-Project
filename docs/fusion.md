# SE + SEP hybrid (late fusion)

Implements `docs/fusion_analysis_plan.md` (read it first: it fixes the method and the decision rule before any result).

## Run it

```powershell
python -m experiments.run_fusion `
  --input se  <SE validation run> <SE test run> `
  --input sep outputs/detectors/sep/mean_answer_layer_16/validation.jsonl outputs/detectors/sep/mean_answer_layer_16/test.jsonl `
  --logit sep --n-boot 5000 --out results/fusion/se-sep.json
```

Then compare with the baselines:

```powershell
python -m experiments.compare_detectors length_only/chars transformer/distilroberta-base_a transformer/distilroberta-base_qak hybrid/se-sep-lr hybrid/se-sep-avg --n-boot 5000
```

Primary hybrid (plan): candidate-aware SE + SEP. Secondary: global SE + SEP (`--input seg ...`), and `--add-length` for the length check.

## What each input run must contain

- A **validation run** and a **test run** on exactly the same samples as the official splits (2,002 / 2,000 samples), written with `schema.write_detector_output`. Subsets are rejected (coverage check).
- Same `config_hash` / `prompt_version` across inputs (otherwise the run stops unless `--allow-mixed-provenance`).
- A threshold is optional (SE has none; the fusion then uses Youden on that input's validation scores for accuracy numbers only).
- Pair-level cost: if a run writes the same generation cost on both samples of a pair, its meta must say `generation_statistics_unit: "pair"` and each record `extra.pair_id` (the SE runners already do); costs are then counted once.

## Method in one paragraph

Inputs are oriented (higher = more hallucinated), optionally logit-transformed, standardized with validation mean/sd, and combined by an L2 logistic regression fitted on **validation only**. C is chosen by 5-fold CV grouped by question pair. The threshold is selected on the out-of-fold validation scores. Test labels are never read by the fitting code (`tests/test_fusion.py` checks this by flipping them). The hybrid is written as ordinary detector runs (`hybrid/<names>-lr` and the unfitted average `hybrid/<names>-avg`) with weights, C and provenance in the run metadata.

## Reading the output

1. Fitted weights (standardized and raw-scale), C, and any negative weight (reported, never clipped).
2. Validation diagnostics: each input's AUROC, number of distinct values (SE has at most 11), correlations with each other and with answer length.
3. Metrics with 95% bootstrap CIs on test and on the length-controlled set.
4. Paired differences hybrid minus each comparator.
5. **Primary decision** (plan section 5): IMPROVES only if the hybrid beats every input on the length-controlled set with a CI above 0. INCONCLUSIVE is not evidence of "no benefit": that set has only 137 matched pairs.
6. Error overlap with the oracle ceiling, and the cost of the hybrid.

A pair-constant input (global SE) cancels inside a pair and cannot change pair-level accuracy.

## Partial coverage (when SE was scored on fewer samples)

The SE test runner scores only the 274 length-controlled test samples, and SE validation may cover only some pairs. Use:

```powershell
python -m experiments.run_fusion --allow-partial --min-val-pairs 100 --input se <val> <test> --input sep <val> <test> --logit sep
python -m experiments.compare_detectors length_only/chars transformer/distilroberta-base_a transformer/distilroberta-base_qak sep/mean_answer_layer_16 hybrid/se-sep-lr --scope controlled
```

The fusion is fitted on the validation samples every input covers, scored on the test samples every input covers, and the report says exactly how many were used. Full-test metrics are skipped when the covered test samples are not complete question pairs. See amendment 4 in the plan.

## Dry run

`--dry-run` fits on validation and prints the weights and validation diagnostics, writes no files and reads no test labels. Use it to check the pipeline on real files before running the analysis. Step-by-step order of operations: `docs/FUSION_RUNBOOK.md`.
