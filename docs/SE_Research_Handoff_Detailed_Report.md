# Semantic Entropy (SE) — Research Handoff & Technical Report
**Project:** HalluciDetect-Project

**Workstream:** Person 2 — Semantic Entropy (SE)

**Branch:** `person2/semantic-entropy`

**Status date:** 18 October 2026

**Latest confirmed commit:** `0e1dec2` — docs: clarify exploratory semantic entropy evaluation

**Current experiment tags:** `validation_full_v1`, `lc_full_v1`, `test_full_anchor_v1`

> **Scope:** This report covers Semantic Entropy (SE), not Semantic Entropy Probes (SEP). SE uses sampled generations, semantic clusters, entropy, and candidate-aware scoring; SEP uses hidden-state features and a supervised probe.

## Executive Summary
The Person 2 Semantic Entropy workstream implements shared generation, data, schema, and evaluation interfaces; semantic clustering; Shannon entropy over cluster probabilities; candidate-aware scoring; NLI-based semantic equivalence; and resumable validation and length-controlled evaluation runners.

The latest recorded regression test run passed **173 tests in 218.44 seconds**.

Two test evaluations have been completed:

- **Full held-out test:** 2,000 examples, comprising 1,000 factual and 1,000 hallucinated examples.

- **Length-controlled test subset:** 274 examples, comprising 137 matched pairs and 137 examples per label.

For the full held-out test, candidate-aware SE achieved **86.05% accuracy and 0.8583 AUROC**. Global SE achieved **49.85% accuracy and 0.4962 AUROC**.

For the length-controlled subset, candidate-aware SE achieved **74.82% accuracy and 0.7472 AUROC**. Global SE achieved **45.26% accuracy and 0.4450 AUROC**.

Thresholds were selected using validation data only and frozen before test evaluation. These findings describe performance on the evaluated HaluEval QA dataset; they do not establish universal hallucination-detection performance.

## Current Status
| Area | Status | Evidence |

|---|---|---|

| Implementation | Implemented | Shared pipeline, entropy, clustering, candidate-aware scoring, NLI equivalence, and runners |

| Automated tests | Passed | `python -m pytest -q` — 173 passed in 218.44s |

| Validation | Completed | 1,001 pairs; 2,002 records per detector |

| Validation thresholds | Selected and frozen | Thresholds selected using validation data only |

| Full held-out test | Completed | 2,000 samples per detector |

| Full-test evaluation | Completed | `outputs/evaluation/full_test_anchor_v1_summary.json` |

| Length-controlled test | Completed | 274 records per detector; 137 matched pairs |

| Length-controlled evaluation | Completed | Evaluation JSON files saved for both detectors |

| Git publication | Final artifacts published | Commit `0e1dec2` is synchronized with the remote; local intermediate untracked artifacts remain and should be reviewed separately |

## 1. Workstream Definition and Terminology
Semantic entropy measures uncertainty over semantic answer clusters rather than raw token-level variation. The pipeline samples multiple answers for an input, groups semantically equivalent answers, and computes entropy over cluster probabilities. Candidate-aware scoring measures how represented a supplied candidate is among generated answers.

**Interpretation caution:** Low global semantic entropy is not proof that an answer is factual. A model can be confidently wrong. Candidate-aware score is not a truth oracle; it measures candidate representation relative to the generated sample set. Keep token entropy, global semantic entropy, and candidate-aware score distinct.

## 2. Data and Split Contract
The project contract records HaluEval QA with 10,000 raw question/answer pairs expanded into 20,000 candidate examples.

- Train: 7,999 pairs

- Validation: 1,001 pairs

- Test: 1,000 pairs

Label convention: **1 = hallucinated**, **0 = factual**. Higher detector scores should indicate a greater likelihood of hallucination.

The length-controlled test set contains **274 examples**, arranged as **137 matched pairs**, with 137 examples per label. Answer lengths within each matched pair differ by at most one character. Some matched pairs may use different original questions, so matched-pair accuracy requires cautious interpretation.

Preserve sample IDs, pair IDs, match IDs, split labels, config hash, prompt version, label semantics, and output metadata. Do not select or tune thresholds using test labels.

## 3. Implementation and Experiment Artifacts
Relevant implementation files include:

- `src/common/generation.py` — generation outputs and statistics.

- `src/common/data.py` — split loading, pair grouping, controlled-test loading, and match grouping.

- `src/common/schema.py` — detector records, JSONL/metadata read/write, efficiency and coverage helpers.

- `src/common/evaluator.py` — score alignment, AUROC, classification metrics, threshold selection, pair/match metrics, length strata, and run evaluation.

- `src/detectors/semantic_entropy/entropy.py` — Shannon entropy over cluster probabilities.

- `src/detectors/semantic_entropy/clustering.py` — exact baseline clustering with optional equivalence predicate.

- `src/detectors/semantic_entropy/candidate_aware.py` — candidate-cluster probability and score `1 - probability`.

- `src/detectors/semantic_entropy/nli_equivalence.py` — NLI equivalence and short-answer anchoring behavior.

The NLI model loader pins `facebook/bart-large-mnli` to revision `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce` for both tokenizer and model loading. Reconstructed historical metadata does not independently prove which revision was used by earlier runs.

- `src/detectors/semantic_entropy/runner_pair_checkpoint.py` — resumable pair-aware validation runner.

- `src/detectors/semantic_entropy/runner_controlled_test.py` — resumable length-controlled test runner.

Recorded experiment configuration:

- Config hash: `435a3e4fb7b8`

- Prompt version: `v1`

- Model: Llama 3.2 3B Instruct NF4

- Dataset: HaluEval QA

### Validation Outputs
- `outputs/detectors/se_global/nli/validation_validation_full_v1.jsonl`

- `outputs/detectors/se_candidate/nli/validation_validation_full_v1.jsonl`

- Checkpoint: `outputs/detectors/\_checkpoints/se_pair_validation_full_v1.jsonl`

- Global threshold: `outputs/detectors/se_global/nli/threshold_validation_full_v1.json`

- Candidate threshold: `outputs/detectors/se_candidate/nli/threshold_validation_full_v1.json`

Validation covers 1,001 pairs, or 2,002 records per detector. Thresholds were saved separately from detector output metadata.

### Full Held-Out Test Outputs
- `outputs/detectors/se_global/nli/test_full_anchor_v1.jsonl`

- `outputs/detectors/se_global/nli/test_full_anchor_v1.meta.json`

- `outputs/detectors/se_candidate/nli/test_full_anchor_v1.jsonl`

- `outputs/detectors/se_candidate/nli/test_full_anchor_v1.meta.json`

- `outputs/evaluation/full_test_anchor_v1_summary.json`

### Length-Controlled Test Outputs
- `outputs/detectors/se_global/nli/test_length_controlled_lc_full_v1.jsonl`

- `outputs/detectors/se_global/nli/test_length_controlled_lc_full_v1.meta.json`

- `outputs/detectors/se_candidate/nli/test_length_controlled_lc_full_v1.jsonl`

- `outputs/detectors/se_candidate/nli/test_length_controlled_lc_full_v1.meta.json`

- `outputs/evaluation/se_global_length_controlled_lc_full_v1.json`

- `outputs/evaluation/se_candidate_length_controlled_lc_full_v1.json`

## 4. Validation-Only Threshold Selection
Threshold selection used validation records only and the Youden criterion. The thresholds were frozen before the test evaluations.

| Detector | Frozen threshold | Validation Youden J |

|---|---:|---:|

| Global SE | `0.1625414866957241` | `0.0` |

| Candidate-aware SE | `0.95` | `0.6313686313686313` |

The experimental anchoring variant independently selected a candidate-aware threshold of `0.95`, with validation Youden J `0.714286`.

The global SE threshold has validation Youden J equal to zero. Its score distribution, direction, and validation behavior warrant further investigation. Do not tune the threshold using test labels.

## 5. Full Held-Out Test Evaluation — Anchoring Matcher
The full held-out test evaluation used 2,000 samples: 1,000 factual and 1,000 hallucinated. Thresholds were frozen from validation. No threshold was selected or tuned using test labels.

| Metric | Global SE | Candidate-aware SE |

|---|---:|---:|

| Threshold | 0.162541 | 0.95 |

| Accuracy | 49.85% | 86.05% |

| Precision | 49.91% | 81.54% |

| Recall | 86.60% | 93.20% |

| F1 | 63.33% | 86.98% |

| Specificity | 13.10% | 78.90% |

| AUROC | 0.4962 | 0.8583 |

| Pair-level accuracy | 49.70% | 85.25% |

### Confusion Matrices
Rows are actual labels; columns are predicted labels.

| Detector | TN | FP | FN | TP |

|---|---:|---:|---:|---:|

| Global SE | 131 | 869 | 134 | 866 |

| Candidate-aware SE | 789 | 211 | 68 | 932 |

### Interpretation
Candidate-aware SE substantially outperformed global SE on this held-out test split. Its AUROC of 0.8583 indicates useful ranking discrimination on this dataset. Global SE's AUROC of 0.4962 is close to random ranking, and its specificity is very low at the frozen validation threshold.

Candidate-aware SE still produced 211 false positives and 68 false negatives. These results support further investigation of candidate-aware scoring, not a claim of universal hallucination detection. Dataset label noise, answer length, the semantic-equivalence heuristic, and score distribution may affect performance.

## 6. Length-Controlled Test Results
Evaluation subset: **274 examples**, comprising 137 factual and 137 hallucinated examples, arranged as 137 matched pairs. Higher scores are intended to indicate hallucination.

| Metric | Global SE | Candidate-aware SE |

|---|---:|---:|

| Threshold | 0.1625414866957241 | 0.95 |

| Accuracy | 45.26% | 74.82% |

| Precision | 47.21% | 67.89% |

| Recall | 80.29% | 94.16% |

| F1 | 59.46% | 78.90% |

| Specificity | 10.22% | 55.47% |

| AUROC | 0.4450 | 0.7472 |

| Matched-pair accuracy | 43.43% | 74.09% |

| Matched pairs | 137 | 137 |

| Matched-pair ties | 1 | 57 |

### Confusion Matrices
| Detector | TN | FP | FN | TP |

|---|---:|---:|---:|---:|

| Global SE | 14 | 123 | 27 | 110 |

| Candidate-aware SE | 76 | 61 | 8 | 129 |

Candidate-aware SE outperformed global SE on this controlled subset. However, it still produced 61 false positives and 57 matched-pair ties. Global SE had very low specificity and AUROC below 0.5 at its frozen validation threshold.

Matched pairs control answer length within a pair, not all confounders. Some pairs may use different original questions, so matched-pair metrics require cautious interpretation. The controlled subset should not be treated as a replacement for the full test split.

### Global SE Controlled-Subset Score Discrepancy

A verification compared two saved score sources on the same 274 length-controlled
samples. The dedicated `lc_full_v1` run gives AUROC **0.4450423571**, while
restricting the full-test scores to those same samples gives AUROC
**0.4460546646**. The two score sources differ for one sample, `S009097_0`;
the other 273 scores match. The dedicated run metadata records `sample_id` as
the generation seed key and `match_id_used_for_generation` as `false`.
The reason for the single-sample score difference has not been established.
The reported controlled-run metrics above remain tied to the dedicated
length-controlled evaluation file; the discrepancy should be resolved before
treating the two score sources as interchangeable.


### Selected SE Variant

The selected SE variant for Person 2 is candidate-aware Semantic Entropy using the `anchor_v1` short-answer anchoring matcher, subject to team review. Global SE is retained as the reference baseline. Candidate-split experiments are exploratory only and must not be used for final reported results.

## 7. Experimental Follow-Up: Short-Answer Anchoring Matcher
A short-answer anchoring fallback was added to `src/detectors/semantic_entropy/nli_equivalence.py`. Saved generations from the existing validation checkpoint were rescored without regenerating answers.

### Validation Comparison
| Candidate-aware validation metric | Original matcher | Anchoring matcher |

|---|---:|---:|

| AUROC | 0.816634 | 0.858234 |

| Accuracy at fixed threshold 0.95 | 81.57% | 85.71% |

| F1 at fixed threshold 0.95 | 0.8352 | 0.8673 |

| Factual candidate match rate | 69.73% | 78.02% |

| Hallucinated candidate match rate | 6.59% | 6.59% |

The changed-match review reported 83 decisions, all label-0 candidates changing from unmatched to matched. Several inspected examples appeared plausible, but this limited review does not establish that every new match is correct.

The anchoring rule is heuristic and may fail on negation, ambiguity, or statements that begin with a candidate but later qualify or contradict it. Fixed-threshold figures above are comparative validation results, not held-out test performance.

The anchoring variant selected its threshold using validation data only: `0.95`, with Youden J `0.714286`. The original threshold file remains available. Historical preservation of all baseline outputs has not been independently verified.

## 8. Reproducible Commands
Run commands from the repository root in PowerShell.

Activate the virtual environment:

```powershell

.\\.venv\Scripts\Activate.ps1

$env:PYTHONPATH = (Get-Location).Path

```

Run regression tests:

```powershell

python -m pytest -q

```

Latest recorded result: `173 passed in 218.44s`.

Inspect the current branch, latest commit, and working tree:

```powershell

git branch --show-current

git log -1 --oneline

git status --short

```

Check validation record counts:

```powershell

(Get-Content outputs/detectors/se_global/nli/validation_validation_full_v1.jsonl | Measure-Object -Line).Lines

(Get-Content outputs/detectors/se_candidate/nli/validation_validation_full_v1.jsonl | Measure-Object -Line).Lines

```

Check length-controlled record counts:

```powershell

(Get-Content outputs/detectors/se_global/nli/test_length_controlled_lc_full_v1.jsonl | Measure-Object -Line).Lines

(Get-Content outputs/detectors/se_candidate/nli/test_length_controlled_lc_full_v1.jsonl | Measure-Object -Line).Lines

```

Check full-test record counts:

```powershell

(Get-Content outputs/detectors/se_global/nli/test_full_anchor_v1.jsonl | Measure-Object -Line).Lines

(Get-Content outputs/detectors/se_candidate/nli/test_full_anchor_v1.jsonl | Measure-Object -Line).Lines

```

Validate the full-test summary JSON:

```powershell

python -m json.tool outputs/evaluation/full_test_anchor_v1_summary.json

```

These checks are useful but do not alone prove complete provenance, unique IDs, metadata consistency, or sample-ID alignment.

## 9. Risks and Limitations
- NLI equivalence can merge non-equivalent answers or split paraphrases.

- The anchoring matcher is heuristic and can incorrectly match answers that are qualified or contradicted later.

- Generation settings and sampling variability affect cluster probabilities and entropy.

- The controlled set controls answer length within a pair, not every confounder; some pairs may use different questions.

- Global SE has AUROC close to 0.5 on the full test split (0.4962) and the controlled subset (0.4450). These results indicate near-random ranking in this evaluation; the cause requires further investigation.

- Candidate-aware SE still produces false positives and false negatives.

- The original threshold file remains available, but historical preservation of all baseline outputs has not been independently verified.

- Per-length-stratum metrics and verified efficiency details, including generation counts, tokens, and runtime, are not included in the recorded results.

- Any mismatch in expected coverage or metadata must stop evaluation instead of silently dropping samples.

- Local backups, diagnostics, and generated outputs should be reviewed before committing. Avoid broad staging commands such as `git add .`.

## 10. Git and Publication Status
The latest confirmed commit is `0e1dec2`, pushed to `origin/person2/semantic-entropy`.

The length-controlled evaluation artifacts were committed and pushed. Full-test anchoring evaluation artifacts were also recorded as committed and pushed in the current work history.

Intermediate local files remain untracked in the working tree. They include diagnostic outputs, temporary validation runs, backup scripts, and additional test files. They have not been automatically staged or deleted. Review each file before deciding whether it belongs in version control.

Before making another commit, inspect:

```powershell

git status --short

git diff --check

git diff

```

Stage only explicitly reviewed files by their paths. Do not stage or delete all untracked files indiscriminately.

## 11. Final Acceptance Checklist
- [x] Validation covers all 1,001 pairs, or 2,002 records per detector.

- [x] Thresholds were selected using validation data only and frozen before test evaluation.

- [x] The length-controlled evaluation covers all 274 samples and 137 matched pairs.

- [x] Global and candidate-aware controlled outputs have recorded coverage and matching IDs.

- [x] Controlled-test evaluation uses frozen validation thresholds.

- [x] The full held-out test evaluation covers 2,000 samples.

- [x] Full-test output files and evaluation summary are recorded.

- [x] Regression tests passed: 173 passed in 218.44 seconds.

- [x] Length-controlled results are reported separately from full-test results.

- [ ] Add length-stratified metrics and verified efficiency measurements if available.

- [ ] Independently verify historical preservation of all baseline outputs.

- [ ] Review remaining local untracked artifacts before any further publication.

## 12. Safe Teammate Summary
The Person 2 Semantic Entropy workstream has an implemented and tested detector pipeline, a resumable validation runner, and a resumable length-controlled test runner. The recorded configuration is `435a3e4fb7b8`, prompt version `v1`, using Llama 3.2 3B Instruct NF4 on HaluEval QA. The latest recorded regression run passed 173 tests in 218.44 seconds.

Full validation completed over 1,001 pairs. The global threshold was `0.1625414866957241` with validation Youden J `0.0`; the candidate-aware threshold was `0.95` with validation Youden J `0.6313686313686313`. The anchoring variant's validation threshold was also `0.95`, with validation Youden J `0.714286`.

On the full held-out test split of 2,000 examples, candidate-aware SE achieved 86.05% accuracy and 0.8583 AUROC; global SE achieved 49.85% accuracy and 0.4962 AUROC. On the separate 274-example length-controlled subset, candidate-aware SE achieved 74.82% accuracy and 0.7472 AUROC; global SE achieved 45.26% accuracy and 0.4450 AUROC.

Thresholds were selected using validation data only. Candidate-aware scoring merits further investigation, while global SE's near-random ranking and low specificity require diagnosis. The anchoring matcher remains heuristic, and these results should not be generalized beyond the evaluated data without additional evidence.

**Primary full-test summary:** `outputs/evaluation/full_test_anchor_v1_summary.json`

**Length-controlled evaluation files:**

- `outputs/evaluation/se_global_length_controlled_lc_full_v1.json`

- `outputs/evaluation/se_candidate_length_controlled_lc_full_v1.json`

**Latest confirmed commit:** `0e1dec2` — `docs: clarify exploratory semantic entropy evaluation`

## Full Test Evaluation — 18 October 2026
### Evaluation protocol
- Test split: 2,000 examples (1,000 factual and 1,000 hallucinated).

- Configuration hash: `435a3e4fb7b8`; prompt version: `v1`.

- Semantic equivalence variant: experimental NLI short-answer anchor.

- Thresholds were frozen using validation data and were not tuned on the test set:

  - Global SE: `0.1625414866957241`

  - Candidate-aware SE: `0.95`

### Full test results
| Metric | Global SE | Candidate-aware SE |

|---|---:|---:|

| Accuracy | 49.85% | 86.05% |

| Precision | 49.91% | 81.54% |

| Recall | 86.60% | 93.20% |

| F1 | 0.6333 | 0.8698 |

| Specificity | 13.10% | 78.90% |

| AUROC | 0.4962 | 0.8583 |

| Pair-level accuracy | 49.70% | 85.25% |

| Pair-level ties | 156 | 199 |

### Length-stratified evaluation
The evaluator used four answer-length bins. Bin AUROC values:

| Answer length (characters) | Global SE AUROC | Candidate-aware SE AUROC |

|---|---:|---:|

| `[2, 12)` | 0.3623 | 0.8280 |

| `[12, 28)` | 0.4336 | 0.8967 |

| `[28, 57)` | 0.4795 | 0.6963 |

| `[57, 313]` | 0.5999 | 0.4656 |

The bins have substantially different label distributions, so their AUROC values should be interpreted cautiously. The longest bin has only eight factual examples, making its specificity and AUROC estimates particularly unstable.

### Efficiency
- Total test records: 2,000.

- Total generations recorded: 20,000.

- Total generated tokens recorded: 403,825.

- Summed recorded runtime: 53,269.02 seconds (approximately 14.80 hours).

- Mean recorded runtime per test record: 26.63 seconds.

- Maximum recorded peak GPU memory: 4,283.76 MB.

The global and candidate-aware outputs share the same generation run. Their efficiency figures must not be added together.

### Interpretation and limitations
Candidate-aware SE substantially outperformed global SE on this test set. Global SE achieved AUROC 0.4962 and specificity 13.10%, indicating poor discrimination at the frozen threshold. Its high recall does not compensate for the large number of false positives.

Candidate-aware SE achieved AUROC 0.8583, but it is not a truth oracle. Its errors, length dependence, and 199 pair-level ties should remain part of the interpretation.

Semantic equivalence relies on an experimental NLI short-answer matcher, which may incorrectly merge or separate answers. These findings apply to this dataset, configuration, matcher, and frozen thresholds; they do not establish generalization to other models or datasets.

### Evaluation artifacts
- `outputs/evaluation/full_test_anchor_v1_metrics_verified.json`

- `outputs/evaluation/se_global_full_test_anchor_v1_metrics.json`

- `outputs/evaluation/se_candidate_full_test_anchor_v1_metrics.json`

The two detector-specific reports contain the detailed metrics, length-stratified breakdowns, and efficiency statistics.

## Anchored Validation Diagnostic — 17 October 2026
- Anchored validation contains 2,002 records across 1,001 matched pairs.

- Global semantic entropy assigns identical scores to both members of all 1,001 pairs. Validation AUROC is 0.5000 and Youden's J is 0.0000. Threshold tuning cannot resolve this paired-label limitation.

- Candidate-aware SE produces different scores for 759/1,001 pairs (75.8%).

- Candidate-aware threshold selected on validation: 0.95 (Youden's J = 0.714286).

- Candidate-aware validation metrics at threshold 0.95: accuracy 0.8571, precision 0.8095, recall 0.9341, specificity 0.7802, F1 0.8673, AUROC 0.8582.

- Global anchored validation metadata was reconstructed because the original metadata file was missing. It must not be represented as original run-emitted metadata.

- No threshold was tuned using test labels. Global SE's paired-label limitation is specific to this evaluation setup and does not establish that global SE is universally ineffective.
