# HalluciDetect: SE + SEP Hybrid (Late Fusion): Analysis Plan

Written 2026-10-12, **before any SE or SEP result exists**. Owner: Person 1.
Purpose: fix the fusion method, the comparisons and the decision rule in advance, so the analysis cannot be chosen after seeing results.
Any change after this date goes into the amendment log (section 10) with its date and reason, and is reported in the final write-up.

## 1. Question and hypotheses

Does combining Semantic Entropy (SE) and Semantic Entropy Probes (SEP) detect hallucination better than either alone, **beyond what answer length and answer form explain**?

- **H1 (primary):** the hybrid has higher AUROC on the **length-controlled test set** than *each* of its two inputs.
- **H2 (secondary):** the same holds on the full test set and on pair-level accuracy.
- **H3 (secondary):** the hybrid's gain does not come from answer length (section 7).

"Higher" always means a paired-bootstrap 95% CI for the AUROC difference that excludes 0. A CI that includes 0 is reported as **inconclusive**, never as "no benefit": the length-controlled set has only 137 matched pairs, so power is limited.

## 2. Inputs

Each input is one standard schema run (`src/common/schema.py`) with validation **and** test files, each written with a validation-selected threshold.

| Hybrid | Inputs | Status |
|---|---|---|
| **H-B (primary)** | candidate-aware SE + SEP | the candidate-aware SE variant Person 2 declares final |
| H-A (secondary) | global SE + SEP | |

Reason for H-B being primary: global SE depends only on the question and knowledge, so both answers of a pair receive the same score (AUROC exactly 0.5 on this data, by construction). In a linear fusion it cancels inside a pair and can only help by calibrating across questions. H-A is still run, as a check on this.

**Variant selection rule.** The SE variant and the SEP variant (layer, pooling, probe settings, SE clustering settings) are chosen by their owners using validation data only, and are frozen before fusion starts. Fusion never re-selects them, and never looks at test results to pick them. If an owner submits several variants, the one named in their hand-off is used; the others are reported as exploratory.

## 3. Fusion model (fixed)

1. Orient: every input score is "higher = more likely hallucinated" (project convention). No sign flipping.
2. Standardize each input using the **validation** mean and standard deviation (the same numbers applied to test).
3. Fit L2-regularized logistic regression on validation: `S_hybrid = w0 + w1·S_SE + w2·S_SEP` (standardized inputs). The regularization strength C is chosen from `{0.01, 0.1, 1, 10, 100}` by 5-fold cross-validation **grouped by `pair_id`** (both answers of a question always in the same fold), minimizing log-loss.
4. Threshold: Youden's J, computed on the **out-of-fold** validation scores (so the threshold is not tuned on the data the weights were fitted to), then applied unchanged to test.
5. Weights are reported as fitted. Negative weights are not clipped; they are reported and discussed.
6. The hybrid is written as a normal detector run (`detector="hybrid"`, variant naming the inputs), with weights, C and standardization constants in the run metadata, so `compare_detectors` treats it like any other detector.

Test labels are never used for fitting, selection or tuning of anything in this plan.

## 4. Comparators

| Comparator | Why |
|---|---|
| Each input alone (SE, SEP) | the H1 test |
| **Unfitted average** of the two standardized scores (equal weights) | does fitting weights matter at all? |
| Length-only; answer-only Transformer; full Transformer | the reference bars (length-controlled AUROC 0.51, 0.84, 0.99) |
| Oracle (right whenever any detector is right) | ceiling on what any fusion of these detectors could gain |

## 5. Primary decision rule

Hybrid H-B **improves** on its inputs if the paired-bootstrap CI (5,000 resamples, seed 0) of `AUROC(hybrid) - AUROC(SE)` **and** of `AUROC(hybrid) - AUROC(SEP)` both exclude 0 in the positive direction on the length-controlled set. Requiring both is an intersection test, so no multiplicity correction is needed for the primary claim. Everything else is secondary and labelled as such.

## 6. What is reported (always with 95% CIs)

AUROC, accuracy, F1, pair/match accuracy on test and on the length-controlled set; the length-stratified table with minority-class counts; fitted weights and chosen C; error overlap and McNemar tests between hybrid and each input; the oracle ceiling; efficiency (hybrid cost = SE cost + SEP cost: latency, number of generations, peak GPU memory).

## 7. Length checks (H3)

- Correlation between each input score, the hybrid score and answer length (validation).
- Performance of the hybrid inside length bins and on the length-controlled set (primary).
- Secondary fit: add `log(answer length in characters)` as an extra input. If the SE and SEP weights stay similar, they carry information beyond length; if they collapse, the gain was length.
- Compare to the answer-only Transformer bar (0.84): a detector that does not exceed it has not shown that it reads anything beyond the form of the answer.

## 8. Data and leakage rules

- Fit on validation only; `train` is not used by fusion (SEP uses it to train its own probe, owned by Person 3).
- Test is scored once with the frozen fusion. The hybrid is not re-fitted after any look at test results.
- All inputs must cover exactly the same samples (`schema.check_coverage`); failed samples must be scored and written by their owner, not dropped.
- `config_hash` / `prompt_version` mismatches between inputs are reported, not hidden.

## 9. Known limits (stated in advance)

- The length-controlled set (274 samples) is small; confidence intervals will be wide for detectors with AUROC well below 0.99.
- The Transformer baselines reach 0.99 on the length-controlled set, so SE, SEP and the hybrid are expected to rank below the supervised model; the comparison of interest is among SE, SEP and the hybrid, and against the length-only and answer-only bars.
- Global SE cannot separate the two answers of a question (section 2).

## 10. Amendment log

| Date | Change | Reason | Made before seeing test results? |
|---|---|---|---|
| | | | |
