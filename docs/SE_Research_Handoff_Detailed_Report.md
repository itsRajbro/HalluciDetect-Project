# Semantic Entropy (SE) --- Research Handoff & Technical Report

**Project:** HalluciDetect-Project\
**Workstream:** Person 2 --- Semantic Entropy (SE)\
**Branch:** `person2/semantic-entropy`\
**Status date:** 15 October 2026\
**Latest previously reported commit:** `ff4ffa0` --- Add resumable
controlled-test semantic entropy runner\
**Current experiment tags:** `validation_full_v1`, `lc_full_v1`

> **Scope:** This report covers Semantic Entropy (SE), not Semantic
> Entropy Probes (SEP). SE uses sampled generations, semantic clusters,
> entropy, and candidate-aware scoring; SEP uses hidden-state features
> and a supervised probe.

## Executive summary

The SE workstream includes shared generation/data/schema/evaluator
interfaces, semantic clustering, Shannon entropy over cluster
probabilities, candidate-aware scoring, NLI-based semantic equivalence,
a resumable pair-aware validation runner, and a resumable
length-controlled test runner.

The latest recorded regression test run passed **148 tests in 30.76
seconds**.

Full validation and length-controlled test runs have completed: -
Validation: 1,001 pairs, or 2,002 records per detector. -
Length-controlled test: 274 examples, comprising 137 matched pairs and
137 examples per label. - Validation-only thresholds were selected and
saved separately. - Controlled-test evaluation was performed using those
frozen thresholds.

**Important scope limitation:** the results below cover the 274-example
length-controlled subset, not the full 2,000-example test split. Do not
present these as full-test performance.

## Current status

  -----------------------------------------------------------------------
  Area                    Status                  Evidence
  ----------------------- ----------------------- -----------------------
  Implementation          Implemented             Shared pipeline,
                                                  entropy, clustering,
                                                  candidate-aware
                                                  scoring, NLI
                                                  equivalence, and
                                                  runners

  Automated tests         Passed                  `python -m pytest -q`
                                                  --- 148 passed in
                                                  30.76s

  Validation run          Completed               1,001 pairs; 2,002
                                                  records per detector

  Validation thresholds   Selected and frozen     Threshold files record
                                                  validation-only
                                                  provenance

  Length-controlled run   Completed               274 records per
                                                  detector; 137 matched
                                                  pairs

  Controlled-test         Completed               Evaluation JSON files
  evaluation                                      saved for both
                                                  detectors

  Full 2,000-example test Not completed           Only the
  evaluation                                      length-controlled
                                                  subset has been
                                                  evaluated

  Git publication         Pending review          Working tree contains
                                                  modified tests and
                                                  untracked local
                                                  artifacts; stage
                                                  selectively
  -----------------------------------------------------------------------

## 1. Workstream definition and terminology

Semantic entropy measures uncertainty over semantic answer clusters
rather than raw token-level variation. The pipeline samples multiple
answers for an input, groups semantically equivalent answers, and
computes entropy over cluster probabilities. Candidate-aware scoring
measures how represented a supplied candidate is among generated
answers.

> **Interpretation caution:** Low global semantic entropy is not proof
> that an answer is factual. A model can be confidently wrong.
> Candidate-aware score is not a truth oracle; it measures candidate
> representation relative to the generated sample set. Keep token
> entropy, global semantic entropy, and candidate-aware score distinct.

## 2. Data and split contract

The project contract records HaluEval QA with 10,000 raw question/answer
pairs expanded into 20,000 candidate examples. Pair-level split:

-   Train: 7,999 pairs
-   Validation: 1,001 pairs
-   Test: 1,000 pairs

Label convention: **1 = hallucinated**, **0 = factual**. The shared
contract specifies that higher detector scores should indicate a greater
likelihood of hallucination.

The length-controlled test set contains **274 examples**, arranged as
**137 matched pairs**, with 137 examples per label. Answer lengths
within each matched pair differ by at most one character. Some matched
pairs may use different original questions, so matched-pair accuracy
requires cautious interpretation.

Preserve sample IDs, pair IDs, match IDs, split labels, config hash,
prompt version, label semantics, and output metadata. Do not select or
tune thresholds using test labels.

## 3. Implementation and experiment artifacts

Relevant implementation files include:

-   `src/common/generation.py` --- generation outputs and statistics.
-   `src/common/data.py` --- split loading, pair grouping,
    controlled-test loading, and match grouping.
-   `src/common/schema.py` --- detector records, JSONL/metadata
    read/write, efficiency and coverage helpers.
-   `src/common/evaluator.py` --- score alignment, AUROC, classification
    metrics, threshold selection, pair/match metrics, length strata, and
    run evaluation.
-   `src/detectors/semantic_entropy/entropy.py` --- Shannon entropy over
    cluster probabilities.
-   `src/detectors/semantic_entropy/clustering.py` --- exact baseline
    clustering with optional equivalence predicate.
-   `src/detectors/semantic_entropy/candidate_aware.py` ---
    candidate-cluster probability and score `1 - probability`.
-   `src/detectors/semantic_entropy/nli_equivalence.py` --- reported NLI
    model `facebook/bart-large-mnli`, bidirectional entailment threshold
    0.85, and short-answer fallback.
-   `src/detectors/semantic_entropy/runner_pair_checkpoint.py` ---
    resumable pair-aware validation runner.
-   `src/detectors/semantic_entropy/runner_controlled_test.py` ---
    resumable length-controlled test runner.

Recorded experiment configuration: - Config hash: `435a3e4fb7b8` -
Prompt version: `v1` - Model: Llama 3.2 3B Instruct NF4 - Dataset:
HaluEval QA

### Validation outputs

-   `outputs/detectors/se_global/nli/validation_validation_full_v1.jsonl`
-   `outputs/detectors/se_candidate/nli/validation_validation_full_v1.jsonl`
-   Checkpoint:
    `outputs/detectors/_checkpoints/se_pair_validation_full_v1.jsonl`

Recorded validation metadata checks: - 2,002 records per detector -
Split: `validation` - Detectors: `se_global` and `se_candidate` -
Variant: `nli` - Config hash: `435a3e4fb7b8` - Prompt version: `v1` -
Output metadata threshold: `None` (thresholds saved separately)

### Length-controlled test outputs

-   `outputs/detectors/se_global/nli/test_length_controlled_lc_full_v1.jsonl`
-   `outputs/detectors/se_candidate/nli/test_length_controlled_lc_full_v1.jsonl`
-   Evaluation:
    `outputs/evaluation/se_global_length_controlled_lc_full_v1.json`
-   Evaluation:
    `outputs/evaluation/se_candidate_length_controlled_lc_full_v1.json`

Recorded controlled-test metadata checks: - 274 records per detector -
Split: `test` - Detectors: `se_global` and `se_candidate` - Variant:
`nli` - Config hash: `435a3e4fb7b8` - Prompt version: `v1` - Output
metadata threshold: `None`; the evaluator used the separately saved
validation threshold.

## 4. Validation-only threshold selection

Threshold selection used validation records only and the Youden
criterion. These thresholds were then frozen before controlled-test
evaluation.

  Detector                   Frozen threshold    Validation Youden J Threshold source
  -------------------- ---------------------- ---------------------- ------------------
  Global SE              `0.1625414866957241`                  `0.0` Validation only
  Candidate-aware SE                   `0.95`   `0.6313686313686313` Validation only

Threshold files: -
`outputs/detectors/se_global/nli/threshold_validation_full_v1.json` -
`outputs/detectors/se_candidate/nli/threshold_validation_full_v1.json`

The global SE threshold has a validation Youden J of zero. This warrants
investigation of score distributions, score direction, and validation
behavior, without tuning against test labels.

## 5. Length-controlled test results

Evaluation subset: **274 examples** (137 positive, 137 negative), with
137 matched pairs. Higher scores are intended to indicate hallucination.

  Metric                             Global SE   Candidate-aware SE
  ----------------------- -------------------- --------------------
  Threshold                 0.1625414866957241                 0.95
  Accuracy                              45.26%               74.82%
  Precision                             47.21%               67.89%
  Recall                                80.29%               94.16%
  F1                                    59.46%               78.90%
  Specificity                           10.22%               55.47%
  AUROC                                 0.4450               0.7472
  Matched-pair accuracy                 43.43%               74.09%
  Matched pairs                            137                  137
  Matched-pair ties                          1                   57

### Confusion matrices

Rows are actual labels; columns are predicted labels.

**Global SE**

                              Predicted factual (0)   Predicted hallucinated (1)
  ------------------------- ----------------------- ----------------------------
  Actual factual (0)                        TN = 14                     FP = 123
  Actual hallucinated (1)                   FN = 27                     TP = 110

**Candidate-aware SE**

                              Predicted factual (0)   Predicted hallucinated (1)
  ------------------------- ----------------------- ----------------------------
  Actual factual (0)                        TN = 76                      FP = 61
  Actual hallucinated (1)                    FN = 8                     TP = 129

### Interpretation

-   Candidate-aware SE outperformed global SE on this length-controlled
    subset: accuracy was about **29.56 percentage points higher**, and
    AUROC was about **0.3022 higher**.
-   Candidate-aware SE had high recall (94.16%) but still produced 61
    false positives.
-   Candidate-aware matched-pair accuracy was 74.09%, with 57 ties among
    137 pairs.
-   Global SE had AUROC below 0.5 and very low specificity at its frozen
    validation threshold. Investigate score direction and validation
    behavior; do not reverse scores or choose a new threshold based on
    these test results without a separately specified, validation-only
    procedure.
-   Matched pairs control answer length within a pair, not all
    confounders; some pairs may contain different original questions.

**These are controlled-subset results only.** They do not establish
performance on the full test split.

## 6. Reproducible commands

Run commands from the repository root with `.venv` activated. In a new
PowerShell terminal:

``` powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path
```

Run regression tests:

``` powershell
python -m pytest -q
```

Expected latest recorded result: `148 passed in 30.76s`.

Inspect the current checkout and working tree:

``` powershell
git branch --show-current
git log -1 --oneline
git status --short
```

Inspect the validation output record counts:

``` powershell
(Get-Content outputs/detectors/se_global/nli/validation_validation_full_v1.jsonl | Measure-Object -Line).Lines
(Get-Content outputs/detectors/se_candidate/nli/validation_validation_full_v1.jsonl | Measure-Object -Line).Lines
```

Inspect the controlled-test output record counts:

``` powershell
(Get-Content outputs/detectors/se_global/nli/test_length_controlled_lc_full_v1.jsonl | Measure-Object -Line).Lines
(Get-Content outputs/detectors/se_candidate/nli/test_length_controlled_lc_full_v1.jsonl | Measure-Object -Line).Lines
```

These are line counts, not complete provenance or ID-alignment proofs.
Validate unique IDs, expected coverage, metadata, finite scores, and
matching IDs before publication.

## 7. Risks and limitations

-   NLI equivalence can merge non-equivalent answers or split
    paraphrases; document model, threshold, and fallback behavior.
-   Generation settings and sampling variability affect cluster
    probabilities and entropy.
-   The controlled set controls answer length within a pair, not every
    confounder; pairs may have different questions.
-   The global SE validation threshold has Youden J = 0, and its
    controlled-test AUROC is below 0.5; investigate validation behavior
    and score semantics.
-   Candidate-aware results still include false positives and many
    matched-pair ties.
-   The full 2,000-example test split has not been evaluated in the
    recorded results.
-   Per-length-stratum metrics and verified efficiency details
    (generation counts, tokens, runtime) are not included here because
    they were not supplied in the recorded evaluation results.
-   Any mismatch in expected coverage or metadata must stop evaluation
    instead of silently dropping samples.
-   Local backups, diagnostics, and generated outputs should be reviewed
    before committing; do not broadly stage with `git add .`.

## 8. Final acceptance checklist

-   [x] Validation covers all 1,001 pairs (2,002 records per detector)
    with recorded metadata checks.
-   [x] Thresholds selected on validation only and frozen with
    provenance.
-   [x] Controlled test covers all 274 samples, 137 matched pairs, and
    balanced label counts.
-   [x] Global and candidate-aware controlled outputs have expected
    coverage and recorded matching IDs.
-   [x] Controlled-test evaluation uses frozen validation thresholds,
    not thresholds tuned on test.
-   [x] Regression tests pass: 148 passed.
-   [ ] Evaluate the full 2,000-example test split before making
    full-test performance claims.
-   [ ] Add length-stratified metrics and verified efficiency
    measurements if available.
-   [ ] Review and commit only intended code, test, and documentation
    changes; exclude backups and unrelated generated artifacts.

## 9. Safe teammate summary

The Person 2 Semantic Entropy workstream has an implemented and tested
detector pipeline, a resumable validation runner, and a resumable
length-controlled test runner. The recorded configuration is
`435a3e4fb7b8` with prompt version `v1`, using Llama 3.2 3B Instruct NF4
on HaluEval QA. The latest reported regression run passed 150 tests.

Full validation completed over 1,001 pairs. The original thresholds were
selected on validation only: global SE `0.1625414866957241` (Youden J
`0.0`) and candidate-aware SE `0.95` (Youden J `0.6313686313686313`).
The length-controlled test evaluation covers 274 examples, not the full
test split. Candidate-aware SE achieved 74.82% accuracy and 0.7472 AUROC;
global SE achieved 45.26% accuracy and 0.4450 AUROC. Interpret these only
as results on the controlled subset.

### Experimental follow-up: short-answer anchoring matcher

A short-answer anchoring fallback was added to
`src/detectors/semantic_entropy/nli_equivalence.py`. Saved generations from
the existing validation checkpoint were rescored without regenerating
answers. The original baseline outputs and threshold files were preserved.

| Candidate-aware validation metric | Original matcher | Anchoring matcher |
|---|---:|---:|
| AUROC | 0.816634 | 0.858234 |
| Accuracy at fixed threshold 0.95 | 81.57% | 85.71% |
| F1 at fixed threshold 0.95 | 0.8352 | 0.8673 |
| Factual candidate match rate | 69.73% | 78.02% |
| Hallucinated candidate match rate | 6.59% | 6.59% |

The changed-match review reported 83 decisions, all label-0 candidates
changing from unmatched to matched. Several inspected examples appeared
plausible, but this limited review does not establish that every new match
is correct. The anchoring rule is heuristic and may fail on negation,
ambiguity, or statements that begin with a candidate but later qualify or
contradict it. The fixed-threshold figures are comparative validation
results, not held-out test performance.

The updated score distribution differs from the original. Using the shared
evaluator and validation data only, the experimental anchoring variant
independently selected a Youden-J threshold of **0.95**, with a validation
Youden-J value of **0.714286**. This threshold is recorded in
outputs/detectors/se_candidate/nli/threshold_validation_anchor_v1.json.
The original baseline threshold and outputs remain preserved. The held-out
test split has not yet been evaluated with the updated matcher; evaluate it
using the recorded validation-selected threshold without tuning on test labels.

The full 2,000-example test split remains unevaluated with the updated
matcher. Preserve the original baseline outputs and thresholds, review
the working tree before staging, and confirm output metadata and Git state
before publication.
*This handoff reflects the currently reported experiment results. Recheck
the live checkout, output metadata, and Git state before publication.*
