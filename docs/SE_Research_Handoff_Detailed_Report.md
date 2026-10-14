# Semantic Entropy (SE) --- Research Handoff & Technical Report

**Project:** HalluciDetect-Project\
**Workstream:** Person 2 --- Semantic Entropy (SE)\
**Branch:** `person2/semantic-entropy`\
**Status date:** 14 October 2026\
**Latest known commit:** `ff4ffa0` --- Add resumable controlled-test
semantic entropy runner

> **Purpose:** Document the SE implementation, data contract, experiment
> status, evaluation protocol, reproducible commands, risks, and
> remaining work.
>
> **Important distinction:** This is for **Semantic Entropy (SE)**, not
> the separate **Semantic Entropy Probes (SEP)** workstream. SEP uses
> hidden-state features and a supervised probe; this SE workstream uses
> sampled generations, semantic clusters, entropy, and candidate-aware
> scoring.
>
> Full-scale performance results are pending. The two-sample controlled
> smoke run is an integration check only, not final performance
> evidence.

## Executive summary

The Person 2 SE workstream includes shared
generation/data/schema/evaluator interfaces, semantic clustering,
Shannon entropy over cluster probabilities, candidate-aware scoring,
NLI-based semantic equivalence, a resumable pair-aware validation
runner, and a resumable length-controlled test runner.

The reported full test suite passes **148 tests**. Commit `ff4ffa0` was
pushed to `origin/person2/semantic-entropy`.

Full-scale experiments remain incomplete. The validation run was
previously interrupted after 26 pairs out of 1,001. The
length-controlled test set contains 274 samples, of which two were
processed in the `lc_v1` smoke run.

## Current status

  ----------------------------------------------------------------------------
  Area                    Status                  Evidence / note
  ----------------------- ----------------------- ----------------------------
  Implementation          Implemented             Shared pipeline, entropy,
                                                  clustering, candidate-aware
                                                  scoring, NLI equivalence,
                                                  and runners exist.

  Automated tests         148 passed              Last reported command:
                                                  `python -m pytest -q`.

  Git                     Pushed                  Commit `ff4ffa0`; branch
                                                  `person2/semantic-entropy`
                                                  aligned with origin at last
                                                  check.

  Validation              Pending                 Previously checkpointed 26
                                                  pairs; target 1,001
                                                  validation pairs.

  Controlled test         Pending                 274 samples total; `lc_v1`
                                                  smoke outputs contain 2
                                                  samples.

  Final metrics           Pending                 Wait for full coverage and
                                                  validation-only threshold
                                                  selection.
  ----------------------------------------------------------------------------

## 1. Workstream definition and terminology

Semantic entropy measures uncertainty over semantic answer clusters
rather than raw token-level variation. The pipeline samples multiple
answers for an input, groups semantically equivalent answers, and
computes entropy over cluster probabilities. Candidate-aware scoring
separately measures how represented a supplied candidate is among
generated answers.

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

Label convention: **1 = hallucinated**, **0 = factual**. Higher detector
score should mean more likely hallucinated.

The length-controlled test set has **274 examples**, arranged as **137
matched pairs**, with 137 examples per label. Answer lengths within each
matched pair differ by at most one character. Some matched pairs may use
different original questions, so interpret matched-pair accuracy
carefully.

Preserve sample IDs, pair IDs, match IDs, split labels, config hash,
prompt version, label semantics, and output metadata. Do not select or
tune thresholds using test labels.

## 3. Implementation file map

  ------------------------------------------------------------------------------------------------------------------
  Component                                                    Role                    Current understanding
  ------------------------------------------------------------ ----------------------- -----------------------------
  `src/common/generation.py`                                   Shared generation       Generation outputs and
                                                                                       statistics; sample-level
                                                                                       seeding supports
                                                                                       reproducibility.

  `src/common/data.py`                                         Data loading            Split loading, pair grouping,
                                                                                       controlled-test loading, and
                                                                                       match grouping.

  `src/common/schema.py`                                       Output schema           Detector records,
                                                                                       JSONL/metadata read/write,
                                                                                       efficiency and coverage
                                                                                       helpers.

  `src/common/evaluator.py`                                    Evaluation              Score alignment, AUROC,
                                                                                       classification metrics,
                                                                                       threshold selection,
                                                                                       pair/match metrics, length
                                                                                       strata, run evaluation.

  `src/detectors/semantic_entropy/entropy.py`                  Entropy                 Shannon entropy over cluster
                                                                                       probabilities in nats and
                                                                                       normalized entropy.

  `src/detectors/semantic_entropy/clustering.py`               Clustering              Whitespace/case exact
                                                                                       baseline plus optional
                                                                                       equivalence predicate.

  `src/detectors/semantic_entropy/candidate_aware.py`          Candidate score         Candidate-cluster probability
                                                                                       and score `1 - probability`;
                                                                                       not a factuality guarantee.

  `src/detectors/semantic_entropy/nli_equivalence.py`          NLI equivalence         Reported model
                                                                                       `facebook/bart-large-mnli`,
                                                                                       bidirectional entailment
                                                                                       threshold 0.85, short-answer
                                                                                       fallback.

  `runner_pair_checkpoint.py`                                  Validation runner       Pair-aware resumable
                                                                                       validation; last known
                                                                                       checkpoint had 26 completed
                                                                                       pairs.

  `src/detectors/semantic_entropy/runner_controlled_test.py`   Controlled runner       Resumable per-sample
                                                                                       checkpoints, unique tags,
                                                                                       metadata checks, overwrite
                                                                                       protection.
  ------------------------------------------------------------------------------------------------------------------

## 4. Controlled-test smoke run

The `lc_v1` smoke run wrote global-SE and candidate-aware outputs with
two records each. Recorded IDs: `S009005_0` and `S009305_1`. Both
outputs used `split=test`, config hash `435a3e4fb7b8`, prompt version
`v1`, finite scores, and no selected threshold. The two checkpoints
contained the same IDs. The evaluator smoke check reported one matched
pair and match accuracy 1.0 for each detector.

> **Do not interpret this as performance:** one matched pair is only a
> pipeline integration test. Existing `lc_v1` final outputs exist, and
> the runner protects against overwriting them. Inspect CLI help and use
> a fresh tag for a full run if required.

## 5. Remaining work in order

### A. Resume full validation

Inspect
`outputs/detectors/_checkpoints/se_pair_validation_full_v1.jsonl`,
validate unique pair IDs and metadata, and resume the pair-aware runner
until all 1,001 validation pairs are covered. Do not restart from
scratch if checkpoint validation allows resume.

### B. Verify validation output

Check expected pair/sample coverage, duplicates, missing IDs, label
distribution, finite scores, config hash, prompt version, split, both
detector variants, and generation statistics.

### C. Freeze threshold on validation

Use validation records only for threshold selection. Save threshold and
provenance in metadata. Test labels must not influence threshold choice.

### D. Complete controlled test

Inspect `lc_v1` checkpoint/output state. Verify global and candidate
checkpoint IDs match. Use a fresh tag if needed and process all 274
samples, expecting 137 matched pairs and 137 examples per label.

### E. Verify final outputs

Enforce exact sample coverage, unique IDs, `split=test`, config hash,
prompt version, finite scores, label semantics, and metadata
consistency. Stop if coverage differs; never silently discard mismatched
IDs.

### F. Run final evaluation

Use the frozen validation threshold. Report AUROC, classification
metrics, matched-pair accuracy, ties, length-stratified metrics,
efficiency, and counts for both detectors.

### G. Package and hand off

Document run settings, model/generation config, seeds,
software/hardware, limitations, and output locations. Review local Git
changes and stage only intended files; do not use `git add .`.

## 6. Reproducible commands

Run from the repository root with `.venv` activated. The committed
`pytest.ini` configures `testpaths = tests` and
`addopts = --import-mode=importlib`.

### 6.1 Verify checkout and tests

``` powershell
git branch --show-current
git log -1 --oneline
git status -sb
python -m pytest -q
```

Expected branch: `person2/semantic-entropy`; latest known commit
`ff4ffa0`. The last reported normal test run passed 148 tests. At the
last status check, `tests/tests/test_se_nli.py` was modified and
backups, diagnostics, `outputs/`, and `data/processed_backup/` were
untracked. Preserve them while reviewing; do not broadly stage.

### 6.2 Inspect validation checkpoint

``` powershell
Get-Content outputs\detectors\_checkpoints\se_pair_validation_full_v1.jsonl | Measure-Object -Line
```

This is a quick line count only, not a coverage proof. Before resuming,
inspect runner CLI/help and checkpoint validation for unique IDs, config
hash `435a3e4fb7b8`, prompt `v1`, and split `validation`.

### 6.3 Inspect controlled-runner CLI

``` powershell
python -m src.detectors.semantic_entropy.runner_controlled_test --help
```

Use actual CLI help as the source of truth for full-run flags and tag
syntax. The known runner has a smoke limit of 2 and an `--all` option
for the full 274 samples, but verify flags against the current checkout.

### 6.4 Existing smoke-output verifier

``` powershell
python verify_controlled_evaluator.py
```

This helper currently checks `lc_v1` smoke outputs. After a full run,
point the verifier/evaluator at the full-run tag as appropriate; never
treat the two-record files as full coverage.

## 7. Evaluation rules

-   **Global SE:** Entropy over semantic cluster probabilities; low
    entropy does not prove factual correctness.
-   **Candidate-aware score:** Candidate representation in sampled
    clusters; interpret according to the current implementation and
    score direction.
-   **Score direction:** Higher detector score means more likely
    hallucinated under the shared contract.
-   **Threshold provenance:** Select on validation only, freeze it, and
    record provenance.
-   **Matched pairs:** Report number of matches and ties; explain that
    matched questions may differ.
-   **Length strata:** Report per-stratum sample counts as well as
    metric values.
-   **Efficiency:** Report actual recorded generation
    counts/tokens/runtime; do not invent missing values.
-   **Reproducibility:** Record config hash `435a3e4fb7b8`, prompt
    version `v1`, model checkpoint, sampling settings, seeds, software
    versions, hardware, and run tag.

## 8. Risks and limitations

-   NLI equivalence can merge non-equivalent answers or split
    paraphrases; document model, threshold, and fallback behavior.
-   Generation settings and sampling variability affect cluster
    probabilities and entropy.
-   The controlled set controls answer length within a pair, not every
    confounder; pairs may have different questions.
-   The two-sample smoke run is integration evidence only.
-   Any mismatch in expected coverage or metadata must stop evaluation
    instead of dropping samples silently.
-   The last known working tree had an uncommitted nested NLI test and
    untracked local artifacts; review carefully and avoid destructive
    cleanup.

## 9. Final acceptance checklist

-   [ ] Validation covers all 1,001 pairs and passes metadata/coverage
    checks.
-   [ ] Threshold selected on validation and frozen with provenance.
-   [ ] Controlled test covers all 274 samples, 137 matches, and
    expected label counts.
-   [ ] Global and candidate-aware outputs have matching IDs and exact
    coverage.
-   [ ] Final evaluation uses frozen validation threshold, not a
    test-tuned threshold.
-   [ ] Report includes metrics, counts, match/tie results, length
    strata, efficiency, and limitations.
-   [ ] Final claims rely on full runs, not the two-sample smoke run.
-   [ ] Required code/tests/docs are committed and pushed; backups and
    generated outputs are excluded unless intentionally needed.

## 10. Safe teammate summary

The Semantic Entropy workstream has an implemented and tested detector
pipeline, a resumable validation runner, and a resumable
length-controlled test runner. Latest known commit: `ff4ffa0` on
`person2/semantic-entropy`; 148 tests pass. Full-scale validation and
controlled-test experiments remain pending. Final performance claims
require full coverage, validation-only threshold selection, and verified
evaluator output. The two-sample controlled smoke run demonstrates
integration only.

*This handoff reflects the recorded project status and contract. Verify
commands and output metadata against the live checkout before
publication.*