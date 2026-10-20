# SE–SEP Fusion — Research Handoff and Technical Report

**Project:** HalluciDetect-Project  
**Workstream:** SE–SEP fusion  
**Status date:** 20 October 2026  
**Status:** Implemented and evaluated locally; final provenance/release checks remain open  
**Primary result artifact:** `results/fusion/se-sep.json`  
**Fusion outputs:** `outputs/detectors/hybrid/se-sep-lr/` and `outputs/detectors/hybrid/se-sep-avg/`

> **Scope and reporting note.** This document summarizes the recorded SE–SEP fusion implementation and evaluation. It distinguishes measured results from interpretation and open provenance questions. The values below are the reported local run outputs shared during the handoff; re-run the commands in the project checkout to verify against the exact current files before publication.

## 1. Executive summary

This work combines two complementary hallucination-risk signals on the HaluEval QA benchmark:

- **Candidate-aware Semantic Entropy (SE):** a sampling-based, candidate-aware score using semantic-equivalence clustering and an NLI short-answer anchoring matcher.
- **Semantic Entropy Probe (SEP):** a supervised probe over frozen Llama 3.2 3B Instruct hidden-state representations, using mean-answer pooling and layer 16.

Two fusion baselines were evaluated on aligned SE and SEP validation/test records:

1. **Learned fusion:** L2-regularized logistic regression on standardized SE and SEP scores.
2. **Equal-weight average:** an unfitted baseline that averages the standardized input scores, with its decision threshold selected on validation data.

The reported logistic-regression hybrid reached **0.9996 test AUROC** and **0.9946 AUROC on the 274-sample length-controlled subset**. On the full test set, this was very similar to SEP alone (AUROC 0.9996); on the length-controlled subset, the hybrid and SEP were also statistically indistinguishable by the reported paired AUROC confidence interval. The learned hybrid did outperform the equal-weight average on AUROC in both reported comparisons.

The main conclusion is therefore cautious: **the learned fusion improves on the simple average, but the current results do not establish a clear AUROC improvement over SEP alone.** Because SE adds substantial generation cost, SEP remains a strong operational baseline unless further evidence demonstrates that fusion's incremental benefit justifies the cost.

The fusion pipeline passed the reported project test suite (**204 passed**) and the four generated hybrid output files passed schema validation. Before final release or strong research claims, the SE variant provenance mapping and repository/release checks should be completed.

## 2. Research question

Does combining a candidate-aware semantic-uncertainty signal with a supervised hidden-state probe improve hallucination detection compared with either component alone, while retaining performance when answer length is controlled?

The experiment compares:

- SE alone;
- SEP alone;
- learned SE+SEP logistic-regression fusion;
- an equal-weight standardized-score average;
- reference baselines from the broader detector comparison, including a length-only baseline and Transformer detectors.

The primary controlled comparison is the length-controlled test subset. The full held-out test is reported as a complementary evaluation because the original dataset exhibits a strong answer-length association with labels.

## 3. Dataset, labels, and split contract

The project data contract records HaluEval QA with 10,000 underlying question/answer pairs expanded to 20,000 candidate examples.

| Split | Records | Pair count |
|---|---:|---:|
| Train | 15,998 | 7,999 |
| Validation | 2,002 | 1,001 |
| Test | 2,000 | 1,000 |

**Label convention:** `1 = hallucinated`, `0 = factual`. Higher detector scores are intended to indicate greater hallucination risk.

The length-controlled test subset contains **274 samples in 137 matched groups**, with exactly one factual and one hallucinated sample per group. Answer lengths within each matched group differ by at most one character. The controlled subset is useful for testing sensitivity to answer length, but it does not control every possible confound. Some matched groups may contain different original questions; interpret match-level metrics accordingly.

The reported fusion run used full validation/test coverage: 2,002/2,002 validation records and 2,000/2,000 test records. The fusion artifact reports `partial: false`.

## 4. Component methods

### 4.1 Candidate-aware Semantic Entropy (SE)

The SE workstream samples multiple answers, groups semantically equivalent answers, and uses semantic-cluster probabilities to characterize uncertainty. Candidate-aware scoring measures the representation of a supplied candidate among generated answers; in the implementation described in the SE handoff, the candidate-aware score is `1 - candidate-cluster probability`.

Semantic equivalence uses an NLI-based short-answer anchoring matcher. The recorded model identifier is `facebook/bart-large-mnli`, with revision `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce` recorded in the test metadata. The matcher is heuristic: it can merge non-equivalent answers or split paraphrases, and anchoring can fail for negation, ambiguity, or later qualifications.

The SE handoff reports the selected candidate-aware `anchor_v1` variant at 0.8583 AUROC on the full test and 0.7472 AUROC on the length-controlled subset in its earlier evaluation. The fusion run's SE input is the later anchor-v1 candidate-aware run with a fixed threshold of 0.95.

### 4.2 Semantic Entropy Probe (SEP)

SEP is a supervised hidden-state probe, not the sampling-based semantic-entropy algorithm itself. It uses a frozen Llama 3.2 3B Instruct model configured for 4-bit NF4, mean-answer pooling, and selected layer 16. A lightweight supervised classifier is trained on the extracted representation. The selected layer and threshold are based on validation data.

The recorded SEP threshold is `0.9854559515186279`. Its recorded full-test AUROC is approximately `0.9996`; its length-controlled AUROC is approximately `0.9947`.

The SEP handoff notes that its reported probe-only runtime is based on cached feature vectors and excludes model loading, tokenization, Llama forward passes, and hidden-state extraction. It must not be represented as end-to-end inference latency.

## 5. Fusion methods

### 5.1 Input alignment and score transforms

The run aligned SE and SEP records by sample ID, with complete coverage on validation and test. The saved input manifest records these sources:

| Input | Validation file | Test file | Transform |
|---|---|---|---|
| SE | `outputs/detectors/se_candidate/nli/validation_validation_anchor_v1.jsonl` | `outputs/detectors/se_candidate/nli/test_full_anchor_v1.jsonl` | none |
| SEP | `results/detector_runs/sep/mean_answer_layer_16/validation.jsonl` | `results/detector_runs/sep/mean_answer_layer_16/test.jsonl` | logit |

The learned model standardizes both input scores using validation-derived means and standard deviations. The reported standardization parameters are:

| Feature | Mean | Standard deviation |
|---|---:|---:|
| SE | 0.796753 | 0.316519 |
| SEP (after logit transform) | 0.309647 | 17.328890 |

No answer-length feature was included (`include_length: false`).

### 5.2 Learned logistic-regression fusion

The learned fusion is an L2-regularized logistic regression using standardized SE and SEP scores. The selected regularization parameter was **C = 10.0**, chosen from the reported cross-validation log-loss comparison. Five-fold cross-validation used seed 0. The fusion artifact records that fitting uses validation labels only and that validation scores used for the fusion fit are out-of-fold.

Reported cross-validation log-loss:

| C | CV log-loss |
|---:|---:|
| 0.01 | 0.166325 |
| 0.1 | 0.053165 |
| 1.0 | 0.026308 |
| 10.0 | 0.023310 |
| 100.0 | 0.023693 |

The learned standardized coefficients were:

| Parameter | Value |
|---|---:|
| SE weight | +0.714500 |
| SEP weight | +8.246454 |
| Standardized intercept | −0.254326 |

The equivalent reported raw-scale coefficients were:

| Parameter | Value |
|---|---:|
| Intercept (`w0`) | −2.200248 |
| SE weight | +2.257370 |
| SEP weight | +0.475879 |

Both learned weights were positive, consistent with the intended convention that higher input scores indicate greater hallucination risk. The large difference between standardized and raw coefficients reflects the different input scales; coefficients should be interpreted with their parameterization stated.

### 5.3 Threshold selection

The learned hybrid threshold was selected on validation data using Youden's J criterion:

- **Threshold:** `1.7544161276613728`
- **Validation Youden J:** `0.986014`
- **Selection source:** validation

The same threshold is recorded in both hybrid validation and test metadata, with `threshold_source: validation`. The test set was not used to select the fusion threshold.

The equal-weight average is an unfitted baseline over standardized inputs. Its reported validation-selected threshold is `0.18390119337141347`, also marked as validation-derived.

### 5.4 Why include an equal-weight average?

The average provides a simple reference point for whether learned weights improve over a transparent, unfitted combination. It is not a replacement for the component ablations. Results should be compared on the same IDs and with threshold selection confined to validation.

## 6. Main results

### 6.1 Full held-out test (2,000 samples)

Reported point estimates and 95% bootstrap confidence intervals:

| Detector | AUROC (95% CI) | Accuracy (95% CI) | Pair accuracy (95% CI) |
|---|---:|---:|---:|
| SE | 0.8583 [0.8429, 0.8736] | 0.8605 [0.8460, 0.8755] | 0.8525 [0.8355, 0.8695] |
| SEP | 0.9996 [0.9989, 1.0000] | 0.9890 [0.9845, 0.9935] | 1.0000 [1.0000, 1.0000] |
| SE+SEP logistic regression | **0.9996 [0.9990, 0.9999]** | **0.9895 [0.9850, 0.9935]** | 1.0000 [1.0000, 1.0000] |
| Standardized equal-weight average | 0.9821 [0.9760, 0.9880] | 0.9710 [0.9635, 0.9780] | 0.9830 [0.9750, 0.9910] |

The logistic-regression hybrid's reported AUROC improvement over the equal-weight average was **+0.0175** (95% paired CI [0.0116, 0.0236]). The interval excludes zero.

Compared with SEP, the reported test AUROC difference for hybrid minus SEP was approximately **+0.0000** (95% paired CI [−0.0002, 0.0002]); the interval includes zero. Thus, the results do not show a statistically clear AUROC improvement over SEP on the full test set.

### 6.2 Length-controlled test (274 samples; 137 matched groups)

| Detector | AUROC (95% CI) | Accuracy (95% CI) | Match accuracy (95% CI) |
|---|---:|---:|---:|
| SE | 0.7512 [0.7016, 0.7983] | 0.7518 [0.7044, 0.7993] | 0.7445 [0.6934, 0.7920] |
| SEP | 0.9947 [0.9863, 0.9995] | 0.9599 [0.9343, 0.9818] | 1.0000 [1.0000, 1.0000] |
| SE+SEP logistic regression | **0.9946 [0.9871, 0.9994]** | 0.9635 [0.9416, 0.9818] | 1.0000 [1.0000, 1.0000] |
| Standardized equal-weight average | 0.9769 [0.9533, 0.9941] | **0.9672 [0.9453, 0.9854]** | 0.9708 [0.9416, 0.9927] |

On the controlled subset, the hybrid's paired AUROC difference over the equal-weight average was **+0.0177** (95% paired CI [0.0022, 0.0394]), excluding zero. The reported hybrid-minus-SEP difference was approximately **−0.0001** (95% paired CI [−0.0023, 0.0018]), which includes zero.

The learned hybrid has slightly higher point-estimate AUROC than SEP on the full test, but a slightly lower point estimate on the controlled subset. Neither comparison provides clear evidence of AUROC improvement over SEP. The average baseline has marginally higher controlled-set accuracy than the learned hybrid, despite lower AUROC; this illustrates why thresholded accuracy and ranking metrics should be reported separately.

### 6.3 Paired error analysis

Reported full-test error-overlap results:

| Comparison | Only first model correct | Only second model correct | McNemar exact p-value |
|---|---:|---:|---:|
| Hybrid vs. equal-weight average | Hybrid: 47 | Average: 10 | 7.51 × 10⁻⁷ |
| Hybrid vs. SE | Hybrid: 270 | SE: 12 | 1.12 × 10⁻⁶⁴ |
| Hybrid vs. SEP | Hybrid: 4 | SEP: 3 | 1.00 |

For hybrid versus SEP, only seven samples were discordant (four hybrid-only correct and three SEP-only correct), consistent with the near-identical overall test performance. The McNemar result does not establish AUROC superiority; it evaluates paired binary classification errors at the selected thresholds.

The reported oracle test accuracy was 0.9970, with six samples wrong for all evaluated detectors. This is a descriptive upper-bound diagnostic for this set of predictions, not a deployable detector.

### 6.4 Broader comparison context

A separate detector comparison reported these additional AUROC values:

| Detector | Full test AUROC | Length-controlled AUROC |
|---|---:|---:|
| Length-only / character baseline | 0.9665 | 0.5097 |
| Transformer A (`distilroberta-base_a`) | 0.9855 | 0.8375 |
| Transformer QAK (`distilroberta-base_qak`) | 0.9994 | 0.9899 |
| SEP (`mean_answer_layer_16`) | 0.9996 | 0.9947 |
| SE+SEP logistic regression | 0.9996 | 0.9946 |
| SE+SEP equal-weight average | 0.9821 | 0.9769 |

The length-only baseline's AUROC drops close to chance on the controlled subset, indicating that answer length is a strong confound in the full test evaluation. SEP and the hybrid retain high ranking performance on the controlled subset. This does not rule out other shortcuts, duplicates, label artifacts, or leakage pathways.

## 7. Efficiency and operational trade-off

The recorded test cost metadata for the SE component reports:

- Total recorded runtime: **53,269.02 seconds** (approximately 14.80 hours).
- Recorded generations: **20,000**.
- Generated tokens: **403,825**.
- Maximum recorded peak GPU memory: **4,283.76 MB**.

The SEP metadata reports approximately **0.0549 seconds** for scoring 2,000 cached feature vectors, but this is probe-only scoring time. It excludes Llama model loading, tokenization, forward passes, and hidden-state extraction; it is not comparable to end-to-end SE generation latency.

The hybrid requires the SE and SEP scores as inputs, so its practical inference cost includes the SE generation/sampling work unless an application already has those signals available. The current results do not establish that the very small performance difference versus SEP alone justifies this additional cost. A fair deployment comparison requires end-to-end latency and compute accounting under the same hardware and batching conditions.

**Operational implication:** SEP is the stronger default baseline when latency/cost matters. Hybrid fusion is still useful as a research comparison and may be justified in settings where marginal classification behavior, robustness, or other validated benefits matter more than the added SE cost.

## 8. Statistical and methodological notes

- The report records 5,000 bootstrap replicates for the fusion evaluation.
- The evaluator uses pair-cluster bootstrap intervals for full-test metrics and matched-group bootstrap intervals for the length-controlled subset.
- Paired AUROC intervals compare models on the same samples.
- Thresholds are selected on validation data only and held fixed for test evaluation.
- Validation fusion fitting is reported to use out-of-fold validation scores, reducing the risk of evaluating the fitted fusion on in-sample predictions.
- The length-controlled set is small (274 samples); its confidence intervals are consequently wider and its point estimates less stable.
- An interval that includes zero means the evaluation does not demonstrate a clear difference under this analysis; it does **not** prove the models are equivalent or that the hybrid has no possible benefit.

## 9. Provenance, reproducibility, and current caveats

### 9.1 SE variant naming requires clarification

The fusion input paths refer to files named `validation_validation_anchor_v1.jsonl` and `test_full_anchor_v1.jsonl`. The SE threshold artifact identifies the matcher as `nli_short_answer_anchor_experimental`, while the SE output metadata uses `variant: nli` and stores the matcher name under `matching_rule` or `experimental_matcher`. The SE metadata also records `metadata_reconstructed: true` for validation.

The SE owner has confirmed `anchor_v1` as the selected candidate-aware variant. However, reconstructed metadata and a matching configuration hash do not, by themselves, independently prove which exact implementation generated historical scores. Before publication, inspect the committed source and run records to verify the mapping between the approved `anchor_v1` variant and `nli_short_answer_anchor_experimental`. If verified, document that mapping consistently. If not, state the provenance limitation explicitly.

Do not regenerate SE/SEP runs merely to normalize naming, and do not alter saved scores, IDs, thresholds, or schemas as part of a documentation-only clarification.

### 9.2 Git/release state

At the last reported check, the following fusion files were modified or untracked:

- `docs/fusion_analysis_plan.md`
- `experiments/compare_detectors.py`
- `src/common/comparison.py`
- `src/common/evaluator.py`
- `docs/FUSION_RUNBOOK.md`
- `docs/fusion.md`
- `experiments/run_fusion.py`
- `results/fusion/`
- `src/detectors/fusion/`
- `tests/test_fusion.py`

The exact working tree may have changed since that check. Review `git status --short`, `git diff --check`, and the relevant diffs before staging or committing. Do not use broad staging commands without reviewing generated artifacts.

### 9.3 Remaining scientific limitations

- The SE short-answer anchoring matcher is heuristic and may make incorrect semantic-equivalence decisions.
- Strong full-test metrics do not rule out duplicate-content leakage, dataset shortcuts, or all other forms of contamination.
- The length-controlled set controls answer character length within matched groups, not every possible confound.
- The controlled set is small and should be reported separately, not as a replacement for the official full test.
- The test AUROC of SEP is already extremely high; ceiling effects make small differences difficult to estimate.
- The equal-weight average is a baseline, not a separately trained classifier.
- The reported cost metadata has different levels of completeness for SE and SEP; missing SEP generation-cost values must not be interpreted as proof that SEP has zero end-to-end cost.
- The fusion results are from one dataset/configuration and do not establish generalization to other models, datasets, prompts, or real-world deployment conditions.

## 10. Recommended paper wording

A cautious summary suitable as a starting point for the research paper:

> We evaluated a learned fusion of candidate-aware Semantic Entropy (SE) and a supervised Semantic Entropy Probe (SEP) on HaluEval QA. The fusion model was an L2-regularized logistic regression trained on standardized SE and SEP scores, with regularization selected by five-fold cross-validation and the decision threshold selected on validation data. On the 2,000-example held-out test set, the learned fusion achieved AUROC 0.9996 (95% bootstrap CI 0.9990–0.9999), compared with 0.9996 (0.9989–1.0000) for SEP and 0.9821 (0.9760–0.9880) for a standardized equal-weight average. On a 274-example length-controlled subset, fusion achieved AUROC 0.9946 (0.9871–0.9994), compared with 0.9947 (0.9863–0.9995) for SEP and 0.9769 (0.9533–0.9941) for the average. Paired AUROC intervals supported improvement over the equal-weight average but did not establish a clear AUROC improvement over SEP alone. Because SE incurs substantial generation cost, the results do not yet justify claiming that fusion is operationally superior to SEP. These findings are specific to the evaluated dataset, model configuration, and semantic-equivalence matcher.

Before using this wording in a submitted paper, verify the SE provenance mapping and rerun the final comparison from the exact release candidate. Add details required by the paper venue, including data licensing, complete model/prompt/sampling settings, calibration analysis if relevant, and a documented leakage/duplicate-content audit.

## 11. Reproducibility commands

Run from the repository root in PowerShell, with the project virtual environment activated.

### Run the reported SE+SEP fusion evaluation

```powershell
python -m experiments.run_fusion --n-boot 5000 --out results/fusion/se-sep.json `
  --input se outputs/detectors/se_candidate/nli/validation_validation_anchor_v1.jsonl outputs/detectors/se_candidate/nli/test_full_anchor_v1.jsonl `
  --input sep results/detector_runs/sep/mean_answer_layer_16/validation.jsonl results/detector_runs/sep/mean_answer_layer_16/test.jsonl `
  --logit sep
```

This command writes/updates the fusion outputs and result artifact. Do not run it if the goal is strictly read-only inspection or if overwriting the existing artifacts is not intended.

### Run the broader detector comparison

```powershell
python -m experiments.compare_detectors length_only/chars transformer/distilroberta-base_a transformer/distilroberta-base_qak sep/mean_answer_layer_16 hybrid/se-sep-lr hybrid/se-sep-avg --n-boot 5000
```

### Run tests

```powershell
python -m pytest tests -q
```

The reported test result after the fusion implementation was **204 passed in 283.27 seconds**. Re-run in the release candidate to verify current state.

### Check repository state

```powershell
git status --short
git diff --check
git diff -- docs/fusion_analysis_plan.md experiments/compare_detectors.py src/common/comparison.py src/common/evaluator.py
```

### Verify saved output schema

```powershell
@'
from pathlib import Path
from src.common.schema import read_detector_output, validate_run

for variant in ("se-sep-lr", "se-sep-avg"):
    root = Path("outputs/detectors/hybrid") / variant
    for split in ("validation", "test"):
        records, meta = read_detector_output(root / f"{split}.jsonl")
        validate_run(records, meta)
        print(variant, split, len(records), meta.threshold_source, meta.threshold)
'@ | python -
```

## 12. Current acceptance checklist

- [x] Fusion implementation and CLI run completed.
- [x] Learned logistic-regression fusion and equal-weight average evaluated.
- [x] Validation/test coverage reported as complete.
- [x] Threshold selection recorded as validation-only.
- [x] Full-test and length-controlled metrics reported separately.
- [x] Paired AUROC comparisons and error-overlap analysis recorded.
- [x] Reported test suite passed (204 tests).
- [x] Four hybrid JSONL/metadata pairs passed schema validation.
- [ ] Verify the `anchor_v1` to `nli_short_answer_anchor_experimental` provenance mapping from source/run records.
- [ ] Review current Git diffs and decide which fusion artifacts should be version-controlled.
- [ ] Re-run final tests and comparisons from the exact release candidate.
- [ ] Complete/document duplicate-content and leakage audit before strong generalization claims.
- [ ] Measure comparable end-to-end cost if making deployment-efficiency claims.

## 13. Final handoff statement

The SE–SEP fusion pipeline is implemented and evaluated. The learned logistic-regression hybrid achieved AUROC 0.9996 on the 2,000-sample test set and 0.9946 on the 274-sample length-controlled subset. It improved over the equal-weight standardized average in paired AUROC comparisons, but it did not demonstrate a clear AUROC improvement over SEP alone. Given SE's substantial generation cost and SEP's already strong controlled-set performance, SEP remains a strong operational baseline. The fusion system is suitable for continued research and ablation analysis; final release and paper claims should wait for SE provenance clarification, final repository checks, and an explicit leakage/duplicate-content audit.
