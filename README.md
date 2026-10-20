# HalluciDetect — Hallucination Detection in Large Language Models

HalluciDetect is a research project evaluating automated methods for identifying hallucinated answers from large language models (LLMs). The current work compares Semantic Entropy (SE), Semantic Entropy Probes (SEP), lightweight baselines, and a validation-trained fusion model.

> **Current status (20 October 2026):** The dataset and frozen evaluation splits are in place. SE and SEP detector runs are available, and the SE–SEP fusion implementation has been added and evaluated locally. The current fusion code passes the project test suite (**204 passed** in the latest reported run). The fusion work is **not yet documented here as merged or released**; check Git status and branch history before treating it as committed project state.

## Research goals

- Evaluate hallucination detection on the HaluEval QA benchmark.
- Compare Semantic Entropy and Semantic Entropy Probes.
- Measure robustness on a frozen length-controlled test subset.
- Compare detector scores, calibration/thresholded decisions, and operational costs.
- Evaluate whether validation-trained fusion adds measurable value over individual detectors and simple score averaging.
- Establish a foundation for future multilingual evaluation (Hindi and Hinglish).

## Dataset and evaluation protocol

**Dataset:** [HaluEval](https://huggingface.co/datasets/pminervini/HaluEval), QA task.

The dataset contains question–answer examples with binary labels:

- `0`: factual / non-hallucinated
- `1`: hallucinated
- **Score convention:** higher detector scores indicate greater hallucination risk.

| Split / artifact | Samples | Question pairs | Role |
|---|---:|---:|---|
| Train | 15,998 | 7,999 | Detector training where applicable |
| Validation | 2,002 | 1,001 | Model selection, fusion fitting, and threshold selection |
| Test | 2,000 | 1,000 | Primary held-out benchmark |
| Length-controlled test | 274 | 137 matched factual/hallucinated pairs | Secondary robustness evaluation |

The original test split remains the primary benchmark. The length-controlled subset is frozen and matches factual/hallucinated answers by answer character length (`len(answer)`) within **1 character**. It contains 137 factual and 137 hallucinated samples. Do not use the test or length-controlled test data for fitting, hyperparameter selection, or threshold tuning.

Pair-level grouping matters because the two labeled answers associated with a question are not independent observations. Use grouped resampling for uncertainty estimates where implemented. The length-controlled set is a matched subset and should be interpreted as a smaller, secondary robustness check rather than a replacement for the full test set.

## Methods evaluated

### Semantic Entropy (SE)

The candidate-aware SE run uses the `anchor_v1` validation and test artifacts under:

```text
outputs/detectors/se_candidate/nli/
```

Key files include:

- `validation_validation_anchor_v1.jsonl` and its `.meta.json`
- `test_full_anchor_v1.jsonl` and its `.meta.json`
- `test_length_controlled_lc_full_v1.jsonl` and its `.meta.json`
- `threshold_validation_anchor_v1.json`

The threshold artifact records a validation-selected threshold of **0.95** using the Youden criterion. The reported SE test AUROC is approximately **0.8583** on the full test set and **0.7512** on the length-controlled subset in the fusion comparison run.

**Provenance note:** the owner confirmed `anchor_v1` as the final candidate-aware SE variant, and the validation artifacts are present. However, some metadata was reconstructed and naming differs across the artifact path, `variant`, and matcher fields. Before making a definitive reproducibility claim in a paper, audit the run code and provenance chain. Do not regenerate or alter scores to resolve this discrepancy.

### Semantic Entropy Probes (SEP)

The saved SEP run is located at:

```text
results/detector_runs/sep/mean_answer_layer_16/
```

It uses the `mean_answer_layer_16` variant. The associated run metadata records a validation-selected threshold of approximately **0.985456**.

Reported results:

| Evaluation set | AUROC | Accuracy |
|---|---:|---:|
| Full test | 0.9996 | 0.9890 |
| Length-controlled test | 0.9947 | 0.9599 |

The full-test confusion counts in the supplied SEP handoff were TN=999, FP=1, FN=21, TP=979. The length-controlled set is small, so its confidence intervals and matched-pair metrics should be reported alongside point estimates.

### SE–SEP fusion

The fusion experiment combines SE and SEP scores using **L2-regularized logistic regression**. Inputs are standardized, and SEP scores are transformed to logits for the main run. The regularization parameter is selected using five-fold cross-validation on validation data, grouped by question pair. Out-of-fold validation scores are used for validation threshold selection, with the Youden criterion. The fitted threshold is then applied to the held-out test set.

The recorded main run selected **C=10**. Standardized coefficients were approximately:

- SE: `+0.715`
- SEP: `+8.246`
- Intercept: `−0.254`

The corresponding raw-input coefficients depend on the standardization and SEP logit transform; see `results/fusion/se-sep.json` for the saved run details.

Two fusion variants were compared:

- `hybrid/se-sep-lr`: validation-trained logistic-regression fusion
- `hybrid/se-sep-avg`: simple score-average baseline

The fusion run writes detector artifacts under `outputs/detectors/hybrid/` and a comparison/results artifact at `results/fusion/se-sep.json`.

## Latest reported results

These are results from the current local fusion evaluation, not a claim that all changes have been merged or independently reproduced on another machine. AUROC confidence intervals shown are 95% intervals from the reported 5,000-resample analysis.

### Full test set

| Detector | AUROC (95% CI) | Accuracy (95% CI) | Pair accuracy |
|---|---:|---:|---:|
| Length-only baseline | 0.9665 [0.9582, 0.9744] | 0.9345 | — |
| Transformer baseline A | 0.9855 [0.9797, 0.9910] | 0.9750 | — |
| Transformer baseline QAK | 0.9994 [0.9986, 0.9999] | 0.9940 | — |
| SE candidate-aware | 0.8583 [0.8429, 0.8736] | 0.8605 | 0.8525 |
| SEP | 0.9996 [0.9989, 1.0000] | 0.9890 | 1.0000 |
| SE–SEP logistic fusion | 0.9996 [0.9990, 0.9999] | 0.9895 | 1.0000 |
| SE–SEP score average | 0.9821 [0.9760, 0.9880] | 0.9710 | 0.9830 |

### Length-controlled test subset

| Detector | AUROC (95% CI) | Accuracy (95% CI) | Matched-pair accuracy |
|---|---:|---:|---:|
| Length-only baseline | 0.5097 [0.5047, 0.5177] | 0.5219 | 0.5876 |
| Transformer baseline A | 0.8375 [0.7900, 0.8846] | 0.8321 | 0.8978 |
| Transformer baseline QAK | 0.9899 [0.9766, 0.9989] | 0.9745 | 1.0000 |
| SE candidate-aware | 0.7512 [0.7016, 0.7983] | 0.7518 | 0.7445 |
| SEP | 0.9947 [0.9863, 0.9995] | 0.9599 | 1.0000 |
| SE–SEP logistic fusion | 0.9946 [0.9871, 0.9994] | 0.9635 | 1.0000 |
| SE–SEP score average | 0.9769 [0.9533, 0.9941] | 0.9672 | 0.9708 |

### Interpretation

- Logistic-regression fusion substantially outperformed the simple score-average baseline in AUROC on the full test set and on the length-controlled subset in the paired comparison reported for the run.
- Fusion did **not** demonstrate a statistically clear AUROC improvement over SEP on the primary length-controlled benchmark: the paired AUROC-difference confidence interval included zero.
- SEP alone is already a very strong detector in these results. Since SE adds substantial inference cost, the operational case for fusion is not established by AUROC alone.
- The length-controlled subset has only 274 samples. Report its uncertainty and avoid treating a non-significant result as proof that two methods are equivalent.
- The near-perfect detector results warrant a careful leakage, duplicate-content, and split-provenance audit before making broad generalization claims.

These numbers are research results, not a deployment guarantee.

## Runtime and cost considerations

In the reported SE test run, metadata recorded approximately:

- 53,269 seconds total runtime
- 20,000 generations
- 403,825 tokens
- 4,284 MB peak GPU memory

SEP metadata did not provide equivalent runtime/generation/token fields in the reviewed artifact, so missing values should not be interpreted as zero cost. Collect comparable measurements before making a direct cost-efficiency claim.

## Repository map

The repository contains the data pipeline, detector implementations, shared evaluation/schema utilities, experiment scripts, tests, and research documentation. Key current paths include:

```text
data/                         Dataset artifacts and splits
src/common/schema.py          Detector record/metadata schema and validation
src/common/evaluator.py       Shared detector evaluation and uncertainty estimates
src/common/comparison.py      Paired detector comparisons
src/detectors/fusion/         Fusion detector implementation
experiments/run_fusion.py     Fit/evaluate fusion and write detector outputs
experiments/compare_detectors.py
tests/test_fusion.py          Fusion tests
outputs/detectors/            Detector outputs, including local fusion runs
results/detector_runs/sep/    Saved SEP run
results/fusion/se-sep.json    Saved fusion evaluation and comparisons
docs/fusion.md                Fusion method and implementation notes
docs/fusion_analysis_plan.md  Analysis plan and decision criteria
docs/FUSION_RUNBOOK.md        Fusion run instructions
```

Generated outputs under `outputs/` and `results/` may be local or uncommitted. Use `git status` to confirm what is tracked and committed before citing repository state.

## Setup

Python 3.10+ is recommended. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If the environment is already set up, activate it before running the commands below.

## Running fusion evaluation

Run the validation/test fusion experiment using the saved SE and SEP records:

```powershell
python -m experiments.run_fusion --n-boot 5000 --out results/fusion/se-sep.json `
  --input se outputs/detectors/se_candidate/nli/validation_validation_anchor_v1.jsonl outputs/detectors/se_candidate/nli/test_full_anchor_v1.jsonl `
  --input sep results/detector_runs/sep/mean_answer_layer_16/validation.jsonl results/detector_runs/sep/mean_answer_layer_16/test.jsonl `
  --logit sep
```

A dry run that checks coverage and reports the fitted configuration without writing outputs:

```powershell
python -m experiments.run_fusion --dry-run `
  --input se outputs/detectors/se_candidate/nli/validation_validation_anchor_v1.jsonl outputs/detectors/se_candidate/nli/test_full_anchor_v1.jsonl `
  --input sep results/detector_runs/sep/mean_answer_layer_16/validation.jsonl results/detector_runs/sep/mean_answer_layer_16/test.jsonl `
  --logit sep
```

Compare detectors, including the fusion outputs:

```powershell
python -m experiments.compare_detectors length_only/chars transformer/distilroberta-base_a transformer/distilroberta-base_qak sep/mean_answer_layer_16 hybrid/se-sep-lr hybrid/se-sep-avg --n-boot 5000
```

Run tests:

```powershell
python -m pytest tests -q
```

The latest reported test run completed with **204 passed**. Re-run the suite on the current checkout before committing or publishing results.

## Data integrity and reporting safeguards

- Preserve the frozen train/validation/test splits and existing detector score artifacts.
- Do not fit fusion coefficients, select hyperparameters, or tune thresholds on test data.
- Keep the higher-score-means-more-hallucinated convention consistent across detectors.
- Validate detector records and metadata with the shared schema utilities.
- Use pair-grouped analysis for the full test set and matched-group analysis for the length-controlled subset where supported.
- Do not treat candidate-split exploratory results as official benchmark results.
- Do not regenerate SE/SEP runs merely to make metadata names look consistent. Audit provenance first and document any remaining limitation.
- The leakage/near-duplicate audit and cleanup tasks described in the project workflow should be completed and reviewed before final paper claims.
- Report confidence intervals, paired comparisons, and the small size of the length-controlled set. Distinguish “inconclusive” from “no effect.”

## Current roadmap

### Completed / available

- [x] HaluEval preprocessing and official split infrastructure
- [x] Frozen length-controlled test subset
- [x] Saved candidate-aware SE validation/test artifacts
- [x] Saved SEP `mean_answer_layer_16` validation/test run
- [x] SE–SEP fusion implementation and local evaluation
- [x] Fusion tests passing in the latest reported run (204 tests)

### In progress / required before final publication

- [ ] Audit SE artifact provenance and resolve/document metadata naming differences
- [ ] Complete leakage and near-duplicate audits and preserve exclusion lists
- [ ] Finish pending repository cleanup and review changes
- [ ] Re-run tests and integrity checks on the final merged checkout
- [ ] Review statistical reporting and confirm paper tables from saved artifacts
- [ ] Decide whether fusion offers enough validated benefit to justify additional SE inference cost

### Future work

- [ ] Extend evaluation to Hindi and Hinglish
- [ ] Assess robustness on additional datasets and model families
- [ ] Improve cost reporting and calibration analysis
- [ ] Evaluate deployment constraints after reproducibility and leakage audits

## Documentation

Useful project documents include:

- `docs/fusion.md` — fusion method and implementation
- `docs/fusion_analysis_plan.md` — analysis plan and decision criteria
- `docs/FUSION_RUNBOOK.md` — fusion run instructions
- `results/fusion/se-sep.json` — saved fusion metrics and comparisons
- `results/length_controlled_test_report.md` — length-controlled subset report, if present in the checkout
- `PREPROCESSING_EDA_SUMMARY.md` — preprocessing and exploratory analysis summary, if present in the checkout

## Citation

Draft project citation:

```text
HalluciDetect: Automated Hallucination Detection in Large Language Models
Research team, 2026
```

Replace this placeholder with the final author list, venue, and publication details when available.

---

**Last updated:** 2026-10-20  
**Status:** SE, SEP, and SE–SEP fusion have been evaluated locally; provenance/leakage review and final repository integration remain important open tasks.
