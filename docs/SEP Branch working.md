HALLUCIDETECT
# Person 3 — SEP Technical Handoff
# & SE–SEP Fusion Plan
Implementation summary, results, reproducibility notes, limitations, and next steps

| Project field | Current recorded value |
| --- | --- |
| Workstream | Person 3 — Semantic Entropy Probes (SEP) |
| Repository branch | sep |
| Latest recorded commit | 6ecbea8 — Add SEP evaluator predictions and integration report |
| Base model configuration | meta-llama/Llama-3.2-3B-Instruct; 4-bit NF4 (as recorded in project notes) |
| Probe configuration | mean_answer pooling; selected layer 16 |
| Document purpose | Enable team members to understand the current SEP implementation and continue with full Semantic Entropy (SE) and SE–SEP fusion. |

Status: Core SEP pipeline and evaluator handoff are implemented. Fusion and full SE comparison remain follow-up work.

## 1. Executive summary

This document records the work completed for Person 3’s SEP workstream in HalluciDetect. The implementation extracts hidden-state features from a frozen Llama model, trains a supervised probe, selects a layer and threshold on validation data, produces standardized prediction records, evaluates results through Person 1’s evaluator, and includes a matched-answer-length control experiment.
Important terminology: the current implementation is a supervised hidden-state probe using correctness/hallucination labels. It is not, by itself, a full traditional Semantic Entropy (SE) implementation that samples multiple answers, groups semantically equivalent answers, and computes entropy over semantic clusters. The upcoming work should implement SE as a separate signal and then combine the SE and SEP signals in a fusion layer.

## 2. Goal and approach

The Person 3 task was to test whether useful reliability-related information is present in the LLM’s internal representations.
- Keep the Llama model frozen; train only a lightweight supervised probe.
- Extract hidden-state representations across transformer layers.
- Use an explicit pooling strategy to turn answer-token representations into a fixed-size feature vector.
- Train on the training split, use validation for model/layer/threshold selection, and reserve test data for final reporting.
- Export continuous scores and predicted labels using the shared evaluator’s standardized schema.
- Investigate whether strong performance is explained by answer length alone.

## 3. Implementation completed


| Area | Work completed / recorded | Status |
| --- | --- | --- |
| Model & representation | Llama 3.2 3B Instruct, 4-bit NF4 configuration; frozen hidden states; explicit mean_answer pooling. | Implemented |
| Feature extraction | Layer-wise feature cache; representation shape recorded as (29, 3072) in the integrity check. | Implemented on GPU system; caches are local/ignored by Git |
| Data splits | Train 15,998; validation 2,002; test 2,000. Sample-ID overlap checks reported zero overlap across split pairs. | Checked |
| Probe training | Supervised classifier trained over frozen features; probe artifact saved. | Implemented |
| Layer selection | Layer 16 selected using validation performance; layer identity retained in output paths/metadata. | Implemented |
| Threshold selection | Threshold 0.9854559515186279 selected on validation. | Implemented |
| Inference records | Validation/test JSONL records and accompanying metadata exported for Person 1’s evaluator. | Implemented |
| Evaluation | Person 1 evaluator ran successfully; warnings list was empty. | Integrated |
| Length confound | Length-only baseline and matched-character-length controlled comparison generated. | Implemented |
| Git handoff | Prediction records, metadata, and evaluator report committed and pushed to branch sep; latest known commit 6ecbea8. | Pushed |


## 4. Repository paths and artifacts


| Path | Purpose |
| --- | --- |
| experiments/sep_extract_features.py | Feature extraction workflow for SEP hidden states. |
| experiments/sep_train_evaluate.py | Probe training/evaluation workflow. |
| experiments/sep_length_baseline.py | Length-only baseline experiment. |
| src/common/comparison.py | Shared comparison/controlled evaluation utilities. |
| outputs/detector_models/sep/mean_answer_layer_16/probe.joblib | Trained probe model artifact. |
| outputs/detector_models/sep/mean_answer_layer_16/frozen_config.json | Frozen model/probe configuration metadata. |
| outputs/detectors/sep/mean_answer_layer_16/validation.jsonl | Validation prediction records. |
| outputs/detectors/sep/mean_answer_layer_16/validation.meta.json | Validation run metadata. |
| outputs/detectors/sep/mean_answer_layer_16/test.jsonl | Test prediction records; latency field was populated with amortized probe-only time per sample. |
| outputs/detectors/sep/mean_answer_layer_16/test.meta.json | Test run metadata. |
| results/sep_evaluator_integration.json | Person 1 evaluator integration report, including metrics, controlled evaluation, efficiency values, and warnings. |
| results/sep_length_controlled_eval.json | Length-controlled SEP evaluation results. |
| results/sep_vs_length_controlled.json | SEP versus length-only controlled comparison. |

Git note: the prediction JSONL files, their metadata, and results/sep_evaluator_integration.json were committed in 6ecbea8. The feature caches under outputs/sep_features/ and model artifacts under outputs/detector_models/ may be ignored by Git and may not be present in a fresh clone. Check the current .gitignore and repository before assuming these large/local artifacts are available to every teammate.

## 5. Recorded results


| Metric | Recorded value | Interpretation |
| --- | --- | --- |
| Selected layer | 16 | Chosen using validation, not test labels. |
| Decision threshold | 0.9854559515186279 | Selected on validation. |
| Validation F1 | 0.99397 | High validation classification performance. |
| Test accuracy | 0.9890 | 98.9% on 2,000 test samples. |
| Test precision | 0.99898 | Very few false positives in the recorded test run. |
| Test recall | 0.9790 | Some positive examples were missed. |
| Test F1 | 0.98889 | Recorded test F1. |
| Test AUROC | 0.999554 | Strong ranking performance on the recorded test split. |
| Test confusion counts | TN=999, FP=1, FN=21, TP=979 | Counts from evaluator report; total 2,000. |
| Length-only baseline AUROC | 0.966518 | Length alone is predictive on the unfiltered test set. |
| Matched-length SEP AUROC | 0.9947 | SEP retains strong discrimination in the recorded matched-length subset. |
| Matched-length SEP accuracy | 0.9599 | Accuracy on controlled subset. |
| Matched-length match accuracy | 1.0000 | The controlled comparison records perfect matched-pair agreement for SEP. |
| Matched-length length-only AUROC | 0.5097 | Near-chance ranking on the controlled subset. |
| Matched-length length-only accuracy | 0.5219 | Near-chance classification on the controlled subset. |


## 6. Evaluator integration and efficiency

The integration command used was:
python -c "from pathlib import Path; from src.common.evaluator import evaluate_run, print_summary; d=Path('outputs/detectors/sep/mean_answer_layer_16'); r=evaluate_run(val_path=d/'validation.jsonl', test_path=d/'test.jsonl', save=True, out_path='results/sep_evaluator_integration.json'); print_summary(r); print('WARNINGS:', r.get('warnings')); print('EFFICIENCY:', r.get('efficiency'))"
Recorded evaluator summary:
validation acc=0.9940 P=1.0000 R=0.9880 F1=0.9940 AUROC=0.9996 pair_acc=0.9990
test       acc=0.9890 P=0.9990 R=0.9790 F1=0.9889 AUROC=0.9996 pair_acc=1.0000
len-ctrl   acc=0.9599 AUROC=0.9947 match_acc=1.0000
WARNINGS: []
Efficiency values in the report: test_total_runtime_sec ≈ 0.054919 seconds for 2,000 cached feature vectors; test_mean_runtime_sec_per_sample ≈ 2.74595e-05 seconds (about 27.46 microseconds/sample). This is probe-only batch scoring on cached features. It excludes tokenization, Llama model loading, Llama forward passes, hidden-state extraction, and any semantic sampling/clustering. Do not present it as end-to-end inference latency.

## 7. Length-bias experiment

The original dataset showed a notable answer-length correlation with labels, so a high-performing detector could exploit a shortcut. A length-only baseline was evaluated, followed by a matched-character-length control subset containing 274 samples / 137 matched pairs.
- On the full test set, the length-only baseline had AUROC 0.966518 and accuracy about 0.9345, indicating length is a strong confound.
- On the matched-length subset, the length-only baseline fell to AUROC 0.5097 and accuracy 0.5219, near chance.
- SEP on the same controlled subset retained AUROC about 0.9947 and accuracy about 0.9599.
- This is evidence that answer character length alone does not explain the recorded SEP result. It does not rule out every possible dataset shortcut, duplicate-content issue, prompt artifact, or other leakage.

## 8. Reproducibility and quality checks


| Check | Recorded outcome | Caveat / action |
| --- | --- | --- |
| Unit tests | python -m pytest tests/test_sep_probe.py -v → 3 passed in the earlier verification. | Rerun in the current environment/clone. |
| JSONL/metadata | Validation: 2,002 records; test: 2,000 records; threshold source recorded as validation. | Recheck after any regeneration. |
| Sample-ID overlap | Train/validation, train/test, and validation/test overlap reported as zero in cached-feature check. | This is not the same as checking duplicate question/answer content. |
| Git diff checks | git diff --cached --check returned no output before the latest commit. | Run git status after later changes. |
| Evaluator warnings | Empty warnings list in integration report. | Warnings-free execution does not prove the dataset has no leakage. |
| Fresh-clone reproducibility | Feature caches/model artifacts may not be tracked. | Transfer/recreate required artifacts and rerun on another machine. |

Important unresolved validation item: audit duplicate question/answer content across train, validation, and test, and inspect the split-generation and label-handling pipeline. Until that is done, describe the metrics as the recorded evaluation results, not as fully leakage-verified results.

## 9. Known limitations and correct claims

- The present detector is a supervised hidden-state probe; it should not be described as a full implementation of sampling-based semantic entropy.
- The recorded metrics are very strong, but dataset content duplicates and all possible leakage/shortcut pathways have not been ruled out by the checks summarized here.
- The reported latency is probe-only on cached features; end-to-end Llama inference latency is not yet measured.
- The matched-length test contains 274 samples / 137 pairs, which is useful as a control but smaller than the full test set.
- A confidence score/probability is a model score, not a guarantee that a response is factually true.
- The frozen base LLM configuration and shared tokenizer/prompt conventions should remain aligned with Person 1’s implementation.

## 10. How teammates can continue from this work


### A. Verify the current checkout

git checkout sep
git pull origin sep
git log -1 --oneline
git status -sb
Expected latest commit from this handoff: 6ecbea8. If the remote has advanced since this document was created, use the newer branch state.

### B. Check environment and tests

.venv\Scripts\Activate.ps1
python -m pytest tests/test_sep_probe.py -v
Use the project’s requirements and model-access instructions. The recorded laptop environment was Windows and CPU-only; Llama hidden-state extraction was previously run on a GPU-enabled system. A fresh laptop clone may not have the local feature cache or trained probe artifact.

### C. Locate artifacts before rerunning

Get-ChildItem outputs\detectors\sep\mean_answer_layer_16
Get-ChildItem outputs\detector_models\sep\mean_answer_layer_16
Get-ChildItem outputs\sep_features
If the feature caches are absent, transfer the agreed cache from the GPU machine or rerun the extraction workflow with the required model access and hardware. Do not silently compare a newly generated run with old metrics unless configuration, data split, prompt version, and feature variant match.

## 11. Proposed next phase: full SE and SE–SEP fusion

The following is a proposed engineering plan, not work already completed. Keep the existing SEP path intact as a baseline and implement full SE as a separate detector before fusion.

### Phase 1 — Implement and validate full Semantic Entropy (SE)

1. For each question/context, generate multiple answer samples using a documented sampling configuration (model revision, temperature, top-p, number of samples, max tokens, seed policy).
1. Normalize/prepare generated answers and group semantically equivalent answers using a documented clustering or entailment method.
1. Compute entropy over the semantic clusters from the model probabilities or a clearly documented approximation. Document assumptions if exact sequence probabilities are unavailable.
1. Export one SE score per sample with common sample_id, split, model/config identifiers, prompt version, and generation/clustering metadata.
1. Evaluate SE using the same fixed splits and shared evaluator. Select any threshold on validation only; do not tune using test labels.
1. Measure end-to-end runtime and generation count/cost because sampling-based SE is expected to cost more than a single-generation probe.

### Phase 2 — Align SE and SEP outputs

1. Use common sample_id and ensure one-to-one alignment of SEP and SE records for each split.
1. Confirm score direction and semantics: for example, whether larger scores consistently mean greater uncertainty/risk.
1. Check missing records, duplicate IDs, and split consistency; do not impute missing values without documenting the rule.
1. Fit any score normalization using training data only, or fit it on validation only if that is the predeclared protocol; never use test labels to choose normalization.
1. Keep raw SE and SEP scores in the output so fusion can be audited and ablated.

### Phase 3 — Build fusion baselines


| Fusion option | How it works | Recommended use |
| --- | --- | --- |
| Rule-based | Normalize scores and combine with a fixed weighted sum or conservative OR/AND rule. | Easy baseline; weights/rules must be fixed before final test evaluation. |
| Logistic-regression fusion | Use [SEP score, SE score] as two input features; fit a small classifier on training records. | Strong interpretable first learned fusion baseline. |
| Validation-selected threshold | Choose the fused model’s decision threshold on validation data. | Use validation only; report the selected threshold and criterion. |
| Ablation | Evaluate SEP alone, SE alone, and SE+SEP fusion on the same samples. | Required to show whether fusion adds value beyond either component. |


### Phase 4 — Evaluate fairly

1. Report accuracy, precision, recall, F1, AUROC, confusion matrix, and calibration metrics if relevant.
1. Use identical held-out test IDs for SEP, SE, and fusion whenever possible.
1. Report coverage/missingness and matched-subset results; preserve sample IDs and label mapping.
1. Compare latency, number of model generations, and compute cost for SEP-only, SE-only, and fusion.
1. Audit duplicate content across splits and document limitations before making research claims.

## 12. Suggested output schema for fusion

Use a joined record with one row per sample. Preserve raw signals and provenance; exact field names should be reconciled with Person 1’s evaluator schema before implementation.
{
  "sample_id": "...",
  "split": "validation|test",
  "sep_score": 0.0,
  "sep_pred_label": 0,
  "se_score": 0.0,
  "fusion_score": 0.0,
  "fusion_pred_label": 0,
  "threshold_source": "validation",
  "config_hash": "...",
  "prompt_version": "v1",
  "se_generation_config": {},
  "fusion_method": "logistic_regression"
}

## 13. Recommended team task split


| Work item | Suggested owner | Deliverable |
| --- | --- | --- |
| Freeze and document SEP baseline | Person 3 / current owner | This handoff, stable outputs, exact config and metrics. |
| Duplicate-content/split audit | Data/evaluation owner with Person 3 | Audit report and any corrected split decision before final claims. |
| Full SE implementation | SE owner / assigned teammate | Sampling, semantic clustering, entropy calculation, tests, SE JSONL outputs. |
| SE–SEP alignment and fusion | Person 3 + evaluator owner | Joined records, fusion baseline, ablations, validation-selected threshold. |
| Cost/latency benchmarking | Shared | Comparable end-to-end measurements for SEP, SE, and fusion. |
| Final reporting | Team | Method comparison, limitations, reproducibility steps, and final test results. |


## 14. Completion checklist

- ☐ SEP extraction, training, validation-based layer/threshold selection, and standardized inference outputs are implemented.
- ☐ Person 1 evaluator integration report exists and recorded warnings are empty.
- ☐ Length-only baseline and matched-length control are available.
- ☐ Prediction outputs and integration report are committed/pushed in the sep branch (latest known commit 6ecbea8).
- ☐ Full end-to-end latency is still pending.
- ☐ Duplicate question/answer content audit and broader leakage review remain important before final scientific claims.
- ☐ Full sampling-based Semantic Entropy implementation is separate follow-up work.
- ☐ SE–SEP fusion, component ablations, and comparative cost/performance analysis are future work.

## 15. Short handoff statement

Person 3’s current SEP baseline uses frozen Llama 3.2 3B Instruct hidden states with mean-answer pooling and a supervised probe. Layer 16 and threshold 0.9854559515 were selected on validation. The recorded test metrics are accuracy 0.989, F1 0.98889, and AUROC 0.999554. A matched-length control retained SEP AUROC around 0.9947 while the length-only baseline was near chance (AUROC around 0.5097). Standardized validation/test outputs are integrated with Person 1’s evaluator and the latest known commit is 6ecbea8. These are promising recorded results, but duplicate-content/leakage audit and end-to-end latency remain open. The next research phase is to implement full sampling-based SE, align its scores with SEP, and evaluate SE-only, SEP-only, and fused systems on the same held-out samples.
