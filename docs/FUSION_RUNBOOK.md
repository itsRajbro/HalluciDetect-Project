# Fusion runbook (Person 1)

What to run, in what order, and which steps wait for Person 2. All commands are PowerShell, run from the repo root with the virtual environment active.
The method and the decision rule are fixed in `docs/fusion_analysis_plan.md`; this file only says how to execute them.

## Timeline at a glance

| Phase | When | Needs Person 2? | What it gives you |
|---|---|---|---|
| 0. Install and commit the fusion code | now | no | the code on your branch, tests green |
| 1. Dry run on real files | now | no | proof the pipeline reads real runs correctly (no SE, no test labels, no files written) |
| 2. Person 2 hand-off | Person 2's side | it IS Person 2 | SE validation score files committed + the final SE variant confirmed |
| 3. Check the SE files, dry run with SE | right after phase 2 | yes | first check of the real SE files |
| 4. The real fusion (primary + secondary) | after phase 3 | yes | the results |
| 5. Compare, commit, write up | after phase 4 | yes | final tables |
| 6. Sensitivity with the baseline `v1` SE (optional) | only if its validation file is committed | yes | robustness check |

## Phase 0: install and commit (about 10 minutes, now)

```powershell
git fetch origin
git merge origin/main                         # brings in Person 2's and Person 3's latest work
# unzip fusion_update_v3.zip into the repo root (it replaces src/common/evaluator.py with main's version + one small fix)
git diff origin/main -- src/common/evaluator.py   # expect roughly +30 / -8 lines, only the cost change
python -m pytest tests -q
git add src/common/comparison.py src/common/evaluator.py src/detectors/fusion experiments/run_fusion.py experiments/compare_detectors.py tests/test_fusion.py docs/fusion.md docs/fusion_analysis_plan.md docs/FUSION_RUNBOOK.md
git commit -m "Add fusion module, partial-coverage mode, comparison module, pair-level cost"
git push -u origin fusion
```

## Phase 1: dry run on real files (about 2 minutes, now)

This checks the pipeline on real SEP scores (saturated probabilities, logit transform, cross-validation) before SE exists. It pairs SEP with the full Transformer purely as a **code test**: it writes no files, reads no test labels, and is not an analysis, so it does not touch the pre-registered plan.

```powershell
python -m experiments.run_fusion --dry-run `
  --input sep  results/detector_runs/sep/mean_answer_layer_16/validation.jsonl results/detector_runs/sep/mean_answer_layer_16/test.jsonl `
  --input tqak results/detector_runs/transformer/distilroberta-base_qak/validation.jsonl results/detector_runs/transformer/distilroberta-base_qak/test.jsonl `
  --logit sep
```

What good looks like:
- `coverage: validation 2002 of 2002 samples, test 2000 of 2000 samples` (no PARTIAL).
- A chosen C and two weights. Usually both are positive; with two highly correlated inputs one can come out negative, which is reported as fitted and is not an error.
- Validation diagnostics: both AUROCs very high (about 0.99+) and a strong correlation between the two inputs. For SEP the line also shows `clipped X%`: the share of its probabilities so extreme that the logit clip applies. A small share (under a few percent) is fine; if it is large, tell me, because those scores lose their ordering.
- It ends with `DRY RUN: nothing was written and no test label was used.`

If it stops with a message, the message says why (coverage, provenance, a constant input, scores outside [0, 1] for `--logit`). Paste it to me.

## Phase 2: what Person 2 must deliver (blocks everything below)

State of SE on `main` (checked against the committed files): SE has a **full-test run with the anchored matcher** (2,000 samples, threshold 0.95 selected on validation) and a control-set run with the baseline matcher. Person 2's report names the anchored full-test run as the primary result. Its test AUROC (0.8583) matches its validation AUROC (0.8582), so there is no sign the matcher rule was tuned to validation, and on the 274 control samples the anchored and baseline scores differ on only 3 samples (AUROC 0.7512 vs 0.7472). Using the anchored variant therefore costs nothing and gives full coverage, so **no partial mode is needed**.

1. **Confirm in writing that the anchored variant (`anchor_v1`) is the final SE variant.** Do this before you run any fusion.
2. **Commit the SE validation score files.** They are blocked by `.gitignore`, so the repo currently has only SE's test files and a validation meta file. Person 2 runs, from the repo root:
   ```powershell
   git add -f outputs/detectors/se_candidate/nli/validation_validation_anchor_v1.jsonl outputs/detectors/se_candidate/nli/validation_validation_anchor_v1.meta.json
   git add -f outputs/detectors/se_global/nli/validation_validation_full_v1.jsonl outputs/detectors/se_global/nli/validation_validation_full_v1.meta.json
   git commit -m "Add SE validation score files needed for fusion"
   git push
   ```
3. Remove the duplicate `tests/tests/` folder (same file name as `tests/test_se_candidate.py`).

## Phase 3: check the SE files (about 5 minutes, after phase 2)

```powershell
git pull origin main
(Get-Content outputs/detectors/se_candidate/nli/validation_validation_anchor_v1.jsonl | Measure-Object -Line).Lines   # 2002
(Get-Content outputs/detectors/se_candidate/nli/test_full_anchor_v1.jsonl | Measure-Object -Line).Lines                # 2000
(Get-Content outputs/detectors/se_global/nli/validation_validation_full_v1.jsonl | Measure-Object -Line).Lines         # 2002
(Get-Content outputs/detectors/se_global/nli/test_full_anchor_v1.jsonl | Measure-Object -Line).Lines                   # 2000
```

Then the dry run with the real SE files (no test labels, no files written):

```powershell
python -m experiments.run_fusion --dry-run `
  --input se  outputs/detectors/se_candidate/nli/validation_validation_anchor_v1.jsonl outputs/detectors/se_candidate/nli/test_full_anchor_v1.jsonl `
  --input sep results/detector_runs/sep/mean_answer_layer_16/validation.jsonl results/detector_runs/sep/mean_answer_layer_16/test.jsonl `
  --logit sep
```

Expect `validation 2002 of 2002`, `test 2000 of 2000` (not partial), SE showing 11 distinct values and a validation AUROC of about 0.858. A coverage or provenance error means a file is wrong; fix it before phase 4.

## Phase 4: the real fusion (a few minutes)

Run these three, in this order. Do not change any option after seeing results (that needs an entry in the plan's amendment log).

```powershell
# PRIMARY (plan H-B): candidate-aware SE + SEP
python -m experiments.run_fusion --n-boot 5000 --out results/fusion/se-sep.json `
  --input se  outputs/detectors/se_candidate/nli/validation_validation_anchor_v1.jsonl outputs/detectors/se_candidate/nli/test_full_anchor_v1.jsonl `
  --input sep results/detector_runs/sep/mean_answer_layer_16/validation.jsonl results/detector_runs/sep/mean_answer_layer_16/test.jsonl `
  --logit sep

# SECONDARY (H-A): global SE + SEP
python -m experiments.run_fusion --n-boot 5000 --out results/fusion/seg-sep.json `
  --input seg outputs/detectors/se_global/nli/validation_validation_full_v1.jsonl outputs/detectors/se_global/nli/test_full_anchor_v1.jsonl `
  --input sep results/detector_runs/sep/mean_answer_layer_16/validation.jsonl results/detector_runs/sep/mean_answer_layer_16/test.jsonl `
  --logit sep

# LENGTH CHECK (plan section 7): primary inputs + log answer length
python -m experiments.run_fusion --add-length --n-boot 5000 --out results/fusion/se-sep-len.json `
  --input se  outputs/detectors/se_candidate/nli/validation_validation_anchor_v1.jsonl outputs/detectors/se_candidate/nli/test_full_anchor_v1.jsonl `
  --input sep results/detector_runs/sep/mean_answer_layer_16/validation.jsonl results/detector_runs/sep/mean_answer_layer_16/test.jsonl `
  --logit sep
```

Use the short name `seg` for global SE so its outputs do not overwrite the primary's.

How to read the output (section 4, **PRIMARY DECISION**):
- `IMPROVES`: the hybrid beats both SE and SEP on the length-controlled set, with the 95% CI above zero.
- `INCONCLUSIVE`: at least one CI includes zero. Report it as such, **not** as "no benefit". SEP is at about 0.995 on this set, so there is almost no room to improve; this is the expected outcome.
- `WORSE`: the hybrid is below an input (CI below zero).
- Global SE should get a weight near zero. Its validation score is identical for both answers of a question (generations shared per pair), so its AUROC there is exactly 0.5 and Youden J is 0, as Person 2's report shows; on the test run the two answers get separately sampled scores, so about 16% of pairs tie and the rest differ by sampling noise (AUROC 0.496). This is expected, not a bug, and the scores must not be reversed.

## Phase 5: compare, commit, write up

```powershell
python -m experiments.compare_detectors length_only/chars transformer/distilroberta-base_a transformer/distilroberta-base_qak sep/mean_answer_layer_16 hybrid/se-sep-lr hybrid/se-sep-avg --n-boot 5000 --out results/comparison/final.json
```

(The hybrid now covers the full test set, so no special scope is needed. Raw SE runs use tag-based file names, so they appear in the fusion output rather than in this table.)

Then publish the final runs and results:

```powershell
mkdir results\detector_runs\hybrid\se-sep-lr
copy outputs\detectors\hybrid\se-sep-lr\* results\detector_runs\hybrid\se-sep-lr\
mkdir results\detector_runs\hybrid\se-sep-avg
copy outputs\detectors\hybrid\se-sep-avg\* results\detector_runs\hybrid\se-sep-avg\
git add results/detector_runs/hybrid results/fusion results/comparison/final.json
git commit -m "Add hybrid runs and final fusion results"
git push
```

## Phase 6 (optional): baseline-matcher sensitivity

SE's earlier baseline matcher (`v1`) has a control-set test run but, like the anchored validation file, its validation file is uncommitted. If Person 2 commits `validation_validation_full_v1.jsonl` for candidate-aware SE, rerun the primary fusion with it and `--allow-partial` (its test run covers only the 274 control samples), using a different alias such as `sev1`, and report it as a sensitivity analysis next to the primary. With only 3 of 274 control scores differing, expect nearly the same result. Do not replace the primary after seeing which looks better.

## Rules that keep the analysis valid

- Fit and select on validation only; test labels are read only by the analysis step, after the fusion is written.
- No option changes after seeing results. Any deviation goes in the plan's amendment log (section 10) with the date and reason.
- The SE variant is declared once, before phase 4.
