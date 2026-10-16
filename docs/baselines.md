# Baselines (Person 1)

Both baselines write standard detector runs (`src/common/schema.py`) and are evaluated with the common evaluator.

## Length-only (`src/detectors/baselines/length_only.py`)

- Feature: `answer_length_chars = len(answer)` (official). `--feature tokens` adds a Llama-token variant.
- Selected on **validation only**: direction (+1 if longer answers are more often hallucinated, else -1) and the threshold (Youden's J).
- Score = direction x length, so "higher = more likely hallucinated" always holds.
- Run: `python -m experiments.run_length_only [--plot]`
- Output: `outputs/detectors/length_only/chars/{validation,test}.jsonl`, report `results/eval/length_only/chars.json`.
- Reading the result: full-test performance reflects the length confound; the length-controlled set and the length-stratified table show what is left.

## Transformer (`src/detectors/baselines/transformer.py`)

- Fine-tunes a pretrained encoder (default `distilroberta-base`) on (question, answer, knowledge) -> label. Score = logit margin.
- Protocol, enforced in code:
  1. `select`: train on **train**, choose (learning rate, epoch) and the threshold on **validation**, write `frozen_config.json` (hyperparameters, threshold, checkpoint SHA-256, project `config_hash`). Never loads test.
  2. `test`: refuses unless the frozen config exists and the checkpoint hash matches. A changed project `config_hash` only warns (the baseline uses no Llama/SE/SEP setting) and results keep the hash they were frozen under. Scores test once, evaluates, writes `test_evaluated.json`.
  3. After that marker exists, `select` refuses to run again unless `--force` (recorded as `reselected_after_test` in the frozen config).
- Run (GPU recommended):

```powershell
python -m experiments.run_transformer smoke     # tiny run, checks the training loop, never touches test
python -m experiments.run_transformer select    # full grid, ~2 learning rates x 3 epochs
python -m experiments.run_transformer test      # only after select
```

- Artifacts (not committed): `outputs/transformer/<variant>/` (best.pt, frozen_config.json, training_log.json).
- Final runs go to `results/detector_runs/transformer/<variant>/` (copy and commit when final).
- Caveat: like any text classifier it can learn answer length; judge it on the length-controlled set and the length-stratified results.

## Comparing detectors (`experiments/compare_detectors.py`)

```powershell
python -m experiments.compare_detectors length_only/chars transformer/distilroberta-base_qak se_candidate/<variant> sep/<variant> --n-boot 5000 --out results/comparison/final.json
```

- Reads each run's test file from `results/detector_runs/` (then `outputs/detectors/`); each run needs a validation-selected threshold.
- **CIs:** 95% bootstrap intervals that resample question pairs (test) or length matches (controlled set), never single samples.
- **Paired differences** against a reference detector, on identical resamples. "CI excludes 0" is the criterion for claiming a difference.
- **Error overlap:** right/wrong table per detector pair, exact McNemar test, and the accuracy of an oracle that is right whenever any detector is: the ceiling for any fusion.
- Single-detector reports (`evaluate_run`) now include the same CIs, a minority-class count per length bin (`n_min`, AUROC flagged `*` when `n_min < 30`), and full-text error listings.

## Input ablations (same Transformer, fewer inputs)

The official baseline sees question + answer + knowledge (`qak`). Ablations show what it actually relies on. They are
reported as ablations; the official variant is fixed in advance and is never swapped after seeing test results.

| Variant | Command | Question it answers |
|---|---|---|
| `a` (answer only) | `select --no-question --no-knowledge` | Can the answer text alone (style, form) separate the classes? |
| `qa` | `select --no-knowledge` | Does the knowledge passage matter? |
| `ak` | `select --no-question` | Does the answer need its question, or only the passage? |

Each variant has its own artifact folder, frozen config and test lock. For every variant run `select`, then `test` with the
same flags, then compare with `experiments/compare_detectors.py` (e.g. `transformer/distilroberta-base_a`).
