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
  2. `test`: refuses unless the frozen config exists, the checkpoint hash matches and `config_hash` is unchanged. Scores test once, evaluates, writes `test_evaluated.json`.
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
