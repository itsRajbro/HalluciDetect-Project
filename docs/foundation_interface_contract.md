# HalluciDetect: Common Foundation Interface Contract

Applies to Person 1 (baselines + evaluation), Person 2 (Semantic Entropy) and Person 3 (Semantic Entropy Probes).
Written against the repository state at the `v0.1-foundation` freeze.
After the tag, **anything in sections 1 and 2 changes only with agreement from all three people.**
Setup, branching and merging commands are in `docs/TEAMMATE_WORKFLOW.md`.

## 0. Freeze reference

| Item | Value |
|---|---|
| `config_hash` | `435a3e4fb7b8` (print with `python -m src.common.config`; must match on every machine) |
| Model | `meta-llama/Llama-3.2-3B-Instruct`, NF4 4-bit, revision `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| Prompt template | `v1` (`src/common/prompts.py`, includes `knowledge`) |
| Data | HaluEval QA, 10,000 pairs, pair-level split seed 42: 7,999 / 1,001 / 1,000 pairs (15,998 / 2,002 / 2,000 samples) |
| Length-controlled test set | 274 samples (137 factual / 137 hallucinated), lengths matched within 1 character |
| Environment | Python 3.11, torch 2.6.0+cu124, transformers 5.17.0, bitsandbytes 0.50.1, scikit-learn 1.9.0 (see `requirements.txt`) |

## 1. Frozen modules

| Area | Module | Public interface |
|---|---|---|
| Configuration | `src/common/config.py` | `get_config()`, `Config.config_hash()`, label / split / length constants |
| Dataset | `src/common/data.py` | `load_split`, `load_all`, `load_length_controlled`, `group_by_pair`, `group_by_match`, `attach_token_lengths` |
| Splits | `data/split_manifest.json`, `src/data/split.py`, `src/data/make_split_manifest.py` | `python -m src.data.make_split_manifest verify` |
| Model | `src/common/llama_loader.py` | `load_llama()`, `set_seed()`, `verify_model()` |
| Prompts | `src/common/prompts.py` | `build_generation_prompt()`, `encode_with_answer()`, `PROMPT_TEMPLATE_VERSION` |
| Generation | `src/common/generation.py` | `generate_greedy()`, `generate_samples()` returning `GenerationOutput(generations, stats)` |
| Hidden states | `src/common/hidden_states.py` | `extract_hidden_states()`, `save_hidden_states()`, `load_hidden_states()` |
| Detector output | `src/common/schema.py` | `DetectorRecord`, `write_detector_output()`, `read_detector_output()`, `efficiency_from_gen_stats()`, `check_coverage()` |
| Evaluation | `src/common/evaluator.py` | `evaluate_run()`, `evaluate_detector()`, `select_threshold()`, `print_summary()` |
| Gate | `src/common/smoke_test.py` | `python -m src.common.smoke_test [--freeze]` |

Rules: nobody calls `AutoModelForCausalLM.from_pretrained`, `model.generate`, or builds a Llama prompt string anywhere else. Nobody splits the data again. Nobody computes metrics with private code.

## 2. Conventions (do not change)

- Label **1 = hallucinated**, 0 = factual. Splits: `train`, `validation`, `test`.
- Every detector's score: **higher = more likely hallucinated.** No sign flipping in the evaluator. If your raw signal runs the other way, negate it before writing.
- Decision rule: `pred_label = 1 iff score >= threshold` (`schema.predict_label`).
- Thresholds, layers, poolings, probe hyperparameters and fusion weights are chosen on **validation only**. Test is used for final numbers after the configuration is frozen. The schema rejects any threshold whose source is not `validation`.
- Official length feature: `answer_length_chars = len(answer)`. Token count is secondary.
- Every record carries `config_hash` and `prompt_version`. `evaluate_run` rejects validation/test runs that differ in detector, variant, config hash, or prompt version.
- A failed sample is never dropped silently: decide how it is scored, write it. The schema rejects NaN/inf, the evaluator rejects missing samples.

## 3. Bar to beat: length-only baseline (real results at freeze)

| | AUROC | Accuracy | F1 | Pair accuracy |
|---|---|---|---|---|
| Validation | 0.9727 | 0.9446 | 0.9444 | 0.9700 |
| **Test** | **0.9665** | 0.9345 | 0.9345 | 0.9705 |
| **Length-controlled set (274)** | **0.5097** | 0.5219 | n/a | 0.5876 (match-level) |

Threshold: 27.5 characters, selected on validation (Youden). Answer length alone separates the classes at 0.97 AUROC, so a detector's full-test score means little by itself. **Judge detectors on the length-controlled set and the length-stratified results** (and report confidence intervals: the controlled set has only 274 samples, roughly +/-0.07 on AUROC). Pair accuracy is NOT length-controlled: within a pair the hallucinated answer is almost always the longer one.

## 4. Usage recipes

Common to all (setup in `docs/TEAMMATE_WORKFLOW.md`):

```python
from src.common.config import get_config
from src.common.data import load_split
from src.common.schema import DetectorRecord, write_detector_output, efficiency_from_gen_stats
from src.common import evaluator as E

cfg = get_config()
val, test = load_split("validation"), load_split("test")    # verified against the manifest
```

Person 2, Semantic Entropy:

```python
from src.common.llama_loader import load_llama
from src.common.generation import generate_samples

model, tok = load_llama()
out = generate_samples(model, tok, s.question, s.knowledge, sample_id=pair_or_sample_id)
# out.generations: Generation(text, token_logprobs, sum_logprob, ...) x cfg.sampled.num_samples
# out.stats: GenStats -> DetectorRecord(..., **efficiency_from_gen_stats(out.stats))
```

Person 3, SEP:

```python
from src.common.hidden_states import extract_hidden_states, save_hidden_states

hs = extract_hidden_states(model, tok, s.question, s.knowledge, s.answer, sample_id=s.sample_id,
                           layers=[...], poolings=["last_answer", "mean_answer"])
hs.features["last_answer"]     # (len(layers), 3072) CPU tensor
hs.answer_num_tokens           # keep it: needed to control for length
```

Everyone, writing and evaluating a detector:

```python
def make_records(samples, split, scores, **extra):
    return [DetectorRecord(sample_id=s.sample_id, split=split, detector="sep",     # [a-z0-9_]+
                           variant="layer20_last_answer",                          # [A-Za-z0-9_.-]+
                           score=sc, config_hash=cfg.config_hash(),
                           prompt_version=cfg.prompt.version, **extra)
            for s, sc in zip(samples, scores)]

val_recs = make_records(val, "validation", val_scores)
y_val, x_val = E.align(val, E.records_to_scores(val_recs))
thr = E.select_threshold(y_val, x_val)["value"]                  # validation only
write_detector_output(val_recs, threshold=thr)                   # -> outputs/detectors/sep/<variant>/validation.jsonl
write_detector_output(make_records(test, "test", test_scores), threshold=thr)
report = E.evaluate_run(val_path=".../validation.jsonl", test_path=".../test.jsonl", save=True)
E.print_summary(report)
```

Suggested detector names: `length_only`, `transformer`, `se_global`, `se_candidate`, `sep`, `hybrid`.
Scratch work goes to `outputs/` (not committed). When a run is final, copy its `.jsonl` + `.meta.json` files to `results/detector_runs/<detector>/<variant>/` and commit them, so the hybrid and the final comparison can read every detector's scores.

## 5. Design notes that follow from the dataset (read before building)

1. **Each question appears twice, once factual and once hallucinated.** Anything that depends only on the question (+ knowledge) is identical for both samples of a pair. Consequence for **global SE**: both answers get the same score, so its AUROC is exactly 0.5 on this data by construction, and its pair accuracy is 0.5 (all ties). Global SE is still worth computing (it is the reference method and needed for the hybrid), but only **candidate-aware SE** can discriminate within a pair. Generate samples once per `pair_id`, not once per `sample_id`, to halve the cost.
2. **The same applies to the `tbg` pooling** in `hidden_states.py` (last prompt token, before the answer): it is identical for both samples of a pair, so a label-supervised probe on `tbg` cannot separate them. Use `last_answer`, `mean_answer` or `eot` for label prediction; `tbg` is only meaningful for the original SEP formulation (predicting the question's semantic entropy).
3. **Length leaks into answer-level features** (`mean_answer`, `last_answer`). Always check probe performance on the length-controlled set and in length-stratified bins, and consider `answer_num_tokens` as a covariate or a control model.
4. **Storage.** Train-split hidden states for all 29 layers and 4 poolings are about 21 GiB in float32. Choose layers/poolings, store float16, and keep caches in `outputs/` (not committed). One pooling over 8 layers in float16 is about 0.7 GiB.

## 6. Decisions to confirm before the tag

| Decision | Current value | Who | Notes |
|---|---|---|---|
| Prompt includes `knowledge` | `True` (`PromptConfig.include_knowledge`) | all three | Consistent with `docs/dataset_selection.md` (input is question + knowledge + answer; SE generation uses knowledge and question). Confirm, then replace the "DECISION NEEDED" comment in `config.py` (a comment change does not alter `config_hash`). Changing the value later invalidates every generation and cached hidden state. |
| SE sampling settings | `num_samples=10`, `temperature=1.0`, `top_p=0.9`, `top_k=50`, `max_new_tokens=64`, `chunk_size=5` | Person 2 | Marked PROPOSED in `config.py`. All are part of `config_hash`, including `chunk_size`. Fix before generating at scale. |
| Greedy `max_new_tokens` | `64` | Person 2 | Same. |

Hidden-state layers and poolings are **not** frozen choices: Person 3 selects them on validation.

## 7. Freeze checklist

| Step | Status |
|---|---|
| `python -m pytest tests -q` passes | Person 1: done. Persons 2, 3: |
| `python -m src.data.make_split_manifest verify` says OK | Person 1: done. Persons 2, 3: |
| `python -m src.common.config` prints `435a3e4fb7b8` | Person 1: done. Persons 2, 3: |
| Llama GPU smoke tests pass (`python -m scripts.smoke_step4_loader`, `scripts.smoke_generation`, `scripts.smoke_step7_hidden`) | Person 1: |
| `python -m src.common.smoke_test --freeze` ends with FREEZE GATE PASSED | Person 1: done |
| Section 6 decisions confirmed | |
| Tag `v0.1-foundation` pushed | |

| Person | Sign-off (date) |
|---|---|
| Person 1 | |
| Person 2 | |
| Person 3 | |

## 8. Change policy after the freeze

- A change to anything in sections 1 or 2 needs agreement from all three people.
- A change that alters results (prompt text, generation settings, model revision, split, length definition) requires a new `config_hash` or `prompt_version`, and affected detectors must be re-run before results are compared.
- Bug fixes that change no result can be merged normally with tests passing.
- Never edit `src/data/split.py` or its ratios. A different split invalidates every result.
