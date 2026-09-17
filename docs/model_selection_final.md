# HalluciDetect — Final Base Model Selection

## 1. Purpose

This document records the final model-identification and base-model selection decision for HalluciDetect.

Three candidate base LLMs were evaluated:

1. Qwen3-4B
2. Gemma 3 4B IT
3. Llama 3.2 3B Instruct

The selection is based on measured benchmark results from the project's actual local hardware environment rather than model specifications alone.

> **Phase boundary:** This document closes the **base-model selection phase**. Semantic Entropy (SE), Semantic Entropy Probe (SEP), hallucination-detector implementation, training/fine-tuning, and final detector evaluation are outside the scope of this phase.

---

## 2. Benchmark Environment

| Component | Configuration |
|---|---|
| CPU | AMD Ryzen 5 7535HS |
| RAM | Approximately 16 GB |
| GPU | NVIDIA RTX 3050 Laptop GPU |
| VRAM | 6 GB |
| CUDA | 12.4 |
| PyTorch | 2.6.0+cu124 |
| Benchmark script | `src/models/model_identification.py` |
| Prompt set | Controlled 5-prompt model-identification set |

The benchmark was performed on the project's actual local GPU environment.

---

## 3. Candidate Models

| Model | Parameters | Architecture | License / Access |
|---|---:|---|---|
| Qwen3-4B | 4B | Text LLM | Apache 2.0, ungated |
| Gemma 3 4B IT | 4B | Multimodal vision + text | Gemma Terms of Use, gated |
| Llama 3.2 3B Instruct | 3B | Text-only causal LM | Llama 3.2 Community License, gated |

### Model repositories

- Qwen3-4B: https://huggingface.co/Qwen/Qwen3-4B
- Gemma 3 4B IT: https://huggingface.co/google/gemma-3-4b-it
- Llama 3.2 3B Instruct: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

---

## 4. Resource Comparison

Based on the measured VRAM usage, **4-bit NF4 is the practical configuration for running all three candidates within the available 6 GB GPU memory.**

| Metric | Qwen3-4B | Gemma 3 4B IT | Llama 3.2 3B |
|---|---:|---:|---:|
| 4-bit peak VRAM (MB) | 2,642 | 3,107 | **2,178** |
| 4-bit RAM (MB) | 1,598 | 1,746 | **1,544** |
| 4-bit generation speed (tok/s) | 8.46 | 5.54 | **10.92** |
| 4-bit generation time, 5 prompts (sec) | 106.8 | 19.7 | **9.8** |
| 4-bit fits 6 GB VRAM | Yes | Yes | Yes |

### Gemma throughput variance

A repeat 4-bit Gemma 3 4B IT run using identical settings measured 7.92 tok/s rather than 5.54 tok/s. Peak VRAM remained 3,107 MB in both runs.

This demonstrates that measured generation throughput can vary across runs on the same hardware. The VRAM measurements were stable, so the variation does not change the conclusion that the 4-bit configuration fits the available GPU memory.

### BF16 measurements

| Metric | Qwen3-4B | Gemma 3 4B IT | Llama 3.2 3B |
|---|---:|---:|---:|
| BF16 peak VRAM (MB) | 7,735 | 8,241 | 6,157 |
| BF16 fits nominal 6 GB capacity | No | No | Borderline |

Qwen3-4B and Gemma 3 4B IT exceed the nominal 6,144 MB capacity in bf16. Llama 3.2 3B is approximately 13 MB above that nominal boundary and is therefore also not treated as a practical bf16 configuration for this project.

**Selected runtime configuration:** 4-bit NF4.

---

## 5. Model Architecture and Required Inference Access

Hidden-state and logit access were treated as technical requirements for the planned downstream work.

| Property | Qwen3-4B | Gemma 3 4B IT | Llama 3.2 3B |
|---|---:|---:|---:|
| Hidden-state layers | 37 | 35 | 29 |
| Hidden size | 2,560 | 2,560 | 3,072 |
| Vocabulary size | 151,936 | 262,208 | 128,256 |
| Logits available | Yes | Yes | Yes |
| Hidden states available | Yes | Yes | Yes |

### Logit tensor verification

| Model | Observed logits shape | Logits dtype |
|---|---|---|
| Qwen3-4B | `[1, 12, 151936]` | `torch.bfloat16` |
| Gemma 3 4B IT | `[1, 13, 262208]` | `torch.bfloat16` |
| Llama 3.2 3B | `[1, 13, 128256]` | `torch.bfloat16` |

### Hidden-state verification

| Model | Last hidden-state shape | Hidden-state dtype |
|---|---|---|
| Qwen3-4B | `[1, 12, 2560]` | `torch.bfloat16` |
| Gemma 3 4B IT | `[1, 13, 2560]` | `torch.bfloat16` |
| Llama 3.2 3B | `[1, 13, 3072]` | `torch.bfloat16` |

All three candidates satisfied the tested logit and hidden-state access requirements through the standard Transformers inference path.

---

## 6. Controlled Language Capability Test

The benchmark tested:

- English
- Hindi in Devanagari
- Roman Hindi
- Hinglish / Hindi-English code-mixed input
- False-premise / uncertainty handling

| Test | Qwen3-4B | Gemma 3 4B IT | Llama 3.2 3B |
|---|---|---|---|
| English | Correct | Correct | Correct |
| Hindi / Devanagari | Correct | Correct | Correct |
| Roman Hindi | Correct | Partial | Correct |
| Hinglish | Correct | Partial | Correct |
| False-premise / uncertainty | Correct | Correct | Correct |

### Observed language behavior

**Qwen3-4B:** Produced correct responses for English, Hindi, Roman Hindi, and Hinglish in the tested prompts. It also correctly identified the false premise in the Mars prompt.

**Gemma 3 4B IT:** Produced correct English and Hindi responses. Roman Hindi and Hinglish showed naming/script inconsistencies: it used "Delhi" rather than "New Delhi", and the 4-bit Roman Hindi response switched to Devanagari.

**Llama 3.2 3B Instruct:** Produced correct English, Hindi, Roman Hindi, and Hinglish responses in the tested runs. "New Delhi" was preserved across the tested language forms. The false-premise prompt was correctly identified as false.

---

## 7. Qwen3-4B Thinking-Mode Consideration

Qwen3-4B has a generation-behavior consideration that was not observed in the same way for the other candidates.

Its default chat template can generate a reasoning trace inside `<think>...</think>` before the final answer.

Observed behavior:

- With `max_new_tokens=128`, 2–3 of the 5 prompts reached the token budget while still inside the thinking block.
- With `max_new_tokens=512`, all 5 prompts completed.
- The uncertainty prompt used approximately 450+ tokens of reasoning before producing an approximately 90-token answer.

This creates two relevant considerations:

1. Reasoning traces can increase generation cost and latency.
2. Raw output length can become confounded with the project's observed answer-length signal.

The project EDA found that hallucinated answers were substantially longer than factual answers. Therefore, reasoning traces should not automatically be treated as part of the answer in later length-sensitive analysis.

### Status of the optional Qwen control run

The dedicated `--no-thinking` control run was not required to make the current base-model selection.

It remains an **optional follow-up experiment** and does not block closure of this model-identification phase.

---

## 8. Selection Criteria Comparison

No separate weighted scoring system was introduced.

| # | Evaluation criterion | Qwen3-4B | Gemma 3 4B IT | Llama 3.2 3B Instruct |
|---:|---|---|---|---|
| 1 | Hardware feasibility | Pass in 4-bit; bf16 exceeds 6 GB | Pass in 4-bit; bf16 exceeds 6 GB | Pass in 4-bit; bf16 not treated as practical |
| 2 | Model size (~3–4B) | Pass — 4B | Pass — 4B | Pass — 3B |
| 3 | English capability | Pass | Pass | Pass |
| 4 | Hindi capability | Pass | Pass | Pass |
| 5 | Roman Hindi / Hinglish | Pass in tested prompts | Partial in tested prompts | Pass in tested prompts |
| 6 | Hidden-state access | Pass | Pass | Pass |
| 7 | Logit / probability access | Pass | Pass | Pass |
| 8 | Tooling / reproducibility | Pass; Transformers/PyTorch + bitsandbytes | Pass; Transformers/PyTorch + bitsandbytes | Pass; Transformers/PyTorch + bitsandbytes |
| 9 | License / access | Apache 2.0; ungated | Gated; Gemma Terms of Use | Gated; Llama 3.2 Community License |

---

## 9. Final Base Model Selection

### Selected model

**Llama 3.2 3B Instruct**

### Selected runtime configuration

**4-bit NF4 quantization**

### Selection rationale

The selection is based on the measured evidence from the model-identification benchmark.

Llama 3.2 3B Instruct:

- had the **lowest measured 4-bit peak VRAM usage** at 2,178 MB;
- had the **lowest measured 4-bit RAM usage** at 1,544 MB;
- recorded the **highest measured 4-bit generation throughput** at 10.92 tok/s;
- completed the tested English, Hindi, Roman Hindi, and Hinglish prompts correctly;
- correctly handled the tested false-premise / uncertainty prompt;
- provided both logits and hidden states through the tested Transformers inference path;
- satisfied the target model-size range at 3B parameters;
- did not exhibit the Qwen-specific thinking-mode behavior observed during this benchmark.

These observations provide the basis for selecting Llama 3.2 3B Instruct as the project's base model for the next development phase.

### Trade-off

Llama 3.2 3B Instruct uses gated access under the Llama 3.2 Community License. This is a project-access consideration and should be preserved in the reproducibility record.

### Scope of the conclusion

This selection means that **Llama 3.2 3B Instruct is the selected base model based on the completed model-identification benchmark**.

It does **not** establish that Llama 3.2 3B Instruct will achieve the best hallucination-detection accuracy. Detector performance remains a separate empirical question for a later project phase.

---

## 10. Evidence Summary

| Dimension | Qwen3-4B | Gemma 3 4B IT | Llama 3.2 3B Instruct |
|---|---|---|---|
| 4-bit VRAM | 2.64 GB | 3.11 GB | **2.18 GB** |
| 4-bit throughput | 8.46 tok/s | 5.54 tok/s; 5.54–7.92 across repeat runs | **10.92 tok/s** |
| English | Correct | Correct | Correct |
| Hindi | Correct | Correct | Correct |
| Roman Hindi | Correct | Partial | Correct |
| Hinglish | Correct | Partial | Correct |
| Logits | Available | Available | Available |
| Hidden states | Available | Available | Available |
| Thinking-mode complication | Yes | No | No |
| Access | Ungated | Gated | Gated |

---

## 11. Reproducibility and Access Record

The benchmark should preserve the following metadata:

- exact model ID
- model revision, where available
- PyTorch version
- Transformers version
- device
- dtype
- generation settings
- relevant chat-template configuration
- quantization configuration

The benchmark used local Transformers/PyTorch inference with bitsandbytes for the 4-bit tests.

---

## 12. Phase Closure

**Model-identification benchmark:** Complete for all three candidates.

**Final base model:** Llama 3.2 3B Instruct.

**Selected runtime configuration:** 4-bit NF4.

**Base-model selection phase:** Complete.

### Out of scope for this phase

The following work is intentionally not started as part of this document:

- Semantic Entropy (SE)
- Semantic Entropy Probe (SEP)
- hallucination-detector implementation
- detector training/fine-tuning
- final detector evaluation
- downstream multilingual detector experiments

The project can resume from the selected base model in a future phase without reopening the completed model-identification benchmark unless new requirements or hardware conditions arise.

---

## 13. Final Decision Record

> **Decision:** Select **Llama 3.2 3B Instruct** as the HalluciDetect base model and use **4-bit NF4 quantization** for the project's current local GPU configuration.
>
> **Basis:** Measured hardware efficiency, generation throughput, tested English/Hindi/Roman Hindi/Hinglish behavior, false-premise handling, and verified logits/hidden-state access.
>
> **Status:** **Base-model selection phase complete.**
>
> **Note:** The optional Qwen3-4B no-thinking control run remains a non-blocking follow-up experiment and does not alter the recorded selection decision.
