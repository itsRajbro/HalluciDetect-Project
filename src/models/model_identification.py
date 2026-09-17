"""
HalluciDetect — Model Identification Benchmark
================================================

Reusable benchmark script for the base-model identification phase.

Usage:
    python src/models/model_identification.py --model Qwen/Qwen3-4B
    python src/models/model_identification.py --model google/gemma-3-4b-it
    python src/models/model_identification.py --model meta-llama/Llama-3.2-3B-Instruct

Notes:
    - Designed for a single local GPU (target: RTX 3050 6GB VRAM, ~16GB RAM).
    - Uses a fixed controlled prompt set (English / Hindi / Roman Hindi /
      Hinglish / uncertainty) and greedy (non-sampling) decoding so results
      are comparable across candidates (see Rule 6 in project notes).
    - Every phase (load / generate / logits / hidden states / resources) is
      wrapped so that a failure in one phase does not stop the others, and
      failures are recorded in the result record rather than discarded
      (Rule 8: failures are useful evidence).
    - Writes one JSON record per run to results/model_identification/.
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Controlled prompt set (Section 11 of the project spec)
# ---------------------------------------------------------------------------

PROMPTS = {
    "english": "What is the capital of France? Answer in one sentence.",
    "hindi": "भारत की राजधानी क्या है? एक वाक्य में उत्तर दें।",
    "roman_hindi": "Bharat ki rajdhani kya hai? Ek vakya mein uttar dein.",
    "hinglish": "India ki capital kya hai? Please answer in one short sentence.",
    "uncertainty": "Who was the first person to walk on Mars? If the premise is false, say so.",
}

# Fixed, consistent generation settings for every candidate (Rule 6).
GEN_KWARGS = dict(
    max_new_tokens=128,
    do_sample=False,
    num_beams=1,
    temperature=None,
    top_p=None,
    top_k=None,
)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def sanitize_model_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace(" ", "_")


def get_ram_mb():
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 2)
    except Exception:
        return None


def get_vram_mb(torch):
    try:
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    except Exception:
        pass
    return None


def build_result_skeleton(model_id, revision):
    return {
        "model_id": model_id,
        "model_revision": revision,
        "load_status": "not_started",
        "dtype": None,
        "model_class_used": None,
        "device": None,
        "load_time_sec": None,
        "peak_vram_mb": None,
        "ram_mb": None,
        "generation_time_sec": None,
        "english_output": None,
        "hindi_output": None,
        "roman_hindi_output": None,
        "hinglish_output": None,
        "uncertainty_output": None,
        "logits_available": None,
        "logits_shape": None,
        "logits_dtype": None,
        "hidden_states_available": None,
        "hidden_states_shape": None,
        "hidden_state_layers": None,
        "hidden_states_dtype": None,
        "notes": [],
        "quantization": "none",
        "timestamp_utc": now_iso(),
        "generation_settings": GEN_KWARGS,
    }


def add_note(result, msg):
    print(f"[NOTE] {msg}")
    result["notes"].append(msg)


def run_generation(model, tokenizer, prompt, device, use_chat_template, max_new_tokens, disable_thinking):
    """Run one generation and return the exact decoded text (new tokens only).

    Note: we get the chat-formatted *string* from apply_chat_template
    (tokenize=False) and then tokenize it with a normal tokenizer() call.
    This is deliberate — depending on the transformers/tokenizer version,
    apply_chat_template(..., return_tensors="pt") can silently return a
    plain Python list instead of a tensor, which breaks any downstream
    tensor ops (this is what caused the earlier 'bool has no attribute
    long' failures). A plain tokenizer() call reliably returns a tensor
    plus a correct attention_mask.

    Some chat models (e.g. Qwen3) default to an extended "thinking" mode
    that wraps a reasoning trace in <think>...</think> before the actual
    answer. If disable_thinking is True, we try to pass
    enable_thinking=False to apply_chat_template; if the tokenizer's
    template doesn't support that kwarg, we fall back silently.
    """
    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        try:
            if disable_thinking:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            else:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except TypeError:
            # Template doesn't accept enable_thinking (or another unsupported kwarg) — fall back.
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    else:
        text = prompt

    enc = tokenizer(text, return_tensors="pt").to(device)
    inputs = dict(enc)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    import torch
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=GEN_KWARGS["do_sample"],
            num_beams=GEN_KWARGS["num_beams"],
            pad_token_id=pad_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    text_out = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text_out, len(new_tokens)


def check_truncated_thinking(text):
    """Return True if a <think> block was opened but never closed — meaning
    the token budget ran out mid-reasoning and no final answer was produced."""
    if text is None:
        return False
    return "<think>" in text and "</think>" not in text


def main():
    parser = argparse.ArgumentParser(description="HalluciDetect model-identification benchmark")
    parser.add_argument("--model", required=True, help="Hugging Face model id, e.g. Qwen/Qwen3-4B")
    parser.add_argument("--revision", default=None, help="Optional model revision/commit hash")
    parser.add_argument("--output-dir", default="results/model_identification",
                         help="Directory to write the JSON result record")
    parser.add_argument("--hf-token", default=None,
                         help="Optional HF token for gated models (or set HF_TOKEN env var)")
    parser.add_argument("--no-chat-template", action="store_true",
                         help="Force raw prompting instead of tokenizer.apply_chat_template")
    parser.add_argument("--load-in-4bit", action="store_true",
                         help="Load with bitsandbytes 4-bit quantization (requires `pip install bitsandbytes`). "
                              "Use this if bf16/fp16 loading overflows VRAM into shared/system memory.")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                         help="Max new tokens per generation. Reasoning models (e.g. Qwen3 with thinking "
                              "enabled) may need 512+ to get past the <think> block to an actual answer.")
    parser.add_argument("--no-thinking", action="store_true",
                         help="Attempt to disable extended 'thinking' mode via enable_thinking=False in "
                              "apply_chat_template (supported by some models, e.g. Qwen3). Ignored if the "
                              "tokenizer's template doesn't support this kwarg.")
    args = parser.parse_args()

    result = build_result_skeleton(args.model, args.revision)

    # -------------------------------------------------------------------
    # Phase A: imports + environment
    # -------------------------------------------------------------------
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        result["load_status"] = "failed_import"
        add_note(result, f"Import failure: {e}")
        save_result(result, args.output_dir)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result["device"] = device
    if device == "cpu":
        add_note(result, "CUDA not available in this environment — running on CPU (not representative of RTX 3050 target).")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # -------------------------------------------------------------------
    # Phase B: model + tokenizer loading
    # -------------------------------------------------------------------
    tokenizer = None
    model = None
    load_start = time.time()
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        tokenizer = AutoTokenizer.from_pretrained(
            args.model, revision=args.revision, token=hf_token, trust_remote_code=True
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        quant_config = None
        if args.load_in_4bit:
            if device != "cuda":
                add_note(result, "--load-in-4bit requested but no CUDA device available; ignoring flag.")
            else:
                from transformers import BitsAndBytesConfig
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )

        model_load_kwargs = dict(
            revision=args.revision,
            token=hf_token,
            dtype=None if quant_config else dtype,
            quantization_config=quant_config,
            device_map=device if device == "cuda" else None,
            trust_remote_code=True,
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, **model_load_kwargs)
            result["model_class_used"] = "AutoModelForCausalLM"
        except ValueError as e:
            # Some checkpoints (e.g. Gemma 3 4B/12B/27B, which are multimodal
            # vision+text models) are not registered under AutoModelForCausalLM.
            # Fall back to AutoModelForImageTextToText, which covers these
            # conditional-generation architectures; text-only generation still
            # works normally without supplying pixel_values.
            if "Unrecognized configuration class" in str(e) or "AutoModelForCausalLM" in str(e):
                add_note(
                    result,
                    "AutoModelForCausalLM did not recognize this checkpoint (likely a multimodal "
                    "vision+text architecture) — retrying with AutoModelForImageTextToText.",
                )
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(args.model, **model_load_kwargs)
                result["model_class_used"] = "AutoModelForImageTextToText"
            else:
                raise
        if device == "cpu":
            model.to(device)
        model.eval()

        result["dtype"] = str(next(model.parameters()).dtype)
        result["load_status"] = "success"
        result["quantization"] = "4bit-nf4" if quant_config else "none"
    except Exception as e:
        result["load_status"] = "failed"
        add_note(result, f"Load failure: {e}")
        add_note(result, traceback.format_exc(limit=3))
        save_result(result, args.output_dir)
        return
    finally:
        result["load_time_sec"] = round(time.time() - load_start, 2)

    has_chat_template = getattr(tokenizer, "chat_template", None) is not None
    use_chat_template = has_chat_template and not args.no_chat_template
    if not has_chat_template:
        add_note(result, "No chat template found on tokenizer — using raw prompts instead.")

    # -------------------------------------------------------------------
    # Phase C: generation (English / Hindi / Roman Hindi / Hinglish / uncertainty)
    # -------------------------------------------------------------------
    gen_start = time.time()
    total_new_tokens = 0
    output_map = {
        "english": "english_output",
        "hindi": "hindi_output",
        "roman_hindi": "roman_hindi_output",
        "hinglish": "hinglish_output",
        "uncertainty": "uncertainty_output",
    }
    for key, prompt in PROMPTS.items():
        field = output_map[key]
        try:
            text, n_tokens = run_generation(
                model, tokenizer, prompt, device, use_chat_template,
                max_new_tokens=args.max_new_tokens,
                disable_thinking=args.no_thinking,
            )
            result[field] = text
            total_new_tokens += n_tokens
            if check_truncated_thinking(text):
                add_note(
                    result,
                    f"Prompt '{key}': hit max_new_tokens ({args.max_new_tokens}) while still inside a "
                    "<think> block — no final answer was produced. Re-run with a higher --max-new-tokens "
                    "and/or --no-thinking.",
                )
        except Exception as e:
            result[field] = None
            add_note(result, f"Generation failed for prompt '{key}': {e}")
    result["generation_time_sec"] = round(time.time() - gen_start, 2)
    result["generation_settings"]["max_new_tokens"] = args.max_new_tokens
    result["generation_settings"]["thinking_disabled_requested"] = args.no_thinking
    if total_new_tokens > 0 and result["generation_time_sec"]:
        result["tokens_per_sec_approx"] = round(total_new_tokens / result["generation_time_sec"], 2)

    # -------------------------------------------------------------------
    # Phase D: logits availability
    # -------------------------------------------------------------------
    try:
        probe_text = PROMPTS["english"]
        enc = tokenizer(probe_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        if hasattr(out, "logits") and out.logits is not None:
            result["logits_available"] = True
            result["logits_shape"] = list(out.logits.shape)
            result["logits_dtype"] = str(out.logits.dtype)
        else:
            result["logits_available"] = False
            add_note(result, "Model forward pass returned no logits attribute.")
    except Exception as e:
        result["logits_available"] = False
        add_note(result, f"Logits probe failed: {e}")

    # -------------------------------------------------------------------
    # Phase E: hidden-state availability
    # -------------------------------------------------------------------
    try:
        enc = tokenizer(PROMPTS["english"], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs = getattr(out, "hidden_states", None)
        if hs is not None:
            result["hidden_states_available"] = True
            result["hidden_state_layers"] = len(hs)
            result["hidden_states_shape"] = list(hs[-1].shape)
            result["hidden_states_dtype"] = str(hs[-1].dtype)
        else:
            result["hidden_states_available"] = False
            add_note(result, "output_hidden_states=True returned no hidden_states.")
    except Exception as e:
        result["hidden_states_available"] = False
        add_note(result, f"Hidden-states probe failed: {e}")

    # -------------------------------------------------------------------
    # Phase F: resource measurements
    # -------------------------------------------------------------------
    result["peak_vram_mb"] = get_vram_mb(torch)
    result["ram_mb"] = get_ram_mb()

    # Sanity check: flag likely VRAM overflow into shared/system memory.
    # torch.cuda.max_memory_allocated() reports allocations Torch believes are
    # on-GPU, but on Windows, CUDA can silently spill into "shared GPU memory"
    # (system RAM) when a card's physical VRAM is exceeded. A reported value
    # close to or above the card's physical VRAM, paired with an unusually
    # long load/generation time, is a strong signal of this — not a genuine
    # in-VRAM measurement.
    vram = result["peak_vram_mb"]
    if vram is not None and vram > 5800:
        add_note(
            result,
            f"peak_vram_mb ({vram:.0f} MB) is at or above typical 6GB-card physical capacity "
            "(~6144 MB). If load_time_sec was also unusually high, this likely reflects "
            "overflow into shared/system memory rather than a clean in-VRAM fit. Consider "
            "re-running with --load-in-4bit for an honest fit check on 6GB VRAM.",
        )

    tok_per_sec = result.get("tokens_per_sec_approx")
    if device == "cuda" and tok_per_sec is not None and tok_per_sec < 8:
        vram_is_high = vram is not None and vram > 5800
        if vram_is_high:
            add_note(
                result,
                f"Approx generation speed ({tok_per_sec} tok/s) is unusually slow for a GPU-resident "
                f"model of this size, and peak_vram_mb ({vram:.0f} MB) is at/above typical 6GB-card "
                "capacity. Together this is consistent with VRAM overflow into shared/system memory "
                "rather than a clean in-VRAM fit.",
            )
        else:
            vram_str = f"{vram:.0f} MB" if vram is not None else "unavailable"
            add_note(
                result,
                f"Approx generation speed ({tok_per_sec} tok/s) is slower than a typical GPU-resident "
                f"model of this size, but peak_vram_mb ({vram_str}) is well within capacity — this "
                "does NOT look like VRAM overflow. More likely explanations: quantization/dequantization "
                "compute overhead (common with 4-bit on older GPU architectures), thermal throttling, "
                "or other processes competing for the GPU. Not necessarily a red flag for hardware fit.",
            )

    save_result(result, args.output_dir)

    # cleanup
    try:
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass


def save_result(result, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{sanitize_model_name(result['model_id'])}.json"
    path = os.path.join(output_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n=== Result saved to {path} ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()