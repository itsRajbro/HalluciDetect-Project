"""Shared generation utility (Step 6). Shared by Person 1, 2 and 3.

Rule: nobody calls model.generate() for the project model anywhere else.
Every greedy answer and every Semantic-Entropy sample goes through here, so
prompt, stop tokens, seeding and log-probs are identical for all detectors.

    from src.common.llama_loader import load_llama
    from src.common.generation import generate_greedy, generate_samples

    model, tok = load_llama()
    g = generate_greedy(model, tok, question, knowledge, sample_id="abc").generations[0]
    s = generate_samples(model, tok, question, knowledge, sample_id="abc")  # SE
    for gen in s.generations: print(gen.text, gen.sum_logprob)

Conventions
-----------
* Prompt: always `build_generation_prompt` (Step 5).
* Stop: generation ends at <|eot_id|> OR the base EOS id. Both are passed to
  generate() explicitly (the checkpoint's own generation_config is not relied on).
* Log-probs: log p(token | prefix) under the RAW model distribution
  (temperature 1, no top-k/top-p truncation), taken from `output_logits`.
  The stop token is included in `token_ids` and `token_logprobs` (it is part of
  the sequence probability). `text` excludes special tokens.
* `truncated=True` means no stop token appeared within max_new_tokens.
* Seeds: sampled chunks are seeded with derive_seed(cfg.run.seed, sample_id,
  first_sample_index_of_chunk), so reruns (and other machines) draw the same
  samples. HF generate() cannot seed individual rows of a batch, so the draw for
  a chunk depends on `cfg.sampled.chunk_size`; that value is part of
  config_hash(). Bit-identical results across different GPU models are not
  guaranteed (bf16 kernels), identical results on the same machine are.
* Greedy decoding is one prompt at a time (no padding), which keeps the
  canonical answer independent of batch composition.
"""
from __future__ import annotations

import gc
import hashlib
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from src.common.config import Config, get_config
from src.common.prompts import build_generation_prompt

_EOT = "<|eot_id|>"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class Generation:
    sample_id: str
    mode: str                      # "greedy" | "sampled"
    sample_index: int              # 0 for greedy; 0..n-1 for sampled
    seed: Optional[int]            # chunk seed for sampled, None for greedy
    text: str
    token_ids: List[int]           # includes the stop token if one was produced
    token_logprobs: List[float]    # same length as token_ids
    num_tokens: int
    sum_logprob: float
    mean_logprob: float
    truncated: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenStats:
    wall_time_s: float
    num_sequences: int
    new_tokens_total: int
    prompt_num_tokens: int
    peak_gpu_mem_mb: Optional[float]
    config_hash: str
    prompt_version: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationOutput:
    generations: List[Generation]
    stats: GenStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def derive_seed(base_seed: int, sample_id, index: int) -> int:
    """Deterministic 32-bit seed from (base seed, sample id, sample index)."""
    key = f"{base_seed}:{sample_id}:{index}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def stop_token_ids(tok) -> List[int]:
    """[<|eot_id|>, eos] (deduplicated). Fails loudly if <|eot_id|> is missing."""
    eot = tok.convert_tokens_to_ids(_EOT)
    unk = getattr(tok, "unk_token_id", None)
    if eot is None or (unk is not None and eot == unk):
        raise RuntimeError(f"{_EOT} not found in tokenizer vocab")
    ids = [eot]
    if tok.eos_token_id is not None and tok.eos_token_id not in ids:
        ids.append(tok.eos_token_id)
    return ids


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def _generate_chunk(model, tok, enc, gen_kwargs: dict, stop_ids: Sequence[int]):
    """One model.generate() call -> list of (token_ids, logprobs, truncated)."""
    out = model.generate(
        **enc,
        **gen_kwargs,
        eos_token_id=list(stop_ids),
        pad_token_id=tok.pad_token_id,
        return_dict_in_generate=True,
        output_logits=True,
    )
    prompt_len = enc["input_ids"].shape[1]
    new = out.sequences[:, prompt_len:]                 # (N, T)
    T = new.shape[1]
    if len(out.logits) != T:
        raise RuntimeError(f"logits steps ({len(out.logits)}) != new tokens ({T})")
    # raw-distribution log-prob of each generated token
    lps = torch.stack(
        [
            torch.log_softmax(step.float(), dim=-1)
            .gather(-1, new[:, t : t + 1].to(step.device))
            .squeeze(-1)
            for t, step in enumerate(out.logits)
        ],
        dim=1,
    )                                                   # (N, T)

    stop_set = set(int(s) for s in stop_ids)
    rows: List[Tuple[List[int], List[float], bool]] = []
    for i in range(new.shape[0]):
        ids = new[i].tolist()
        end = None
        for j, t in enumerate(ids):
            if t in stop_set:
                end = j + 1                             # keep the stop token
                break
        truncated = end is None
        if end is None:
            end = len(ids)
        rows.append((ids[:end], lps[i, :end].tolist(), truncated))
    return rows


def _make_generation(tok, sample_id, mode, index, seed, ids, lps, truncated) -> Generation:
    s = float(sum(lps))
    return Generation(
        sample_id=str(sample_id),
        mode=mode,
        sample_index=index,
        seed=seed,
        text=tok.decode(ids, skip_special_tokens=True).strip(),
        token_ids=list(ids),
        token_logprobs=[float(x) for x in lps],
        num_tokens=len(ids),
        sum_logprob=s,
        mean_logprob=s / max(len(ids), 1),
        truncated=truncated,
    )


def _encode_prompt(model, tok, question, knowledge, cfg):
    prompt = build_generation_prompt(question, knowledge, cfg)
    enc = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    return enc


def _stats(t0, generations, prompt_len, cfg) -> GenStats:
    _cuda_sync()
    peak = (
        torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
    )
    return GenStats(
        wall_time_s=time.perf_counter() - t0,
        num_sequences=len(generations),
        new_tokens_total=sum(g.num_tokens for g in generations),
        prompt_num_tokens=prompt_len,
        peak_gpu_mem_mb=peak,
        config_hash=cfg.config_hash(),
        prompt_version=cfg.prompt.version,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_greedy(
    model,
    tok,
    question: str,
    knowledge: Optional[str] = None,
    sample_id="",
    cfg: Optional[Config] = None,
    clear_cache: bool = True,
) -> GenerationOutput:
    """The model's single deterministic answer (greedy decoding)."""
    cfg = cfg or get_config()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _cuda_sync()
    t0 = time.perf_counter()

    enc = _encode_prompt(model, tok, question, knowledge, cfg)
    g = cfg.greedy
    gen_kwargs = dict(
        max_new_tokens=g.max_new_tokens,
        do_sample=False,
        num_beams=g.num_beams,
        temperature=None,   # override the checkpoint's sampling defaults
        top_p=None,
        top_k=None,
    )
    (ids, lps, trunc), = _generate_chunk(model, tok, enc, gen_kwargs, stop_token_ids(tok))
    gens = [_make_generation(tok, sample_id, "greedy", 0, None, ids, lps, trunc)]
    stats = _stats(t0, gens, enc["input_ids"].shape[1], cfg)
    if clear_cache:
        _cleanup()
    return GenerationOutput(gens, stats)


def generate_samples(
    model,
    tok,
    question: str,
    knowledge: Optional[str] = None,
    sample_id="",
    num_samples: Optional[int] = None,
    cfg: Optional[Config] = None,
    clear_cache: bool = True,
) -> GenerationOutput:
    """`num_samples` stochastic answers for one input (Semantic Entropy).

    Sampling settings come from cfg.sampled. Samples are drawn in chunks of
    cfg.sampled.chunk_size to bound GPU memory.
    """
    cfg = cfg or get_config()
    s = cfg.sampled
    n = s.num_samples if num_samples is None else num_samples
    if n < 1:
        raise ValueError("num_samples must be >= 1")
    if s.chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _cuda_sync()
    t0 = time.perf_counter()

    enc = _encode_prompt(model, tok, question, knowledge, cfg)
    stop_ids = stop_token_ids(tok)
    gens: List[Generation] = []
    for start in range(0, n, s.chunk_size):
        size = min(s.chunk_size, n - start)
        seed = derive_seed(cfg.run.seed, sample_id, start)
        torch.manual_seed(seed)                         # also seeds CUDA
        gen_kwargs = dict(
            max_new_tokens=s.max_new_tokens,
            do_sample=True,
            temperature=s.temperature,
            top_p=s.top_p,
            top_k=s.top_k,
            num_return_sequences=size,
        )
        rows = _generate_chunk(model, tok, enc, gen_kwargs, stop_ids)
        for k, (ids, lps, trunc) in enumerate(rows):
            gens.append(
                _make_generation(tok, sample_id, "sampled", start + k, seed, ids, lps, trunc)
            )
    stats = _stats(t0, gens, enc["input_ids"].shape[1], cfg)
    if clear_cache:
        _cleanup()
    return GenerationOutput(gens, stats)
