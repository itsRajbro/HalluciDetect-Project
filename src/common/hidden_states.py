"""Shared hidden-state extraction (Step 7). Shared by Person 1, 2 and 3.

Rule: nobody runs their own forward pass to get hidden states of
(question, knowledge, candidate answer). Use `extract_hidden_states()`.

    from src.common.llama_loader import load_llama
    from src.common.hidden_states import extract_hidden_states

    model, tok = load_llama()
    hs = extract_hidden_states(model, tok, question, knowledge, answer, sample_id="abc")
    hs.features["last_answer"]      # tensor (num_layers, 3072), CPU
    hs.answer_logprobs              # log p of each answer token (+ <|eot_id|>)

What is computed
----------------
One teacher-forced forward pass over the Step 5 sequence
    prompt + candidate answer + <|eot_id|>
(`prompts.encode_with_answer`, so the answer span is exact and identical to what
the model sees when it generates). From the token-level hidden states we pool
answer-level vectors, one per requested pooling:

    tbg          last prompt token (state BEFORE the answer; independent of the
                 candidate answer, depends only on question [+ knowledge])
    last_answer  last answer token
    mean_answer  mean over the answer tokens
    eot          the <|eot_id|> token placed right after the answer

Layer indexing: indices into `hidden_states` as returned by HF, i.e. 0 =
embedding output, 1..28 = decoder layers, 28 = the LAST entry (after the final
RMSNorm). 29 entries total for Llama 3.2 3B.

Length confound
---------------
Answer length is strongly tied to the label in this dataset (hallucinated
answers are ~4x longer). `mean_answer` and `last_answer` can both encode
length. `answer_num_tokens` is returned with every extraction so downstream
code can control for it (e.g. length-stratified evaluation, or including it as
a covariate / checking probe performance on the length-controlled test set).

Single sequence per forward pass (no padding) on purpose: results do not depend
on batch composition.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from src.common.config import Config, get_config
from src.common.prompts import encode_with_answer

POOLINGS = ("tbg", "last_answer", "mean_answer", "eot")


@dataclass
class HiddenStates:
    sample_id: str
    layers: List[int]                      # hidden_states indices, in feature row order
    poolings: List[str]
    features: Dict[str, torch.Tensor]      # pooling -> (len(layers), hidden_size), CPU
    answer_start: int                      # index of first answer token in the sequence
    answer_end: int                        # exclusive; token at answer_end is <|eot_id|>
    answer_num_tokens: int
    total_tokens: int
    answer_logprobs: Optional[List[float]]  # len = answer_num_tokens + 1 (includes <|eot_id|>)
    config_hash: str
    prompt_version: str

    def metadata(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "features"}
        return d


def _resolve_layers(layers: Optional[Sequence[int]], n_entries: int) -> List[int]:
    if layers is None:
        return list(range(n_entries))
    out = [int(l) for l in layers]
    bad = [l for l in out if not 0 <= l < n_entries]
    if bad:
        raise ValueError(f"layer indices {bad} out of range [0, {n_entries - 1}]")
    return out


def _resolve_poolings(poolings: Optional[Sequence[str]]) -> List[str]:
    out = list(poolings)
    bad = [p for p in out if p not in POOLINGS]
    if bad or not out:
        raise ValueError(f"poolings must be a non-empty subset of {POOLINGS}; got {out}")
    return out


@torch.no_grad()
def extract_hidden_states(
    model,
    tok,
    question: str,
    knowledge: Optional[str],
    answer: str,
    sample_id="",
    layers: Optional[Sequence[int]] = None,
    poolings: Optional[Sequence[str]] = None,
    with_logprobs: bool = True,
    cfg: Optional[Config] = None,
    clear_cache: bool = False,
) -> HiddenStates:
    """Hidden-state features of one (question, knowledge, candidate answer)."""
    cfg = cfg or get_config()
    layers = cfg.hidden.layers if layers is None else layers
    poolings = _resolve_poolings(cfg.hidden.poolings if poolings is None else poolings)
    store_dtype = getattr(torch, cfg.hidden.store_dtype)

    enc = encode_with_answer(tok, question, knowledge, answer, cfg)
    ids: List[int] = enc["input_ids"]
    a0, a1 = enc["answer_start"], enc["answer_end"]
    input_ids = torch.tensor([ids], device=model.device)

    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    layer_idx = _resolve_layers(layers, len(hs))

    # (L, S, H) in float32 for pooling accuracy
    stack = torch.stack([hs[l][0] for l in layer_idx]).float()

    feats: Dict[str, torch.Tensor] = {}
    for p in poolings:
        if p == "tbg":
            v = stack[:, a0 - 1]
        elif p == "last_answer":
            v = stack[:, a1 - 1]
        elif p == "mean_answer":
            v = stack[:, a0:a1].mean(dim=1)
        else:  # "eot"
            v = stack[:, a1]
        feats[p] = v.to(store_dtype).cpu()

    logprobs = None
    if with_logprobs:
        # logits at position i predict token i+1: positions a0-1 .. a1-1 predict
        # answer tokens a0..a1-1 and the <|eot_id|> at a1.
        logits = out.logits[0, a0 - 1 : a1].float()
        targets = torch.tensor(ids[a0 : a1 + 1], device=logits.device)
        logprobs = (
            torch.log_softmax(logits, dim=-1)
            .gather(-1, targets.unsqueeze(-1))
            .squeeze(-1)
            .cpu()
            .tolist()
        )

    result = HiddenStates(
        sample_id=str(sample_id),
        layers=layer_idx,
        poolings=poolings,
        features=feats,
        answer_start=a0,
        answer_end=a1,
        answer_num_tokens=a1 - a0,
        total_tokens=len(ids),
        answer_logprobs=logprobs,
        config_hash=cfg.config_hash(),
        prompt_version=cfg.prompt.version,
    )
    del out, hs, stack
    if clear_cache:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def save_hidden_states(items: Sequence[HiddenStates], path) -> None:
    """Save a list of HiddenStates (tensors + metadata) with torch.save."""
    payload = [{"meta": it.metadata(), "features": it.features} for it in items]
    torch.save(payload, path)


def load_hidden_states(path) -> List[HiddenStates]:
    payload = torch.load(path, map_location="cpu")
    return [HiddenStates(features=p["features"], **p["meta"]) for p in payload]
