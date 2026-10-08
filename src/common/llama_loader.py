"""Canonical Llama loader (Step 4). Shared by Person 1, 2 and 3.

Rule: nobody calls AutoModelForCausalLM / AutoTokenizer.from_pretrained for
the project model anywhere else. Use `load_llama()` so everyone gets the same
quantization, pad token, padding side, revision and seed handling.

    from src.common.llama_loader import load_llama, set_seed
    model, tokenizer = load_llama()
"""
from __future__ import annotations

import os
import warnings
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers import set_seed as _hf_set_seed

from src.common.config import Config, get_config


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_seed(seed: Optional[int] = None, cfg: Optional[Config] = None) -> int:
    """Seed python, numpy and torch (CPU + CUDA). Defaults to cfg.run.seed."""
    cfg = cfg or get_config()
    seed = cfg.run.seed if seed is None else seed
    _hf_set_seed(seed)
    return seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_hf_token(cfg: Optional[Config] = None) -> Optional[str]:
    """Read the HF token from the env var named in config (never stored)."""
    cfg = cfg or get_config()
    return os.environ.get(cfg.run.hf_token_env)


def _check_revision(cfg: Config, allow_unpinned: bool) -> None:
    if cfg.model.revision is None:
        msg = (
            "ModelConfig.revision is None (Llama commit hash not pinned). "
            "Pin it in src/common/config.py before the foundation freeze."
        )
        if not allow_unpinned:
            raise RuntimeError(msg + " Pass allow_unpinned_revision=True only for local smoke tests.")
        warnings.warn(msg)


def build_bnb_config(cfg: Optional[Config] = None) -> BitsAndBytesConfig:
    cfg = cfg or get_config()
    m = cfg.model
    return BitsAndBytesConfig(
        load_in_4bit=m.load_in_4bit,
        bnb_4bit_quant_type=m.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, m.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=m.bnb_4bit_use_double_quant,
    )


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def load_tokenizer(
    cfg: Optional[Config] = None, allow_unpinned_revision: bool = False
) -> PreTrainedTokenizerBase:
    cfg = cfg or get_config()
    _check_revision(cfg, allow_unpinned_revision)
    m = cfg.model
    tok = AutoTokenizer.from_pretrained(
        m.model_id, revision=m.revision, token=get_hf_token(cfg),
        clean_up_tokenization_spaces=False,
    )
    # Llama 3.2 ships without a pad token. Use the dedicated pad token
    # (distinct from EOS, so EOS handling and attention masks stay unambiguous).
    if m.pad_token not in tok.get_vocab():
        raise RuntimeError(f"pad token {m.pad_token!r} not in tokenizer vocab")
    tok.pad_token = m.pad_token
    tok.padding_side = m.padding_side
    return tok
    

# ---------------------------------------------------------------------------
# Model + tokenizer
# ---------------------------------------------------------------------------
def load_llama(
    cfg: Optional[Config] = None,
    allow_unpinned_revision: bool = False,
    seed: bool = True,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the canonical Llama 3.2 3B Instruct (NF4) and its tokenizer."""
    cfg = cfg or get_config()
    if seed:
        set_seed(cfg=cfg)
    m = cfg.model
    tok = load_tokenizer(cfg, allow_unpinned_revision)
    model = AutoModelForCausalLM.from_pretrained(
        m.model_id,
        revision=m.revision,
        token=get_hf_token(cfg),
        quantization_config=build_bnb_config(cfg),
        device_map=m.device_map,
        dtype=getattr(torch, m.bnb_4bit_compute_dtype),
    )
    model.eval()
    model.generation_config.pad_token_id = tok.pad_token_id
    return model, tok


# ---------------------------------------------------------------------------
# Structural verification (used by the smoke test and Step 10)
# ---------------------------------------------------------------------------
@torch.no_grad()
def verify_model(model, tok, cfg: Optional[Config] = None) -> dict:
    """Check the loaded model matches the benchmarked configuration."""
    cfg = cfg or get_config()
    enc = tok("Hello", return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model(**enc, output_hidden_states=True)
    report = {
        "num_hidden_state_entries": len(out.hidden_states),
        "hidden_size": out.hidden_states[-1].shape[-1],
        "vocab_size_logits": out.logits.shape[-1],
        "pad_token": tok.pad_token,
        "pad_token_id": tok.pad_token_id,
        "eos_token_id": tok.eos_token_id,
        "padding_side": tok.padding_side,
        "config_hash": cfg.config_hash(),
    }
    assert report["num_hidden_state_entries"] == cfg.model.num_hidden_state_entries, report
    assert report["hidden_size"] == cfg.model.hidden_size, report
    assert tok.pad_token_id != tok.eos_token_id, "pad token must differ from EOS"
    assert tok.padding_side == cfg.model.padding_side, report
    return report
