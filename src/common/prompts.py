"""Canonical prompts (Step 5). Shared by Person 1, 2 and 3.

Rule: nobody builds Llama prompts anywhere else. Every generation, log-prob
and hidden-state extraction goes through these functions, so all detectors see
exactly the same text.

The Llama 3 chat format is written out explicitly (instead of calling
tokenizer.apply_chat_template) so the frozen text cannot change with a
tokenizer/template update, and so the default "Cutting Knowledge Date / Today
Date" system header is not injected.

Two entry points:
  build_generation_prompt(...)  -> text the model continues (SE sampling, greedy).
  encode_with_answer(...)       -> token ids of prompt + a candidate answer, plus
                                   the exact answer token span (SEP, candidate-aware SE).

Tokenize prompt strings with add_special_tokens=False: <|begin_of_text|> is
already in the text.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from src.common.config import Config, get_config

# Bump this (and PromptConfig.version in config.py) whenever ANY text below changes.
PROMPT_TEMPLATE_VERSION = "v1"

SYSTEM_PROMPT = "You are a helpful assistant. Answer the question in one short, factual sentence."

_BOS = "<|begin_of_text|>"
_EOT = "<|eot_id|>"


def _header(role: str) -> str:
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n"


def _check_version(cfg: Config) -> None:
    if cfg.prompt.version != PROMPT_TEMPLATE_VERSION:
        raise RuntimeError(
            f"config PromptConfig.version={cfg.prompt.version!r} but prompts.py is "
            f"{PROMPT_TEMPLATE_VERSION!r}. Set PromptConfig.version to "
            f"{PROMPT_TEMPLATE_VERSION!r} in config.py (it is part of config_hash)."
        )


def build_user_message(question: str, knowledge: Optional[str], cfg: Optional[Config] = None) -> str:
    cfg = cfg or get_config()
    question = (question or "").strip()
    if not question:
        raise ValueError("empty question")
    if cfg.prompt.include_knowledge:
        knowledge = (knowledge or "").strip()
        if not knowledge:
            raise ValueError("include_knowledge=True but knowledge is empty")
        return f"Knowledge: {knowledge}\n\nQuestion: {question}"
    return f"Question: {question}"


def build_generation_prompt(
    question: str, knowledge: Optional[str] = None, cfg: Optional[Config] = None
) -> str:
    """Full chat prompt ending at the assistant header; the model writes the answer."""
    cfg = cfg or get_config()
    _check_version(cfg)
    user = build_user_message(question, knowledge, cfg)
    return (
        _BOS
        + _header("system") + SYSTEM_PROMPT + _EOT
        + _header("user") + user + _EOT
        + _header("assistant")
    )


def encode_with_answer(
    tokenizer,
    question: str,
    knowledge: Optional[str],
    answer: str,
    cfg: Optional[Config] = None,
) -> Dict[str, object]:
    """Token ids for prompt + candidate answer + <|eot_id|>, with the answer span.

    The prompt and the answer are tokenized separately and concatenated, so the
    answer boundary is exact (the same boundary the model sees when it
    generates its own answer).

    Returns:
        input_ids    list[int]
        answer_start int, index of the first answer token
        answer_end   int, exclusive; input_ids[answer_end] is <|eot_id|>
    """
    cfg = cfg or get_config()
    answer = (answer or "").strip()
    if not answer:
        raise ValueError("empty answer")
    prefix = build_generation_prompt(question, knowledge, cfg)
    prefix_ids: List[int] = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    answer_ids: List[int] = tokenizer(answer, add_special_tokens=False)["input_ids"]
    eot_id = tokenizer.convert_tokens_to_ids(_EOT)
    return {
        "input_ids": prefix_ids + answer_ids + [eot_id],
        "answer_start": len(prefix_ids),
        "answer_end": len(prefix_ids) + len(answer_ids),
    }
