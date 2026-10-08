"""Step 5 tests. The string tests need no GPU and no network.
The tokenizer test is skipped unless the Llama tokenizer can be loaded (needs HF_TOKEN).
"""
from dataclasses import replace

import pytest

from src.common.config import Config, PromptConfig
from src.common.prompts import (
    PROMPT_TEMPLATE_VERSION,
    build_generation_prompt,
    encode_with_answer,
)

Q = "Which country is Paris the capital of?"
K = "Paris is the capital and largest city of France."
A = "Paris is the capital of France."


def cfg(include_knowledge=True, version=PROMPT_TEMPLATE_VERSION) -> Config:
    return Config(prompt=PromptConfig(version=version, include_knowledge=include_knowledge))


def test_ends_with_assistant_header_and_single_bos():
    p = build_generation_prompt(Q, K, cfg())
    assert p.startswith("<|begin_of_text|>")
    assert p.count("<|begin_of_text|>") == 1
    assert p.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


def test_knowledge_flag_controls_content():
    with_k = build_generation_prompt(Q, K, cfg(True))
    without_k = build_generation_prompt(Q, K, cfg(False))
    assert K in with_k and Q in with_k
    assert K not in without_k and Q in without_k


def test_missing_knowledge_raises_when_required():
    with pytest.raises(ValueError):
        build_generation_prompt(Q, "", cfg(True))
    build_generation_prompt(Q, None, cfg(False))  # fine


def test_version_mismatch_raises():
    with pytest.raises(RuntimeError):
        build_generation_prompt(Q, K, cfg(version="v0-unfrozen"))


def test_prompt_is_deterministic_and_strips_whitespace():
    a = build_generation_prompt(Q, K, cfg())
    b = build_generation_prompt(f"  {Q}\n", f"{K}  ", cfg())
    assert a == b


def test_answer_span_with_real_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    except Exception as e:  # no token / no network
        pytest.skip(f"tokenizer unavailable: {e}")
    c = cfg()
    enc = encode_with_answer(tok, Q, K, A, c)
    ids, s, e = enc["input_ids"], enc["answer_start"], enc["answer_end"]
    prefix = tok(build_generation_prompt(Q, K, c), add_special_tokens=False)["input_ids"]
    assert ids[:s] == prefix
    assert tok.decode(ids[s:e]).strip() == A
    assert ids[e] == tok.convert_tokens_to_ids("<|eot_id|>")
    assert ids.count(tok.convert_tokens_to_ids("<|begin_of_text|>")) == 1
