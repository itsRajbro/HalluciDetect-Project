"""GPU smoke test for the canonical Llama loader (Step 4).

Run from the repo root (needs GPU + HF_TOKEN):
    python -m scripts.smoke_step4_loader --allow-unpinned     # before the hash is pinned
    python -m scripts.smoke_step4_loader                      # after pinning (strict)

Not collected by pytest on purpose (needs a GPU).
"""
from __future__ import annotations

import argparse
import time

import torch

from src.common.config import get_config
from src.common.llama_loader import load_llama, set_seed, verify_model

QUESTIONS = [
    "What is the capital of France?",
    "Who wrote the novel 'Pride and Prejudice'? Answer briefly.",
]


def chat(tok, q: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
    )


def greedy(model, tok, texts, max_new_tokens):
    # add_special_tokens=False: the chat template already contains <|begin_of_text|>.
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        temperature=None,  # silence Llama generation_config sampling defaults
        top_p=None,
    )
    return tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-unpinned", action="store_true")
    args = ap.parse_args()

    cfg = get_config()
    print("config_hash:", cfg.config_hash())
    print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
    assert torch.cuda.is_available(), "GPU required"

    t0 = time.time()
    model, tok = load_llama(cfg, allow_unpinned_revision=args.allow_unpinned)
    print(f"loaded in {time.time() - t0:.1f}s")

    print("verify:", verify_model(model, tok, cfg))

    texts = [chat(tok, q) for q in QUESTIONS]
    n = cfg.greedy.max_new_tokens

    singles = [greedy(model, tok, [t], n)[0] for t in texts]
    batched = greedy(model, tok, texts, n)
    for q, s, b in zip(QUESTIONS, singles, batched):
        print(f"\nQ: {q}\n  single : {s!r}\n  batched: {b!r}\n  match  : {s == b}")

    # Seeding check: same seed -> same sampled output.
    def sample_once():
        set_seed(cfg=cfg)
        enc = tok(texts[:1], return_tensors="pt", add_special_tokens=False).to(model.device)
        out = model.generate(
            **enc, max_new_tokens=32, do_sample=True,
            temperature=cfg.sampled.temperature, top_p=cfg.sampled.top_p, top_k=cfg.sampled.top_k,
        )
        return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

    a, b = sample_once(), sample_once()
    print("\nseeded sampling reproducible:", a == b)

    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
