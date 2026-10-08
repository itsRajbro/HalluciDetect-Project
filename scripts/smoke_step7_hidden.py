"""GPU smoke test for Step 7 (hidden-state extraction). Run from repo root:

    python -m scripts.smoke_step7_hidden

Checks: feature shapes, finiteness, run-to-run determinism, pooling modes differ,
and consistency with Step 6 (teacher-forced log-probs of the greedy answer should
match the log-probs recorded during generation).
"""
import torch

from src.common.config import get_config
from src.common.generation import generate_greedy
from src.common.hidden_states import extract_hidden_states
from src.common.llama_loader import load_llama

Q = "What is the capital of France?"
K = "Paris is the capital and most populous city of France."
FACTUAL = "Paris"
HALLUCINATED = "The capital of France is Marseille, a port city on the Mediterranean coast."


def main():
    cfg = get_config()
    model, tok = load_llama(cfg)
    n_entries = cfg.model.num_hidden_state_entries
    hid = cfg.model.hidden_size

    h1 = extract_hidden_states(model, tok, Q, K, FACTUAL, sample_id="f", cfg=cfg)
    h2 = extract_hidden_states(model, tok, Q, K, FACTUAL, sample_id="f", cfg=cfg)
    hh = extract_hidden_states(model, tok, Q, K, HALLUCINATED, sample_id="h", cfg=cfg)

    for p in h1.poolings:
        f = h1.features[p]
        assert f.shape == (n_entries, hid), (p, f.shape)
        assert torch.isfinite(f).all(), f"non-finite values in {p}"
        assert torch.equal(f, h2.features[p]), f"{p} not reproducible"
    print("shapes OK, finite, reproducible")

    # pooling modes should not be identical for a multi-token answer
    assert not torch.equal(hh.features["last_answer"], hh.features["mean_answer"])
    assert not torch.equal(hh.features["last_answer"], hh.features["eot"])
    # the tbg state does not see the answer: same prompt -> same tbg state
    # (bf16 + different sequence lengths -> tiny numeric noise, so compare relatively)
    a, b = h1.features["tbg"], hh.features["tbg"]
    rel = ((a - b).abs().max() / a.abs().max()).item()
    print(f"tbg relative diff between two different answers (should be ~0): {rel:.4g}")
    assert rel < 0.02
    print("answer tokens:", h1.answer_num_tokens, "vs", hh.answer_num_tokens)

    # Consistency with Step 6: greedy answer log-probs vs teacher-forced log-probs
    g = generate_greedy(model, tok, Q, K, sample_id="g", cfg=cfg).generations[0]
    hg = extract_hidden_states(model, tok, Q, K, g.text, sample_id="g", cfg=cfg)
    print(f"greedy answer: {g.text!r}")
    if len(hg.answer_logprobs) == len(g.token_logprobs):
        diff = max(abs(a - b) for a, b in zip(hg.answer_logprobs, g.token_logprobs))
        print(f"max |teacher-forced - generation| logprob diff: {diff:.4f}")
        assert diff < 0.5
    else:
        print("token counts differ (re-tokenization); skipping logprob comparison:",
              len(hg.answer_logprobs), len(g.token_logprobs))

    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
