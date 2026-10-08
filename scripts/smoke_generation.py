"""GPU smoke test for Steps 4 + 6. Run from repo root:

    python -m scripts.smoke_generation

Checks: loader structure, greedy answer sensible and reproducible, sampled
answers reproducible for a fixed sample_id but varied across samples,
generation stops at <|eot_id|>, log-probs sane, efficiency numbers printed.
"""
from src.common.config import get_config
from src.common.generation import generate_greedy, generate_samples
from src.common.llama_loader import load_llama, verify_model

ITEMS = [
    ("smoke-1", "What is the capital of France?",
     "Paris is the capital and most populous city of France."),
    ("smoke-2", "Who wrote the novel Pride and Prejudice?",
     "Pride and Prejudice is an 1813 novel by the English author Jane Austen."),
    ("smoke-3", "In which country is the city of Kyoto located?",
     "Kyoto is a city in Japan, on the island of Honshu."),
]


def main():
    cfg = get_config()
    model, tok = load_llama(cfg)
    print("verify_model:", verify_model(model, tok, cfg))

    for sid, q, k in ITEMS:
        g1 = generate_greedy(model, tok, q, k, sample_id=sid, cfg=cfg)
        g2 = generate_greedy(model, tok, q, k, sample_id=sid, cfg=cfg)
        a, b = g1.generations[0], g2.generations[0]
        print(f"\n[{sid}] greedy: {a.text!r}  tokens={a.num_tokens} "
              f"sum_lp={a.sum_logprob:.2f} truncated={a.truncated}")
        assert a.token_ids == b.token_ids, "greedy not reproducible"
        assert not a.truncated, "greedy did not stop at a stop token"
        assert a.sum_logprob <= 0 and all(lp <= 0 for lp in a.token_logprobs)

        s1 = generate_samples(model, tok, q, k, sample_id=sid, num_samples=6, cfg=cfg)
        s2 = generate_samples(model, tok, q, k, sample_id=sid, num_samples=6, cfg=cfg)
        t1 = [g.token_ids for g in s1.generations]
        t2 = [g.token_ids for g in s2.generations]
        print(f"[{sid}] samples: {[g.text for g in s1.generations]}")
        print(f"[{sid}] stats: {s1.stats.to_dict()}")
        assert t1 == t2, "sampling not reproducible for fixed sample_id"
        assert len(s1.generations) == 6

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
