"""Step 6 tests: logic of src/common/generation.py with a fake model (no GPU)."""
import dataclasses
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.common import generation as G
from src.common.config import get_config

V, PAD, EOT, EOS = 30, 0, 3, 4


class Enc(dict):
    def to(self, device):
        return self


class FakeTok:
    pad_token_id = PAD
    eos_token_id = EOS
    unk_token_id = None

    def convert_tokens_to_ids(self, t):
        return EOT if t == "<|eot_id|>" else None

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [5 + (ord(c) % 20) for c in text]
        return Enc(
            input_ids=torch.tensor([ids]),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long),
        )

    def decode(self, ids, skip_special_tokens=True):
        return "".join(
            chr(97 + i) for i in ids if not (skip_special_tokens and i in (PAD, EOT, EOS))
        )


class FakeModel:
    """Returns scripted rows if given, else random tokens (using the global RNG)."""

    device = torch.device("cpu")

    def __init__(self, script=None):
        self.script = script
        self.calls = []
        self.last_logits = None

    def generate(self, input_ids, attention_mask, **kw):
        self.calls.append(kw)
        n = kw.get("num_return_sequences", 1)
        if self.script is not None:
            rows = self.script.pop(0)
            T = max(len(r) for r in rows)
            toks = torch.full((len(rows), T), PAD)
            for i, r in enumerate(rows):
                toks[i, : len(r)] = torch.tensor(r)
        else:
            T = 8
            toks = torch.randint(0, V, (n, T))
            for i in range(n):
                for j in range(T):
                    if int(toks[i, j]) in (EOT, EOS):
                        toks[i, j + 1 :] = PAD
                        break
        N = toks.shape[0]
        logits = tuple(torch.randn(N, V) for _ in range(T))
        self.last_logits = logits
        seq = torch.cat([input_ids.expand(N, -1), toks], dim=1)
        return SimpleNamespace(sequences=seq, logits=logits)


def _cfg(chunk_size=3, num_samples=7):
    cfg = get_config()
    return dataclasses.replace(
        cfg,
        sampled=dataclasses.replace(cfg.sampled, chunk_size=chunk_size, num_samples=num_samples),
    )


Q, K = "who?", "some fact"


def test_greedy_stops_at_eot_and_logprobs_match():
    m = FakeModel(script=[[[10, 11, EOT]]])
    out = G.generate_greedy(m, FakeTok(), Q, K, sample_id="a")
    g = out.generations[0]
    assert g.token_ids == [10, 11, EOT] and g.text == "kl"
    assert g.num_tokens == 3 and not g.truncated and g.mode == "greedy" and g.seed is None
    for t, tid in enumerate(g.token_ids):
        expected = torch.log_softmax(m.last_logits[t][0].float(), -1)[tid].item()
        assert g.token_logprobs[t] == pytest.approx(expected, abs=1e-5)
    assert g.sum_logprob == pytest.approx(sum(g.token_logprobs))
    assert g.mean_logprob == pytest.approx(g.sum_logprob / 3)


def test_greedy_truncated_when_no_stop_token():
    m = FakeModel(script=[[[10, 11, 12, 13]]])
    g = G.generate_greedy(m, FakeTok(), Q, K).generations[0]
    assert g.truncated and g.num_tokens == 4


def test_base_eos_also_stops():
    m = FakeModel(script=[[[10, EOS]]])
    g = G.generate_greedy(m, FakeTok(), Q, K).generations[0]
    assert g.token_ids == [10, EOS] and not g.truncated


def test_greedy_kwargs():
    m = FakeModel(script=[[[10, EOT]]])
    G.generate_greedy(m, FakeTok(), Q, K)
    kw = m.calls[0]
    assert kw["do_sample"] is False and kw["num_beams"] == 1
    assert kw["temperature"] is None and kw["top_p"] is None and kw["top_k"] is None
    assert set(kw["eos_token_id"]) == {EOT, EOS}
    assert kw["pad_token_id"] == PAD and kw["output_logits"] is True
    assert kw.get("num_return_sequences", 1) == 1


def test_sampled_chunking_and_kwargs():
    cfg = _cfg(chunk_size=3, num_samples=7)
    m = FakeModel()
    out = G.generate_samples(m, FakeTok(), Q, K, sample_id="a", cfg=cfg)
    assert [c["num_return_sequences"] for c in m.calls] == [3, 3, 1]
    gens = out.generations
    assert len(gens) == 7 and [g.sample_index for g in gens] == list(range(7))
    assert len({g.seed for g in gens}) == 3            # one seed per chunk
    kw = m.calls[0]
    assert kw["do_sample"] is True and kw["temperature"] == cfg.sampled.temperature
    assert kw["top_p"] == cfg.sampled.top_p and kw["top_k"] == cfg.sampled.top_k
    assert out.stats.num_sequences == 7
    assert out.stats.new_tokens_total == sum(g.num_tokens for g in gens)
    assert out.stats.config_hash == cfg.config_hash()


def test_mixed_lengths_in_one_chunk():
    cfg = _cfg(chunk_size=2, num_samples=2)
    m = FakeModel(script=[[[10, EOT], [10, 11, 12, EOT]]])
    gens = G.generate_samples(m, FakeTok(), Q, K, cfg=cfg).generations
    assert [g.num_tokens for g in gens] == [2, 4]
    assert [len(g.token_logprobs) for g in gens] == [2, 4]
    assert all(not g.truncated for g in gens)


def test_seeding_reproducible_and_id_dependent():
    cfg = _cfg(chunk_size=2, num_samples=4)
    a1 = G.generate_samples(FakeModel(), FakeTok(), Q, K, sample_id="a", cfg=cfg).generations
    a2 = G.generate_samples(FakeModel(), FakeTok(), Q, K, sample_id="a", cfg=cfg).generations
    b = G.generate_samples(FakeModel(), FakeTok(), Q, K, sample_id="b", cfg=cfg).generations
    assert [g.token_ids for g in a1] == [g.token_ids for g in a2]
    assert [g.token_ids for g in a1] != [g.token_ids for g in b]


def test_derive_seed():
    assert G.derive_seed(42, "a", 0) == G.derive_seed(42, "a", 0)
    assert G.derive_seed(42, "a", 0) != G.derive_seed(42, "a", 1)
    assert G.derive_seed(42, "a", 0) != G.derive_seed(43, "a", 0)
    assert 0 <= G.derive_seed(42, "a", 0) < 2**32


def test_num_samples_override_and_validation():
    m = FakeModel()
    out = G.generate_samples(m, FakeTok(), Q, K, num_samples=2, cfg=_cfg(chunk_size=5))
    assert len(out.generations) == 2
    with pytest.raises(ValueError):
        G.generate_samples(FakeModel(), FakeTok(), Q, K, num_samples=0)
