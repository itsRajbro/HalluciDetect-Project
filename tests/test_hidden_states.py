"""Step 7 tests: src/common/hidden_states.py with a fake model (no GPU)."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.common import hidden_states as H
from src.common.prompts import encode_with_answer
from src.common.config import get_config

NL, HID, VOC = 4, 6, 40        # hidden_states entries, hidden size, vocab
EOT = 3


class FakeTok:
    def convert_tokens_to_ids(self, t):
        return EOT if t == "<|eot_id|>" else None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [5 + (ord(c) % 30) for c in text]}


class FakeModel:
    """hidden_states[l][0, pos] = 1000*l + pos (all dims), so pooling is checkable."""

    device = torch.device("cpu")

    def __init__(self):
        self.last_input = None
        self.last_logits = None

    def __call__(self, input_ids, output_hidden_states=True, use_cache=False):
        self.last_input = input_ids
        S = input_ids.shape[1]
        hs = tuple(
            (1000.0 * l + torch.arange(S, dtype=torch.float32)).view(1, S, 1).expand(1, S, HID).clone()
            for l in range(NL)
        )
        self.last_logits = torch.randn(1, S, VOC)
        return SimpleNamespace(hidden_states=hs, logits=self.last_logits)


Q, K, A = "who?", "fact", "abc"


def _extract(**kw):
    m = FakeModel()
    tok = FakeTok()
    return m, tok, H.extract_hidden_states(m, tok, Q, K, A, sample_id="s1", **kw)


def test_span_and_shapes():
    m, tok, hs = _extract()
    enc = encode_with_answer(tok, Q, K, A)
    assert hs.answer_start == enc["answer_start"] and hs.answer_end == enc["answer_end"]
    assert hs.answer_num_tokens == len(A) and hs.total_tokens == len(enc["input_ids"])
    assert hs.layers == list(range(NL))
    assert set(hs.poolings) == set(H.POOLINGS)
    for p in H.POOLINGS:
        assert hs.features[p].shape == (NL, HID)
        assert hs.features[p].device.type == "cpu"
    assert m.last_input.tolist()[0] == enc["input_ids"]


def test_pooling_positions():
    _, _, hs = _extract()
    a0, a1 = hs.answer_start, hs.answer_end
    for row, l in enumerate(hs.layers):
        base = 1000.0 * l
        assert hs.features["tbg"][row, 0] == base + (a0 - 1)
        assert hs.features["last_answer"][row, 0] == base + (a1 - 1)
        assert hs.features["eot"][row, 0] == base + a1
        expected_mean = base + sum(range(a0, a1)) / (a1 - a0)
        assert hs.features["mean_answer"][row, 0] == pytest.approx(expected_mean)


def test_layer_and_pooling_subsets():
    _, _, hs = _extract(layers=[3, 1], poolings=["last_answer"])
    assert hs.layers == [3, 1] and hs.poolings == ["last_answer"]
    assert list(hs.features) == ["last_answer"]
    assert hs.features["last_answer"].shape == (2, HID)
    assert hs.features["last_answer"][0, 0] == 3000.0 + (hs.answer_end - 1)
    assert hs.features["last_answer"][1, 0] == 1000.0 + (hs.answer_end - 1)


def test_answer_logprobs_alignment():
    m, tok, hs = _extract()
    ids = encode_with_answer(tok, Q, K, A)["input_ids"]
    a0, a1 = hs.answer_start, hs.answer_end
    assert len(hs.answer_logprobs) == (a1 - a0) + 1          # answer tokens + eot
    for k, tgt_pos in enumerate(range(a0, a1 + 1)):
        expected = torch.log_softmax(m.last_logits[0, tgt_pos - 1].float(), -1)[ids[tgt_pos]].item()
        assert hs.answer_logprobs[k] == pytest.approx(expected, abs=1e-5)
    assert ids[a1] == EOT


def test_without_logprobs():
    _, _, hs = _extract(with_logprobs=False)
    assert hs.answer_logprobs is None


def test_invalid_args():
    m, tok = FakeModel(), FakeTok()
    with pytest.raises(ValueError):
        H.extract_hidden_states(m, tok, Q, K, A, layers=[NL])        # out of range
    with pytest.raises(ValueError):
        H.extract_hidden_states(m, tok, Q, K, A, layers=[-1])
    with pytest.raises(ValueError):
        H.extract_hidden_states(m, tok, Q, K, A, poolings=["bogus"])
    with pytest.raises(ValueError):
        H.extract_hidden_states(m, tok, Q, K, "   ")                  # empty answer


def test_metadata_and_save_load_roundtrip(tmp_path):
    _, _, hs = _extract(layers=[0, 2], poolings=["tbg", "eot"])
    cfg = get_config()
    assert hs.config_hash == cfg.config_hash() and hs.prompt_version == cfg.prompt.version
    p = tmp_path / "hs.pt"
    H.save_hidden_states([hs], p)
    (back,) = H.load_hidden_states(p)
    assert back.sample_id == "s1" and back.layers == [0, 2] and back.poolings == ["tbg", "eot"]
    assert back.answer_logprobs == pytest.approx(hs.answer_logprobs)
    for k in hs.features:
        assert torch.equal(back.features[k], hs.features[k])
