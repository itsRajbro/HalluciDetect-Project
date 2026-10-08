"""Central configuration for HalluciDetect (shared by Person 1, 2 and 3).

Rule: nobody hardcodes model IDs, generation settings, seeds, split names,
paths, or label/score conventions anywhere else. Import them from here.

Every experiment output should record `config_hash()` so results can be traced
back to the exact configuration that produced them.

Items marked PROPOSED are defaults chosen by Person 1 so the foundation can be
tested; the owner named in the comment must confirm them BEFORE the foundation
is frozen (they change every generation / hidden state downstream).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Project-wide conventions (decided once; do not change after freeze)
# ---------------------------------------------------------------------------
LABEL_FACTUAL = 0          # right_answer        -> "Non-Hallucination"
LABEL_HALLUCINATED = 1     # hallucinated_answer -> "Hallucination"

# EVERY detector must output a score where HIGHER = MORE LIKELY HALLUCINATED.
# (SE: higher entropy -> higher score. SEP / Transformer: P(label=1).
#  Length-only: longer answer -> higher score.) AUROC is computed with
# label 1 as the positive class on this score, with no sign flipping.
SCORE_HIGHER_MEANS_HALLUCINATED = True

SPLIT_TRAIN = "train"
SPLIT_VAL = "validation"   # matches the existing file name validation.jsonl
SPLIT_TEST = "test"
SPLITS = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

# Official length feature = characters, i.e. len(answer) on the cleaned answer.
# This is the same metric used to build test_length_controlled.jsonl, so the
# length-only baseline and the length-controlled test are consistent.
# Token length (Llama tokenizer) is stored alongside as a secondary feature.
OFFICIAL_LENGTH_FIELD = "answer_length_chars"
SECONDARY_LENGTH_FIELD = "answer_length_tokens"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PathsConfig:
    processed_dir: str = "data/processed"          # gitignored, local
    split_manifest: str = "data/split_manifest.json"   # committed
    length_controlled_test: str = "data/processed/test_length_controlled.jsonl"
    outputs_dir: str = "outputs"                   # detector outputs, caches
    results_dir: str = "results"

    def resolve(self, rel: str) -> Path:
        return REPO_ROOT / rel

    def split_file(self, split: str) -> Path:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        return self.resolve(self.processed_dir) / f"{split}.jsonl"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    # TODO(Person 1): pin the exact commit hash before freeze. The benchmark
    # ran with revision=None. On a machine with HF access to the gated repo:
    #   python -c "from huggingface_hub import model_info; \
    #              print(model_info('meta-llama/Llama-3.2-3B-Instruct').sha)"
    revision: Optional[str] = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
    # NF4 settings copied from src/models/model_identification.py (the
    # configuration that was actually benchmarked and selected).
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    device_map: str = "cuda"
    pad_token: str = "<|finetune_right_pad_id|>"
    padding_side: str = "left"
    # Verified in the benchmark: 29 hidden_states entries (embeddings + 28
    # layers), hidden size 3072, vocab 128256.
    num_hidden_state_entries: int = 29
    hidden_size: int = 3072


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GreedyGenConfig:
    """Single deterministic answer (benchmark-style decoding)."""
    max_new_tokens: int = 64          # PROPOSED (Person 2): HaluEval answers are short
    do_sample: bool = False
    num_beams: int = 1


@dataclass(frozen=True)
class SampledGenConfig:
    """Multiple stochastic answers per question, used by Semantic Entropy."""
    num_samples: int = 10             # PROPOSED (Person 2)
    temperature: float = 1.0          # PROPOSED (Person 2)
    top_p: float = 0.9                # PROPOSED (Person 2)
    top_k: int = 50                   # PROPOSED (Person 2)
    max_new_tokens: int = 64          # PROPOSED (Person 2)
    do_sample: bool = True
    chunk_size: int = 5


# ---------------------------------------------------------------------------
# Prompt / hidden states (filled in at Steps 5 and 7)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PromptConfig:
    # Bump this string whenever ANY template text changes; it is recorded in
    # every output so results can be traced to a template.
    version: str = "v1"
    include_knowledge: bool = True    # DECISION NEEDED (all three): freeze before Step 5 ends
    

@dataclass(frozen=True)
class HiddenStateConfig:
    """Defaults for src/common/hidden_states.py (Step 7)."""
    poolings: tuple = ("tbg", "last_answer", "mean_answer", "eot")
    layers: Optional[tuple] = None    # None = all 29 hidden_states entries
    store_dtype: str = "float32"      # downcast to float16 yourself when caching


@dataclass(frozen=True)
class RunConfig:
    seed: int = 42                    # same seed value used by split.py
    hf_token_env: str = "HF_TOKEN"    # token is read from this env var, never stored


@dataclass(frozen=True)
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    greedy: GreedyGenConfig = field(default_factory=GreedyGenConfig)
    sampled: SampledGenConfig = field(default_factory=SampledGenConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    hidden: HiddenStateConfig = field(default_factory=HiddenStateConfig)   # <- new
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        """Short, stable hash of everything that affects model behaviour.

        Paths are excluded on purpose: they differ between machines but do
        not change results.
        """
        d = self.to_dict()
        d.pop("paths", None)
        blob = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def get_config() -> Config:
    """Return the project configuration (immutable)."""
    return Config()


if __name__ == "__main__":
    cfg = get_config()
    print(json.dumps(cfg.to_dict(), indent=2))
    print("config_hash:", cfg.config_hash())
