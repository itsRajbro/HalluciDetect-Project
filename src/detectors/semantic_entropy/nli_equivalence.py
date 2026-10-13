
"""NLI-based semantic-equivalence checks for SE clustering."""

from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_ID = "facebook/bart-large-mnli"
BIDIRECTIONAL_THRESHOLD = 0.85
SHORT_ANSWER_THRESHOLD = 0.85


class NLISemanticEquivalence:
    """Treat bidirectional entailment as a heuristic for equivalence."""

    def __init__(self, model_id: str = MODEL_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.eval()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        labels = {
            int(index): str(label).lower()
            for index, label in self.model.config.id2label.items()
        }

        self.entailment_id = next(
            (
                index
                for index, label in labels.items()
                if "entail" in label
            ),
            None,
        )

        if self.entailment_id is None:
            raise ValueError(
                "Could not identify the NLI entailment label"
            )

    @lru_cache(maxsize=4096)
    def _entailment_probability(
        self,
        premise: str,
        hypothesis: str,
    ) -> float:
        """Return the NLI model's entailment probability."""
        if not premise.strip() or not hypothesis.strip():
            return 0.0

        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            logits = self.model(**inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1)

        return float(
            probabilities[self.entailment_id].item()
        )

    def __call__(
        self,
        answer_a: str,
        answer_b: str,
    ) -> bool:
        """Check semantic equivalence using NLI entailment."""
        a = (answer_a or "").strip()
        b = (answer_b or "").strip()

        if not a or not b:
            return False

        # Exact match needs no model inference.
        if a.casefold() == b.casefold():
            return True

        a_to_b = self._entailment_probability(a, b)
        b_to_a = self._entailment_probability(b, a)

        # Standard rule: both answers must entail each other.
        if (
            a_to_b >= BIDIRECTIONAL_THRESHOLD
            and b_to_a >= BIDIRECTIONAL_THRESHOLD
        ):
            return True

        # Short-answer fallback:
        # A longer answer must strongly entail the short answer.
        a_word_count = len(a.split())
        b_word_count = len(b.split())

        if (
            a_word_count <= 3
            and b_word_count > a_word_count
            and b_to_a >= SHORT_ANSWER_THRESHOLD
        ):
            return True

        if (
            b_word_count <= 3
            and a_word_count > b_word_count
            and a_to_b >= SHORT_ANSWER_THRESHOLD
        ):
            return True

        return False
