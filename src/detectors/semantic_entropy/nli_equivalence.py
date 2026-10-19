
"""NLI-based semantic-equivalence checks for SE clustering."""

import re
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_ID = "facebook/bart-large-mnli"
MODEL_REVISION = "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce"
BIDIRECTIONAL_THRESHOLD = 0.85
SHORT_ANSWER_THRESHOLD = 0.85


class NLISemanticEquivalence:
    """Treat bidirectional entailment as a heuristic for equivalence."""

    # Negation patterns checked by the experimental short-answer fallback.
    NEGATION_PATTERN = re.compile(
        r"^(?:"
        r"is\s+not\b|isn't\b|"
        r"was\s+not\b|wasn't\b|"
        r"are\s+not\b|aren't\b|"
        r"were\s+not\b|weren't\b|"
        r"does\s+not\b|doesn't\b|"
        r"do\s+not\b|don't\b|"
        r"did\s+not\b|didn't\b|"
        r"has\s+not\b|hasn't\b|"
        r"have\s+not\b|haven't\b|"
        r"had\s+not\b|hadn't\b|"
        r"cannot\b|can't\b|"
        r"will\s+not\b|won't\b|"
        r"would\s+not\b|wouldn't\b|"
        r"could\s+not\b|couldn't\b|"
        r"should\s+not\b|shouldn't\b|"
        r"not\b"
        r")",
        flags=re.IGNORECASE,
    )

    def __init__(self, model_id: str = MODEL_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=MODEL_REVISION,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            revision=MODEL_REVISION,
        )
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

        self.model_id = model_id
        self.model_revision = MODEL_REVISION

    def provenance(self) -> dict:
        """Return NLI model identity for embedding in run metadata."""
        return {
            "nli_model_id": self.model_id,
            "nli_model_revision": self.model_revision,
        }

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

    @classmethod
    def _anchored_short_answer(
        cls,
        short_answer: str,
        long_answer: str,
    ) -> bool:
        """Experimental fallback for a short answer anchoring a sentence.

        Accepts a short answer when the longer answer begins with that
        answer as a standalone term, unless a recognized negation follows.
        This is a heuristic and is not a general factuality check.
        """
        short = (short_answer or "").strip()
        long = (long_answer or "").strip()

        if not short or not long:
            return False

        if len(short.split()) > 3:
            return False

        if len(long.split()) <= len(short.split()):
            return False

        # The short answer must appear at the beginning as a standalone
        # term, not merely as a substring of another word.
        pattern = re.compile(
            r"^\s*" + re.escape(short) + r"\b",
            flags=re.IGNORECASE,
        )
        match = pattern.match(long)

        if match is None:
            return False

        remainder = long[match.end():].lstrip(" \t:,-–—")

        # Reject explicit negation immediately after the anchored answer.
        if cls.NEGATION_PATTERN.match(remainder):
            return False

        return True

    def __call__(
        self,
        answer_a: str,
        answer_b: str,
    ) -> bool:
        """Check semantic equivalence using NLI and short-answer heuristics."""
        a = (answer_a or "").strip()
        b = (answer_b or "").strip()

        if not a or not b:
            return False

        # Exact match needs no model inference.
        if a.casefold() == b.casefold():
            return True

        # Experimental short-answer anchoring fallback.
        # Keep this narrowly scoped; it is not a substitute for NLI.
        if (
            self._anchored_short_answer(a, b)
            or self._anchored_short_answer(b, a)
        ):
            return True

        a_to_b = self._entailment_probability(a, b)
        b_to_a = self._entailment_probability(b, a)

        # Standard rule: both answers must entail each other.
        if (
            a_to_b >= BIDIRECTIONAL_THRESHOLD
            and b_to_a >= BIDIRECTIONAL_THRESHOLD
        ):
            return True

        # Existing short-answer fallback: the longer answer must strongly
        # entail the short answer.
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
