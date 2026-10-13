
"""Semantic clustering utilities for generated answers."""

import re
from collections.abc import Callable, Sequence


def normalize_answer(text: str) -> str:
    """Normalize case and whitespace for baseline matching."""
    text = (text or "").casefold().strip()
    return re.sub(r"\s+", " ", text)


def cluster_answers(
    answers: Sequence[str],
    equivalent: Callable[[str, str], bool] | None = None,
) -> list[int]:
    """Group answers using an equivalence predicate.

    Without a supplied predicate, this uses normalized exact matching.
    That fallback is a baseline, not a genuine semantic similarity model.
    """
    if not answers:
        raise ValueError("answers must not be empty")

    predicate = equivalent or (
        lambda a, b: normalize_answer(a) == normalize_answer(b)
    )

    representatives: list[str] = []
    cluster_ids: list[int] = []

    for answer in answers:
        answer = (answer or "").strip()

        for cluster_id, representative in enumerate(representatives):
            if predicate(answer, representative):
                cluster_ids.append(cluster_id)
                break
        else:
            cluster_ids.append(len(representatives))
            representatives.append(answer)

    return cluster_ids
