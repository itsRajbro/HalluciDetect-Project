
"""Candidate-aware semantic entropy scoring."""

from collections.abc import Callable, Sequence

from src.detectors.semantic_entropy.clustering import cluster_answers
from src.detectors.semantic_entropy.entropy import (
    cluster_probabilities,
    semantic_entropy,
)


def score_candidate(
    generated_answers: Sequence[str],
    candidate_answer: str,
    equivalent: Callable[[str, str], bool] | None = None,
) -> dict:
    """Score a candidate against the semantic distribution of generations.

    Higher candidate_score means the candidate is less represented by
    the generated semantic distribution.

    The supplied equivalence predicate determines semantic matching.
    Without one, normalized exact matching is used as a baseline.
    """
    if not generated_answers:
        raise ValueError("generated_answers must not be empty")

    if not candidate_answer or not candidate_answer.strip():
        raise ValueError("candidate_answer must not be empty")

    # Append the candidate only to identify its cluster.
    # It is excluded from the probability calculation.
    all_cluster_ids = cluster_answers(
        [*generated_answers, candidate_answer],
        equivalent=equivalent,
    )

    generation_cluster_ids = all_cluster_ids[:-1]
    candidate_cluster_id = all_cluster_ids[-1]

    probabilities = cluster_probabilities(generation_cluster_ids)
    candidate_probability = probabilities.get(candidate_cluster_id, 0.0)

    return {
        "global_entropy": semantic_entropy(generation_cluster_ids),
        "candidate_cluster_id": (
            candidate_cluster_id
            if candidate_probability > 0
            else None
        ),
        "candidate_cluster_probability": candidate_probability,
        "candidate_score": 1.0 - candidate_probability,
        "num_generations": len(generated_answers),
        "num_clusters": len(probabilities),
        "candidate_matched": candidate_probability > 0,
    }
