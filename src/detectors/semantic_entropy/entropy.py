
"""Semantic entropy calculations for generated answer clusters."""

import math
from collections import Counter
from collections.abc import Sequence


def cluster_probabilities(cluster_ids: Sequence[int]) -> dict[int, float]:
    """Estimate probability mass from generation frequencies."""
    if not cluster_ids:
        raise ValueError("cluster_ids must not be empty")

    counts = Counter(cluster_ids)
    total = len(cluster_ids)

    return {
        cluster_id: count / total
        for cluster_id, count in sorted(counts.items())
    }


def semantic_entropy(cluster_ids: Sequence[int]) -> float:
    """Calculate Shannon entropy in nats."""
    probabilities = cluster_probabilities(cluster_ids)
    return -sum(p * math.log(p) for p in probabilities.values() if p > 0)


def normalized_semantic_entropy(cluster_ids: Sequence[int]) -> float:
    """Normalize entropy to [0, 1] using observed cluster count."""
    probabilities = cluster_probabilities(cluster_ids)
    k = len(probabilities)

    if k <= 1:
        return 0.0

    return semantic_entropy(cluster_ids) / math.log(k)
