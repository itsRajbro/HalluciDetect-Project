
import math
import pytest

from src.detectors.semantic_entropy.entropy import (
    cluster_probabilities,
    semantic_entropy,
    normalized_semantic_entropy,
)


def test_single_cluster_has_zero_entropy():
    assert semantic_entropy([0, 0, 0, 0]) == pytest.approx(0.0)


def test_two_equal_clusters():
    assert semantic_entropy([0, 1]) == pytest.approx(math.log(2))


def test_unequal_cluster_probabilities():
    assert cluster_probabilities([0, 0, 0, 1]) == {
        0: 0.75,
        1: 0.25,
    }


def test_normalized_entropy_for_equal_clusters():
    assert normalized_semantic_entropy([0, 1, 2]) == pytest.approx(1.0)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        semantic_entropy([])


def test_normalized_entropy_for_one_cluster():
    assert normalized_semantic_entropy([4, 4, 4]) == pytest.approx(0.0)
