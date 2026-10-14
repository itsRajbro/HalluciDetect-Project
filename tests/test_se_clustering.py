
import pytest

from src.detectors.semantic_entropy.clustering import cluster_answers


def test_normalized_exact_matching():
    answers = [
        "Paris is the capital.",
        "  PARIS IS THE CAPITAL. ",
        "London",
    ]
    assert cluster_answers(answers) == [0, 0, 1]


def test_custom_equivalence_predicate():
    answers = ["Paris", "PARIS", "London"]

    clusters = cluster_answers(
        answers,
        equivalent=lambda a, b: a.casefold() == b.casefold(),
    )

    assert clusters == [0, 0, 1]


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        cluster_answers([])


def test_paraphrase_clustering_with_test_predicate():
    equivalent_pairs = {
        frozenset((
            "Paris is the capital of France.",
            "France's capital is Paris.",
        ))
    }

    def equivalent(a, b):
        return a == b or frozenset((a, b)) in equivalent_pairs

    answers = [
        "Paris is the capital of France.",
        "France's capital is Paris.",
        "London is the capital of the UK.",
    ]

    assert cluster_answers(answers, equivalent) == [0, 0, 1]
