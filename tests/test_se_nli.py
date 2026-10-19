from src.detectors.semantic_entropy.clustering import cluster_answers
from src.detectors.semantic_entropy.nli_equivalence import (
    NLISemanticEquivalence,
)


def test_nli_equivalence_examples():
    equivalent = NLISemanticEquivalence()

    assert equivalent(
        "Paris is the capital of France.",
        "France's capital is Paris.",
    )

    assert not equivalent(
        "Paris is the capital of France.",
        "Lyon is the capital of France.",
    )


def test_nli_integrates_with_semantic_clustering():
    equivalent = NLISemanticEquivalence()

    answers = [
        "Paris is the capital of France.",
        "France's capital is Paris.",
        "Lyon is the capital of France.",
    ]

    clusters = cluster_answers(answers, equivalent=equivalent)

    assert clusters[0] == clusters[1]
    assert clusters[0] != clusters[2]


def test_nli_short_answer_against_supporting_sentence():
    equivalent = NLISemanticEquivalence()

    assert equivalent(
        "Bul",
        "Bul is a running-fight board game originating in Mesoamerica "
        "and known among Maya peoples of Guatemala.",
    )


def test_nli_short_answer_rejects_negation():
    equivalent = NLISemanticEquivalence()

    assert not equivalent(
        "Paris",
        "Paris is not the capital of France.",
    )


def test_nli_numeric_short_answer():
    equivalent = NLISemanticEquivalence()

    assert equivalent("42", "The answer is 42.")

    assert not equivalent("42", "The answer is not 42.")


def test_nli_exact_match_short_answer():
    """Moved from tests/tests/test_se_candidate.py (duplicate file, E7 cleanup)."""
    equivalent = NLISemanticEquivalence()

    assert equivalent(
        "Albert Hammond was born in 1944.",
        "Albert Hammond was born in 1944.",
    )