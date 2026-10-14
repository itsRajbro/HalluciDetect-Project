
import pytest

from src.detectors.semantic_entropy.candidate_aware import score_candidate


def test_candidate_in_dominant_cluster_gets_lower_score():
    result = score_candidate(
        ["Paris", "Paris", "Paris", "London"],
        "Paris",
    )

    assert result["candidate_cluster_probability"] == pytest.approx(0.75)
    assert result["candidate_score"] == pytest.approx(0.25)
    assert result["candidate_matched"] is True


def test_unmatched_candidate_gets_maximum_initial_score():
    result = score_candidate(
        ["Paris", "Paris", "London"],
        "Berlin",
    )

    assert result["candidate_cluster_probability"] == 0.0
    assert result["candidate_score"] == 1.0
    assert result["candidate_matched"] is False


def test_global_entropy_is_separate():
    result = score_candidate(
        ["Paris", "Paris", "London", "London"],
        "Paris",
    )

    assert result["global_entropy"] == pytest.approx(0.69314718056)
    assert result["candidate_score"] == pytest.approx(0.5)


def test_custom_semantic_equivalence():
    equivalents = {
        frozenset(("Paris", "The capital of France is Paris."))
    }

    def equivalent(a, b):
        return a == b or frozenset((a, b)) in equivalents

    result = score_candidate(
        ["Paris", "Paris", "London"],
        "The capital of France is Paris.",
        equivalent=equivalent,
    )

    assert result["candidate_cluster_probability"] == pytest.approx(2 / 3)
    assert result["candidate_matched"] is True


def test_empty_generations_are_rejected():
    with pytest.raises(ValueError):
        score_candidate([], "Paris")


def test_empty_candidate_is_rejected():
    with pytest.raises(ValueError):
        score_candidate(["Paris"], " ")
