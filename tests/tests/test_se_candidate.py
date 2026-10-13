

def test_nli_short_answer_supporting_sentence():
    from src.detectors.semantic_entropy.nli_equivalence import (
        NLISemanticEquivalence,
    )

    equivalent = NLISemanticEquivalence()

    assert equivalent(
        "Albert Hammond was born in 1944.",
        "Albert Hammond was born in 1944.",
    )
