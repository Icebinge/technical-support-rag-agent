import numpy as np
import pytest

from ts_rag_agent.infrastructure.exact_top_k import (
    exact_top_k_indices,
    full_sort_top_k_indices,
)


def test_exact_top_k_matches_full_sort_with_string_ties_and_boundary_ties() -> None:
    scores = np.asarray([1.0, 4.0, 3.0, 3.0, -2.0, 3.0, 0.5])
    ids = ["g", "a", "z", "b", "c", "a", "d"]
    eligible = np.asarray([0, 1, 2, 3, 5, 6])

    actual = exact_top_k_indices(
        scores,
        top_k=3,
        eligible_indices=eligible,
        string_tie_breaks=ids,
    )
    expected = sorted(eligible, key=lambda index: (-scores[index], ids[index]))[:3]

    assert actual == expected


def test_exact_top_k_matches_stable_index_tie_break_for_dense_scores() -> None:
    scores = np.asarray([0.2, 0.9, 0.9, -0.5, 0.8, 0.8], dtype=np.float32)

    actual = exact_top_k_indices(scores, top_k=4)
    expected = list(np.argsort(-scores, kind="stable")[:4])

    assert actual == expected


def test_exact_top_k_matches_full_sort_reference_across_random_tied_scores() -> None:
    generator = np.random.default_rng(142)
    scores = generator.integers(-5, 6, size=200).astype(np.float64)
    ids = [f"doc-{index % 37:03d}-{index:03d}" for index in range(scores.size)]
    eligible = np.flatnonzero(scores != -5)

    for top_k in (1, 10, 50, 150, 400):
        assert exact_top_k_indices(
            scores,
            top_k=top_k,
            eligible_indices=eligible,
            string_tie_breaks=ids,
        ) == full_sort_top_k_indices(
            scores,
            top_k=top_k,
            eligible_indices=eligible,
            string_tie_breaks=ids,
        )


def test_exact_top_k_returns_all_eligible_rows_when_top_k_is_larger() -> None:
    scores = np.asarray([0.1, 0.4, 0.2])
    eligible = np.asarray([0, 2])

    assert exact_top_k_indices(scores, top_k=10, eligible_indices=eligible) == [2, 0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scores": np.zeros((2, 2)), "top_k": 1}, "one-dimensional"),
        ({"scores": np.zeros(2), "top_k": 0}, "positive"),
        (
            {"scores": np.zeros(2), "top_k": 1, "eligible_indices": np.zeros((1, 1))},
            "eligible_indices",
        ),
        (
            {"scores": np.zeros(2), "top_k": 1, "string_tie_breaks": ["only-one"]},
            "tie-break count",
        ),
    ],
)
def test_exact_top_k_rejects_invalid_inputs(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        exact_top_k_indices(**kwargs)
