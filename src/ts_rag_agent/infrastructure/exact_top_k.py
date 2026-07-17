from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def exact_top_k_indices(
    scores: np.ndarray,
    *,
    top_k: int,
    eligible_indices: np.ndarray | None = None,
    string_tie_breaks: Sequence[str] | None = None,
) -> list[int]:
    """Return exact top-k indices without fully sorting all eligible rows."""

    score_array = np.asarray(scores)
    if score_array.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if string_tie_breaks is not None and len(string_tie_breaks) != score_array.size:
        raise ValueError("tie-break count must match score count")

    if eligible_indices is None:
        eligible = np.arange(score_array.size, dtype=np.intp)
    else:
        eligible = np.asarray(eligible_indices, dtype=np.intp)
        if eligible.ndim != 1:
            raise ValueError("eligible_indices must be one-dimensional")
    if eligible.size == 0:
        return []

    candidates = eligible
    if eligible.size > top_k:
        eligible_scores = score_array[eligible]
        boundary_position = eligible.size - top_k
        boundary_score = np.partition(eligible_scores, boundary_position)[boundary_position]
        candidates = eligible[eligible_scores >= boundary_score]

    if string_tie_breaks is None:
        ranked = sorted(candidates, key=lambda index: (-float(score_array[index]), int(index)))
    else:
        ranked = sorted(
            candidates,
            key=lambda index: (-float(score_array[index]), string_tie_breaks[int(index)]),
        )
    return [int(index) for index in ranked[:top_k]]


def full_sort_top_k_indices(
    scores: np.ndarray,
    *,
    top_k: int,
    eligible_indices: np.ndarray | None = None,
    string_tie_breaks: Sequence[str] | None = None,
) -> list[int]:
    """Reference the historical behavior by sorting every eligible row."""

    score_array = np.asarray(scores)
    if score_array.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if string_tie_breaks is not None and len(string_tie_breaks) != score_array.size:
        raise ValueError("tie-break count must match score count")
    eligible = (
        np.arange(score_array.size, dtype=np.intp)
        if eligible_indices is None
        else np.asarray(eligible_indices, dtype=np.intp)
    )
    if eligible.ndim != 1:
        raise ValueError("eligible_indices must be one-dimensional")
    if string_tie_breaks is None:
        ranked = sorted(eligible, key=lambda index: (-float(score_array[index]), int(index)))
    else:
        ranked = sorted(
            eligible,
            key=lambda index: (-float(score_array[index]), string_tie_breaks[int(index)]),
        )
    return [int(index) for index in ranked[:top_k]]
