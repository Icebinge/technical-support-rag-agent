from __future__ import annotations

import inspect

from scripts.audit_stage178_candidate_alignment import main
from ts_rag_agent.application.primeqa_hybrid_stage178_candidate_alignment import (
    CandidateAlignmentObservation,
    summarize_candidate_alignment,
)


def test_alignment_summary_separates_prefix_and_selection_surfaces() -> None:
    rows = (
        _observation(prefix_exact=False, union_exact=True, missing=0),
        _observation(prefix_exact=True, union_exact=False, missing=2),
    )

    summary = summarize_candidate_alignment(rows)

    assert summary["prefix_sequence_exact_count"] == 1
    assert summary["union_sequence_exact_count"] == 1
    assert summary["live_union_missing_from_offline_question_count"] == 1
    assert summary["live_union_missing_from_offline_pair_count"] == 2
    assert summary["union_gold_hit"] == {
        "offline_count": 2,
        "live_count": 2,
        "live_gain_count": 0,
        "live_loss_count": 0,
    }


def test_alignment_cli_exposes_no_development_or_test_input() -> None:
    assert set(inspect.signature(main).parameters) == {
        "output",
        "encoder_batch_size",
    }


def _observation(
    *,
    prefix_exact: bool,
    union_exact: bool,
    missing: int,
) -> CandidateAlignmentObservation:
    return CandidateAlignmentObservation(
        prefix_sequence_exact=prefix_exact,
        prefix_set_exact=prefix_exact,
        prefix_symmetric_difference_count=0 if prefix_exact else 2,
        original_rrf_top10_exact=True,
        query_overlap_top10_exact=union_exact,
        union_sequence_exact=union_exact,
        union_set_exact=union_exact,
        live_union_missing_from_offline_count=missing,
        offline_original_gold_hit=True,
        live_original_gold_hit=True,
        offline_overlap_gold_hit=True,
        live_overlap_gold_hit=True,
        offline_union_gold_hit=True,
        live_union_gold_hit=True,
        live_retrieval_seconds=0.1,
    )
