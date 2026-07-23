from __future__ import annotations

from ts_rag_agent.application.composition_action_audit import (
    ActionAuditRow,
    CompositionAction,
    build_action_runtime_features,
    classify_action_outcome,
    enumerate_atomic_composition_actions,
    run_action_predictability_oof,
    stage180_action_summary,
    summarize_action_rows,
)
from ts_rag_agent.application.evidence_selection import SentenceEvidenceCandidate
from ts_rag_agent.domain.dataset import PrimeQADocument
from ts_rag_agent.domain.retrieval import RetrievalResult


def test_strict_expected_taxonomy_requires_nonregression_and_gain() -> None:
    assert classify_action_outcome(citation_delta=1, f1_delta=0.2) == (
        "dual_gain",
        True,
    )
    assert classify_action_outcome(citation_delta=1, f1_delta=0.0) == (
        "citation_gain_f1_tied",
        True,
    )
    assert classify_action_outcome(citation_delta=0, f1_delta=0.2) == (
        "f1_gain_citation_preserved",
        True,
    )
    assert classify_action_outcome(citation_delta=1, f1_delta=-0.001) == (
        "citation_gain_f1_loss",
        False,
    )
    assert classify_action_outcome(citation_delta=0, f1_delta=0.0) == (
        "neutral",
        False,
    )


def test_atomic_actions_are_deduplicated_and_keep_stage180_identity() -> None:
    candidates = tuple(_candidate(index, document_id=f"d{index}") for index in range(6))

    actions = enumerate_atomic_composition_actions(
        candidates=candidates,
        stage180_selected_indices=(0, 3, 4),
    )

    assert len({action.selected_indices for action in actions}) == len(actions)
    assert actions[0].family == "baseline"
    assert actions[0].selected_indices == (0, 1, 2)
    stage180_matches = [action for action in actions if action.matches_stage180]
    assert len(stage180_matches) == 1
    assert stage180_matches[0].selected_indices == (0, 3, 4)
    assert any(action.family == "replace_slot_2" for action in actions)
    assert any(action.family == "delete_slot_1" for action in actions)


def test_empty_candidate_pool_keeps_one_baseline_control() -> None:
    actions = enumerate_atomic_composition_actions(
        candidates=(),
        stage180_selected_indices=(),
    )

    assert len(actions) == 1
    assert actions[0].family == "baseline"
    assert actions[0].matches_stage180 is True
    assert "stage180_selected" in actions[0].aliases


def test_action_features_are_runtime_safe_and_track_lead_changes() -> None:
    candidates = tuple(_candidate(index, document_id=f"d{index}") for index in range(5))
    action = CompositionAction(
        action_id="a",
        family="replace_slot_1",
        aliases=("replace_slot_1",),
        selected_indices=(3, 1, 2),
        matches_stage180=False,
    )
    features = build_action_runtime_features(
        action=action,
        candidates=candidates,
        candidate_runtime_features=tuple(_candidate_features(index) for index in range(5)),
        route="how_to_or_lookup",
    )

    assert features["preserves_baseline_lead"] is False
    assert features["added_sentence_count"] == 1
    assert features["removed_sentence_count"] == 1
    assert features["removed_slot_1"] is True
    assert features["selected_answer_signal_score_mean"] == 1.0
    assert not any(
        private_name in features
        for private_name in (
            "answer",
            "answer_doc_id",
            "document_id",
            "gold_answer",
            "question_id",
            "question_key",
            "selected_indices",
        )
    )
    assert not any("citation" in key or key.startswith("gold_") for key in features)


def test_oof_predictability_preserves_question_folds() -> None:
    rows = []
    for fold_index in range(5):
        for question_index in range(2):
            question_key = f"q-{fold_index}-{question_index}"
            for action_index, expected in enumerate((False, True)):
                rows.append(
                    _row(
                        question_key=question_key,
                        fold_id=f"fold_{fold_index + 1}",
                        action_id=f"a-{action_index}",
                        expected=expected,
                        signal=float(expected),
                    )
                )

    result = run_action_predictability_oof(rows, total_question_count=10)

    assert result["model"]["fit_count"] == 5
    assert result["aggregate"]["action_count"] == 20
    assert result["aggregate"]["roc_auc"] > 0.9
    assert result["question_ranking"]["strict_expected_hit_at_1_count"] == 10
    assert len(result["coverage_curve"]) == 4


def test_public_summaries_use_strict_expected_rows() -> None:
    baseline = _row(
        question_key="q",
        fold_id="fold_1",
        action_id="baseline",
        expected=False,
        signal=0.0,
        family="baseline",
        matches_stage180=False,
    )
    expected = _row(
        question_key="q",
        fold_id="fold_1",
        action_id="expected",
        expected=True,
        signal=1.0,
        citation_delta=1,
        f1_delta=0.1,
        matches_stage180=True,
    )

    summary = summarize_action_rows((baseline, expected), total_question_count=1)
    stage180 = stage180_action_summary((baseline, expected))

    assert summary["strict_expected_action_count"] == 1
    assert summary["oracle"]["gold_citation_delta"] == 1
    assert stage180["strict_expected_count"] == 1
    assert stage180["mean_answerable_f1_delta"] == 0.1


def _row(
    *,
    question_key: str,
    fold_id: str,
    action_id: str,
    expected: bool,
    signal: float,
    family: str = "replace_slot_2",
    citation_delta: int = 0,
    f1_delta: float = 0.1,
    matches_stage180: bool = False,
) -> ActionAuditRow:
    return ActionAuditRow(
        question_key=question_key,
        fold_id=fold_id,
        route="how_to_or_lookup",
        action=CompositionAction(
            action_id=action_id,
            family=family,
            aliases=(family,),
            selected_indices=(0, 1),
            matches_stage180=matches_stage180,
        ),
        runtime_features={
            "question_route": "how_to_or_lookup",
            "action_family": family,
            "signal": signal,
            "modified_sentence_count": 1,
            "preserves_baseline_lead": True,
            "added_sentence_count": 1,
            "removed_sentence_count": 1,
        },
        outcome_class="dual_gain" if expected else "citation_preserved_f1_loss",
        strict_expected=expected,
        citation_delta=citation_delta,
        f1_delta=f1_delta if expected else -abs(f1_delta),
    )


def _candidate(index: int, *, document_id: str) -> SentenceEvidenceCandidate:
    sentence = f"Candidate {index} contains enough technical answer text for testing."
    return SentenceEvidenceCandidate(
        sentence=sentence,
        retrieval_result=RetrievalResult(
            document=PrimeQADocument(
                id=document_id,
                title=f"Document {document_id}",
                text=sentence,
            ),
            score=100.0 - index,
            rank=index + 1,
        ),
        score=20.0 - index,
        overlap_terms=("technical",),
    )


def _candidate_features(index: int) -> dict[str, object]:
    return {
        "retrieval_rank": index + 1,
        "retrieval_score": 100.0 - index,
        "candidate_score": 20.0 - index,
        "candidate_token_count": 10,
        "candidate_sentence_count": 1,
        "query_overlap_count": 1,
        "query_overlap_ratio": 0.5,
        "candidate_query_coverage_ratio": 0.2,
        "title_query_overlap_count": 1,
        "title_query_overlap_ratio": 0.5,
        "answer_signal_score": 1.0,
        "problem_noise_score": 0.0,
        "symbol_ratio": 0.0,
        "has_answer_heading": False,
        "has_problem_heading": False,
        "has_question_heading": False,
        "has_url": False,
        "has_trace_noise": False,
    }
