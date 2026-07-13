from ts_rag_agent.application.candidate_reranker_cv import (
    cross_validated_candidate_reranker_selections,
)
from ts_rag_agent.application.candidate_reranker_policy_search import (
    CandidateRerankerPolicyConfig,
    candidate_reranker_policy_decisions_from_selections,
    candidate_reranker_policy_search_to_dict,
    evaluate_candidate_reranker_policy_from_selections,
    search_candidate_reranker_policies,
)


def test_candidate_reranker_policy_search_finds_rank_constrained_gain():
    rows = [
        candidate
        for question_index in range(1, 7)
        for candidate in _question_rows(question_index)
    ]

    result = search_candidate_reranker_policies(
        rows=rows,
        model_name="logistic_best_candidate",
        fold_count=3,
        max_selected_rank_grid=[1, 2],
        min_score_margin_grid=[0.0],
        protect_top1_candidate_score_min_grid=[None],
        blocked_route_sets=[(), ("error_or_log",)],
        top_policy_limit=4,
    )
    result_dict = candidate_reranker_policy_search_to_dict(result)

    assert result.policy_count == 4
    assert result.best_average_delta_policy.config.max_selected_rank == 2
    assert result.best_average_delta_policy.metrics.average_delta_vs_top_candidate == 0.7
    assert result.best_average_delta_policy.metrics.replacement_count == 6
    assert result.best_average_delta_policy.metrics.regressed_count == 0
    assert result.top_policies[0].metrics.policy_average_token_f1 == 0.8
    assert result.unconstrained_metrics.policy_average_token_f1 == 0.8
    assert result_dict["analysis_scope"].startswith("Offline grouped-CV")


def test_candidate_reranker_policy_search_rejects_invalid_grid():
    rows = [
        candidate
        for question_index in range(1, 3)
        for candidate in _question_rows(question_index)
    ]

    try:
        search_candidate_reranker_policies(
            rows=rows,
            model_name="logistic_best_candidate",
            fold_count=2,
            max_selected_rank_grid=[],
        )
    except ValueError as exc:
        assert "max_selected_rank_grid must not be empty" in str(exc)
    else:
        raise AssertionError("empty rank grid should fail")


def test_candidate_reranker_policy_blocks_low_selected_runtime_candidate_score():
    rows = [
        candidate
        for question_index in range(1, 7)
        for candidate in _question_rows(question_index)
    ]
    selections = cross_validated_candidate_reranker_selections(
        rows=rows,
        model_name="logistic_best_candidate",
        fold_count=3,
    )
    config = CandidateRerankerPolicyConfig(
        name="test_selected_score_gate",
        max_selected_rank=2,
        blocked_routes=(),
        min_score_margin_vs_top_candidate=0.0,
        protect_top1_candidate_score_min=None,
        min_selected_candidate_score=19.0,
    )

    evaluation = evaluate_candidate_reranker_policy_from_selections(
        config=config,
        selections=selections,
        rows=rows,
    )
    decisions = candidate_reranker_policy_decisions_from_selections(
        config=config,
        selections=selections,
        rows=rows,
    )

    assert evaluation.metrics.replacement_count == 0
    assert evaluation.metrics.policy_average_token_f1 == 0.1
    assert all(
        "selected_runtime_candidate_score_below_min" in decision.decision_reasons
        for decision in decisions
    )


def _question_rows(question_index: int) -> list[dict]:
    split = "dev" if question_index % 2 else "train"
    route = "error_or_log" if question_index % 2 else "other"
    return [
        _row(
            split=split,
            question_id=f"q{question_index}",
            candidate_rank=1,
            route=route,
            answer_signal_score=0.1,
            problem_noise_score=1.4,
            candidate_token_f1=0.1,
            is_best=False,
        ),
        _row(
            split=split,
            question_id=f"q{question_index}",
            candidate_rank=2,
            route=route,
            answer_signal_score=3.0,
            problem_noise_score=0.0,
            candidate_token_f1=0.8,
            is_best=True,
        ),
    ]


def _row(
    split: str,
    question_id: str,
    candidate_rank: int,
    route: str,
    answer_signal_score: float,
    problem_noise_score: float,
    candidate_token_f1: float,
    is_best: bool,
) -> dict:
    return {
        "split": split,
        "question_id": question_id,
        "candidate_id": f"{question_id}::candidate_{candidate_rank:03d}",
        "candidate_rank": candidate_rank,
        "runtime_features": {
            "selector_name": "test_selector",
            "question_route": route,
            "retrieval_rank": candidate_rank,
            "retrieval_score": 10.0 - candidate_rank,
            "candidate_score": 20.0 - candidate_rank,
            "candidate_token_count": 8,
            "candidate_sentence_count": 1,
            "question_token_count": 6,
            "query_term_count": 4,
            "query_overlap_count": 3 if is_best else 1,
            "query_overlap_ratio": 0.75 if is_best else 0.25,
            "candidate_query_coverage_ratio": 0.6 if is_best else 0.2,
            "title_query_overlap_count": 1,
            "title_query_overlap_ratio": 0.25,
            "answer_signal_score": answer_signal_score,
            "problem_noise_score": problem_noise_score,
            "has_answer_heading": is_best,
            "has_problem_heading": not is_best,
            "has_question_heading": False,
            "has_url": False,
            "has_trace_noise": not is_best,
            "symbol_ratio": 0.02,
        },
        "gold_labels": {
            "candidate_token_f1": candidate_token_f1,
            "is_gold_document": is_best,
            "is_best_candidate_for_question": is_best,
            "best_candidate_token_f1_for_question": 0.8,
            "f1_gap_to_best_candidate": 0.0 if is_best else 0.7,
        },
        "metadata": {
            "question_title": "Test question",
            "document_id": "gold" if is_best else "noise",
            "document_title": "Test document",
            "candidate_sentence": "Candidate text.",
        },
    }
