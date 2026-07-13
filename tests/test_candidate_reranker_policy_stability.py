from ts_rag_agent.application.candidate_reranker_policy_stability import (
    analyze_candidate_reranker_policy_stability,
    candidate_reranker_policy_stability_to_dict,
)


def test_candidate_reranker_policy_stability_compares_fixed_policies():
    rows = [
        candidate
        for question_index in range(1, 7)
        for candidate in _question_rows(question_index)
    ]

    result = analyze_candidate_reranker_policy_stability(
        rows=rows,
        model_name="logistic_best_candidate",
        fold_count=3,
    )
    result_dict = candidate_reranker_policy_stability_to_dict(result)

    assert result.model_name == "logistic_best_candidate"
    assert result.fold_count == 3
    assert result.primary_policy.metrics.question_count == 6
    assert result.challenger_policy.metrics.question_count == 6
    assert len(result.fold_metrics) == 6
    assert any(
        comparison.route == "security_bulletin_affected_product"
        for comparison in result.route_comparisons
    )
    assert result.findings
    assert result_dict["analysis_scope"].startswith("Offline stability analysis")


def _question_rows(question_index: int) -> list[dict]:
    split = "dev" if question_index % 2 else "train"
    route = _route(question_index)
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


def _route(question_index: int) -> str:
    if question_index == 1:
        return "how_to_or_lookup"
    if question_index == 2:
        return "security_bulletin_affected_product"
    return "other"


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
