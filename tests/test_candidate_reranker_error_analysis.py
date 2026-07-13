from ts_rag_agent.application.candidate_reranker_error_analysis import (
    analyze_candidate_reranker_errors,
    candidate_reranker_error_analysis_to_dict,
)


def test_candidate_reranker_error_analysis_summarizes_grouped_cv_cases():
    rows = [
        candidate
        for question_index in range(1, 7)
        for candidate in _question_rows(question_index)
    ]

    result = analyze_candidate_reranker_errors(
        rows=rows,
        model_name="logistic_best_candidate",
        fold_count=3,
        sample_limit=3,
    )
    result_dict = candidate_reranker_error_analysis_to_dict(result)

    assert result.model_name == "logistic_best_candidate"
    assert result.summary.question_count == 6
    assert result.summary.improved_count == 6
    assert result.summary.regressed_count == 0
    assert result.summary.average_delta_vs_top_candidate == 0.7
    assert result.summary.selected_missed_gold_document_count == 0
    assert result.summary.selected_missed_oracle_best_count == 0
    assert result.route_summaries[0].question_count == 3
    assert result.split_summaries[0].question_count == 3
    assert result.selected_rank_summaries[0].segment_name == "rank_2"
    assert result.sample_cases["largest_improvements"]
    assert result.sample_cases["largest_regressions"] == []
    assert result.feature_contrasts
    assert result_dict["analysis_scope"].startswith("Offline grouped-CV")


def test_candidate_reranker_error_analysis_rejects_unknown_model():
    rows = [
        candidate
        for question_index in range(1, 3)
        for candidate in _question_rows(question_index)
    ]

    try:
        analyze_candidate_reranker_errors(
            rows=rows,
            model_name="unknown_model",
            fold_count=2,
        )
    except ValueError as exc:
        assert "Unknown candidate reranker model" in str(exc)
    else:
        raise AssertionError("unknown model should fail")


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
