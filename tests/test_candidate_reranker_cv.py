from pathlib import Path

from ts_rag_agent.application.candidate_reranker_cv import (
    candidate_reranker_cv_result_to_dict,
    cross_validate_candidate_rerankers,
    split_validated_candidate_reranker_selections,
    write_cv_visualizations,
)


def test_cross_validated_candidate_rerankers_improve_on_learnable_signal(
    tmp_path: Path,
):
    rows = [
        candidate
        for question_index in range(1, 7)
        for candidate in _question_rows(question_index)
    ]

    result = cross_validate_candidate_rerankers(
        rows=rows,
        fold_count=3,
        model_names=["logistic_best_candidate", "ridge_candidate_token_f1"],
    )
    result_dict = candidate_reranker_cv_result_to_dict(result)
    visualizations = write_cv_visualizations(result=result, output_dir=tmp_path)

    assert result.fold_count == 3
    assert result.best_model_name in {
        "logistic_best_candidate",
        "ridge_candidate_token_f1",
    }
    assert len(result.models) == 2
    for model in result.models:
        assert model.aggregate_validation.question_count == 6
        assert model.aggregate_validation.selected_average_token_f1 > (
            model.aggregate_validation.baseline_average_token_f1
        )
        assert model.aggregate_validation.average_delta_vs_top_candidate > 0
        assert model.aggregate_validation.selected_best_candidate_rate == 1.0
        assert model.aggregate_validation.selected_rank_distribution == {"rank_2": 6}

    assert result_dict["feature_contract"].startswith("Only row.runtime_features")
    assert {visualization.name for visualization in visualizations} == {
        "candidate_reranker_model_delta.svg",
        "candidate_reranker_model_gap_closed.svg",
        "candidate_reranker_best_model_route_delta.svg",
        "candidate_reranker_best_model_selected_rank.svg",
    }
    assert (
        tmp_path / "candidate_reranker_model_delta.svg"
    ).read_text(encoding="utf-8").startswith("<svg")


def test_cross_validated_candidate_rerankers_reject_invalid_fold_count():
    rows = [
        candidate
        for question_index in range(1, 3)
        for candidate in _question_rows(question_index)
    ]

    try:
        cross_validate_candidate_rerankers(rows=rows, fold_count=3)
    except ValueError as exc:
        assert "fold_count must be no larger" in str(exc)
    else:
        raise AssertionError("fold_count larger than question count should fail")


def test_cross_validated_candidate_rerankers_reject_unknown_model():
    rows = [
        candidate
        for question_index in range(1, 3)
        for candidate in _question_rows(question_index)
    ]

    try:
        cross_validate_candidate_rerankers(
            rows=rows,
            fold_count=2,
            model_names=["unknown_model"],
        )
    except ValueError as exc:
        assert "Unknown candidate reranker model" in str(exc)
    else:
        raise AssertionError("unknown model should fail")


def test_split_validated_candidate_reranker_selections_train_on_train_validate_dev():
    rows = [
        candidate
        for question_index in range(1, 7)
        for candidate in _question_rows(question_index)
    ]

    selections = split_validated_candidate_reranker_selections(
        rows=rows,
        model_name="logistic_best_candidate",
        train_split="train",
        validation_split="dev",
    )

    assert len(selections) == 3
    assert {selection.split for selection in selections} == {"dev"}
    assert all(selection.selected_candidate_rank == 2 for selection in selections)
    assert all(
        selection.selected_candidate_token_f1
        > selection.baseline_candidate_token_f1
        for selection in selections
    )


def test_split_validated_candidate_reranker_selections_reject_same_split():
    rows = [
        candidate
        for question_index in range(1, 3)
        for candidate in _question_rows(question_index)
    ]

    try:
        split_validated_candidate_reranker_selections(
            rows=rows,
            model_name="logistic_best_candidate",
            train_split="train",
            validation_split="train",
        )
    except ValueError as exc:
        assert "must be different" in str(exc)
    else:
        raise AssertionError("same train/validation split should fail")


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
