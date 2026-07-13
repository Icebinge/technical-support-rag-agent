from ts_rag_agent.application.candidate_score_guarded_policy_split_validation import (
    candidate_score_guarded_policy_split_validation_to_dict,
    evaluate_candidate_score_guarded_policy_split_validation,
    write_candidate_score_guarded_policy_split_validation_visualizations,
)


def test_candidate_score_guarded_policy_split_validation_runs_train_to_dev():
    rows = [
        candidate
        for question_index in range(1, 7)
        for candidate in _question_rows(question_index)
    ]

    result = evaluate_candidate_score_guarded_policy_split_validation(
        rows=rows,
        gold_answers_by_question_key=_gold_answers(rows),
        model_name="logistic_best_candidate",
        train_split="train",
        evaluation_split="dev",
        train_fold_count=3,
        max_answer_candidates=1,
    )
    result_dict = candidate_score_guarded_policy_split_validation_to_dict(result)
    holdout_policy_by_label = {
        policy.label: policy for policy in result.holdout_evaluation.policies
    }

    assert result.train_split == "train"
    assert result.evaluation_split == "dev"
    assert result.train_question_count == 3
    assert result.evaluation_question_count == 3
    assert result.train_cv_evaluation.selection_scope == "train_only_grouped_cv"
    assert result.holdout_evaluation.selection_scope == "train_to_validation_holdout"
    assert set(holdout_policy_by_label) == {
        "stage36_main",
        "model_margin_gte_0.10",
        "candidate_score_gte_60",
        "candidate_score_gte_90",
    }
    assert result.holdout_evaluation.evaluation_split == "dev"
    assert result.findings
    assert result_dict["analysis_scope"].startswith("Stage 40")


def test_candidate_score_guarded_policy_split_validation_writes_visualizations(tmp_path):
    rows = [
        candidate
        for question_index in range(1, 7)
        for candidate in _question_rows(question_index)
    ]
    result = evaluate_candidate_score_guarded_policy_split_validation(
        rows=rows,
        gold_answers_by_question_key=_gold_answers(rows),
        model_name="logistic_best_candidate",
        train_split="train",
        evaluation_split="dev",
        train_fold_count=3,
        max_answer_candidates=1,
    )

    visualizations = write_candidate_score_guarded_policy_split_validation_visualizations(
        result=result,
        output_dir=tmp_path,
    )

    assert len(visualizations.train_cv) == 4
    assert len(visualizations.holdout) == 4
    for visualization in [*visualizations.train_cv, *visualizations.holdout]:
        svg_path = tmp_path / visualization.name
        assert svg_path.exists()
        assert svg_path.read_text(encoding="utf-8").startswith("<svg")


def test_candidate_score_guarded_policy_split_validation_rejects_empty_policies():
    rows = [
        candidate
        for question_index in range(1, 7)
        for candidate in _question_rows(question_index)
    ]

    try:
        evaluate_candidate_score_guarded_policy_split_validation(
            rows=rows,
            gold_answers_by_question_key=_gold_answers(rows),
            model_name="logistic_best_candidate",
            train_split="train",
            evaluation_split="dev",
            train_fold_count=3,
            max_answer_candidates=1,
            policies=(),
        )
    except ValueError as exc:
        assert "policies must not be empty" in str(exc)
    else:
        raise AssertionError("empty policies should fail")


def _question_rows(question_index: int) -> list[dict]:
    split = "dev" if question_index % 2 else "train"
    route = "other" if question_index != 1 else "how_to_or_lookup"
    selected_candidate_score = 70.0 if question_index % 2 else 50.0
    return [
        _row(
            split=split,
            question_id=f"q{question_index}",
            candidate_rank=1,
            route=route,
            runtime_candidate_score=95.0,
            answer_signal_score=0.1,
            problem_noise_score=1.4,
            candidate_token_f1=0.1,
            is_best=False,
            sentence="Original weak answer.",
        ),
        _row(
            split=split,
            question_id=f"q{question_index}",
            candidate_rank=2,
            route=route,
            runtime_candidate_score=selected_candidate_score,
            answer_signal_score=3.0,
            problem_noise_score=0.0,
            candidate_token_f1=0.8,
            is_best=True,
            sentence="Correct service fix.",
        ),
    ]


def _row(
    split: str,
    question_id: str,
    candidate_rank: int,
    route: str,
    runtime_candidate_score: float,
    answer_signal_score: float,
    problem_noise_score: float,
    candidate_token_f1: float,
    is_best: bool,
    sentence: str,
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
            "candidate_score": runtime_candidate_score,
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
            "question_route": route,
            "document_id": "gold" if is_best else "noise",
            "document_title": "Test document",
            "candidate_sentence": sentence,
        },
    }


def _gold_answers(rows: list[dict]) -> dict[str, str]:
    return {
        f"{row['split']}::{row['question_id']}": "Correct service fix."
        for row in rows
    }
