from ts_rag_agent.application.candidate_score_guarded_policy_split_validation import (
    candidate_score_guarded_policy_split_validation_to_dict,
    evaluate_candidate_score_guarded_policy_split_validation,
)
from ts_rag_agent.application.candidate_score_holdout_changed_case_audit import (
    analyze_candidate_score_holdout_changed_cases,
    candidate_score_holdout_changed_case_audit_to_dict,
    write_candidate_score_holdout_changed_case_visualizations,
)


def test_candidate_score_holdout_changed_case_audit_finds_changed_and_residual_cases():
    rows = [
        candidate
        for question_index in range(1, 7)
        for candidate in _question_rows(question_index)
    ]
    gold_answers = _gold_answers(rows)
    stage40_report = candidate_score_guarded_policy_split_validation_to_dict(
        evaluate_candidate_score_guarded_policy_split_validation(
            rows=rows,
            gold_answers_by_question_key=gold_answers,
            model_name="logistic_best_candidate",
            train_split="train",
            evaluation_split="dev",
            train_fold_count=3,
            max_answer_candidates=1,
        )
    )

    audit = analyze_candidate_score_holdout_changed_cases(
        stage40_report=stage40_report,
        rows=rows,
        gold_answers_by_question_key=gold_answers,
        model_name="logistic_best_candidate",
        train_split="train",
        evaluation_split="dev",
        max_answer_candidates=1,
    )
    audit_dict = candidate_score_holdout_changed_case_audit_to_dict(audit)

    assert audit.mode_name == "top1_leading_candidate_rewrite"
    assert audit.metrics.question_count == 3
    assert audit.metrics.changed_case_count >= 1
    assert audit.changed_cases
    assert audit.residual_regression_cases
    assert audit.changed_case_route_summaries
    assert audit.blocked_candidate_score_summaries
    assert audit_dict["analysis_scope"].startswith("Offline Stage 41")


def test_candidate_score_holdout_changed_case_audit_writes_visualizations(tmp_path):
    rows = [
        candidate
        for question_index in range(1, 7)
        for candidate in _question_rows(question_index)
    ]
    gold_answers = _gold_answers(rows)
    stage40_report = candidate_score_guarded_policy_split_validation_to_dict(
        evaluate_candidate_score_guarded_policy_split_validation(
            rows=rows,
            gold_answers_by_question_key=gold_answers,
            model_name="logistic_best_candidate",
            train_split="train",
            evaluation_split="dev",
            train_fold_count=3,
            max_answer_candidates=1,
        )
    )
    audit = analyze_candidate_score_holdout_changed_cases(
        stage40_report=stage40_report,
        rows=rows,
        gold_answers_by_question_key=gold_answers,
        model_name="logistic_best_candidate",
        train_split="train",
        evaluation_split="dev",
        max_answer_candidates=1,
    )

    visualizations = write_candidate_score_holdout_changed_case_visualizations(
        audit=audit,
        output_dir=tmp_path,
    )

    assert len(visualizations) == 5
    for visualization in visualizations:
        svg_path = tmp_path / visualization.name
        assert svg_path.exists()
        assert svg_path.read_text(encoding="utf-8").startswith("<svg")


def test_candidate_score_holdout_changed_case_audit_validates_stage40_report():
    rows = [
        candidate
        for question_index in range(1, 7)
        for candidate in _question_rows(question_index)
    ]
    gold_answers = _gold_answers(rows)
    stage40_report = candidate_score_guarded_policy_split_validation_to_dict(
        evaluate_candidate_score_guarded_policy_split_validation(
            rows=rows,
            gold_answers_by_question_key=gold_answers,
            model_name="logistic_best_candidate",
            train_split="train",
            evaluation_split="dev",
            train_fold_count=3,
            max_answer_candidates=1,
        )
    )
    stage40_report["model_name"] = "different_model"

    try:
        analyze_candidate_score_holdout_changed_cases(
            stage40_report=stage40_report,
            rows=rows,
            gold_answers_by_question_key=gold_answers,
            model_name="logistic_best_candidate",
            train_split="train",
            evaluation_split="dev",
            max_answer_candidates=1,
        )
    except ValueError as exc:
        assert "model_name does not match" in str(exc)
    else:
        raise AssertionError("mismatched Stage 40 report should fail")


def _question_rows(question_index: int) -> list[dict]:
    split = "dev" if question_index % 2 else "train"
    dev_case_kind = {
        1: "block_bad_low_score",
        3: "accept_bad_high_score",
        5: "block_good_low_score",
    }.get(question_index, "train_good")
    if dev_case_kind == "block_bad_low_score":
        baseline_sentence = "Correct service fix."
        selected_sentence = "Wrong service fix."
        selected_score = 50.0
    elif dev_case_kind == "accept_bad_high_score":
        baseline_sentence = "Correct service fix."
        selected_sentence = "Wrong service fix."
        selected_score = 70.0
    elif dev_case_kind == "block_good_low_score":
        baseline_sentence = "Original weak answer."
        selected_sentence = "Correct service fix."
        selected_score = 50.0
    else:
        baseline_sentence = "Original weak answer."
        selected_sentence = "Correct service fix."
        selected_score = 70.0
    return [
        _row(
            split=split,
            question_id=f"q{question_index}",
            candidate_rank=1,
            runtime_candidate_score=95.0,
            answer_signal_score=0.1,
            problem_noise_score=1.4,
            candidate_token_f1=0.1,
            is_best=False,
            sentence=baseline_sentence,
        ),
        _row(
            split=split,
            question_id=f"q{question_index}",
            candidate_rank=2,
            runtime_candidate_score=selected_score,
            answer_signal_score=3.0,
            problem_noise_score=0.0,
            candidate_token_f1=0.8,
            is_best=True,
            sentence=selected_sentence,
        ),
    ]


def _row(
    split: str,
    question_id: str,
    candidate_rank: int,
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
            "question_route": "other",
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
            "question_route": "other",
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
