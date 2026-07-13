from ts_rag_agent.application.candidate_score_holdout_changed_case_audit import (
    analyze_candidate_score_holdout_changed_cases,
    candidate_score_holdout_changed_case_audit_to_dict,
)
from ts_rag_agent.application.document_aware_guard_design import (
    document_aware_guard_design_to_dict,
    evaluate_document_aware_holdout_guards,
)
from ts_rag_agent.application.runtime_document_risk_proxy_cv import (
    runtime_document_risk_proxy_cv_to_dict,
    select_runtime_document_risk_proxy_with_train_cv,
    write_runtime_document_risk_proxy_cv_visualizations,
)
from ts_rag_agent.application.runtime_document_risk_proxy_search import (
    runtime_document_risk_proxy_search_to_dict,
    search_runtime_document_risk_proxies,
)


def test_runtime_document_risk_proxy_cv_selects_from_train_only_cv():
    rows = [
        candidate
        for question_index in range(1, 7)
        for candidate in _question_rows(question_index)
    ]
    gold_answers = _gold_answers(rows)

    result = select_runtime_document_risk_proxy_with_train_cv(
        stage43_report=_stage43_report(rows, gold_answers),
        rows=rows,
        gold_answers_by_question_key=gold_answers,
        model_name="logistic_best_candidate",
        train_split="train",
        evaluation_split="dev",
        train_fold_count=3,
        max_answer_candidates=1,
    )
    result_dict = runtime_document_risk_proxy_cv_to_dict(result)
    expected_train_best = max(
        result.train_cv_guard_evaluations,
        key=lambda evaluation: (
            evaluation.metrics.average_delta_vs_baseline,
            -evaluation.metrics.regressed_count,
            -evaluation.metrics.citation_lost_count,
            evaluation.metrics.gold_citation_delta,
            evaluation.metrics.policy_average_answer_token_f1,
            evaluation.label,
        ),
    )

    assert result.selected_guard_label == expected_train_best.label
    assert result.train_split == "train"
    assert result.evaluation_split == "dev"
    assert result.train_question_count == 3
    assert result.evaluation_question_count == 3
    assert {
        evaluation.feature_scope for evaluation in result.train_cv_guard_evaluations
    } == {"runtime_available_features_only"}
    assert result.findings
    assert result_dict["analysis_scope"].startswith("Stage 44")


def test_runtime_document_risk_proxy_cv_writes_visualizations(tmp_path):
    rows = [
        candidate
        for question_index in range(1, 7)
        for candidate in _question_rows(question_index)
    ]
    gold_answers = _gold_answers(rows)
    result = select_runtime_document_risk_proxy_with_train_cv(
        stage43_report=_stage43_report(rows, gold_answers),
        rows=rows,
        gold_answers_by_question_key=gold_answers,
        model_name="logistic_best_candidate",
        train_split="train",
        evaluation_split="dev",
        train_fold_count=3,
        max_answer_candidates=1,
    )

    visualizations = write_runtime_document_risk_proxy_cv_visualizations(
        result=result,
        output_dir=tmp_path,
    )

    all_visualizations = [
        *visualizations.train_cv,
        *visualizations.holdout,
        *visualizations.selected,
    ]
    assert len(all_visualizations) == 5
    for visualization in all_visualizations:
        svg_path = tmp_path / visualization.name
        assert svg_path.exists()
        assert svg_path.read_text(encoding="utf-8").startswith("<svg")


def test_runtime_document_risk_proxy_cv_validates_stage43_report():
    rows = [
        candidate
        for question_index in range(1, 7)
        for candidate in _question_rows(question_index)
    ]
    gold_answers = _gold_answers(rows)
    stage43_report = _stage43_report(rows, gold_answers)
    stage43_report["evaluation_split"] = "train"

    try:
        select_runtime_document_risk_proxy_with_train_cv(
            stage43_report=stage43_report,
            rows=rows,
            gold_answers_by_question_key=gold_answers,
            model_name="logistic_best_candidate",
            train_split="train",
            evaluation_split="dev",
            train_fold_count=3,
            max_answer_candidates=1,
        )
    except ValueError as exc:
        assert "evaluation_split does not match" in str(exc)
    else:
        raise AssertionError("mismatched Stage 43 report should fail")


def _stage43_report(rows: list[dict], gold_answers: dict[str, str]) -> dict:
    return runtime_document_risk_proxy_search_to_dict(
        search_runtime_document_risk_proxies(
            stage42_report=_stage42_report(rows, gold_answers),
            rows=rows,
            gold_answers_by_question_key=gold_answers,
            model_name="logistic_best_candidate",
            train_split="train",
            evaluation_split="dev",
            max_answer_candidates=1,
            probe_question_ids=("q1", "q3"),
        )
    )


def _stage42_report(rows: list[dict], gold_answers: dict[str, str]) -> dict:
    return document_aware_guard_design_to_dict(
        evaluate_document_aware_holdout_guards(
            stage41_report=_stage41_report(rows, gold_answers),
            rows=rows,
            gold_answers_by_question_key=gold_answers,
            model_name="logistic_best_candidate",
            train_split="train",
            evaluation_split="dev",
            max_answer_candidates=1,
        )
    )


def _stage41_report(rows: list[dict], gold_answers: dict[str, str]) -> dict:
    return candidate_score_holdout_changed_case_audit_to_dict(
        analyze_candidate_score_holdout_changed_cases(
            stage40_report=_minimal_stage40_report(rows, gold_answers),
            rows=rows,
            gold_answers_by_question_key=gold_answers,
            model_name="logistic_best_candidate",
            train_split="train",
            evaluation_split="dev",
            max_answer_candidates=1,
        )
    )


def _minimal_stage40_report(rows: list[dict], gold_answers: dict[str, str]) -> dict:
    from ts_rag_agent.application.candidate_score_guarded_policy_split_validation import (
        candidate_score_guarded_policy_split_validation_to_dict,
        evaluate_candidate_score_guarded_policy_split_validation,
    )

    return candidate_score_guarded_policy_split_validation_to_dict(
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


def _question_rows(question_index: int) -> list[dict]:
    split = "dev" if question_index % 2 else "train"
    dev_case_kind = {
        1: "block_bad_low_score",
        3: "accept_bad_mid_score",
        5: "block_good_low_score",
    }.get(question_index, "train_good")
    if dev_case_kind == "block_bad_low_score":
        baseline_sentence = "Correct service fix."
        selected_sentence = "Wrong service fix."
        selected_score = 50.0
    elif dev_case_kind == "accept_bad_mid_score":
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
