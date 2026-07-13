from ts_rag_agent.application.candidate_score_guarded_policy_split_validation import (
    CANDIDATE_SCORE_GTE_60_LABEL,
)
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
)
from ts_rag_agent.application.runtime_document_risk_proxy_search import (
    runtime_document_risk_proxy_search_to_dict,
    search_runtime_document_risk_proxies,
)
from ts_rag_agent.application.runtime_document_risk_risk_aware_selection import (
    PRIMARY_OBJECTIVE_LABEL,
    runtime_document_risk_risk_aware_selection_to_dict,
    select_risk_aware_runtime_document_risk_proxy,
    write_runtime_document_risk_risk_aware_visualizations,
)


def test_risk_aware_runtime_proxy_selection_uses_train_cv_constraints():
    rows = [
        candidate
        for question_index in range(1, 7)
        for candidate in _question_rows(question_index)
    ]
    gold_answers = _gold_answers(rows)
    stage43_report = _stage43_report(rows, gold_answers)
    stage44_report = _stage44_report(rows, gold_answers, stage43_report)

    result = select_risk_aware_runtime_document_risk_proxy(
        stage43_report=stage43_report,
        stage44_report=stage44_report,
        rows=rows,
        gold_answers_by_question_key=gold_answers,
        model_name="logistic_best_candidate",
        train_split="train",
        evaluation_split="dev",
        train_fold_count=3,
        max_answer_candidates=1,
    )
    result_dict = runtime_document_risk_risk_aware_selection_to_dict(result)
    score60_train = _guard_metrics(
        result.train_cv_guard_evaluations,
        CANDIDATE_SCORE_GTE_60_LABEL,
    )

    assert result.primary_objective_label == PRIMARY_OBJECTIVE_LABEL
    assert result.primary_train_cv_metrics.regressed_count <= score60_train.regressed_count
    assert (
        result.primary_train_cv_metrics.citation_lost_count
        <= score60_train.citation_lost_count
    )
    assert (
        result.primary_train_cv_metrics.gold_citation_delta
        >= score60_train.gold_citation_delta
    )
    assert result.objective_evaluations
    assert result.findings
    assert result_dict["analysis_scope"].startswith("Stage 45")


def test_risk_aware_runtime_proxy_selection_writes_visualizations(tmp_path):
    rows = [
        candidate
        for question_index in range(1, 7)
        for candidate in _question_rows(question_index)
    ]
    gold_answers = _gold_answers(rows)
    stage43_report = _stage43_report(rows, gold_answers)
    result = select_risk_aware_runtime_document_risk_proxy(
        stage43_report=stage43_report,
        stage44_report=_stage44_report(rows, gold_answers, stage43_report),
        rows=rows,
        gold_answers_by_question_key=gold_answers,
        model_name="logistic_best_candidate",
        train_split="train",
        evaluation_split="dev",
        train_fold_count=3,
        max_answer_candidates=1,
    )

    visualizations = write_runtime_document_risk_risk_aware_visualizations(
        result=result,
        output_dir=tmp_path,
    )

    all_visualizations = [*visualizations.objectives, *visualizations.guards]
    assert len(all_visualizations) == 5
    for visualization in all_visualizations:
        svg_path = tmp_path / visualization.name
        assert svg_path.exists()
        assert svg_path.read_text(encoding="utf-8").startswith("<svg")


def test_risk_aware_runtime_proxy_selection_validates_stage44_report():
    rows = [
        candidate
        for question_index in range(1, 7)
        for candidate in _question_rows(question_index)
    ]
    gold_answers = _gold_answers(rows)
    stage43_report = _stage43_report(rows, gold_answers)
    stage44_report = _stage44_report(rows, gold_answers, stage43_report)
    stage44_report["selected_guard_label"] = "different_guard"

    try:
        select_risk_aware_runtime_document_risk_proxy(
            stage43_report=stage43_report,
            stage44_report=stage44_report,
            rows=rows,
            gold_answers_by_question_key=gold_answers,
            model_name="logistic_best_candidate",
            train_split="train",
            evaluation_split="dev",
            train_fold_count=3,
            max_answer_candidates=1,
        )
    except ValueError as exc:
        assert "selected_guard_label does not match" in str(exc)
    else:
        raise AssertionError("mismatched Stage 44 report should fail")


def _guard_metrics(evaluations, label):
    for evaluation in evaluations:
        if evaluation.label == label:
            return evaluation.metrics
    raise AssertionError(f"missing guard metrics: {label}")


def _stage44_report(
    rows: list[dict],
    gold_answers: dict[str, str],
    stage43_report: dict,
) -> dict:
    return runtime_document_risk_proxy_cv_to_dict(
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
    )


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
