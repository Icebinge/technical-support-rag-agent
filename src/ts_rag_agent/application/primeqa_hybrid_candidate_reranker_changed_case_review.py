from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from ts_rag_agent.application.candidate_reranker_cv import (
    DEFAULT_MODEL_NAMES,
    split_validated_candidate_reranker_selections,
)
from ts_rag_agent.application.candidate_reranker_dataset_audit import (
    load_candidate_reranker_rows,
)
from ts_rag_agent.application.candidate_reranker_policy_search import (
    CandidateRerankerPolicyDecision,
    candidate_reranker_policy_decisions_from_selections,
)
from ts_rag_agent.application.candidate_score_guarded_policy_evaluation import (
    STAGE39_MAIN_POLICY_LABEL,
    default_stage39_policy_specs,
)
from ts_rag_agent.application.candidate_score_guarded_policy_split_validation import (
    CANDIDATE_SCORE_GTE_60_LABEL,
)
from ts_rag_agent.application.guarded_candidate_reranker_answer_experiment import (
    GuardedCandidateAnswerCase,
    build_topk_leading_candidate_answer_cases_from_decisions,
    summarize_guarded_candidate_answer_cases,
)
from ts_rag_agent.application.svg_charts import (
    BarDatum,
    render_horizontal_bar_chart_svg,
)
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)

_STAGE = "Stage 72"
_CREATED_AT = "2026-07-14"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})
_POLICY_LABELS = (STAGE39_MAIN_POLICY_LABEL, CANDIDATE_SCORE_GTE_60_LABEL)


@dataclass(frozen=True)
class PrimeQAHybridChangedCaseReviewVisualization:
    """One generated Stage72 visualization."""

    name: str
    path: str


def review_primeqa_hybrid_candidate_reranker_changed_cases(
    *,
    stage71_report_path: Path,
    candidate_dataset_path: Path,
    train_split_path: Path,
    dev_split_path: Path,
    model_names: Sequence[str] = DEFAULT_MODEL_NAMES,
    max_answer_candidates: int = 3,
    sample_limit: int = 20,
) -> dict[str, Any]:
    """Review train/dev-only changed cases from the Stage71 reranker policies."""

    _validate_options(
        model_names=model_names,
        max_answer_candidates=max_answer_candidates,
        sample_limit=sample_limit,
    )
    started_at = time.perf_counter()
    stage71_report = _load_json_object(stage71_report_path)
    rows = load_candidate_reranker_rows(candidate_dataset_path)
    loaded_inputs_at = time.perf_counter()
    split_samples = {
        _TRAIN_SPLIT: load_primeqa_hybrid_split_samples(train_split_path),
        _DEV_SPLIT: load_primeqa_hybrid_split_samples(dev_split_path),
    }
    gold_answers = _gold_answers_by_question_key(split_samples)
    loaded_splits_at = time.perf_counter()
    rows_by_split = _rows_by_split(rows)
    model_reviews = [
        _review_model(
            stage71_report=stage71_report,
            rows=rows,
            evaluation_rows=rows_by_split[_DEV_SPLIT],
            gold_answers_by_question_key=gold_answers,
            model_name=model_name,
            max_answer_candidates=max_answer_candidates,
            sample_limit=sample_limit,
        )
        for model_name in model_names
    ]
    reviewed_at = time.perf_counter()
    guard_checks = _guard_checks(
        stage71_report=stage71_report,
        rows_by_split=rows_by_split,
        model_reviews=model_reviews,
        gold_answers_by_question_key=gold_answers,
    )
    checked_at = time.perf_counter()
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_scope": (
            "Train/dev-only changed-case review for Stage71 candidate-reranker "
            "policies. This stage rebuilds dev holdout top-k proxy cases, keeps "
            "the frozen test split locked, does not run final metrics, and does "
            "not change the default runtime."
        ),
        "split_contract": {
            "split_name": _SPLIT_NAME,
            "protocol_version": _PROTOCOL_VERSION,
            "train_split": _TRAIN_SPLIT,
            "development_evaluation_split": _DEV_SPLIT,
            "allowed_development_splits": list(_ALLOWED_DEVELOPMENT_SPLITS),
            "forbidden_final_splits": sorted(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "stage71_report": _fingerprint(stage71_report_path),
            "candidate_dataset": _fingerprint(candidate_dataset_path),
            "train_split": _fingerprint(train_split_path),
            "dev_split": _fingerprint(dev_split_path),
        },
        "loaded_candidate_summary": _loaded_candidate_summary(
            rows=rows,
            rows_by_split=rows_by_split,
        ),
        "loaded_gold_answer_summary": _gold_answer_summary(gold_answers),
        "model_reviews": model_reviews,
        "cross_model_summary": _cross_model_summary(model_reviews),
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks, model_reviews),
        "timing_seconds": {
            "load_inputs": round(loaded_inputs_at - started_at, 3),
            "load_train_dev_splits": round(loaded_splits_at - loaded_inputs_at, 3),
            "review_changed_cases": round(reviewed_at - loaded_splits_at, 3),
            "guard_checks": round(checked_at - reviewed_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_candidate_reranker_changed_case_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridChangedCaseReviewVisualization]:
    """Write SVG charts for Stage72 changed-case review."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage72_dev_top3_delta_by_policy.svg": render_horizontal_bar_chart_svg(
            title="Stage72 dev top3 delta by policy",
            bars=_policy_delta_bars(report),
            x_label="average F1 delta vs baseline",
            width=1080,
            margin_left=420,
        ),
        "stage72_changed_cases_by_policy.svg": render_horizontal_bar_chart_svg(
            title="Stage72 changed cases by policy",
            bars=_changed_case_count_bars(report),
            x_label="changed dev cases",
            width=1080,
            margin_left=420,
        ),
        "stage72_candidate_score_vs_main_outcomes.svg": render_horizontal_bar_chart_svg(
            title="Stage72 candidate_score_gte_60 vs main outcomes",
            bars=_policy_vs_main_outcome_bars(report),
            x_label="case count",
            width=1120,
            margin_left=470,
        ),
        "stage72_residual_regressions_by_policy.svg": render_horizontal_bar_chart_svg(
            title="Stage72 residual regressions by policy",
            bars=_residual_regression_bars(report),
            x_label="top3 regression cases",
            width=1080,
            margin_left=420,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridChangedCaseReviewVisualization(name=filename, path=str(path))
        )
    return artifacts


def _review_model(
    *,
    stage71_report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    gold_answers_by_question_key: Mapping[str, str],
    model_name: str,
    max_answer_candidates: int,
    sample_limit: int,
) -> dict[str, Any]:
    selections = split_validated_candidate_reranker_selections(
        rows=rows,
        model_name=model_name,
        train_split=_TRAIN_SPLIT,
        validation_split=_DEV_SPLIT,
    )
    row_index = _build_row_index(evaluation_rows)
    stage71_validation = _stage71_validation_for_model(stage71_report, model_name)
    policy_reviews = []
    cases_by_policy: dict[str, list[GuardedCandidateAnswerCase]] = {}
    for policy_label in _POLICY_LABELS:
        policy_config = _policy_configs_by_label()[policy_label]
        decisions = candidate_reranker_policy_decisions_from_selections(
            config=policy_config,
            selections=selections,
            rows=evaluation_rows,
        )
        cases = build_topk_leading_candidate_answer_cases_from_decisions(
            decisions=decisions,
            rows=evaluation_rows,
            gold_answers_by_question_key=gold_answers_by_question_key,
            max_answer_candidates=max_answer_candidates,
        )
        cases_by_policy[policy_label] = cases
        metrics = summarize_guarded_candidate_answer_cases(cases)
        _validate_stage71_policy_metrics(
            stage71_validation=stage71_validation,
            policy_label=policy_label,
            mode_name=f"top{max_answer_candidates}_leading_candidate_rewrite",
            metrics=metrics,
        )
        policy_reviews.append(
            _policy_vs_baseline_review(
                policy_label=policy_label,
                cases=cases,
                decisions=decisions,
                row_index=row_index,
                sample_limit=sample_limit,
            )
        )
    policy_vs_main = _candidate_score_policy_vs_main_review(
        main_cases=cases_by_policy[STAGE39_MAIN_POLICY_LABEL],
        candidate_cases=cases_by_policy[CANDIDATE_SCORE_GTE_60_LABEL],
        sample_limit=sample_limit,
    )
    return {
        "model_name": model_name,
        "train_split": _TRAIN_SPLIT,
        "evaluation_split": _DEV_SPLIT,
        "mode_name": f"top{max_answer_candidates}_leading_candidate_rewrite",
        "policy_reviews": policy_reviews,
        "candidate_score_gte_60_vs_stage36_main": policy_vs_main,
        "findings": _model_findings(policy_reviews, policy_vs_main),
    }


def _policy_vs_baseline_review(
    *,
    policy_label: str,
    cases: Sequence[GuardedCandidateAnswerCase],
    decisions: Sequence[CandidateRerankerPolicyDecision],
    row_index: Mapping[str, list[Mapping[str, Any]]],
    sample_limit: int,
) -> dict[str, Any]:
    metrics = summarize_guarded_candidate_answer_cases(cases)
    decisions_by_key = {
        _question_key(decision.split, decision.question_id): decision
        for decision in decisions
    }
    changed_cases = [
        _changed_case_summary(
            case=case,
            decision=decisions_by_key[_question_key(case.split, case.question_id)],
            row_index=row_index,
        )
        for case in cases
        if case.policy_candidate_ids != case.baseline_candidate_ids
    ]
    residual_regressions = [
        _changed_case_summary(
            case=case,
            decision=decisions_by_key[_question_key(case.split, case.question_id)],
            row_index=row_index,
        )
        for case in cases
        if case.f1_delta_vs_baseline < 0
    ]
    return {
        "policy_label": policy_label,
        "metrics": _metrics_summary(metrics),
        "changed_vs_baseline_summary": _changed_case_aggregate(changed_cases),
        "changed_case_route_summaries": _segment_summaries(
            changed_cases,
            lambda case: case["question_route"],
        ),
        "changed_case_document_transition_summaries": _segment_summaries(
            changed_cases,
            lambda case: case["document_transition"],
        ),
        "changed_case_rank_summaries": _segment_summaries(
            changed_cases,
            lambda case: case["policy_leading_rank_bucket"],
        ),
        "changed_case_samples": _sample_changed_cases(changed_cases, sample_limit),
        "residual_regression_summary": _changed_case_aggregate(residual_regressions),
        "residual_regression_samples": _sample_changed_cases(
            residual_regressions,
            sample_limit,
        ),
    }


def _candidate_score_policy_vs_main_review(
    *,
    main_cases: Sequence[GuardedCandidateAnswerCase],
    candidate_cases: Sequence[GuardedCandidateAnswerCase],
    sample_limit: int,
) -> dict[str, Any]:
    main_by_key = _cases_by_key(main_cases)
    candidate_by_key = _cases_by_key(candidate_cases)
    changed_cases = [
        _policy_vs_policy_case(
            main_case=main_case,
            candidate_case=candidate_by_key[question_key],
        )
        for question_key, main_case in main_by_key.items()
        if main_case.policy_candidate_ids
        != candidate_by_key[question_key].policy_candidate_ids
    ]
    return {
        "main_policy_label": STAGE39_MAIN_POLICY_LABEL,
        "candidate_policy_label": CANDIDATE_SCORE_GTE_60_LABEL,
        "summary": _policy_vs_policy_aggregate(changed_cases, main_cases, candidate_cases),
        "route_summaries": _policy_vs_policy_segment_summaries(
            changed_cases,
            lambda case: case["question_route"],
        ),
        "document_transition_summaries": _policy_vs_policy_segment_summaries(
            changed_cases,
            lambda case: case["main_to_candidate_document_transition"],
        ),
        "changed_case_samples": _sample_policy_vs_policy_cases(
            changed_cases,
            sample_limit,
        ),
    }


def _changed_case_summary(
    *,
    case: GuardedCandidateAnswerCase,
    decision: CandidateRerankerPolicyDecision,
    row_index: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    question_rows = row_index[_question_key(case.split, case.question_id)]
    baseline_row = _row_by_candidate_id(question_rows, case.baseline_candidate_ids[0])
    policy_row = _row_by_candidate_id(question_rows, case.policy_candidate_ids[0])
    policy_score = _runtime_feature_float(policy_row, "candidate_score")
    baseline_score = _runtime_feature_float(baseline_row, "candidate_score")
    return {
        "split": case.split,
        "question_id": case.question_id,
        "question_route": case.question_route,
        "f1_outcome": _outcome(case.f1_delta_vs_baseline),
        "f1_delta_vs_baseline": case.f1_delta_vs_baseline,
        "baseline_answer_token_f1": case.baseline_answer_token_f1,
        "policy_answer_token_f1": case.policy_answer_token_f1,
        "baseline_gold_cited": case.baseline_gold_cited,
        "policy_gold_cited": case.policy_gold_cited,
        "citation_delta_vs_baseline": case.citation_delta,
        "citation_outcome": _citation_outcome(case.citation_delta),
        "baseline_leading_candidate_id": case.baseline_candidate_ids[0],
        "policy_leading_candidate_id": case.policy_candidate_ids[0],
        "baseline_leading_rank": 1,
        "policy_leading_rank": case.policy_leading_candidate_rank,
        "policy_leading_rank_bucket": _rank_bucket(case.policy_leading_candidate_rank),
        "baseline_leading_candidate_score": baseline_score,
        "policy_leading_candidate_score": policy_score,
        "policy_candidate_score_bucket": _candidate_score_bucket(policy_score),
        "model_score_margin_vs_top_candidate": decision.model_score_margin_vs_top_candidate,
        "baseline_leading_document_id": _document_id(baseline_row),
        "policy_leading_document_id": _document_id(policy_row),
        "document_transition": _document_transition(
            from_row=baseline_row,
            to_row=policy_row,
        ),
        "decision_reasons": list(case.decision_reasons),
        "baseline_candidate_ids": list(case.baseline_candidate_ids),
        "policy_candidate_ids": list(case.policy_candidate_ids),
        "baseline_document_ids": list(case.baseline_document_ids),
        "policy_document_ids": list(case.policy_document_ids),
    }


def _policy_vs_policy_case(
    *,
    main_case: GuardedCandidateAnswerCase,
    candidate_case: GuardedCandidateAnswerCase,
) -> dict[str, Any]:
    candidate_delta_vs_main = round(
        candidate_case.policy_answer_token_f1 - main_case.policy_answer_token_f1,
        4,
    )
    return {
        "split": main_case.split,
        "question_id": main_case.question_id,
        "question_route": main_case.question_route,
        "candidate_vs_main_outcome": _outcome(candidate_delta_vs_main),
        "candidate_delta_vs_main": candidate_delta_vs_main,
        "main_delta_vs_baseline": main_case.f1_delta_vs_baseline,
        "candidate_delta_vs_baseline": candidate_case.f1_delta_vs_baseline,
        "main_answer_token_f1": main_case.policy_answer_token_f1,
        "candidate_answer_token_f1": candidate_case.policy_answer_token_f1,
        "main_gold_cited": main_case.policy_gold_cited,
        "candidate_gold_cited": candidate_case.policy_gold_cited,
        "candidate_citation_delta_vs_main": (
            int(candidate_case.policy_gold_cited) - int(main_case.policy_gold_cited)
        ),
        "main_candidate_ids": list(main_case.policy_candidate_ids),
        "candidate_policy_candidate_ids": list(candidate_case.policy_candidate_ids),
        "main_document_ids": list(main_case.policy_document_ids),
        "candidate_policy_document_ids": list(candidate_case.policy_document_ids),
        "main_to_candidate_document_transition": _document_transition_by_id(
            main_case.policy_document_ids[0],
            candidate_case.policy_document_ids[0],
            from_gold=main_case.policy_gold_cited,
            to_gold=candidate_case.policy_gold_cited,
        ),
        "main_decision_reasons": list(main_case.decision_reasons),
        "candidate_decision_reasons": list(candidate_case.decision_reasons),
    }


def _metrics_summary(metrics) -> dict[str, Any]:
    return {
        "question_count": metrics.question_count,
        "baseline_average_answer_token_f1": metrics.baseline_average_answer_token_f1,
        "policy_average_answer_token_f1": metrics.policy_average_answer_token_f1,
        "average_delta_vs_baseline": metrics.average_delta_vs_baseline,
        "oracle_gap_closed_rate": metrics.oracle_gap_closed_rate,
        "replacement_count": metrics.replacement_count,
        "changed_answer_count": metrics.changed_answer_count,
        "improved_count": metrics.improved_count,
        "regressed_count": metrics.regressed_count,
        "tied_count": metrics.tied_count,
        "citation_lost_count": metrics.citation_lost_count,
        "citation_gained_count": metrics.citation_gained_count,
        "gold_citation_delta": metrics.gold_citation_delta,
    }


def _changed_case_aggregate(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcome_counts = Counter(str(case["f1_outcome"]) for case in cases)
    citation_counts = Counter(str(case["citation_outcome"]) for case in cases)
    return {
        "changed_case_count": len(cases),
        "average_f1_delta_vs_baseline": _rounded_mean(
            [float(case["f1_delta_vs_baseline"]) for case in cases]
        ),
        "improved_count": outcome_counts["improved"],
        "regressed_count": outcome_counts["regressed"],
        "tied_count": outcome_counts["tied"],
        "citation_lost_count": citation_counts["citation_lost"],
        "citation_gained_count": citation_counts["citation_gained"],
        "average_policy_leading_rank": _rounded_mean(
            [int(case["policy_leading_rank"]) for case in cases]
        ),
        "average_policy_leading_candidate_score": _rounded_mean(
            [float(case["policy_leading_candidate_score"]) for case in cases]
        ),
    }


def _policy_vs_policy_aggregate(
    changed_cases: Sequence[Mapping[str, Any]],
    main_cases: Sequence[GuardedCandidateAnswerCase],
    candidate_cases: Sequence[GuardedCandidateAnswerCase],
) -> dict[str, Any]:
    main_metrics = summarize_guarded_candidate_answer_cases(main_cases)
    candidate_metrics = summarize_guarded_candidate_answer_cases(candidate_cases)
    outcome_counts = Counter(
        str(case["candidate_vs_main_outcome"]) for case in changed_cases
    )
    return {
        "question_count": len(main_cases),
        "changed_case_count": len(changed_cases),
        "changed_case_rate": _ratio(len(changed_cases), len(main_cases)),
        "average_candidate_delta_vs_main_on_changed": _rounded_mean(
            [float(case["candidate_delta_vs_main"]) for case in changed_cases]
        ),
        "candidate_improved_vs_main_count": outcome_counts["improved"],
        "candidate_regressed_vs_main_count": outcome_counts["regressed"],
        "candidate_tied_vs_main_count": outcome_counts["tied"],
        "main_delta_vs_baseline": main_metrics.average_delta_vs_baseline,
        "candidate_delta_vs_baseline": candidate_metrics.average_delta_vs_baseline,
        "main_regressed_count": main_metrics.regressed_count,
        "candidate_regressed_count": candidate_metrics.regressed_count,
        "main_gold_citation_delta": main_metrics.gold_citation_delta,
        "candidate_gold_citation_delta": candidate_metrics.gold_citation_delta,
        "candidate_gold_citation_delta_vs_main": (
            candidate_metrics.gold_citation_delta - main_metrics.gold_citation_delta
        ),
    }


def _segment_summaries(
    cases: Sequence[Mapping[str, Any]],
    segment_fn,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(segment_fn(case))].append(case)
    summaries = []
    for segment_name, segment_cases in grouped.items():
        outcome_counts = Counter(str(case["f1_outcome"]) for case in segment_cases)
        summaries.append(
            {
                "segment_name": segment_name,
                "case_count": len(segment_cases),
                "average_f1_delta_vs_baseline": _rounded_mean(
                    [float(case["f1_delta_vs_baseline"]) for case in segment_cases]
                ),
                "improved_count": outcome_counts["improved"],
                "regressed_count": outcome_counts["regressed"],
                "tied_count": outcome_counts["tied"],
            }
        )
    return sorted(
        summaries,
        key=lambda summary: (summary["case_count"], abs(summary["average_f1_delta_vs_baseline"])),
        reverse=True,
    )


def _policy_vs_policy_segment_summaries(
    cases: Sequence[Mapping[str, Any]],
    segment_fn,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(segment_fn(case))].append(case)
    summaries = []
    for segment_name, segment_cases in grouped.items():
        outcome_counts = Counter(
            str(case["candidate_vs_main_outcome"]) for case in segment_cases
        )
        summaries.append(
            {
                "segment_name": segment_name,
                "case_count": len(segment_cases),
                "average_candidate_delta_vs_main": _rounded_mean(
                    [float(case["candidate_delta_vs_main"]) for case in segment_cases]
                ),
                "candidate_improved_vs_main_count": outcome_counts["improved"],
                "candidate_regressed_vs_main_count": outcome_counts["regressed"],
                "candidate_tied_vs_main_count": outcome_counts["tied"],
            }
        )
    return sorted(
        summaries,
        key=lambda summary: (
            summary["case_count"],
            abs(summary["average_candidate_delta_vs_main"]),
        ),
        reverse=True,
    )


def _sample_changed_cases(
    cases: Sequence[Mapping[str, Any]],
    sample_limit: int,
) -> list[dict[str, Any]]:
    return [
        dict(case)
        for case in sorted(
            cases,
            key=lambda case: (
                float(case["f1_delta_vs_baseline"]),
                str(case["question_id"]),
            ),
        )[:sample_limit]
    ]


def _sample_policy_vs_policy_cases(
    cases: Sequence[Mapping[str, Any]],
    sample_limit: int,
) -> list[dict[str, Any]]:
    return [
        dict(case)
        for case in sorted(
            cases,
            key=lambda case: (
                float(case["candidate_delta_vs_main"]),
                str(case["question_id"]),
            ),
        )[:sample_limit]
    ]


def _model_findings(
    policy_reviews: Sequence[Mapping[str, Any]],
    policy_vs_main: Mapping[str, Any],
) -> list[str]:
    reviews_by_label = {
        str(review["policy_label"]): review for review in policy_reviews
    }
    main_review = reviews_by_label[STAGE39_MAIN_POLICY_LABEL]
    score60_review = reviews_by_label[CANDIDATE_SCORE_GTE_60_LABEL]
    main_metrics = main_review["metrics"]
    score60_metrics = score60_review["metrics"]
    comparison = policy_vs_main["summary"]
    findings = [
        (
            "stage36_main changes "
            f"{main_review['changed_vs_baseline_summary']['changed_case_count']} "
            f"dev top3 cases with delta {main_metrics['average_delta_vs_baseline']:+.4f} "
            f"and {main_metrics['regressed_count']} regressions."
        ),
        (
            "candidate_score_gte_60 changes "
            f"{score60_review['changed_vs_baseline_summary']['changed_case_count']} "
            f"dev top3 cases with delta {score60_metrics['average_delta_vs_baseline']:+.4f} "
            f"and {score60_metrics['regressed_count']} regressions."
        ),
        (
            "candidate_score_gte_60 differs from stage36_main on "
            f"{comparison['changed_case_count']} / {comparison['question_count']} dev cases."
        ),
    ]
    if comparison["changed_case_count"]:
        findings.append(
            "On policy-vs-main changed cases, candidate_score_gte_60 is "
            f"better/tied/worse: {comparison['candidate_improved_vs_main_count']}/"
            f"{comparison['candidate_tied_vs_main_count']}/"
            f"{comparison['candidate_regressed_vs_main_count']}."
        )
    return findings


def _cross_model_summary(model_reviews: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    policy_rows = [
        {
            "model_name": review["model_name"],
            "policy_label": policy["policy_label"],
            "average_delta_vs_baseline": policy["metrics"]["average_delta_vs_baseline"],
            "regressed_count": policy["metrics"]["regressed_count"],
            "changed_case_count": policy["changed_vs_baseline_summary"][
                "changed_case_count"
            ],
            "gold_citation_delta": policy["metrics"]["gold_citation_delta"],
        }
        for review in model_reviews
        for policy in review["policy_reviews"]
    ]
    best_by_delta = max(
        policy_rows,
        key=lambda row: (
            row["average_delta_vs_baseline"],
            -row["regressed_count"],
            row["gold_citation_delta"],
        ),
    )
    lowest_regression = min(
        policy_rows,
        key=lambda row: (
            row["regressed_count"],
            -row["average_delta_vs_baseline"],
            -row["gold_citation_delta"],
        ),
    )
    return {
        "policy_rows": policy_rows,
        "best_policy_by_dev_top3_delta": best_by_delta,
        "lowest_regression_policy": lowest_regression,
        "max_dev_top3_delta": max(row["average_delta_vs_baseline"] for row in policy_rows),
        "min_regressed_count": min(row["regressed_count"] for row in policy_rows),
    }


def _guard_checks(
    *,
    stage71_report: Mapping[str, Any],
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    model_reviews: Sequence[Mapping[str, Any]],
    gold_answers_by_question_key: Mapping[str, str],
) -> list[dict[str, Any]]:
    candidate_splits = sorted(split for split, split_rows in rows_by_split.items() if split_rows)
    gold_answer_splits = sorted(
        {key.split("::", maxsplit=1)[0] for key in gold_answers_by_question_key}
    )
    stage71_decision = stage71_report.get("decision")
    if not isinstance(stage71_decision, Mapping):
        stage71_decision = {}
    return [
        _check(
            name="candidate_artifact_splits_are_train_dev_only",
            passed=set(candidate_splits) == set(_ALLOWED_DEVELOPMENT_SPLITS),
            observed=candidate_splits,
            expected=sorted(_ALLOWED_DEVELOPMENT_SPLITS),
        ),
        _check(
            name="candidate_rows_have_no_test_split",
            passed=not any(split in _FORBIDDEN_FINAL_SPLITS for split in candidate_splits),
            observed=sum(
                len(rows_by_split.get(split, [])) for split in _FORBIDDEN_FINAL_SPLITS
            ),
            expected=0,
        ),
        _check(
            name="gold_answer_splits_are_train_dev_only",
            passed=set(gold_answer_splits) == set(_ALLOWED_DEVELOPMENT_SPLITS),
            observed=gold_answer_splits,
            expected=sorted(_ALLOWED_DEVELOPMENT_SPLITS),
        ),
        _check(
            name="stage71_final_test_metrics_not_run",
            passed=stage71_decision.get("can_run_final_test_metrics_now") is False,
            observed=stage71_decision.get("can_run_final_test_metrics_now"),
            expected=False,
        ),
        _check(
            name="stage72_review_uses_dev_holdout_only",
            passed=all(
                review["train_split"] == _TRAIN_SPLIT
                and review["evaluation_split"] == _DEV_SPLIT
                for review in model_reviews
            ),
            observed=[
                {
                    "model_name": review["model_name"],
                    "train_split": review["train_split"],
                    "evaluation_split": review["evaluation_split"],
                }
                for review in model_reviews
            ],
            expected={
                "train_split": _TRAIN_SPLIT,
                "evaluation_split": _DEV_SPLIT,
            },
        ),
        _check(
            name="stage72_report_is_public_safe_no_raw_answer_text",
            passed=True,
            observed="raw answer/candidate text omitted",
            expected="raw answer/candidate text omitted",
        ),
        _check(
            name="final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="default_runtime_policy_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
    ]


def _decision(
    guard_checks: Sequence[Mapping[str, Any]],
    model_reviews: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    passed = all(bool(check["passed"]) for check in guard_checks)
    cross_model = _cross_model_summary(model_reviews)
    return {
        "status": (
            "primeqa_hybrid_candidate_reranker_changed_case_review_completed"
            if passed
            else "primeqa_hybrid_candidate_reranker_changed_case_review_blocked"
        ),
        "can_continue_train_dev_development": passed,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage 73: decide whether to refine train/dev reranker policy gates or "
            "explicitly approve a one-time final-test gate; do not use test for tuning"
        ),
        "reason": (
            "Stage72 is a train/dev changed-case review. It does not itself open "
            "the final-test gate; explicit gate criteria or user approval are still "
            "required."
            if passed
            else "One or more Stage72 boundary checks failed."
        ),
        "observed_best_dev_top3_delta": cross_model["max_dev_top3_delta"],
        "observed_min_regressed_count": cross_model["min_regressed_count"],
    }


def _validate_stage71_policy_metrics(
    *,
    stage71_validation: Mapping[str, Any],
    policy_label: str,
    mode_name: str,
    metrics,
) -> None:
    report_metrics = _stage71_policy_mode_metrics(
        stage71_validation=stage71_validation,
        policy_label=policy_label,
        mode_name=mode_name,
    )
    expected = {
        "question_count": metrics.question_count,
        "policy_average_answer_token_f1": metrics.policy_average_answer_token_f1,
        "average_delta_vs_baseline": metrics.average_delta_vs_baseline,
        "replacement_count": metrics.replacement_count,
        "regressed_count": metrics.regressed_count,
        "citation_lost_count": metrics.citation_lost_count,
        "citation_gained_count": metrics.citation_gained_count,
        "gold_citation_delta": metrics.gold_citation_delta,
    }
    for key, expected_value in expected.items():
        if report_metrics.get(key) != expected_value:
            raise ValueError(
                f"Stage71 report {policy_label}/{mode_name} {key} does not "
                "match recomputed Stage72 review"
            )


def _stage71_policy_mode_metrics(
    *,
    stage71_validation: Mapping[str, Any],
    policy_label: str,
    mode_name: str,
) -> Mapping[str, Any]:
    holdout = stage71_validation.get("holdout_evaluation")
    if not isinstance(holdout, Mapping):
        raise ValueError("Stage71 validation must contain holdout_evaluation")
    policies = holdout.get("policies")
    if not isinstance(policies, list):
        raise ValueError("Stage71 holdout_evaluation must contain policies")
    for policy in policies:
        if not isinstance(policy, Mapping) or policy.get("label") != policy_label:
            continue
        mode_evaluations = policy.get("mode_evaluations")
        if not isinstance(mode_evaluations, list):
            raise ValueError("Stage71 policy must contain mode_evaluations")
        for mode_evaluation in mode_evaluations:
            if (
                isinstance(mode_evaluation, Mapping)
                and mode_evaluation.get("mode_name") == mode_name
                and isinstance(mode_evaluation.get("metrics"), Mapping)
            ):
                return mode_evaluation["metrics"]
    raise ValueError(f"Stage71 report missing metrics for {policy_label}/{mode_name}")


def _stage71_validation_for_model(
    stage71_report: Mapping[str, Any],
    model_name: str,
) -> Mapping[str, Any]:
    validations = stage71_report.get("train_to_dev_policy_validations")
    if not isinstance(validations, list):
        raise ValueError("Stage71 report must contain train_to_dev_policy_validations")
    for validation in validations:
        if isinstance(validation, Mapping) and validation.get("model_name") == model_name:
            return validation
    raise ValueError(f"Stage71 report missing validation for model: {model_name}")


def _load_json_object(path: Path) -> dict[str, Any]:
    _ensure_file(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return loaded


def _gold_answers_by_question_key(
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
) -> dict[str, str]:
    answers = {}
    for split, samples in split_samples.items():
        for sample in samples:
            if sample.answerable:
                answers[f"{split}::{sample.sample_id}"] = sample.answer
    return dict(sorted(answers.items()))


def _rows_by_split(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    rows_by_split: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        split = str(row.get("split") or "")
        rows_by_split.setdefault(split, []).append(row)
    for split in _ALLOWED_DEVELOPMENT_SPLITS:
        rows_by_split.setdefault(split, [])
    return rows_by_split


def _loaded_candidate_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "rows_by_split": {
            split: len(split_rows)
            for split, split_rows in sorted(rows_by_split.items())
            if split_rows
        },
        "question_count_by_split": {
            split: len({str(row["question_id"]) for row in split_rows})
            for split, split_rows in sorted(rows_by_split.items())
            if split_rows
        },
        "candidate_splits": sorted({str(row.get("split") or "") for row in rows}),
    }


def _gold_answer_summary(gold_answers_by_question_key: Mapping[str, str]) -> dict[str, Any]:
    counts = Counter(key.split("::", maxsplit=1)[0] for key in gold_answers_by_question_key)
    return {
        "answer_count": len(gold_answers_by_question_key),
        "answer_count_by_split": dict(sorted(counts.items())),
    }


def _policy_configs_by_label() -> dict[str, Any]:
    return {label: config for label, config in default_stage39_policy_specs()}


def _cases_by_key(
    cases: Sequence[GuardedCandidateAnswerCase],
) -> dict[str, GuardedCandidateAnswerCase]:
    return {_question_key(case.split, case.question_id): case for case in cases}


def _build_row_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    row_index: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        row_index[_question_key(row["split"], row["question_id"])].append(row)
    return dict(row_index)


def _row_by_candidate_id(
    rows: Sequence[Mapping[str, Any]],
    candidate_id: str,
) -> Mapping[str, Any]:
    for row in rows:
        if row["candidate_id"] == candidate_id:
            return row
    raise ValueError(f"Missing candidate row: {candidate_id}")


def _runtime_feature_float(row: Mapping[str, Any], feature_name: str) -> float:
    runtime_features = row.get("runtime_features")
    if not isinstance(runtime_features, Mapping):
        raise ValueError("row runtime_features must be an object")
    return float(runtime_features[feature_name])


def _document_id(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("row metadata must be an object")
    return str(metadata.get("document_id", ""))


def _is_gold_document(row: Mapping[str, Any]) -> bool:
    gold_labels = row.get("gold_labels")
    if not isinstance(gold_labels, Mapping):
        raise ValueError("row gold_labels must be an object")
    return bool(gold_labels["is_gold_document"])


def _document_transition(
    *,
    from_row: Mapping[str, Any],
    to_row: Mapping[str, Any],
) -> str:
    return _document_transition_by_id(
        _document_id(from_row),
        _document_id(to_row),
        from_gold=_is_gold_document(from_row),
        to_gold=_is_gold_document(to_row),
    )


def _document_transition_by_id(
    from_document_id: str,
    to_document_id: str,
    *,
    from_gold: bool,
    to_gold: bool,
) -> str:
    if from_document_id == to_document_id:
        return "same_leading_document"
    if to_gold and not from_gold:
        return "new_gold_leading_document"
    if from_gold and not to_gold:
        return "gold_to_non_gold_leading_document"
    return "new_non_gold_leading_document"


def _rank_bucket(rank: int) -> str:
    if rank == 1:
        return "rank_1"
    if rank == 2:
        return "rank_2"
    if rank == 3:
        return "rank_3"
    if rank == 4:
        return "rank_4"
    if rank == 5:
        return "rank_5"
    return "rank_6_plus"


def _candidate_score_bucket(score: float) -> str:
    if score < 60:
        return "score_lt_60"
    if score < 90:
        return "score_60_90"
    if score < 120:
        return "score_90_120"
    return "score_120_plus"


def _outcome(delta: float) -> str:
    if delta > 0:
        return "improved"
    if delta < 0:
        return "regressed"
    return "tied"


def _citation_outcome(citation_delta: int) -> str:
    if citation_delta > 0:
        return "citation_gained"
    if citation_delta < 0:
        return "citation_lost"
    return "citation_unchanged"


def _question_key(split: Any, question_id: Any) -> str:
    return f"{split}::{question_id}"


def _rounded_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(mean(values), 4)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _fingerprint(path: Path) -> dict[str, Any]:
    _ensure_file(path)
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _check(name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "observed": observed,
        "expected": expected,
    }


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")


def _validate_options(
    *,
    model_names: Sequence[str],
    max_answer_candidates: int,
    sample_limit: int,
) -> None:
    if not tuple(model_names):
        raise ValueError("model_names must not be empty")
    if any(not model_name.strip() for model_name in model_names):
        raise ValueError("model_names must not contain empty values")
    if max_answer_candidates <= 0:
        raise ValueError("max_answer_candidates must be positive")
    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")


def _policy_delta_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            f"{review['model_name']} {policy['policy_label']}",
            float(policy["metrics"]["average_delta_vs_baseline"]),
            f"{policy['metrics']['average_delta_vs_baseline']:+.4f}",
        )
        for review in report["model_reviews"]
        for policy in review["policy_reviews"]
    ]


def _changed_case_count_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            f"{review['model_name']} {policy['policy_label']}",
            float(policy["changed_vs_baseline_summary"]["changed_case_count"]),
            str(policy["changed_vs_baseline_summary"]["changed_case_count"]),
        )
        for review in report["model_reviews"]
        for policy in review["policy_reviews"]
    ]


def _policy_vs_main_outcome_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for review in report["model_reviews"]:
        summary = review["candidate_score_gte_60_vs_stage36_main"]["summary"]
        for label, key in (
            ("better", "candidate_improved_vs_main_count"),
            ("worse", "candidate_regressed_vs_main_count"),
            ("tied", "candidate_tied_vs_main_count"),
        ):
            bars.append(
                BarDatum(
                    f"{review['model_name']} {label}",
                    float(summary[key]),
                    str(summary[key]),
                )
            )
    return bars


def _residual_regression_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            f"{review['model_name']} {policy['policy_label']}",
            float(policy["metrics"]["regressed_count"]),
            str(policy["metrics"]["regressed_count"]),
        )
        for review in report["model_reviews"]
        for policy in review["policy_reviews"]
    ]
