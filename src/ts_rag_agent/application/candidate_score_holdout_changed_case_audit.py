from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any

from ts_rag_agent.application.candidate_reranker_cv import (
    split_validated_candidate_reranker_selections,
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
    GuardedCandidateAnswerMetrics,
    build_topk_leading_candidate_answer_cases_from_decisions,
    summarize_guarded_candidate_answer_cases,
)


@dataclass(frozen=True)
class HoldoutChangedCaseMetrics:
    """Aggregate comparison between main and candidate-score policies."""

    question_count: int
    changed_case_count: int
    changed_case_rate: float
    main_average_answer_token_f1: float
    candidate_average_answer_token_f1: float
    average_candidate_delta_vs_main: float
    candidate_improved_vs_main_count: int
    candidate_regressed_vs_main_count: int
    candidate_tied_vs_main_count: int
    main_delta_vs_baseline: float
    candidate_delta_vs_baseline: float
    main_regressed_count: int
    candidate_regressed_count: int
    regression_reduction_vs_main: int
    main_citation_lost_count: int
    candidate_citation_lost_count: int
    main_gold_citation_delta: int
    candidate_gold_citation_delta: int
    candidate_gold_citation_delta_vs_main: int


@dataclass(frozen=True)
class HoldoutChangedCaseAttribution:
    """One dev holdout case where candidate-score policy differs from main."""

    split: str
    question_id: str
    question_route: str
    candidate_vs_main_outcome: str
    candidate_delta_vs_main: float
    main_delta_vs_baseline: float
    candidate_delta_vs_baseline: float
    baseline_answer_token_f1: float
    main_answer_token_f1: float
    candidate_answer_token_f1: float
    main_gold_cited: bool
    candidate_gold_cited: bool
    main_citation_delta_vs_baseline: int
    candidate_citation_delta_vs_baseline: int
    candidate_citation_delta_vs_main: int
    main_decision_reasons: list[str]
    candidate_decision_reasons: list[str]
    baseline_leading_candidate_id: str
    main_leading_candidate_id: str
    candidate_leading_candidate_id: str
    blocked_candidate_id: str
    blocked_candidate_rank: int
    blocked_candidate_score: float
    blocked_candidate_score_bucket: str
    blocked_model_score_margin_vs_top_candidate: float
    candidate_leading_rank: int
    candidate_leading_score: float
    baseline_leading_document_id: str
    main_leading_document_id: str
    candidate_leading_document_id: str
    main_to_candidate_document_transition: str
    baseline_to_main_document_transition: str
    baseline_to_candidate_document_transition: str
    blocked_candidate_rank_bucket: str
    main_candidate_ids: list[str]
    candidate_policy_candidate_ids: list[str]
    baseline_candidate_ids: list[str]
    main_document_ids: list[str]
    candidate_policy_document_ids: list[str]
    baseline_document_ids: list[str]
    main_answer_text: str
    candidate_answer_text: str
    baseline_answer_text: str


@dataclass(frozen=True)
class HoldoutResidualRegressionCase:
    """One residual top-k regression left by candidate-score policy."""

    split: str
    question_id: str
    question_route: str
    f1_delta_vs_baseline: float
    baseline_answer_token_f1: float
    candidate_answer_token_f1: float
    baseline_gold_cited: bool
    candidate_gold_cited: bool
    citation_delta: int
    leading_candidate_id: str
    leading_candidate_rank: int
    leading_candidate_score: float
    leading_candidate_score_bucket: str
    leading_document_id: str
    baseline_leading_document_id: str
    baseline_to_candidate_document_transition: str
    decision_reasons: list[str]
    candidate_policy_candidate_ids: list[str]
    baseline_candidate_ids: list[str]
    candidate_policy_document_ids: list[str]
    baseline_document_ids: list[str]
    candidate_answer_text: str
    baseline_answer_text: str


@dataclass(frozen=True)
class HoldoutChangedCaseSegmentSummary:
    """Aggregate summary for changed cases in one segment."""

    segment_name: str
    case_count: int
    average_candidate_delta_vs_main: float
    min_candidate_delta_vs_main: float
    max_candidate_delta_vs_main: float
    candidate_improved_vs_main_count: int
    candidate_regressed_vs_main_count: int
    main_regressed_count: int
    candidate_regressed_count: int
    main_citation_lost_count: int
    candidate_citation_lost_count: int
    average_blocked_candidate_score: float


@dataclass(frozen=True)
class CandidateScoreHoldoutChangedCaseAudit:
    """Stage 41 audit for candidate-score holdout changed cases."""

    model_name: str
    train_split: str
    evaluation_split: str
    mode_name: str
    main_policy_label: str
    candidate_policy_label: str
    main_policy_metrics: GuardedCandidateAnswerMetrics
    candidate_policy_metrics: GuardedCandidateAnswerMetrics
    metrics: HoldoutChangedCaseMetrics
    changed_case_route_summaries: list[HoldoutChangedCaseSegmentSummary]
    blocked_rank_summaries: list[HoldoutChangedCaseSegmentSummary]
    blocked_candidate_score_summaries: list[HoldoutChangedCaseSegmentSummary]
    document_transition_summaries: list[HoldoutChangedCaseSegmentSummary]
    changed_cases: list[HoldoutChangedCaseAttribution]
    residual_regression_cases: list[HoldoutResidualRegressionCase]
    findings: list[str]
    analysis_scope: str


@dataclass(frozen=True)
class VisualizationArtifact:
    """One generated visualization file."""

    name: str
    path: str


def analyze_candidate_score_holdout_changed_cases(
    stage40_report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    gold_answers_by_question_key: Mapping[str, str],
    model_name: str = "logistic_best_candidate",
    train_split: str = "train",
    evaluation_split: str = "dev",
    max_answer_candidates: int = 3,
    sample_limit: int = 50,
) -> CandidateScoreHoldoutChangedCaseAudit:
    """Audit dev holdout cases changed by candidate_score_gte_60 versus main."""

    _validate_options(
        rows=rows,
        max_answer_candidates=max_answer_candidates,
        sample_limit=sample_limit,
    )
    normalized_train_split = _normalize_split_name(train_split)
    normalized_evaluation_split = _normalize_split_name(evaluation_split)
    evaluation_rows = _rows_for_split(rows, normalized_evaluation_split)
    mode_name = f"top{max_answer_candidates}_leading_candidate_rewrite"
    policy_configs = _policy_configs_by_label()
    main_config = policy_configs[STAGE39_MAIN_POLICY_LABEL]
    candidate_config = policy_configs[CANDIDATE_SCORE_GTE_60_LABEL]

    selections = split_validated_candidate_reranker_selections(
        rows=rows,
        model_name=model_name,
        train_split=normalized_train_split,
        validation_split=normalized_evaluation_split,
    )
    main_decisions = candidate_reranker_policy_decisions_from_selections(
        config=main_config,
        selections=selections,
        rows=evaluation_rows,
    )
    candidate_decisions = candidate_reranker_policy_decisions_from_selections(
        config=candidate_config,
        selections=selections,
        rows=evaluation_rows,
    )
    main_cases = build_topk_leading_candidate_answer_cases_from_decisions(
        decisions=main_decisions,
        rows=evaluation_rows,
        gold_answers_by_question_key=gold_answers_by_question_key,
        max_answer_candidates=max_answer_candidates,
    )
    candidate_cases = build_topk_leading_candidate_answer_cases_from_decisions(
        decisions=candidate_decisions,
        rows=evaluation_rows,
        gold_answers_by_question_key=gold_answers_by_question_key,
        max_answer_candidates=max_answer_candidates,
    )
    main_metrics = summarize_guarded_candidate_answer_cases(main_cases)
    candidate_metrics = summarize_guarded_candidate_answer_cases(candidate_cases)
    _validate_stage40_report(
        report=stage40_report,
        mode_name=mode_name,
        main_metrics=main_metrics,
        candidate_metrics=candidate_metrics,
        model_name=model_name,
        train_split=normalized_train_split,
        evaluation_split=normalized_evaluation_split,
        max_answer_candidates=max_answer_candidates,
    )

    row_index = _build_row_index(evaluation_rows)
    main_decisions_by_key = _decisions_by_key(main_decisions)
    candidate_decisions_by_key = _decisions_by_key(candidate_decisions)
    main_cases_by_key = _cases_by_key(main_cases)
    candidate_cases_by_key = _cases_by_key(candidate_cases)
    changed_attributions = [
        _changed_case_attribution(
            main_case=main_case,
            candidate_case=candidate_cases_by_key[question_key],
            main_decision=main_decisions_by_key[question_key],
            candidate_decision=candidate_decisions_by_key[question_key],
            row_index=row_index,
        )
        for question_key, main_case in main_cases_by_key.items()
        if main_case.policy_candidate_ids
        != candidate_cases_by_key[question_key].policy_candidate_ids
    ]
    residual_regression_cases = [
        _residual_regression_case(
            case=case,
            decision=candidate_decisions_by_key[_question_key(case.split, case.question_id)],
            row_index=row_index,
        )
        for case in candidate_cases
        if case.f1_delta_vs_baseline < 0
    ]
    metrics = _metrics(
        main_cases=main_cases,
        candidate_cases=candidate_cases,
        changed_attributions=changed_attributions,
        main_metrics=main_metrics,
        candidate_metrics=candidate_metrics,
    )

    return CandidateScoreHoldoutChangedCaseAudit(
        model_name=model_name,
        train_split=normalized_train_split,
        evaluation_split=normalized_evaluation_split,
        mode_name=mode_name,
        main_policy_label=STAGE39_MAIN_POLICY_LABEL,
        candidate_policy_label=CANDIDATE_SCORE_GTE_60_LABEL,
        main_policy_metrics=main_metrics,
        candidate_policy_metrics=candidate_metrics,
        metrics=metrics,
        changed_case_route_summaries=_segment_summaries(
            changed_attributions,
            lambda attribution: attribution.question_route,
        ),
        blocked_rank_summaries=_segment_summaries(
            changed_attributions,
            lambda attribution: attribution.blocked_candidate_rank_bucket,
        ),
        blocked_candidate_score_summaries=_segment_summaries(
            changed_attributions,
            lambda attribution: attribution.blocked_candidate_score_bucket,
        ),
        document_transition_summaries=_segment_summaries(
            changed_attributions,
            lambda attribution: attribution.main_to_candidate_document_transition,
        ),
        changed_cases=sorted(
            changed_attributions,
            key=lambda attribution: (
                attribution.candidate_delta_vs_main,
                attribution.question_id,
            ),
        )[:sample_limit],
        residual_regression_cases=sorted(
            residual_regression_cases,
            key=lambda case: (case.f1_delta_vs_baseline, case.question_id),
        )[:sample_limit],
        findings=_findings(
            metrics=metrics,
            changed_attributions=changed_attributions,
            residual_regression_cases=residual_regression_cases,
        ),
        analysis_scope=(
            "Offline Stage 41 holdout changed-case audit only. It rebuilds "
            "train-to-dev holdout selections, validates the recomputed metrics "
            "against the Stage 40 report, compares stage36_main with "
            "candidate_score_gte_60 on dev top-k proxy cases, and does not use "
            "held-out test data or change runtime behavior."
        ),
    )


def candidate_score_holdout_changed_case_audit_to_dict(
    audit: CandidateScoreHoldoutChangedCaseAudit,
) -> dict[str, Any]:
    """Convert a Stage 41 audit to a JSON-safe dictionary."""

    return asdict(audit)


def write_candidate_score_holdout_changed_case_visualizations(
    audit: CandidateScoreHoldoutChangedCaseAudit,
    output_dir: Path,
) -> list[VisualizationArtifact]:
    """Write SVG charts for Stage 41 holdout changed-case audit."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage41_candidate_vs_main_outcomes.svg": _render_bar_chart_svg(
            title="Stage 41 candidate-score policy vs main on changed cases",
            bars=[
                _Bar(
                    "candidate better",
                    float(audit.metrics.candidate_improved_vs_main_count),
                    str(audit.metrics.candidate_improved_vs_main_count),
                ),
                _Bar(
                    "candidate worse",
                    float(audit.metrics.candidate_regressed_vs_main_count),
                    str(audit.metrics.candidate_regressed_vs_main_count),
                ),
                _Bar(
                    "candidate tied",
                    float(audit.metrics.candidate_tied_vs_main_count),
                    str(audit.metrics.candidate_tied_vs_main_count),
                ),
            ],
            x_label="changed-case count",
        ),
        "stage41_changed_cases_by_route.svg": _render_summary_chart(
            title="Stage 41 changed cases by route",
            summaries=audit.changed_case_route_summaries,
            x_label="changed-case count",
        ),
        "stage41_changed_cases_by_blocked_score.svg": _render_summary_chart(
            title="Stage 41 changed cases by blocked candidate score",
            summaries=audit.blocked_candidate_score_summaries,
            x_label="changed-case count",
        ),
        "stage41_changed_cases_by_document_transition.svg": _render_summary_chart(
            title="Stage 41 changed cases by document transition",
            summaries=audit.document_transition_summaries,
            x_label="changed-case count",
        ),
        "stage41_residual_regression_routes.svg": _render_bar_chart_svg(
            title="Stage 41 candidate-score residual regressions by route",
            bars=[
                _Bar(label, float(count), str(count))
                for label, count in sorted(
                    Counter(
                        case.question_route for case in audit.residual_regression_cases
                    ).items()
                )
            ],
            x_label="residual regression count",
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(VisualizationArtifact(name=filename, path=str(path)))
    return artifacts


def _validate_stage40_report(
    report: Mapping[str, Any],
    mode_name: str,
    main_metrics: GuardedCandidateAnswerMetrics,
    candidate_metrics: GuardedCandidateAnswerMetrics,
    model_name: str,
    train_split: str,
    evaluation_split: str,
    max_answer_candidates: int,
) -> None:
    if report.get("model_name") != model_name:
        raise ValueError("Stage 40 report model_name does not match recomputed audit")
    if report.get("train_split") != train_split:
        raise ValueError("Stage 40 report train_split does not match recomputed audit")
    if report.get("evaluation_split") != evaluation_split:
        raise ValueError(
            "Stage 40 report evaluation_split does not match recomputed audit"
        )
    if int(report.get("max_answer_candidates", -1)) != max_answer_candidates:
        raise ValueError(
            "Stage 40 report max_answer_candidates does not match recomputed audit"
        )
    holdout = report.get("holdout_evaluation")
    if not isinstance(holdout, Mapping):
        raise ValueError("Stage 40 report must contain holdout_evaluation")
    _validate_report_policy_metrics(
        holdout=holdout,
        policy_label=STAGE39_MAIN_POLICY_LABEL,
        mode_name=mode_name,
        metrics=main_metrics,
    )
    _validate_report_policy_metrics(
        holdout=holdout,
        policy_label=CANDIDATE_SCORE_GTE_60_LABEL,
        mode_name=mode_name,
        metrics=candidate_metrics,
    )


def _validate_report_policy_metrics(
    holdout: Mapping[str, Any],
    policy_label: str,
    mode_name: str,
    metrics: GuardedCandidateAnswerMetrics,
) -> None:
    report_metrics = _report_mode_metrics(
        holdout=holdout,
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
                f"Stage 40 report {policy_label}/{mode_name} {key} does not "
                "match recomputed audit"
            )


def _report_mode_metrics(
    holdout: Mapping[str, Any],
    policy_label: str,
    mode_name: str,
) -> Mapping[str, Any]:
    policies = holdout.get("policies")
    if not isinstance(policies, list):
        raise ValueError("Stage 40 holdout_evaluation must contain policies")
    for policy in policies:
        if not isinstance(policy, Mapping) or policy.get("label") != policy_label:
            continue
        mode_evaluations = policy.get("mode_evaluations")
        if not isinstance(mode_evaluations, list):
            raise ValueError("Stage 40 policy must contain mode_evaluations")
        for mode_evaluation in mode_evaluations:
            if (
                isinstance(mode_evaluation, Mapping)
                and mode_evaluation.get("mode_name") == mode_name
                and isinstance(mode_evaluation.get("metrics"), Mapping)
            ):
                return mode_evaluation["metrics"]
    raise ValueError(f"Stage 40 report missing metrics for {policy_label}/{mode_name}")


def _metrics(
    main_cases: Sequence[GuardedCandidateAnswerCase],
    candidate_cases: Sequence[GuardedCandidateAnswerCase],
    changed_attributions: Sequence[HoldoutChangedCaseAttribution],
    main_metrics: GuardedCandidateAnswerMetrics,
    candidate_metrics: GuardedCandidateAnswerMetrics,
) -> HoldoutChangedCaseMetrics:
    candidate_delta_vs_main = [
        candidate_case.policy_answer_token_f1 - main_case.policy_answer_token_f1
        for main_case, candidate_case in zip(main_cases, candidate_cases, strict=True)
    ]
    changed_delta_vs_main = [
        attribution.candidate_delta_vs_main for attribution in changed_attributions
    ]
    outcome_counts = Counter(_outcome(delta) for delta in changed_delta_vs_main)
    return HoldoutChangedCaseMetrics(
        question_count=len(main_cases),
        changed_case_count=len(changed_attributions),
        changed_case_rate=_ratio(len(changed_attributions), len(main_cases)),
        main_average_answer_token_f1=main_metrics.policy_average_answer_token_f1,
        candidate_average_answer_token_f1=(
            candidate_metrics.policy_average_answer_token_f1
        ),
        average_candidate_delta_vs_main=_rounded_mean(candidate_delta_vs_main),
        candidate_improved_vs_main_count=outcome_counts["improved"],
        candidate_regressed_vs_main_count=outcome_counts["regressed"],
        candidate_tied_vs_main_count=outcome_counts["tied"],
        main_delta_vs_baseline=main_metrics.average_delta_vs_baseline,
        candidate_delta_vs_baseline=candidate_metrics.average_delta_vs_baseline,
        main_regressed_count=main_metrics.regressed_count,
        candidate_regressed_count=candidate_metrics.regressed_count,
        regression_reduction_vs_main=(
            main_metrics.regressed_count - candidate_metrics.regressed_count
        ),
        main_citation_lost_count=main_metrics.citation_lost_count,
        candidate_citation_lost_count=candidate_metrics.citation_lost_count,
        main_gold_citation_delta=main_metrics.gold_citation_delta,
        candidate_gold_citation_delta=candidate_metrics.gold_citation_delta,
        candidate_gold_citation_delta_vs_main=(
            candidate_metrics.gold_citation_delta - main_metrics.gold_citation_delta
        ),
    )


def _changed_case_attribution(
    main_case: GuardedCandidateAnswerCase,
    candidate_case: GuardedCandidateAnswerCase,
    main_decision: CandidateRerankerPolicyDecision,
    candidate_decision: CandidateRerankerPolicyDecision,
    row_index: Mapping[str, list[Mapping[str, Any]]],
) -> HoldoutChangedCaseAttribution:
    question_rows = row_index[_question_key(main_case.split, main_case.question_id)]
    baseline_row = _row_by_candidate_id(question_rows, main_case.baseline_candidate_ids[0])
    main_leading_row = _row_by_candidate_id(question_rows, main_case.policy_candidate_ids[0])
    candidate_leading_row = _row_by_candidate_id(
        question_rows,
        candidate_case.policy_candidate_ids[0],
    )
    blocked_row = _row_by_candidate_id(question_rows, main_case.policy_candidate_ids[0])
    blocked_candidate_score = _runtime_feature_float(blocked_row, "candidate_score")
    candidate_delta_vs_main = round(
        candidate_case.policy_answer_token_f1 - main_case.policy_answer_token_f1,
        4,
    )
    return HoldoutChangedCaseAttribution(
        split=main_case.split,
        question_id=main_case.question_id,
        question_route=main_case.question_route,
        candidate_vs_main_outcome=_outcome(candidate_delta_vs_main),
        candidate_delta_vs_main=candidate_delta_vs_main,
        main_delta_vs_baseline=main_case.f1_delta_vs_baseline,
        candidate_delta_vs_baseline=candidate_case.f1_delta_vs_baseline,
        baseline_answer_token_f1=main_case.baseline_answer_token_f1,
        main_answer_token_f1=main_case.policy_answer_token_f1,
        candidate_answer_token_f1=candidate_case.policy_answer_token_f1,
        main_gold_cited=main_case.policy_gold_cited,
        candidate_gold_cited=candidate_case.policy_gold_cited,
        main_citation_delta_vs_baseline=main_case.citation_delta,
        candidate_citation_delta_vs_baseline=candidate_case.citation_delta,
        candidate_citation_delta_vs_main=(
            int(candidate_case.policy_gold_cited) - int(main_case.policy_gold_cited)
        ),
        main_decision_reasons=list(main_case.decision_reasons),
        candidate_decision_reasons=list(candidate_case.decision_reasons),
        baseline_leading_candidate_id=main_case.baseline_candidate_ids[0],
        main_leading_candidate_id=main_case.policy_candidate_ids[0],
        candidate_leading_candidate_id=candidate_case.policy_candidate_ids[0],
        blocked_candidate_id=main_case.policy_candidate_ids[0],
        blocked_candidate_rank=main_decision.final_candidate_rank,
        blocked_candidate_score=blocked_candidate_score,
        blocked_candidate_score_bucket=_candidate_score_bucket(blocked_candidate_score),
        blocked_model_score_margin_vs_top_candidate=(
            main_decision.model_score_margin_vs_top_candidate
        ),
        candidate_leading_rank=candidate_decision.final_candidate_rank,
        candidate_leading_score=_runtime_feature_float(
            candidate_leading_row,
            "candidate_score",
        ),
        baseline_leading_document_id=_document_id(baseline_row),
        main_leading_document_id=_document_id(main_leading_row),
        candidate_leading_document_id=_document_id(candidate_leading_row),
        main_to_candidate_document_transition=_document_transition(
            from_row=main_leading_row,
            to_row=candidate_leading_row,
        ),
        baseline_to_main_document_transition=_document_transition(
            from_row=baseline_row,
            to_row=main_leading_row,
        ),
        baseline_to_candidate_document_transition=_document_transition(
            from_row=baseline_row,
            to_row=candidate_leading_row,
        ),
        blocked_candidate_rank_bucket=_rank_bucket(main_decision.final_candidate_rank),
        main_candidate_ids=list(main_case.policy_candidate_ids),
        candidate_policy_candidate_ids=list(candidate_case.policy_candidate_ids),
        baseline_candidate_ids=list(main_case.baseline_candidate_ids),
        main_document_ids=list(main_case.policy_document_ids),
        candidate_policy_document_ids=list(candidate_case.policy_document_ids),
        baseline_document_ids=list(main_case.baseline_document_ids),
        main_answer_text=main_case.policy_answer_text,
        candidate_answer_text=candidate_case.policy_answer_text,
        baseline_answer_text=main_case.baseline_answer_text,
    )


def _residual_regression_case(
    case: GuardedCandidateAnswerCase,
    decision: CandidateRerankerPolicyDecision,
    row_index: Mapping[str, list[Mapping[str, Any]]],
) -> HoldoutResidualRegressionCase:
    question_rows = row_index[_question_key(case.split, case.question_id)]
    leading_row = _row_by_candidate_id(question_rows, case.policy_candidate_ids[0])
    baseline_leading_row = _row_by_candidate_id(question_rows, case.baseline_candidate_ids[0])
    leading_score = _runtime_feature_float(leading_row, "candidate_score")
    return HoldoutResidualRegressionCase(
        split=case.split,
        question_id=case.question_id,
        question_route=case.question_route,
        f1_delta_vs_baseline=case.f1_delta_vs_baseline,
        baseline_answer_token_f1=case.baseline_answer_token_f1,
        candidate_answer_token_f1=case.policy_answer_token_f1,
        baseline_gold_cited=case.baseline_gold_cited,
        candidate_gold_cited=case.policy_gold_cited,
        citation_delta=case.citation_delta,
        leading_candidate_id=case.policy_candidate_ids[0],
        leading_candidate_rank=decision.final_candidate_rank,
        leading_candidate_score=leading_score,
        leading_candidate_score_bucket=_candidate_score_bucket(leading_score),
        leading_document_id=_document_id(leading_row),
        baseline_leading_document_id=_document_id(baseline_leading_row),
        baseline_to_candidate_document_transition=_document_transition(
            from_row=baseline_leading_row,
            to_row=leading_row,
        ),
        decision_reasons=list(case.decision_reasons),
        candidate_policy_candidate_ids=list(case.policy_candidate_ids),
        baseline_candidate_ids=list(case.baseline_candidate_ids),
        candidate_policy_document_ids=list(case.policy_document_ids),
        baseline_document_ids=list(case.baseline_document_ids),
        candidate_answer_text=case.policy_answer_text,
        baseline_answer_text=case.baseline_answer_text,
    )


def _segment_summaries(
    attributions: Sequence[HoldoutChangedCaseAttribution],
    segment_fn: Callable[[HoldoutChangedCaseAttribution], str],
) -> list[HoldoutChangedCaseSegmentSummary]:
    by_segment: dict[str, list[HoldoutChangedCaseAttribution]] = defaultdict(list)
    for attribution in attributions:
        by_segment[str(segment_fn(attribution))].append(attribution)
    summaries = []
    for segment_name, segment_cases in by_segment.items():
        deltas = [case.candidate_delta_vs_main for case in segment_cases]
        summaries.append(
            HoldoutChangedCaseSegmentSummary(
                segment_name=segment_name,
                case_count=len(segment_cases),
                average_candidate_delta_vs_main=_rounded_mean(deltas),
                min_candidate_delta_vs_main=round(min(deltas), 4),
                max_candidate_delta_vs_main=round(max(deltas), 4),
                candidate_improved_vs_main_count=sum(
                    case.candidate_vs_main_outcome == "improved"
                    for case in segment_cases
                ),
                candidate_regressed_vs_main_count=sum(
                    case.candidate_vs_main_outcome == "regressed"
                    for case in segment_cases
                ),
                main_regressed_count=sum(
                    case.main_delta_vs_baseline < 0 for case in segment_cases
                ),
                candidate_regressed_count=sum(
                    case.candidate_delta_vs_baseline < 0 for case in segment_cases
                ),
                main_citation_lost_count=sum(
                    case.main_citation_delta_vs_baseline < 0
                    for case in segment_cases
                ),
                candidate_citation_lost_count=sum(
                    case.candidate_citation_delta_vs_baseline < 0
                    for case in segment_cases
                ),
                average_blocked_candidate_score=_rounded_mean(
                    [case.blocked_candidate_score for case in segment_cases]
                ),
            )
        )
    return sorted(
        summaries,
        key=lambda summary: (summary.case_count, summary.average_candidate_delta_vs_main),
        reverse=True,
    )


def _findings(
    metrics: HoldoutChangedCaseMetrics,
    changed_attributions: Sequence[HoldoutChangedCaseAttribution],
    residual_regression_cases: Sequence[HoldoutResidualRegressionCase],
) -> list[str]:
    findings = [
        (
            f"candidate_score_gte_60 changes {metrics.changed_case_count} / "
            f"{metrics.question_count} holdout top-k cases versus stage36_main."
        ),
        (
            "Across all holdout cases it changes top-k delta from "
            f"{metrics.main_delta_vs_baseline:+.4f} to "
            f"{metrics.candidate_delta_vs_baseline:+.4f}."
        ),
        (
            f"Regression count changes from {metrics.main_regressed_count} to "
            f"{metrics.candidate_regressed_count}; citation-loss count changes "
            f"from {metrics.main_citation_lost_count} to "
            f"{metrics.candidate_citation_lost_count}."
        ),
    ]
    if changed_attributions:
        outcome_counts = Counter(
            attribution.candidate_vs_main_outcome
            for attribution in changed_attributions
        )
        findings.append(
            "On changed cases candidate_score_gte_60 is better/tied/worse than "
            f"main: {outcome_counts['improved']}/"
            f"{outcome_counts['tied']}/{outcome_counts['regressed']}."
        )
        route, count = Counter(
            attribution.question_route for attribution in changed_attributions
        ).most_common(1)[0]
        findings.append(
            f"Most changed cases are in route {route}: {count} / "
            f"{len(changed_attributions)}."
        )
    if residual_regression_cases:
        regression = sorted(
            residual_regression_cases,
            key=lambda case: (case.f1_delta_vs_baseline, case.question_id),
        )[0]
        findings.append(
            "Residual candidate_score_gte_60 regression example is "
            f"{regression.split}::{regression.question_id}, route "
            f"{regression.question_route}, delta "
            f"{regression.f1_delta_vs_baseline:+.4f}."
        )
    else:
        findings.append("candidate_score_gte_60 leaves no residual top-k regressions.")
    return findings


def _policy_configs_by_label() -> dict[str, Any]:
    return {label: config for label, config in default_stage39_policy_specs()}


def _rows_for_split(
    rows: Sequence[Mapping[str, Any]],
    split: str,
) -> list[Mapping[str, Any]]:
    split_rows = [row for row in rows if str(row["split"]).lower() == split]
    if not split_rows:
        raise ValueError(f"No rows found for split: {split}")
    return split_rows


def _build_row_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    row_index: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        row_index[_question_key(row["split"], row["question_id"])].append(row)
    return dict(row_index)


def _decisions_by_key(
    decisions: Sequence[CandidateRerankerPolicyDecision],
) -> dict[str, CandidateRerankerPolicyDecision]:
    return {
        _question_key(decision.split, decision.question_id): decision
        for decision in decisions
    }


def _cases_by_key(
    cases: Sequence[GuardedCandidateAnswerCase],
) -> dict[str, GuardedCandidateAnswerCase]:
    return {_question_key(case.split, case.question_id): case for case in cases}


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


def _document_transition(
    from_row: Mapping[str, Any],
    to_row: Mapping[str, Any],
) -> str:
    if _document_id(from_row) == _document_id(to_row):
        return "same_leading_document"
    if _is_gold_document(to_row) and not _is_gold_document(from_row):
        return "new_gold_leading_document"
    if _is_gold_document(from_row) and not _is_gold_document(to_row):
        return "gold_to_non_gold_leading_document"
    return "new_non_gold_leading_document"


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


def _question_key(split: Any, question_id: Any) -> str:
    return f"{split}::{question_id}"


def _normalize_split_name(split_name: str) -> str:
    normalized = split_name.strip().lower()
    if not normalized:
        raise ValueError("split name must not be empty")
    return normalized


def _rounded_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(mean(values), 4)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _validate_options(
    rows: Sequence[Mapping[str, Any]],
    max_answer_candidates: int,
    sample_limit: int,
) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    if max_answer_candidates <= 0:
        raise ValueError("max_answer_candidates must be positive")
    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")


@dataclass(frozen=True)
class _Bar:
    label: str
    value: float
    value_label: str


def _render_summary_chart(
    title: str,
    summaries: Sequence[HoldoutChangedCaseSegmentSummary],
    x_label: str,
) -> str:
    return _render_bar_chart_svg(
        title=title,
        bars=[
            _Bar(
                label=summary.segment_name,
                value=float(summary.case_count),
                value_label=(
                    f"{summary.case_count} "
                    f"(avg {summary.average_candidate_delta_vs_main:+.4f})"
                ),
            )
            for summary in summaries
        ],
        x_label=x_label,
    )


def _render_bar_chart_svg(
    title: str,
    bars: Sequence[_Bar],
    x_label: str,
) -> str:
    width = 1120
    margin_left = 390
    margin_right = 210
    margin_top = 56
    margin_bottom = 56
    row_height = 38
    height = margin_top + margin_bottom + max(1, len(bars)) * row_height
    plot_width = width - margin_left - margin_right
    max_abs_value = max((abs(bar.value) for bar in bars), default=0.0)
    scale_denominator = max_abs_value if max_abs_value > 0 else 1.0
    has_negative = any(bar.value < 0 for bar in bars)
    zero_x = margin_left + (plot_width // 2 if has_negative else 0)
    positive_plot_width = plot_width // 2 if has_negative else plot_width
    chart_id = _svg_id(title)

    bar_lines = []
    for index, bar in enumerate(bars):
        y = margin_top + index * row_height
        bar_width = int(round(positive_plot_width * abs(bar.value) / scale_denominator))
        if bar.value >= 0:
            x = zero_x
            value_x = x + bar_width + 8
            anchor = "start"
        else:
            x = zero_x - bar_width
            value_x = x - 8
            anchor = "end"
        label_y = y + 21
        bar_lines.append(
            "\n".join(
                [
                    f'<text x="{margin_left - 12}" y="{label_y}" text-anchor="end">'
                    f"{escape(bar.label)}</text>",
                    (
                        f'<rect x="{x}" y="{y + 6}" width="{bar_width}" '
                        'height="22" rx="3" fill="#2f6fed" />'
                    ),
                    f'<text x="{value_x}" y="{label_y}" text-anchor="{anchor}">'
                    f"{escape(bar.value_label)}</text>",
                ]
            )
        )

    axis_y = height - margin_bottom + 12
    return "\n".join(
        [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
                f'aria-labelledby="{chart_id}-title {chart_id}-desc">'
            ),
            f'<title id="{chart_id}-title">{escape(title)}</title>',
            (
                f'<desc id="{chart_id}-desc">Horizontal bar chart showing '
                f'{escape(x_label)} for {len(bars)} categories.</desc>'
            ),
            '<rect width="100%" height="100%" fill="#ffffff" />',
            (
                '<style>text{font-family:Arial, sans-serif;font-size:13px;'
                'fill:#1f2937}.title{font-size:18px;font-weight:700}'
                '.axis{fill:#4b5563;font-size:12px}.grid{stroke:#e5e7eb}'
                '</style>'
            ),
            f'<text x="24" y="32" class="title">{escape(title)}</text>',
            (
                f'<line x1="{zero_x}" x2="{zero_x}" y1="{margin_top - 8}" '
                f'y2="{height - margin_bottom}" class="grid" />'
            ),
            *bar_lines,
            f'<text x="{margin_left}" y="{axis_y}" class="axis">{escape(x_label)}</text>',
            "</svg>",
        ]
    )


def _svg_id(title: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in title).strip("-")
