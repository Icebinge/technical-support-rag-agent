from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any

from ts_rag_agent.application.candidate_reranker_cv import (
    cross_validated_candidate_reranker_selections,
)
from ts_rag_agent.application.candidate_reranker_policy_search import (
    CandidateRerankerPolicyConfig,
    CandidateRerankerPolicyDecision,
    candidate_reranker_policy_decisions_from_selections,
)
from ts_rag_agent.application.candidate_reranker_policy_stability import (
    STAGE35_BEST_POLICY,
    STAGE35_BLOCK_HOW_TO_ONLY_POLICY,
)
from ts_rag_agent.application.text_metrics import token_f1

SINGLE_CANDIDATE_MODE = "single_candidate_answer"


@dataclass(frozen=True)
class GuardedCandidateAnswerCase:
    """One question-level answer proxy case for a guarded candidate-reranker policy."""

    mode_name: str
    split: str
    question_id: str
    question_route: str
    action: str
    decision_reasons: list[str]
    baseline_candidate_ids: list[str]
    policy_candidate_ids: list[str]
    oracle_candidate_ids: list[str]
    baseline_document_ids: list[str]
    policy_document_ids: list[str]
    oracle_document_ids: list[str]
    baseline_answer_text: str
    policy_answer_text: str
    oracle_answer_text: str
    baseline_answer_token_f1: float
    policy_answer_token_f1: float
    oracle_answer_token_f1: float
    f1_delta_vs_baseline: float
    baseline_gold_cited: bool
    policy_gold_cited: bool
    citation_delta: int
    policy_leading_candidate_rank: int
    policy_leading_is_oracle_best_f1: bool


@dataclass(frozen=True)
class GuardedCandidateAnswerMetrics:
    """Aggregate metrics for one answer proxy mode."""

    question_count: int
    baseline_average_answer_token_f1: float
    policy_average_answer_token_f1: float
    oracle_average_answer_token_f1: float
    average_delta_vs_baseline: float
    oracle_gap_closed_rate: float
    replacement_count: int
    replacement_rate: float
    changed_answer_count: int
    changed_answer_rate: float
    improved_count: int
    regressed_count: int
    tied_count: int
    regressed_rate: float
    baseline_gold_citation_count: int
    policy_gold_citation_count: int
    gold_citation_delta: int
    baseline_gold_citation_rate: float
    policy_gold_citation_rate: float
    citation_lost_count: int
    citation_gained_count: int
    policy_oracle_best_count: int
    policy_oracle_best_rate: float
    decision_reason_counts: dict[str, int]


@dataclass(frozen=True)
class GuardedCandidateAnswerMetricsBySegment:
    """Compact answer proxy metrics for one split or route."""

    segment_name: str
    question_count: int
    baseline_average_answer_token_f1: float
    policy_average_answer_token_f1: float
    average_delta_vs_baseline: float
    replacement_count: int
    replacement_rate: float
    changed_answer_count: int
    changed_answer_rate: float
    improved_count: int
    regressed_count: int
    regressed_rate: float
    baseline_gold_citation_count: int
    policy_gold_citation_count: int
    gold_citation_delta: int


@dataclass(frozen=True)
class GuardedCandidateAnswerModeEvaluation:
    """One answer proxy mode evaluated for one guarded policy."""

    mode_name: str
    metrics: GuardedCandidateAnswerMetrics
    route_metrics: list[GuardedCandidateAnswerMetricsBySegment]
    split_metrics: list[GuardedCandidateAnswerMetricsBySegment]
    sample_cases: dict[str, list[GuardedCandidateAnswerCase]]


@dataclass(frozen=True)
class GuardedCandidateAnswerPolicyEvaluation:
    """All answer proxy modes for one guarded policy config."""

    config: CandidateRerankerPolicyConfig
    mode_evaluations: list[GuardedCandidateAnswerModeEvaluation]


@dataclass(frozen=True)
class GuardedCandidateAnswerPolicyDelta:
    """Metric deltas between the main and sensitivity policies for one mode."""

    mode_name: str
    main_policy_name: str
    sensitivity_policy_name: str
    policy_average_f1_difference: float
    average_delta_difference: float
    oracle_gap_closed_difference: float
    replacement_count_difference: int
    regressed_count_difference: int
    gold_citation_count_difference: int


@dataclass(frozen=True)
class GuardedCandidateAnswerExperimentResult:
    """Full Stage 37 guarded candidate-reranker answer proxy experiment."""

    model_name: str
    fold_count: int
    max_answer_candidates: int
    main_policy: GuardedCandidateAnswerPolicyEvaluation
    sensitivity_policy: GuardedCandidateAnswerPolicyEvaluation
    main_vs_sensitivity: list[GuardedCandidateAnswerPolicyDelta]
    findings: list[str]
    analysis_scope: str


@dataclass(frozen=True)
class VisualizationArtifact:
    """One generated visualization file."""

    name: str
    path: str


def run_guarded_candidate_reranker_answer_experiment(
    rows: Sequence[Mapping[str, Any]],
    gold_answers_by_question_key: Mapping[str, str] | None = None,
    model_name: str = "logistic_best_candidate",
    fold_count: int = 5,
    main_policy: CandidateRerankerPolicyConfig = STAGE35_BLOCK_HOW_TO_ONLY_POLICY,
    sensitivity_policy: CandidateRerankerPolicyConfig = STAGE35_BEST_POLICY,
    max_answer_candidates: int = 3,
    sample_limit: int = 20,
) -> GuardedCandidateAnswerExperimentResult:
    """Evaluate guarded reranker decisions as offline answer-level proxy cases."""

    _validate_options(
        rows=rows,
        max_answer_candidates=max_answer_candidates,
        sample_limit=sample_limit,
    )
    selections = cross_validated_candidate_reranker_selections(
        rows=rows,
        model_name=model_name,
        fold_count=fold_count,
    )
    main_evaluation = _evaluate_policy(
        config=main_policy,
        rows=rows,
        selections=selections,
        gold_answers_by_question_key=gold_answers_by_question_key,
        max_answer_candidates=max_answer_candidates,
        sample_limit=sample_limit,
    )
    sensitivity_evaluation = _evaluate_policy(
        config=sensitivity_policy,
        rows=rows,
        selections=selections,
        gold_answers_by_question_key=gold_answers_by_question_key,
        max_answer_candidates=max_answer_candidates,
        sample_limit=sample_limit,
    )

    deltas = _policy_deltas(
        main_policy=main_evaluation,
        sensitivity_policy=sensitivity_evaluation,
    )
    return GuardedCandidateAnswerExperimentResult(
        model_name=model_name,
        fold_count=fold_count,
        max_answer_candidates=max_answer_candidates,
        main_policy=main_evaluation,
        sensitivity_policy=sensitivity_evaluation,
        main_vs_sensitivity=deltas,
        findings=_findings(main_evaluation, sensitivity_evaluation, deltas),
        analysis_scope=(
            "Offline answer-level proxy only. The single-candidate mode uses "
            "Stage 31 candidate token-F1 labels. The top-k leading-candidate rewrite "
            "mode recomputes answer token F1 from local gold answers and the "
            "metadata candidate sentences stored in the candidate dataset. The "
            "experiment uses grouped-CV selections and does not change runtime "
            "behavior or the default verified RAG pipeline."
        ),
    )


def guarded_candidate_answer_experiment_to_dict(
    result: GuardedCandidateAnswerExperimentResult,
) -> dict[str, Any]:
    """Convert a guarded answer experiment result to a JSON-safe dictionary."""

    return asdict(result)


def build_single_candidate_answer_cases_from_decisions(
    decisions: Sequence[CandidateRerankerPolicyDecision],
    rows: Sequence[Mapping[str, Any]],
) -> list[GuardedCandidateAnswerCase]:
    """Build full single-candidate answer proxy cases from policy decisions."""

    row_index = _build_row_index(rows)
    return [
        _single_candidate_case(decision=decision, row_index=row_index)
        for decision in decisions
    ]


def build_topk_leading_candidate_answer_cases_from_decisions(
    decisions: Sequence[CandidateRerankerPolicyDecision],
    rows: Sequence[Mapping[str, Any]],
    gold_answers_by_question_key: Mapping[str, str],
    max_answer_candidates: int = 3,
) -> list[GuardedCandidateAnswerCase]:
    """Build full top-k leading-candidate rewrite cases from policy decisions."""

    if max_answer_candidates <= 0:
        raise ValueError("max_answer_candidates must be positive")
    row_index = _build_row_index(rows)
    mode_name = f"top{max_answer_candidates}_leading_candidate_rewrite"
    return [
        _topk_leading_candidate_case(
            decision=decision,
            row_index=row_index,
            gold_answers_by_question_key=gold_answers_by_question_key,
            max_answer_candidates=max_answer_candidates,
            mode_name=mode_name,
        )
        for decision in decisions
    ]


def summarize_guarded_candidate_answer_cases(
    cases: Sequence[GuardedCandidateAnswerCase],
) -> GuardedCandidateAnswerMetrics:
    """Summarize guarded candidate answer proxy cases."""

    return _metrics(cases)


def segment_guarded_candidate_answer_cases(
    cases: Sequence[GuardedCandidateAnswerCase],
    segment_fn,
) -> list[GuardedCandidateAnswerMetricsBySegment]:
    """Summarize guarded candidate answer proxy cases by a caller-defined segment."""

    return _segment_metrics(cases, segment_fn)


def write_guarded_candidate_answer_visualizations(
    result: GuardedCandidateAnswerExperimentResult,
    output_dir: Path,
) -> list[VisualizationArtifact]:
    """Write compact SVG charts for the guarded answer experiment."""

    output_dir.mkdir(parents=True, exist_ok=True)
    main_mode = _preferred_visualization_mode(result.main_policy.mode_evaluations)
    charts = {
        "guarded_answer_policy_delta.svg": _render_bar_chart_svg(
            title="Guarded reranker answer F1 delta by policy and mode",
            bars=_policy_delta_bars(result),
            x_label="average answer token F1 delta vs baseline",
        ),
        "guarded_answer_main_route_delta.svg": _render_bar_chart_svg(
            title=f"Main policy route delta ({main_mode.mode_name})",
            bars=[
                _Bar(
                    label=metric.segment_name,
                    value=metric.average_delta_vs_baseline,
                    value_label=(
                        f"{metric.average_delta_vs_baseline:+.4f} "
                        f"(n={metric.question_count})"
                    ),
                )
                for metric in main_mode.route_metrics
            ],
            x_label="average answer token F1 delta vs baseline",
        ),
        "guarded_answer_main_citation_delta.svg": _render_bar_chart_svg(
            title="Main policy gold-document citation delta by mode",
            bars=[
                _Bar(
                    label=evaluation.mode_name,
                    value=float(evaluation.metrics.gold_citation_delta),
                    value_label=(
                        f"{evaluation.metrics.gold_citation_delta:+d} "
                        f"({evaluation.metrics.policy_gold_citation_count}"
                        f"/{evaluation.metrics.baseline_gold_citation_count})"
                    ),
                )
                for evaluation in result.main_policy.mode_evaluations
            ],
            x_label="gold-document citation count delta",
        ),
    }

    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(VisualizationArtifact(name=filename, path=str(path)))
    return artifacts


def _evaluate_policy(
    config: CandidateRerankerPolicyConfig,
    rows: Sequence[Mapping[str, Any]],
    selections,
    gold_answers_by_question_key: Mapping[str, str] | None,
    max_answer_candidates: int,
    sample_limit: int,
) -> GuardedCandidateAnswerPolicyEvaluation:
    decisions = candidate_reranker_policy_decisions_from_selections(
        config=config,
        selections=selections,
        rows=rows,
    )
    single_cases = build_single_candidate_answer_cases_from_decisions(
        decisions=decisions,
        rows=rows,
    )
    mode_evaluations = [_mode_evaluation(SINGLE_CANDIDATE_MODE, single_cases, sample_limit)]

    if gold_answers_by_question_key is not None:
        topk_mode_name = f"top{max_answer_candidates}_leading_candidate_rewrite"
        topk_cases = build_topk_leading_candidate_answer_cases_from_decisions(
            decisions=decisions,
            rows=rows,
            gold_answers_by_question_key=gold_answers_by_question_key,
            max_answer_candidates=max_answer_candidates,
        )
        mode_evaluations.append(
            _mode_evaluation(topk_mode_name, topk_cases, sample_limit)
        )

    return GuardedCandidateAnswerPolicyEvaluation(
        config=config,
        mode_evaluations=mode_evaluations,
    )


def _single_candidate_case(
    decision: CandidateRerankerPolicyDecision,
    row_index: Mapping[str, list[Mapping[str, Any]]],
) -> GuardedCandidateAnswerCase:
    question_rows = row_index[_question_key(decision.split, decision.question_id)]
    baseline_row = _row_by_candidate_id(question_rows, decision.baseline_candidate_id)
    policy_row = _row_by_candidate_id(question_rows, decision.final_candidate_id)
    oracle_row = _row_by_candidate_id(question_rows, decision.oracle_candidate_id)

    return GuardedCandidateAnswerCase(
        mode_name=SINGLE_CANDIDATE_MODE,
        split=decision.split,
        question_id=decision.question_id,
        question_route=decision.question_route,
        action=decision.action,
        decision_reasons=list(decision.decision_reasons),
        baseline_candidate_ids=[decision.baseline_candidate_id],
        policy_candidate_ids=[decision.final_candidate_id],
        oracle_candidate_ids=[decision.oracle_candidate_id],
        baseline_document_ids=[_document_id(baseline_row)],
        policy_document_ids=[_document_id(policy_row)],
        oracle_document_ids=[_document_id(oracle_row)],
        baseline_answer_text=_answer_text([baseline_row]),
        policy_answer_text=_answer_text([policy_row]),
        oracle_answer_text=_answer_text([oracle_row]),
        baseline_answer_token_f1=decision.baseline_candidate_token_f1,
        policy_answer_token_f1=decision.final_candidate_token_f1,
        oracle_answer_token_f1=decision.oracle_candidate_token_f1,
        f1_delta_vs_baseline=decision.f1_delta_vs_top_candidate,
        baseline_gold_cited=_is_gold_document(baseline_row),
        policy_gold_cited=decision.final_is_gold_document,
        citation_delta=int(decision.final_is_gold_document) - int(_is_gold_document(baseline_row)),
        policy_leading_candidate_rank=decision.final_candidate_rank,
        policy_leading_is_oracle_best_f1=decision.final_is_oracle_best_f1,
    )


def _topk_leading_candidate_case(
    decision: CandidateRerankerPolicyDecision,
    row_index: Mapping[str, list[Mapping[str, Any]]],
    gold_answers_by_question_key: Mapping[str, str],
    max_answer_candidates: int,
    mode_name: str,
) -> GuardedCandidateAnswerCase:
    question_key = _question_key(decision.split, decision.question_id)
    if question_key not in gold_answers_by_question_key:
        raise ValueError(f"Missing gold answer for question: {question_key}")

    question_rows = sorted(
        row_index[question_key],
        key=lambda row: int(row["candidate_rank"]),
    )
    baseline_rows = question_rows[:max_answer_candidates]
    policy_leading_row = _row_by_candidate_id(question_rows, decision.final_candidate_id)
    oracle_leading_row = _row_by_candidate_id(question_rows, decision.oracle_candidate_id)
    policy_rows = _leading_rewrite_rows(
        leading_row=policy_leading_row,
        baseline_rows=baseline_rows,
        limit=max_answer_candidates,
    )
    oracle_rows = _leading_rewrite_rows(
        leading_row=oracle_leading_row,
        baseline_rows=baseline_rows,
        limit=max_answer_candidates,
    )
    gold_answer = gold_answers_by_question_key[question_key]
    baseline_answer = _answer_text(baseline_rows)
    policy_answer = _answer_text(policy_rows)
    oracle_answer = _answer_text(oracle_rows)
    baseline_f1 = round(token_f1(baseline_answer, gold_answer), 4)
    policy_f1 = round(token_f1(policy_answer, gold_answer), 4)
    oracle_f1 = round(token_f1(oracle_answer, gold_answer), 4)
    baseline_gold_cited = _any_gold_document(baseline_rows)
    policy_gold_cited = _any_gold_document(policy_rows)

    return GuardedCandidateAnswerCase(
        mode_name=mode_name,
        split=decision.split,
        question_id=decision.question_id,
        question_route=decision.question_route,
        action=decision.action,
        decision_reasons=list(decision.decision_reasons),
        baseline_candidate_ids=[str(row["candidate_id"]) for row in baseline_rows],
        policy_candidate_ids=[str(row["candidate_id"]) for row in policy_rows],
        oracle_candidate_ids=[str(row["candidate_id"]) for row in oracle_rows],
        baseline_document_ids=[_document_id(row) for row in baseline_rows],
        policy_document_ids=[_document_id(row) for row in policy_rows],
        oracle_document_ids=[_document_id(row) for row in oracle_rows],
        baseline_answer_text=_truncate(baseline_answer),
        policy_answer_text=_truncate(policy_answer),
        oracle_answer_text=_truncate(oracle_answer),
        baseline_answer_token_f1=baseline_f1,
        policy_answer_token_f1=policy_f1,
        oracle_answer_token_f1=oracle_f1,
        f1_delta_vs_baseline=round(policy_f1 - baseline_f1, 4),
        baseline_gold_cited=baseline_gold_cited,
        policy_gold_cited=policy_gold_cited,
        citation_delta=int(policy_gold_cited) - int(baseline_gold_cited),
        policy_leading_candidate_rank=decision.final_candidate_rank,
        policy_leading_is_oracle_best_f1=decision.final_is_oracle_best_f1,
    )


def _mode_evaluation(
    mode_name: str,
    cases: Sequence[GuardedCandidateAnswerCase],
    sample_limit: int,
) -> GuardedCandidateAnswerModeEvaluation:
    return GuardedCandidateAnswerModeEvaluation(
        mode_name=mode_name,
        metrics=_metrics(cases),
        route_metrics=_segment_metrics(cases, lambda case: case.question_route),
        split_metrics=_segment_metrics(cases, lambda case: case.split),
        sample_cases=_sample_cases(cases, sample_limit),
    )


def _metrics(
    cases: Sequence[GuardedCandidateAnswerCase],
) -> GuardedCandidateAnswerMetrics:
    question_count = len(cases)
    baseline_values = [case.baseline_answer_token_f1 for case in cases]
    policy_values = [case.policy_answer_token_f1 for case in cases]
    oracle_values = [case.oracle_answer_token_f1 for case in cases]
    policy_delta_values = [case.f1_delta_vs_baseline for case in cases]
    oracle_delta_values = [
        case.oracle_answer_token_f1 - case.baseline_answer_token_f1
        for case in cases
    ]
    outcome_counts = Counter(_outcome(delta) for delta in policy_delta_values)
    reason_counts = Counter(
        reason
        for case in cases
        for reason in case.decision_reasons
    )
    replacement_count = sum(
        case.action == "replace_with_model_candidate" for case in cases
    )
    changed_answer_count = sum(
        case.baseline_candidate_ids != case.policy_candidate_ids for case in cases
    )
    baseline_gold_citation_count = sum(case.baseline_gold_cited for case in cases)
    policy_gold_citation_count = sum(case.policy_gold_cited for case in cases)

    return GuardedCandidateAnswerMetrics(
        question_count=question_count,
        baseline_average_answer_token_f1=_rounded_mean(baseline_values),
        policy_average_answer_token_f1=_rounded_mean(policy_values),
        oracle_average_answer_token_f1=_rounded_mean(oracle_values),
        average_delta_vs_baseline=_rounded_mean(policy_delta_values),
        oracle_gap_closed_rate=_safe_ratio(
            sum(policy_delta_values),
            sum(oracle_delta_values),
        ),
        replacement_count=replacement_count,
        replacement_rate=_ratio(replacement_count, question_count),
        changed_answer_count=changed_answer_count,
        changed_answer_rate=_ratio(changed_answer_count, question_count),
        improved_count=outcome_counts["improved"],
        regressed_count=outcome_counts["regressed"],
        tied_count=outcome_counts["tied"],
        regressed_rate=_ratio(outcome_counts["regressed"], question_count),
        baseline_gold_citation_count=baseline_gold_citation_count,
        policy_gold_citation_count=policy_gold_citation_count,
        gold_citation_delta=policy_gold_citation_count - baseline_gold_citation_count,
        baseline_gold_citation_rate=_ratio(baseline_gold_citation_count, question_count),
        policy_gold_citation_rate=_ratio(policy_gold_citation_count, question_count),
        citation_lost_count=sum(case.citation_delta < 0 for case in cases),
        citation_gained_count=sum(case.citation_delta > 0 for case in cases),
        policy_oracle_best_count=sum(case.policy_leading_is_oracle_best_f1 for case in cases),
        policy_oracle_best_rate=_ratio(
            sum(case.policy_leading_is_oracle_best_f1 for case in cases),
            question_count,
        ),
        decision_reason_counts=dict(sorted(reason_counts.items())),
    )


def _segment_metrics(
    cases: Sequence[GuardedCandidateAnswerCase],
    segment_fn,
) -> list[GuardedCandidateAnswerMetricsBySegment]:
    cases_by_segment: dict[str, list[GuardedCandidateAnswerCase]] = defaultdict(list)
    for case in cases:
        cases_by_segment[str(segment_fn(case))].append(case)

    metrics = []
    for segment_name, segment_cases in cases_by_segment.items():
        summary = _metrics(segment_cases)
        metrics.append(
            GuardedCandidateAnswerMetricsBySegment(
                segment_name=segment_name,
                question_count=summary.question_count,
                baseline_average_answer_token_f1=(
                    summary.baseline_average_answer_token_f1
                ),
                policy_average_answer_token_f1=summary.policy_average_answer_token_f1,
                average_delta_vs_baseline=summary.average_delta_vs_baseline,
                replacement_count=summary.replacement_count,
                replacement_rate=summary.replacement_rate,
                changed_answer_count=summary.changed_answer_count,
                changed_answer_rate=summary.changed_answer_rate,
                improved_count=summary.improved_count,
                regressed_count=summary.regressed_count,
                regressed_rate=summary.regressed_rate,
                baseline_gold_citation_count=summary.baseline_gold_citation_count,
                policy_gold_citation_count=summary.policy_gold_citation_count,
                gold_citation_delta=summary.gold_citation_delta,
            )
        )
    return sorted(
        metrics,
        key=lambda item: (item.average_delta_vs_baseline, -item.regressed_count),
        reverse=True,
    )


def _sample_cases(
    cases: Sequence[GuardedCandidateAnswerCase],
    sample_limit: int,
) -> dict[str, list[GuardedCandidateAnswerCase]]:
    if sample_limit == 0:
        return {
            "largest_improvements": [],
            "largest_regressions": [],
            "citation_losses": [],
            "citation_gains": [],
        }

    return {
        "largest_improvements": sorted(
            [case for case in cases if case.f1_delta_vs_baseline > 0],
            key=lambda case: (-case.f1_delta_vs_baseline, case.question_id),
        )[:sample_limit],
        "largest_regressions": sorted(
            [case for case in cases if case.f1_delta_vs_baseline < 0],
            key=lambda case: (case.f1_delta_vs_baseline, case.question_id),
        )[:sample_limit],
        "citation_losses": sorted(
            [case for case in cases if case.citation_delta < 0],
            key=lambda case: (case.f1_delta_vs_baseline, case.question_id),
        )[:sample_limit],
        "citation_gains": sorted(
            [case for case in cases if case.citation_delta > 0],
            key=lambda case: (-case.f1_delta_vs_baseline, case.question_id),
        )[:sample_limit],
    }


def _policy_deltas(
    main_policy: GuardedCandidateAnswerPolicyEvaluation,
    sensitivity_policy: GuardedCandidateAnswerPolicyEvaluation,
) -> list[GuardedCandidateAnswerPolicyDelta]:
    sensitivity_by_mode = {
        evaluation.mode_name: evaluation
        for evaluation in sensitivity_policy.mode_evaluations
    }
    deltas = []
    for main_mode in main_policy.mode_evaluations:
        sensitivity_mode = sensitivity_by_mode[main_mode.mode_name]
        main_metrics = main_mode.metrics
        sensitivity_metrics = sensitivity_mode.metrics
        deltas.append(
            GuardedCandidateAnswerPolicyDelta(
                mode_name=main_mode.mode_name,
                main_policy_name=main_policy.config.name,
                sensitivity_policy_name=sensitivity_policy.config.name,
                policy_average_f1_difference=round(
                    main_metrics.policy_average_answer_token_f1
                    - sensitivity_metrics.policy_average_answer_token_f1,
                    4,
                ),
                average_delta_difference=round(
                    main_metrics.average_delta_vs_baseline
                    - sensitivity_metrics.average_delta_vs_baseline,
                    4,
                ),
                oracle_gap_closed_difference=round(
                    main_metrics.oracle_gap_closed_rate
                    - sensitivity_metrics.oracle_gap_closed_rate,
                    4,
                ),
                replacement_count_difference=(
                    main_metrics.replacement_count - sensitivity_metrics.replacement_count
                ),
                regressed_count_difference=(
                    main_metrics.regressed_count - sensitivity_metrics.regressed_count
                ),
                gold_citation_count_difference=(
                    main_metrics.policy_gold_citation_count
                    - sensitivity_metrics.policy_gold_citation_count
                ),
            )
        )
    return deltas


def _findings(
    main_policy: GuardedCandidateAnswerPolicyEvaluation,
    sensitivity_policy: GuardedCandidateAnswerPolicyEvaluation,
    deltas: Sequence[GuardedCandidateAnswerPolicyDelta],
) -> list[str]:
    findings = []
    main_single = _mode_by_name(main_policy, SINGLE_CANDIDATE_MODE)
    findings.append(
        "Main policy single-candidate proxy delta is "
        f"{main_single.metrics.average_delta_vs_baseline:+.4f} with "
        f"{main_single.metrics.regressed_count} regressions."
    )
    topk_mode = _first_topk_mode(main_policy.mode_evaluations)
    if topk_mode:
        findings.append(
            f"Top-k rewrite proxy delta is "
            f"{topk_mode.metrics.average_delta_vs_baseline:+.4f}; this is a "
            "metadata-sentence proxy, not a verified RAG runtime metric."
        )
    single_delta = next(
        delta for delta in deltas if delta.mode_name == SINGLE_CANDIDATE_MODE
    )
    if abs(single_delta.average_delta_difference) <= 0.001:
        findings.append(
            "Main and sensitivity policies are effectively tied in the "
            "single-candidate proxy, so the extra sensitivity route block remains "
            "sample-size sensitive."
        )
    if single_delta.regressed_count_difference > 0:
        findings.append(
            "Main policy has more single-candidate regressions than sensitivity."
        )
    elif single_delta.regressed_count_difference < 0:
        findings.append(
            "Main policy has fewer single-candidate regressions than sensitivity."
        )
    else:
        findings.append(
            "Main and sensitivity policies have the same single-candidate "
            "regression count."
        )
    sensitivity_single = _mode_by_name(sensitivity_policy, SINGLE_CANDIDATE_MODE)
    if (
        main_single.metrics.policy_gold_citation_count
        == sensitivity_single.metrics.policy_gold_citation_count
    ):
        findings.append(
            "Main and sensitivity policies select the same number of "
            "gold-document single-candidate answers."
        )
    return findings


def _build_row_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    row_index: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        row_index[_question_key(row["split"], row["question_id"])].append(row)
    return dict(row_index)


def _leading_rewrite_rows(
    leading_row: Mapping[str, Any],
    baseline_rows: Sequence[Mapping[str, Any]],
    limit: int,
) -> list[Mapping[str, Any]]:
    rows = [leading_row]
    leading_id = str(leading_row["candidate_id"])
    for row in baseline_rows:
        if str(row["candidate_id"]) == leading_id:
            continue
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows[:limit]


def _row_by_candidate_id(
    rows: Sequence[Mapping[str, Any]],
    candidate_id: str,
) -> Mapping[str, Any]:
    for row in rows:
        if row["candidate_id"] == candidate_id:
            return row
    raise ValueError(f"Missing candidate row: {candidate_id}")


def _answer_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return " ".join(_candidate_sentence(row) for row in rows if _candidate_sentence(row))


def _candidate_sentence(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("row metadata must be an object")
    return str(metadata.get("candidate_sentence", ""))


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


def _any_gold_document(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any(_is_gold_document(row) for row in rows)


def _preferred_visualization_mode(
    evaluations: Sequence[GuardedCandidateAnswerModeEvaluation],
) -> GuardedCandidateAnswerModeEvaluation:
    topk_mode = _first_topk_mode(evaluations)
    return topk_mode or evaluations[0]


def _first_topk_mode(
    evaluations: Sequence[GuardedCandidateAnswerModeEvaluation],
) -> GuardedCandidateAnswerModeEvaluation | None:
    for evaluation in evaluations:
        if evaluation.mode_name.startswith("top"):
            return evaluation
    return None


def _mode_by_name(
    policy: GuardedCandidateAnswerPolicyEvaluation,
    mode_name: str,
) -> GuardedCandidateAnswerModeEvaluation:
    for evaluation in policy.mode_evaluations:
        if evaluation.mode_name == mode_name:
            return evaluation
    raise ValueError(f"Missing mode evaluation: {mode_name}")


def _policy_delta_bars(result: GuardedCandidateAnswerExperimentResult) -> list[_Bar]:
    bars = []
    for policy_label, policy in (
        ("main", result.main_policy),
        ("sensitivity", result.sensitivity_policy),
    ):
        for evaluation in policy.mode_evaluations:
            bars.append(
                _Bar(
                    label=f"{policy_label} {evaluation.mode_name}",
                    value=evaluation.metrics.average_delta_vs_baseline,
                    value_label=f"{evaluation.metrics.average_delta_vs_baseline:+.4f}",
                )
            )
    return bars


def _question_key(split: Any, question_id: Any) -> str:
    return f"{split}::{question_id}"


def _outcome(delta: float) -> str:
    if delta > 0:
        return "improved"
    if delta < 0:
        return "regressed"
    return "tied"


def _rounded_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(mean(values), 4)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _truncate(text: str, max_chars: int = 1000) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}..."


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


def _render_bar_chart_svg(
    title: str,
    bars: Sequence[_Bar],
    x_label: str,
) -> str:
    width = 1060
    margin_left = 390
    margin_right = 190
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
