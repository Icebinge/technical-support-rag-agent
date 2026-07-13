from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any

from ts_rag_agent.application.candidate_reranker_cv import (
    cross_validated_candidate_reranker_selections,
)
from ts_rag_agent.application.candidate_reranker_policy_search import (
    CandidateRerankerPolicyDecision,
    candidate_reranker_policy_decisions_from_selections,
)
from ts_rag_agent.application.candidate_reranker_policy_stability import (
    STAGE35_BLOCK_HOW_TO_ONLY_POLICY,
)
from ts_rag_agent.application.guarded_candidate_reranker_answer_experiment import (
    GuardedCandidateAnswerCase,
    build_topk_leading_candidate_answer_cases_from_decisions,
)


@dataclass(frozen=True)
class ChangedCaseMetrics:
    """Aggregate metrics for full top-k guarded answer cases."""

    question_count: int
    changed_case_count: int
    changed_case_rate: float
    baseline_average_answer_token_f1: float
    policy_average_answer_token_f1: float
    average_delta_vs_baseline: float
    improved_count: int
    regressed_count: int
    tied_count: int
    citation_lost_count: int
    citation_gained_count: int
    baseline_gold_citation_count: int
    policy_gold_citation_count: int
    gold_citation_delta: int


@dataclass(frozen=True)
class ChangedCaseAttribution:
    """Feature and transition attribution for one changed top-k answer case."""

    split: str
    question_id: str
    question_route: str
    f1_outcome: str
    citation_outcome: str
    f1_delta_vs_baseline: float
    baseline_answer_token_f1: float
    policy_answer_token_f1: float
    baseline_gold_cited: bool
    policy_gold_cited: bool
    citation_delta: int
    baseline_leading_candidate_id: str
    policy_leading_candidate_id: str
    baseline_leading_document_id: str
    policy_leading_document_id: str
    added_document_ids: list[str]
    removed_document_ids: list[str]
    policy_leading_candidate_rank: int
    policy_leading_candidate_score: float
    baseline_leading_candidate_score: float
    candidate_score_delta_vs_baseline: float
    model_score_margin_vs_top_candidate: float
    rank_bucket: str
    model_margin_bucket: str
    candidate_score_bucket: str
    document_transition: str
    decision_reasons: list[str]
    baseline_candidate_ids: list[str]
    policy_candidate_ids: list[str]
    baseline_document_ids: list[str]
    policy_document_ids: list[str]
    baseline_answer_text: str
    policy_answer_text: str


@dataclass(frozen=True)
class ChangedCaseSegmentSummary:
    """Aggregate changed-case attribution summary for one segment."""

    segment_name: str
    case_count: int
    average_f1_delta: float
    min_f1_delta: float
    max_f1_delta: float
    improved_count: int
    regressed_count: int
    citation_lost_count: int
    citation_gained_count: int
    average_policy_leading_rank: float
    average_model_score_margin_vs_top_candidate: float
    average_policy_leading_candidate_score: float


@dataclass(frozen=True)
class StricterGateAudit:
    """Post-hoc audit for a runtime-feature-only stricter replacement gate."""

    name: str
    description: str
    blocked_replacement_count: int
    remaining_replacement_count: int
    average_delta_vs_baseline: float
    delta_change_vs_main_policy: float
    improved_count: int
    regressed_count: int
    citation_lost_count: int
    citation_gained_count: int
    policy_gold_citation_count: int
    gold_citation_delta: int


@dataclass(frozen=True)
class GuardedCandidateChangedCaseAnalysis:
    """Stage 38 changed-case analysis for guarded top-k candidate answers."""

    model_name: str
    fold_count: int
    mode_name: str
    policy_name: str
    metrics: ChangedCaseMetrics
    changed_case_route_summaries: list[ChangedCaseSegmentSummary]
    regression_route_summaries: list[ChangedCaseSegmentSummary]
    citation_loss_route_summaries: list[ChangedCaseSegmentSummary]
    rank_bucket_summaries: list[ChangedCaseSegmentSummary]
    margin_bucket_summaries: list[ChangedCaseSegmentSummary]
    candidate_score_bucket_summaries: list[ChangedCaseSegmentSummary]
    document_transition_summaries: list[ChangedCaseSegmentSummary]
    stricter_gate_audits: list[StricterGateAudit]
    regression_cases: list[ChangedCaseAttribution]
    citation_loss_cases: list[ChangedCaseAttribution]
    citation_gain_cases: list[ChangedCaseAttribution]
    largest_improvement_cases: list[ChangedCaseAttribution]
    findings: list[str]
    analysis_scope: str


@dataclass(frozen=True)
class VisualizationArtifact:
    """One generated visualization file."""

    name: str
    path: str


@dataclass(frozen=True)
class _GateSpec:
    name: str
    description: str
    should_block: Callable[[ChangedCaseAttribution], bool]


def analyze_guarded_candidate_changed_cases(
    stage37_report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    gold_answers_by_question_key: Mapping[str, str],
    model_name: str = "logistic_best_candidate",
    fold_count: int = 5,
    max_answer_candidates: int = 3,
    sample_limit: int = 20,
) -> GuardedCandidateChangedCaseAnalysis:
    """Analyze top-k changed cases from the Stage 37 guarded answer experiment."""

    _validate_options(rows=rows, max_answer_candidates=max_answer_candidates)
    mode_name = f"top{max_answer_candidates}_leading_candidate_rewrite"
    selections = cross_validated_candidate_reranker_selections(
        rows=rows,
        model_name=model_name,
        fold_count=fold_count,
    )
    decisions = candidate_reranker_policy_decisions_from_selections(
        config=STAGE35_BLOCK_HOW_TO_ONLY_POLICY,
        selections=selections,
        rows=rows,
    )
    cases = build_topk_leading_candidate_answer_cases_from_decisions(
        decisions=decisions,
        rows=rows,
        gold_answers_by_question_key=gold_answers_by_question_key,
        max_answer_candidates=max_answer_candidates,
    )
    metrics = _metrics(cases)
    _validate_stage37_report(
        report=stage37_report,
        mode_name=mode_name,
        computed_metrics=metrics,
    )

    row_index = _build_row_index(rows)
    decisions_by_key = {
        _question_key(decision.split, decision.question_id): decision
        for decision in decisions
    }
    changed_attributions = [
        _case_attribution(
            case=case,
            decision=decisions_by_key[_question_key(case.split, case.question_id)],
            row_index=row_index,
        )
        for case in cases
        if case.baseline_candidate_ids != case.policy_candidate_ids
    ]
    regression_cases = [
        attribution
        for attribution in changed_attributions
        if attribution.f1_outcome == "regressed"
    ]
    citation_loss_cases = [
        attribution
        for attribution in changed_attributions
        if attribution.citation_outcome == "citation_lost"
    ]
    citation_gain_cases = [
        attribution
        for attribution in changed_attributions
        if attribution.citation_outcome == "citation_gained"
    ]

    stricter_gate_audits = _stricter_gate_audits(
        cases=cases,
        changed_attributions=changed_attributions,
        main_metrics=metrics,
    )
    return GuardedCandidateChangedCaseAnalysis(
        model_name=model_name,
        fold_count=fold_count,
        mode_name=mode_name,
        policy_name=STAGE35_BLOCK_HOW_TO_ONLY_POLICY.name,
        metrics=metrics,
        changed_case_route_summaries=_segment_summaries(
            changed_attributions,
            lambda attribution: attribution.question_route,
        ),
        regression_route_summaries=_segment_summaries(
            regression_cases,
            lambda attribution: attribution.question_route,
        ),
        citation_loss_route_summaries=_segment_summaries(
            citation_loss_cases,
            lambda attribution: attribution.question_route,
        ),
        rank_bucket_summaries=_segment_summaries(
            changed_attributions,
            lambda attribution: attribution.rank_bucket,
        ),
        margin_bucket_summaries=_segment_summaries(
            changed_attributions,
            lambda attribution: attribution.model_margin_bucket,
        ),
        candidate_score_bucket_summaries=_segment_summaries(
            changed_attributions,
            lambda attribution: attribution.candidate_score_bucket,
        ),
        document_transition_summaries=_segment_summaries(
            changed_attributions,
            lambda attribution: attribution.document_transition,
        ),
        stricter_gate_audits=stricter_gate_audits,
        regression_cases=sorted(
            regression_cases,
            key=lambda attribution: (
                attribution.f1_delta_vs_baseline,
                attribution.question_id,
            ),
        )[:sample_limit],
        citation_loss_cases=sorted(
            citation_loss_cases,
            key=lambda attribution: (
                attribution.f1_delta_vs_baseline,
                attribution.question_id,
            ),
        )[:sample_limit],
        citation_gain_cases=sorted(
            citation_gain_cases,
            key=lambda attribution: (
                -attribution.f1_delta_vs_baseline,
                attribution.question_id,
            ),
        )[:sample_limit],
        largest_improvement_cases=sorted(
            [
                attribution
                for attribution in changed_attributions
                if attribution.f1_outcome == "improved"
            ],
            key=lambda attribution: (
                -attribution.f1_delta_vs_baseline,
                attribution.question_id,
            ),
        )[:sample_limit],
        findings=_findings(
            metrics=metrics,
            changed_attributions=changed_attributions,
            regression_cases=regression_cases,
            citation_loss_cases=citation_loss_cases,
            gate_audits=stricter_gate_audits,
        ),
        analysis_scope=(
            "Offline changed-case analysis only. It validates Stage 37 aggregate "
            "metrics against the Stage 37 report, rebuilds full top-k proxy cases "
            "from the ignored Stage 31 candidate dataset, and audits stricter "
            "runtime-feature gates post hoc. It does not change runtime behavior."
        ),
    )


def guarded_candidate_changed_case_analysis_to_dict(
    analysis: GuardedCandidateChangedCaseAnalysis,
) -> dict[str, Any]:
    """Convert a changed-case analysis result to a JSON-safe dictionary."""

    return asdict(analysis)


def write_changed_case_visualizations(
    analysis: GuardedCandidateChangedCaseAnalysis,
    output_dir: Path,
) -> list[VisualizationArtifact]:
    """Write SVG charts for the changed-case analysis."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage38_changed_case_outcomes.svg": _render_bar_chart_svg(
            title="Stage 38 top-k changed-case outcomes",
            bars=[
                _Bar(
                    "improved",
                    float(analysis.metrics.improved_count),
                    str(analysis.metrics.improved_count),
                ),
                _Bar(
                    "regressed",
                    float(analysis.metrics.regressed_count),
                    str(analysis.metrics.regressed_count),
                ),
                _Bar(
                    "citation lost",
                    float(analysis.metrics.citation_lost_count),
                    str(analysis.metrics.citation_lost_count),
                ),
                _Bar(
                    "citation gained",
                    float(analysis.metrics.citation_gained_count),
                    str(analysis.metrics.citation_gained_count),
                ),
            ],
            x_label="case count",
        ),
        "stage38_regressions_by_route.svg": _render_bar_chart_svg(
            title="Stage 38 top-k regressions by route",
            bars=[
                _Bar(
                    label=summary.segment_name,
                    value=float(summary.case_count),
                    value_label=(
                        f"{summary.case_count} "
                        f"(avg {summary.average_f1_delta:+.4f})"
                    ),
                )
                for summary in analysis.regression_route_summaries
            ],
            x_label="regression cases",
        ),
        "stage38_gate_audit_delta.svg": _render_bar_chart_svg(
            title="Stage 38 stricter gate average F1 delta",
            bars=[
                _Bar(
                    label=audit.name,
                    value=audit.average_delta_vs_baseline,
                    value_label=(
                        f"{audit.average_delta_vs_baseline:+.4f} "
                        f"(blocked {audit.blocked_replacement_count})"
                    ),
                )
                for audit in analysis.stricter_gate_audits
            ],
            x_label="average F1 delta vs baseline",
        ),
        "stage38_gate_audit_regressions.svg": _render_bar_chart_svg(
            title="Stage 38 stricter gate regression count",
            bars=[
                _Bar(
                    label=audit.name,
                    value=float(audit.regressed_count),
                    value_label=str(audit.regressed_count),
                )
                for audit in analysis.stricter_gate_audits
            ],
            x_label="regression cases",
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(VisualizationArtifact(name=filename, path=str(path)))
    return artifacts


def _validate_stage37_report(
    report: Mapping[str, Any],
    mode_name: str,
    computed_metrics: ChangedCaseMetrics,
) -> None:
    mode_metrics = _stage37_main_mode_metrics(report, mode_name)
    if int(mode_metrics["question_count"]) != computed_metrics.question_count:
        raise ValueError("Stage 37 report question_count does not match recomputed cases")
    if int(mode_metrics["regressed_count"]) != computed_metrics.regressed_count:
        raise ValueError("Stage 37 report regressed_count does not match recomputed cases")
    if int(mode_metrics["citation_lost_count"]) != computed_metrics.citation_lost_count:
        raise ValueError("Stage 37 report citation_lost_count does not match recomputed cases")
    if int(mode_metrics["citation_gained_count"]) != computed_metrics.citation_gained_count:
        raise ValueError("Stage 37 report citation_gained_count does not match recomputed cases")
    if (
        round(float(mode_metrics["average_delta_vs_baseline"]), 4)
        != computed_metrics.average_delta_vs_baseline
    ):
        raise ValueError(
            "Stage 37 report average_delta_vs_baseline does not match recomputed cases"
        )


def _stage37_main_mode_metrics(
    report: Mapping[str, Any],
    mode_name: str,
) -> Mapping[str, Any]:
    main_policy = report.get("main_policy")
    if not isinstance(main_policy, Mapping):
        raise ValueError("Stage 37 report must contain main_policy")
    config = main_policy.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Stage 37 main_policy must contain config")
    if config.get("name") != STAGE35_BLOCK_HOW_TO_ONLY_POLICY.name:
        raise ValueError("Stage 37 report main policy does not match Stage 38 policy")
    mode_evaluations = main_policy.get("mode_evaluations")
    if not isinstance(mode_evaluations, list):
        raise ValueError("Stage 37 main_policy must contain mode_evaluations")
    for mode_evaluation in mode_evaluations:
        if (
            isinstance(mode_evaluation, Mapping)
            and mode_evaluation.get("mode_name") == mode_name
            and isinstance(mode_evaluation.get("metrics"), Mapping)
        ):
            return mode_evaluation["metrics"]
    raise ValueError(f"Stage 37 report missing main mode metrics: {mode_name}")


def _metrics(cases: Sequence[GuardedCandidateAnswerCase]) -> ChangedCaseMetrics:
    changed_cases = [
        case for case in cases if case.baseline_candidate_ids != case.policy_candidate_ids
    ]
    f1_deltas = [case.f1_delta_vs_baseline for case in cases]
    outcome_counts = Counter(_f1_outcome(case.f1_delta_vs_baseline) for case in cases)
    baseline_gold_count = sum(case.baseline_gold_cited for case in cases)
    policy_gold_count = sum(case.policy_gold_cited for case in cases)
    return ChangedCaseMetrics(
        question_count=len(cases),
        changed_case_count=len(changed_cases),
        changed_case_rate=_ratio(len(changed_cases), len(cases)),
        baseline_average_answer_token_f1=_rounded_mean(
            [case.baseline_answer_token_f1 for case in cases]
        ),
        policy_average_answer_token_f1=_rounded_mean(
            [case.policy_answer_token_f1 for case in cases]
        ),
        average_delta_vs_baseline=_rounded_mean(f1_deltas),
        improved_count=outcome_counts["improved"],
        regressed_count=outcome_counts["regressed"],
        tied_count=outcome_counts["tied"],
        citation_lost_count=sum(case.citation_delta < 0 for case in cases),
        citation_gained_count=sum(case.citation_delta > 0 for case in cases),
        baseline_gold_citation_count=baseline_gold_count,
        policy_gold_citation_count=policy_gold_count,
        gold_citation_delta=policy_gold_count - baseline_gold_count,
    )


def _case_attribution(
    case: GuardedCandidateAnswerCase,
    decision: CandidateRerankerPolicyDecision,
    row_index: Mapping[str, list[Mapping[str, Any]]],
) -> ChangedCaseAttribution:
    question_rows = row_index[_question_key(case.split, case.question_id)]
    baseline_leading_row = _row_by_candidate_id(
        question_rows,
        case.baseline_candidate_ids[0],
    )
    policy_leading_row = _row_by_candidate_id(
        question_rows,
        case.policy_candidate_ids[0],
    )
    policy_candidate_score = _runtime_feature_float(policy_leading_row, "candidate_score")
    baseline_candidate_score = _runtime_feature_float(
        baseline_leading_row,
        "candidate_score",
    )
    added_document_ids = sorted(set(case.policy_document_ids) - set(case.baseline_document_ids))
    removed_document_ids = sorted(
        set(case.baseline_document_ids) - set(case.policy_document_ids)
    )

    return ChangedCaseAttribution(
        split=case.split,
        question_id=case.question_id,
        question_route=case.question_route,
        f1_outcome=_f1_outcome(case.f1_delta_vs_baseline),
        citation_outcome=_citation_outcome(case.citation_delta),
        f1_delta_vs_baseline=case.f1_delta_vs_baseline,
        baseline_answer_token_f1=case.baseline_answer_token_f1,
        policy_answer_token_f1=case.policy_answer_token_f1,
        baseline_gold_cited=case.baseline_gold_cited,
        policy_gold_cited=case.policy_gold_cited,
        citation_delta=case.citation_delta,
        baseline_leading_candidate_id=case.baseline_candidate_ids[0],
        policy_leading_candidate_id=case.policy_candidate_ids[0],
        baseline_leading_document_id=case.baseline_document_ids[0],
        policy_leading_document_id=case.policy_document_ids[0],
        added_document_ids=added_document_ids,
        removed_document_ids=removed_document_ids,
        policy_leading_candidate_rank=case.policy_leading_candidate_rank,
        policy_leading_candidate_score=policy_candidate_score,
        baseline_leading_candidate_score=baseline_candidate_score,
        candidate_score_delta_vs_baseline=round(
            policy_candidate_score - baseline_candidate_score,
            4,
        ),
        model_score_margin_vs_top_candidate=decision.model_score_margin_vs_top_candidate,
        rank_bucket=_rank_bucket(case.policy_leading_candidate_rank),
        model_margin_bucket=_model_margin_bucket(
            decision.model_score_margin_vs_top_candidate
        ),
        candidate_score_bucket=_candidate_score_bucket(policy_candidate_score),
        document_transition=_document_transition(
            baseline_row=baseline_leading_row,
            policy_row=policy_leading_row,
        ),
        decision_reasons=list(case.decision_reasons),
        baseline_candidate_ids=list(case.baseline_candidate_ids),
        policy_candidate_ids=list(case.policy_candidate_ids),
        baseline_document_ids=list(case.baseline_document_ids),
        policy_document_ids=list(case.policy_document_ids),
        baseline_answer_text=case.baseline_answer_text,
        policy_answer_text=case.policy_answer_text,
    )


def _segment_summaries(
    attributions: Sequence[ChangedCaseAttribution],
    segment_fn: Callable[[ChangedCaseAttribution], str],
) -> list[ChangedCaseSegmentSummary]:
    by_segment: dict[str, list[ChangedCaseAttribution]] = defaultdict(list)
    for attribution in attributions:
        by_segment[str(segment_fn(attribution))].append(attribution)
    summaries = []
    for segment_name, segment_cases in by_segment.items():
        deltas = [case.f1_delta_vs_baseline for case in segment_cases]
        summaries.append(
            ChangedCaseSegmentSummary(
                segment_name=segment_name,
                case_count=len(segment_cases),
                average_f1_delta=_rounded_mean(deltas),
                min_f1_delta=round(min(deltas), 4) if deltas else 0.0,
                max_f1_delta=round(max(deltas), 4) if deltas else 0.0,
                improved_count=sum(case.f1_outcome == "improved" for case in segment_cases),
                regressed_count=sum(
                    case.f1_outcome == "regressed" for case in segment_cases
                ),
                citation_lost_count=sum(
                    case.citation_outcome == "citation_lost" for case in segment_cases
                ),
                citation_gained_count=sum(
                    case.citation_outcome == "citation_gained" for case in segment_cases
                ),
                average_policy_leading_rank=_rounded_mean(
                    [case.policy_leading_candidate_rank for case in segment_cases]
                ),
                average_model_score_margin_vs_top_candidate=_rounded_mean(
                    [
                        case.model_score_margin_vs_top_candidate
                        for case in segment_cases
                    ]
                ),
                average_policy_leading_candidate_score=_rounded_mean(
                    [case.policy_leading_candidate_score for case in segment_cases]
                ),
            )
        )
    return sorted(
        summaries,
        key=lambda summary: (summary.case_count, abs(summary.average_f1_delta)),
        reverse=True,
    )


def _stricter_gate_audits(
    cases: Sequence[GuardedCandidateAnswerCase],
    changed_attributions: Sequence[ChangedCaseAttribution],
    main_metrics: ChangedCaseMetrics,
) -> list[StricterGateAudit]:
    changed_by_key = {
        _question_key(attribution.split, attribution.question_id): attribution
        for attribution in changed_attributions
    }
    return [
        _stricter_gate_audit(
            spec=spec,
            cases=cases,
            changed_by_key=changed_by_key,
            main_metrics=main_metrics,
        )
        for spec in _default_gate_specs()
    ]


def _stricter_gate_audit(
    spec: _GateSpec,
    cases: Sequence[GuardedCandidateAnswerCase],
    changed_by_key: Mapping[str, ChangedCaseAttribution],
    main_metrics: ChangedCaseMetrics,
) -> StricterGateAudit:
    blocked_replacement_count = 0
    effective_cases = []
    for case in cases:
        attribution = changed_by_key.get(_question_key(case.split, case.question_id))
        if attribution and spec.should_block(attribution):
            blocked_replacement_count += 1
            effective_cases.append(_baseline_effective_case(case))
        else:
            effective_cases.append(case)
    metrics = _metrics(effective_cases)
    return StricterGateAudit(
        name=spec.name,
        description=spec.description,
        blocked_replacement_count=blocked_replacement_count,
        remaining_replacement_count=(
            main_metrics.changed_case_count - blocked_replacement_count
        ),
        average_delta_vs_baseline=metrics.average_delta_vs_baseline,
        delta_change_vs_main_policy=round(
            metrics.average_delta_vs_baseline - main_metrics.average_delta_vs_baseline,
            4,
        ),
        improved_count=metrics.improved_count,
        regressed_count=metrics.regressed_count,
        citation_lost_count=metrics.citation_lost_count,
        citation_gained_count=metrics.citation_gained_count,
        policy_gold_citation_count=metrics.policy_gold_citation_count,
        gold_citation_delta=metrics.gold_citation_delta,
    )


def _default_gate_specs() -> list[_GateSpec]:
    return [
        _GateSpec(
            name="rank_lte_3",
            description="Reject replacement candidates below rank 3.",
            should_block=lambda attribution: attribution.policy_leading_candidate_rank > 3,
        ),
        _GateSpec(
            name="rank_lte_4",
            description="Reject replacement candidates below rank 4.",
            should_block=lambda attribution: attribution.policy_leading_candidate_rank > 4,
        ),
        _GateSpec(
            name="model_margin_gte_0.10",
            description="Reject replacements with model score margin below 0.10.",
            should_block=lambda attribution: (
                attribution.model_score_margin_vs_top_candidate < 0.10
            ),
        ),
        _GateSpec(
            name="model_margin_gte_0.20",
            description="Reject replacements with model score margin below 0.20.",
            should_block=lambda attribution: (
                attribution.model_score_margin_vs_top_candidate < 0.20
            ),
        ),
        _GateSpec(
            name="rank_lte_4_and_margin_gte_0.10",
            description="Reject replacements below rank 4 or with margin below 0.10.",
            should_block=lambda attribution: (
                attribution.policy_leading_candidate_rank > 4
                or attribution.model_score_margin_vs_top_candidate < 0.10
            ),
        ),
        _GateSpec(
            name="candidate_score_gte_60",
            description="Reject replacements whose selected candidate score is below 60.",
            should_block=lambda attribution: (
                attribution.policy_leading_candidate_score < 60
            ),
        ),
        _GateSpec(
            name="candidate_score_gte_90",
            description="Reject replacements whose selected candidate score is below 90.",
            should_block=lambda attribution: (
                attribution.policy_leading_candidate_score < 90
            ),
        ),
        _GateSpec(
            name="candidate_score_gte_baseline",
            description="Reject replacements whose selected candidate score is below top1.",
            should_block=lambda attribution: (
                attribution.policy_leading_candidate_score
                < attribution.baseline_leading_candidate_score
            ),
        ),
        _GateSpec(
            name="same_leading_document_only",
            description="Reject replacements whose leading document changes.",
            should_block=lambda attribution: (
                attribution.baseline_leading_document_id
                != attribution.policy_leading_document_id
            ),
        ),
    ]


def _baseline_effective_case(
    case: GuardedCandidateAnswerCase,
) -> GuardedCandidateAnswerCase:
    return GuardedCandidateAnswerCase(
        mode_name=case.mode_name,
        split=case.split,
        question_id=case.question_id,
        question_route=case.question_route,
        action="keep_top_candidate",
        decision_reasons=["blocked_by_stricter_gate_audit"],
        baseline_candidate_ids=list(case.baseline_candidate_ids),
        policy_candidate_ids=list(case.baseline_candidate_ids),
        oracle_candidate_ids=list(case.oracle_candidate_ids),
        baseline_document_ids=list(case.baseline_document_ids),
        policy_document_ids=list(case.baseline_document_ids),
        oracle_document_ids=list(case.oracle_document_ids),
        baseline_answer_text=case.baseline_answer_text,
        policy_answer_text=case.baseline_answer_text,
        oracle_answer_text=case.oracle_answer_text,
        baseline_answer_token_f1=case.baseline_answer_token_f1,
        policy_answer_token_f1=case.baseline_answer_token_f1,
        oracle_answer_token_f1=case.oracle_answer_token_f1,
        f1_delta_vs_baseline=0.0,
        baseline_gold_cited=case.baseline_gold_cited,
        policy_gold_cited=case.baseline_gold_cited,
        citation_delta=0,
        policy_leading_candidate_rank=1,
        policy_leading_is_oracle_best_f1=False,
    )


def _findings(
    metrics: ChangedCaseMetrics,
    changed_attributions: Sequence[ChangedCaseAttribution],
    regression_cases: Sequence[ChangedCaseAttribution],
    citation_loss_cases: Sequence[ChangedCaseAttribution],
    gate_audits: Sequence[StricterGateAudit],
) -> list[str]:
    findings = [
        (
            f"Main policy changes {metrics.changed_case_count} / "
            f"{metrics.question_count} top-k proxy answers."
        ),
        (
            f"Top-k proxy has {metrics.regressed_count} regressions and "
            f"{metrics.citation_lost_count} citation losses."
        ),
    ]
    if regression_cases:
        route_counts = Counter(case.question_route for case in regression_cases)
        dominant_route, dominant_count = route_counts.most_common(1)[0]
        findings.append(
            f"Most regressions are in route {dominant_route}: "
            f"{dominant_count} / {len(regression_cases)}."
        )
    if citation_loss_cases:
        findings.append(
            "Citation losses are few but non-zero, so unchanged citation totals hide "
            "case-level swaps."
        )
    rank_gt_3_regressions = sum(
        case.policy_leading_candidate_rank > 3 for case in regression_cases
    )
    if regression_cases:
        findings.append(
            f"{rank_gt_3_regressions} / {len(regression_cases)} regressions use "
            "a replacement below rank 3."
        )
    best_regression_gate = min(
        gate_audits,
        key=lambda audit: (
            audit.regressed_count,
            -audit.average_delta_vs_baseline,
            audit.blocked_replacement_count,
        ),
    )
    findings.append(
        f"Best audited gate by regression count is {best_regression_gate.name}: "
        f"regressions {best_regression_gate.regressed_count}, "
        f"average delta {best_regression_gate.average_delta_vs_baseline:+.4f}."
    )
    positive_delta_lower_regression = [
        audit
        for audit in gate_audits
        if (
            audit.average_delta_vs_baseline > 0
            and audit.regressed_count < metrics.regressed_count
        )
    ]
    if positive_delta_lower_regression:
        best_balanced = max(
            positive_delta_lower_regression,
            key=lambda audit: (
                audit.average_delta_vs_baseline,
                -audit.regressed_count,
            ),
        )
        findings.append(
            f"Balanced audited gate candidate is {best_balanced.name}: "
            f"delta {best_balanced.average_delta_vs_baseline:+.4f}, "
            f"regressions {best_balanced.regressed_count}."
        )
    else:
        findings.append(
            "No audited stricter gate both keeps positive delta and reduces regression count."
        )
    if not changed_attributions:
        findings.append("No changed cases were available for attribution.")
    return findings


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


def _is_gold_document(row: Mapping[str, Any]) -> bool:
    gold_labels = row.get("gold_labels")
    if not isinstance(gold_labels, Mapping):
        raise ValueError("row gold_labels must be an object")
    return bool(gold_labels["is_gold_document"])


def _document_transition(
    baseline_row: Mapping[str, Any],
    policy_row: Mapping[str, Any],
) -> str:
    if _document_id(baseline_row) == _document_id(policy_row):
        return "same_leading_document"
    if _is_gold_document(policy_row) and not _is_gold_document(baseline_row):
        return "new_gold_leading_document"
    if _is_gold_document(baseline_row) and not _is_gold_document(policy_row):
        return "gold_to_non_gold_leading_document"
    return "new_non_gold_leading_document"


def _document_id(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("row metadata must be an object")
    return str(metadata.get("document_id", ""))


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


def _model_margin_bucket(margin: float) -> str:
    if margin < 0.10:
        return "margin_0.05_0.10"
    if margin < 0.20:
        return "margin_0.10_0.20"
    if margin < 0.40:
        return "margin_0.20_0.40"
    return "margin_0.40_plus"


def _candidate_score_bucket(score: float) -> str:
    if score < 60:
        return "score_lt_60"
    if score < 90:
        return "score_60_90"
    if score < 120:
        return "score_90_120"
    return "score_120_plus"


def _f1_outcome(delta: float) -> str:
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


def _validate_options(rows: Sequence[Mapping[str, Any]], max_answer_candidates: int) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    if max_answer_candidates <= 0:
        raise ValueError("max_answer_candidates must be positive")


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
    width = 1100
    margin_left = 370
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
