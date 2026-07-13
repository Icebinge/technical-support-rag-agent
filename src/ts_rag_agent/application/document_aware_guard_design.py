from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
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
class DocumentAwareGuardContext:
    """Offline-only features used to design and audit a guard candidate."""

    split: str
    question_id: str
    question_route: str
    replacement_candidate_id: str
    replacement_candidate_rank: int
    replacement_candidate_score: float
    replacement_candidate_score_bucket: str
    model_score_margin_vs_top_candidate: float
    baseline_to_replacement_document_transition: str
    replacement_citation_delta_vs_baseline: int


@dataclass(frozen=True)
class DocumentAwareGuardSpec:
    """One offline diagnostic guard design."""

    label: str
    description: str
    feature_scope: str
    accept_replacement: Callable[[DocumentAwareGuardContext], bool]


@dataclass(frozen=True)
class DocumentAwareGuardProbe:
    """One tracked question under one guard."""

    question_key: str
    guard_label: str
    action: str
    decision_reasons: list[str]
    final_candidate_id: str
    final_candidate_rank: int
    final_candidate_score: float
    final_document_id: str
    baseline_to_final_document_transition: str
    answer_token_f1: float
    delta_vs_baseline: float
    gold_cited: bool
    citation_delta: int


@dataclass(frozen=True)
class DocumentAwareGuardEvaluation:
    """Metrics for one Stage 42 guard candidate."""

    label: str
    description: str
    feature_scope: str
    metrics: GuardedCandidateAnswerMetrics
    changed_count_vs_main: int
    changed_count_vs_score60: int
    fixed_main_regression_count: int
    introduced_regression_vs_main_count: int
    probe_cases: list[DocumentAwareGuardProbe]


@dataclass(frozen=True)
class DocumentAwareGuardDelta:
    """Metric differences between one guard and a reference guard."""

    guard_label: str
    reference_label: str
    average_delta_difference: float
    policy_average_f1_difference: float
    regressed_count_difference: int
    citation_lost_count_difference: int
    gold_citation_delta_difference: int
    replacement_count_difference: int


@dataclass(frozen=True)
class DocumentAwareGuardDesignResult:
    """Stage 42 document/citation-aware guard design result."""

    model_name: str
    train_split: str
    evaluation_split: str
    mode_name: str
    max_answer_candidates: int
    guard_evaluations: list[DocumentAwareGuardEvaluation]
    deltas_vs_main: list[DocumentAwareGuardDelta]
    deltas_vs_score60: list[DocumentAwareGuardDelta]
    findings: list[str]
    analysis_scope: str


@dataclass(frozen=True)
class VisualizationArtifact:
    """One generated visualization file."""

    name: str
    path: str


DEFAULT_PROBE_QUESTION_IDS = ("DEV_Q119", "DEV_Q201", "DEV_Q261")
DOCUMENT_RISK_GUARD_LABEL = "citation_doc_risk_v1"


def evaluate_document_aware_holdout_guards(
    stage41_report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    gold_answers_by_question_key: Mapping[str, str],
    model_name: str = "logistic_best_candidate",
    train_split: str = "train",
    evaluation_split: str = "dev",
    max_answer_candidates: int = 3,
    probe_question_ids: Sequence[str] = DEFAULT_PROBE_QUESTION_IDS,
) -> DocumentAwareGuardDesignResult:
    """Evaluate offline document/citation-aware guard designs on dev holdout."""

    _validate_options(rows=rows, max_answer_candidates=max_answer_candidates)
    normalized_train_split = _normalize_split_name(train_split)
    normalized_evaluation_split = _normalize_split_name(evaluation_split)
    evaluation_rows = _rows_for_split(rows, normalized_evaluation_split)
    mode_name = f"top{max_answer_candidates}_leading_candidate_rewrite"
    policy_configs = _policy_configs_by_label()

    selections = split_validated_candidate_reranker_selections(
        rows=rows,
        model_name=model_name,
        train_split=normalized_train_split,
        validation_split=normalized_evaluation_split,
    )
    main_decisions = candidate_reranker_policy_decisions_from_selections(
        config=policy_configs[STAGE39_MAIN_POLICY_LABEL],
        selections=selections,
        rows=evaluation_rows,
    )
    main_cases = build_topk_leading_candidate_answer_cases_from_decisions(
        decisions=main_decisions,
        rows=evaluation_rows,
        gold_answers_by_question_key=gold_answers_by_question_key,
        max_answer_candidates=max_answer_candidates,
    )
    main_metrics = summarize_guarded_candidate_answer_cases(main_cases)
    score60_decisions = candidate_reranker_policy_decisions_from_selections(
        config=policy_configs[CANDIDATE_SCORE_GTE_60_LABEL],
        selections=selections,
        rows=evaluation_rows,
    )
    score60_cases = build_topk_leading_candidate_answer_cases_from_decisions(
        decisions=score60_decisions,
        rows=evaluation_rows,
        gold_answers_by_question_key=gold_answers_by_question_key,
        max_answer_candidates=max_answer_candidates,
    )
    score60_metrics = summarize_guarded_candidate_answer_cases(score60_cases)
    _validate_stage41_report(
        report=stage41_report,
        model_name=model_name,
        train_split=normalized_train_split,
        evaluation_split=normalized_evaluation_split,
        mode_name=mode_name,
        main_metrics=main_metrics,
        score60_metrics=score60_metrics,
    )

    row_index = _build_row_index(evaluation_rows)
    main_cases_by_key = _cases_by_key(main_cases)
    score60_cases_by_key = _cases_by_key(score60_cases)
    guard_evaluations = [
        _evaluate_guard(
            spec=spec,
            main_decisions=main_decisions,
            main_cases=main_cases,
            main_cases_by_key=main_cases_by_key,
            score60_cases_by_key=score60_cases_by_key,
            rows=evaluation_rows,
            row_index=row_index,
            gold_answers_by_question_key=gold_answers_by_question_key,
            max_answer_candidates=max_answer_candidates,
            probe_question_ids=probe_question_ids,
        )
        for spec in _default_guard_specs()
    ]
    main_evaluation = _evaluation_by_label(guard_evaluations, STAGE39_MAIN_POLICY_LABEL)
    score60_evaluation = _evaluation_by_label(guard_evaluations, CANDIDATE_SCORE_GTE_60_LABEL)
    deltas_vs_main = [
        _delta(evaluation, main_evaluation) for evaluation in guard_evaluations
    ]
    deltas_vs_score60 = [
        _delta(evaluation, score60_evaluation) for evaluation in guard_evaluations
    ]

    return DocumentAwareGuardDesignResult(
        model_name=model_name,
        train_split=normalized_train_split,
        evaluation_split=normalized_evaluation_split,
        mode_name=mode_name,
        max_answer_candidates=max_answer_candidates,
        guard_evaluations=guard_evaluations,
        deltas_vs_main=deltas_vs_main,
        deltas_vs_score60=deltas_vs_score60,
        findings=_findings(
            guard_evaluations=guard_evaluations,
            score60_label=CANDIDATE_SCORE_GTE_60_LABEL,
        ),
        analysis_scope=(
            "Offline Stage 42 guard design only. Some guard variants use gold/citation "
            "or gold-document transition labels as diagnostic design evidence, so they "
            "are explicitly not runtime policies. Held-out test data is not used and "
            "runtime behavior is not changed."
        ),
    )


def document_aware_guard_design_to_dict(
    result: DocumentAwareGuardDesignResult,
) -> dict[str, Any]:
    """Convert a Stage 42 guard design result to a JSON-safe dictionary."""

    return asdict(result)


def write_document_aware_guard_visualizations(
    result: DocumentAwareGuardDesignResult,
    output_dir: Path,
) -> list[VisualizationArtifact]:
    """Write SVG charts for Stage 42 guard design."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage42_guard_delta.svg": _render_bar_chart_svg(
            title="Stage 42 guard average F1 delta",
            bars=[
                _Bar(
                    evaluation.label,
                    evaluation.metrics.average_delta_vs_baseline,
                    f"{evaluation.metrics.average_delta_vs_baseline:+.4f}",
                )
                for evaluation in result.guard_evaluations
            ],
            x_label="average answer token F1 delta vs baseline",
        ),
        "stage42_guard_regressions.svg": _render_bar_chart_svg(
            title="Stage 42 guard regression count",
            bars=[
                _Bar(
                    evaluation.label,
                    float(evaluation.metrics.regressed_count),
                    str(evaluation.metrics.regressed_count),
                )
                for evaluation in result.guard_evaluations
            ],
            x_label="regression cases",
        ),
        "stage42_guard_citation_loss.svg": _render_bar_chart_svg(
            title="Stage 42 guard citation-loss count",
            bars=[
                _Bar(
                    evaluation.label,
                    float(evaluation.metrics.citation_lost_count),
                    str(evaluation.metrics.citation_lost_count),
                )
                for evaluation in result.guard_evaluations
            ],
            x_label="citation-loss cases",
        ),
        "stage42_guard_changed_vs_score60.svg": _render_bar_chart_svg(
            title="Stage 42 guard changed cases vs candidate_score_gte_60",
            bars=[
                _Bar(
                    evaluation.label,
                    float(evaluation.changed_count_vs_score60),
                    str(evaluation.changed_count_vs_score60),
                )
                for evaluation in result.guard_evaluations
            ],
            x_label="changed cases vs candidate_score_gte_60",
        ),
        "stage42_probe_delta.svg": _render_bar_chart_svg(
            title="Stage 42 tracked probe deltas",
            bars=[
                _Bar(
                    f"{probe.guard_label} {probe.question_key}",
                    probe.delta_vs_baseline,
                    f"{probe.delta_vs_baseline:+.4f}",
                )
                for evaluation in result.guard_evaluations
                for probe in evaluation.probe_cases
            ],
            x_label="answer token F1 delta vs baseline",
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(VisualizationArtifact(name=filename, path=str(path)))
    return artifacts


def _evaluate_guard(
    spec: DocumentAwareGuardSpec,
    main_decisions: Sequence[CandidateRerankerPolicyDecision],
    main_cases: Sequence[GuardedCandidateAnswerCase],
    main_cases_by_key: Mapping[str, GuardedCandidateAnswerCase],
    score60_cases_by_key: Mapping[str, GuardedCandidateAnswerCase],
    rows: Sequence[Mapping[str, Any]],
    row_index: Mapping[str, list[Mapping[str, Any]]],
    gold_answers_by_question_key: Mapping[str, str],
    max_answer_candidates: int,
    probe_question_ids: Sequence[str],
) -> DocumentAwareGuardEvaluation:
    if spec.label == STAGE39_MAIN_POLICY_LABEL:
        decisions = list(main_decisions)
    else:
        decisions = [
            _guarded_decision(
                spec=spec,
                decision=decision,
                main_case=main_cases_by_key[_question_key(decision.split, decision.question_id)],
                row_index=row_index,
            )
            for decision in main_decisions
        ]
    cases = build_topk_leading_candidate_answer_cases_from_decisions(
        decisions=decisions,
        rows=rows,
        gold_answers_by_question_key=gold_answers_by_question_key,
        max_answer_candidates=max_answer_candidates,
    )
    metrics = summarize_guarded_candidate_answer_cases(cases)
    cases_by_key = _cases_by_key(cases)
    main_regression_keys = {
        _question_key(case.split, case.question_id)
        for case in main_cases
        if case.f1_delta_vs_baseline < 0
    }
    guard_regression_keys = {
        _question_key(case.split, case.question_id)
        for case in cases
        if case.f1_delta_vs_baseline < 0
    }
    introduced_regression_keys = guard_regression_keys - main_regression_keys
    fixed_regression_keys = main_regression_keys - guard_regression_keys
    return DocumentAwareGuardEvaluation(
        label=spec.label,
        description=spec.description,
        feature_scope=spec.feature_scope,
        metrics=metrics,
        changed_count_vs_main=_changed_count(cases_by_key, main_cases_by_key),
        changed_count_vs_score60=_changed_count(cases_by_key, score60_cases_by_key),
        fixed_main_regression_count=len(fixed_regression_keys),
        introduced_regression_vs_main_count=len(introduced_regression_keys),
        probe_cases=_probe_cases(
            spec=spec,
            cases_by_key=cases_by_key,
            decisions_by_key=_decisions_by_key(decisions),
            row_index=row_index,
            probe_question_ids=probe_question_ids,
        ),
    )


def _guarded_decision(
    spec: DocumentAwareGuardSpec,
    decision: CandidateRerankerPolicyDecision,
    main_case: GuardedCandidateAnswerCase,
    row_index: Mapping[str, list[Mapping[str, Any]]],
) -> CandidateRerankerPolicyDecision:
    if decision.action != "replace_with_model_candidate":
        return decision
    context = _guard_context(decision=decision, main_case=main_case, row_index=row_index)
    if spec.accept_replacement(context):
        return replace(
            decision,
            decision_reasons=[*decision.decision_reasons, f"{spec.label}_accepted"],
        )
    question_rows = row_index[_question_key(decision.split, decision.question_id)]
    baseline_row = _row_by_candidate_id(question_rows, decision.baseline_candidate_id)
    baseline_is_gold_document = _is_gold_document(baseline_row)
    baseline_is_oracle_best = (
        decision.baseline_candidate_id == decision.oracle_candidate_id
        or decision.baseline_candidate_token_f1 == decision.oracle_candidate_token_f1
    )
    return replace(
        decision,
        action="keep_top_candidate",
        decision_reasons=[f"{spec.label}_blocked"],
        final_candidate_id=decision.baseline_candidate_id,
        final_candidate_rank=decision.baseline_candidate_rank,
        final_candidate_token_f1=decision.baseline_candidate_token_f1,
        f1_delta_vs_top_candidate=0.0,
        final_is_gold_document=baseline_is_gold_document,
        final_is_oracle_best_f1=baseline_is_oracle_best,
        final_missed_gold_document=(
            decision.gold_document_candidate_count > 0
            and not baseline_is_gold_document
        ),
        final_deep_rank=False,
    )


def _guard_context(
    decision: CandidateRerankerPolicyDecision,
    main_case: GuardedCandidateAnswerCase,
    row_index: Mapping[str, list[Mapping[str, Any]]],
) -> DocumentAwareGuardContext:
    question_rows = row_index[_question_key(decision.split, decision.question_id)]
    baseline_row = _row_by_candidate_id(question_rows, decision.baseline_candidate_id)
    replacement_row = _row_by_candidate_id(question_rows, decision.final_candidate_id)
    replacement_score = _runtime_feature_float(replacement_row, "candidate_score")
    return DocumentAwareGuardContext(
        split=decision.split,
        question_id=decision.question_id,
        question_route=decision.question_route,
        replacement_candidate_id=decision.final_candidate_id,
        replacement_candidate_rank=decision.final_candidate_rank,
        replacement_candidate_score=replacement_score,
        replacement_candidate_score_bucket=_candidate_score_bucket(replacement_score),
        model_score_margin_vs_top_candidate=decision.model_score_margin_vs_top_candidate,
        baseline_to_replacement_document_transition=_document_transition(
            from_row=baseline_row,
            to_row=replacement_row,
        ),
        replacement_citation_delta_vs_baseline=main_case.citation_delta,
    )


def _default_guard_specs() -> tuple[DocumentAwareGuardSpec, ...]:
    return (
        DocumentAwareGuardSpec(
            label=STAGE39_MAIN_POLICY_LABEL,
            description="Stage 36 main policy from the holdout evaluation.",
            feature_scope="runtime_policy_baseline",
            accept_replacement=lambda context: True,
        ),
        DocumentAwareGuardSpec(
            label=CANDIDATE_SCORE_GTE_60_LABEL,
            description="Reject replacements whose selected candidate score is below 60.",
            feature_scope="runtime_available_candidate_score",
            accept_replacement=lambda context: context.replacement_candidate_score >= 60,
        ),
        DocumentAwareGuardSpec(
            label="score60_or_new_gold",
            description=(
                "Reject low-score replacements unless the replacement moves the "
                "leading document from non-gold to gold."
            ),
            feature_scope="diagnostic_offline_only_gold_document_transition",
            accept_replacement=lambda context: (
                context.replacement_candidate_score >= 60
                or context.baseline_to_replacement_document_transition
                == "new_gold_leading_document"
            ),
        ),
        DocumentAwareGuardSpec(
            label="citation_preserving_score60",
            description=(
                "Reject citation-losing replacements, keep citation-gaining "
                "replacements, and otherwise require candidate score >= 60."
            ),
            feature_scope="diagnostic_offline_only_gold_citation",
            accept_replacement=lambda context: (
                context.replacement_citation_delta_vs_baseline > 0
                or (
                    context.replacement_citation_delta_vs_baseline == 0
                    and context.replacement_candidate_score >= 60
                )
            ),
        ),
        DocumentAwareGuardSpec(
            label="document_risk_v1",
            description=(
                "Keep new-gold replacements, block gold-to-non-gold replacements, "
                "and require score >= 90 for rank>=4 new-non-gold replacements."
            ),
            feature_scope="diagnostic_offline_only_gold_document_transition",
            accept_replacement=_document_risk_accepts,
        ),
        DocumentAwareGuardSpec(
            label=DOCUMENT_RISK_GUARD_LABEL,
            description=(
                "Keep citation gains, block citation losses, and block rank>=4 "
                "new-non-gold replacements below score 90."
            ),
            feature_scope="diagnostic_offline_only_gold_citation_and_document_transition",
            accept_replacement=_citation_doc_risk_accepts,
        ),
    )


def _document_risk_accepts(context: DocumentAwareGuardContext) -> bool:
    transition = context.baseline_to_replacement_document_transition
    if transition == "new_gold_leading_document":
        return True
    if transition == "gold_to_non_gold_leading_document":
        return False
    if (
        transition == "new_non_gold_leading_document"
        and context.replacement_candidate_rank >= 4
        and context.replacement_candidate_score < 90
    ):
        return False
    return context.replacement_candidate_score >= 60


def _citation_doc_risk_accepts(context: DocumentAwareGuardContext) -> bool:
    if context.replacement_citation_delta_vs_baseline > 0:
        return True
    if context.replacement_citation_delta_vs_baseline < 0:
        return False
    if (
        context.baseline_to_replacement_document_transition
        == "new_non_gold_leading_document"
        and context.replacement_candidate_rank >= 4
        and context.replacement_candidate_score < 90
    ):
        return False
    return context.replacement_candidate_score >= 60


def _validate_stage41_report(
    report: Mapping[str, Any],
    model_name: str,
    train_split: str,
    evaluation_split: str,
    mode_name: str,
    main_metrics: GuardedCandidateAnswerMetrics,
    score60_metrics: GuardedCandidateAnswerMetrics,
) -> None:
    if report.get("model_name") != model_name:
        raise ValueError("Stage 41 report model_name does not match recomputed design")
    if report.get("train_split") != train_split:
        raise ValueError("Stage 41 report train_split does not match recomputed design")
    if report.get("evaluation_split") != evaluation_split:
        raise ValueError(
            "Stage 41 report evaluation_split does not match recomputed design"
        )
    if report.get("mode_name") != mode_name:
        raise ValueError("Stage 41 report mode_name does not match recomputed design")
    _validate_report_metrics(
        report_metrics=report.get("main_policy_metrics"),
        expected_metrics=main_metrics,
        label="main_policy_metrics",
    )
    _validate_report_metrics(
        report_metrics=report.get("candidate_policy_metrics"),
        expected_metrics=score60_metrics,
        label="candidate_policy_metrics",
    )


def _validate_report_metrics(
    report_metrics: Any,
    expected_metrics: GuardedCandidateAnswerMetrics,
    label: str,
) -> None:
    if not isinstance(report_metrics, Mapping):
        raise ValueError(f"Stage 41 report must contain {label}")
    expected = {
        "question_count": expected_metrics.question_count,
        "policy_average_answer_token_f1": (
            expected_metrics.policy_average_answer_token_f1
        ),
        "average_delta_vs_baseline": expected_metrics.average_delta_vs_baseline,
        "replacement_count": expected_metrics.replacement_count,
        "regressed_count": expected_metrics.regressed_count,
        "citation_lost_count": expected_metrics.citation_lost_count,
        "citation_gained_count": expected_metrics.citation_gained_count,
        "gold_citation_delta": expected_metrics.gold_citation_delta,
    }
    for key, expected_value in expected.items():
        if report_metrics.get(key) != expected_value:
            raise ValueError(f"Stage 41 report {label}.{key} does not match")


def _findings(
    guard_evaluations: Sequence[DocumentAwareGuardEvaluation],
    score60_label: str,
) -> list[str]:
    best_delta = max(
        guard_evaluations,
        key=lambda evaluation: (
            evaluation.metrics.average_delta_vs_baseline,
            -evaluation.metrics.regressed_count,
            -evaluation.metrics.citation_lost_count,
        ),
    )
    best_regression = min(
        guard_evaluations,
        key=lambda evaluation: (
            evaluation.metrics.regressed_count,
            -evaluation.metrics.average_delta_vs_baseline,
        ),
    )
    score60 = _evaluation_by_label(guard_evaluations, score60_label)
    findings = [
        (
            f"Best average-delta guard is {best_delta.label}: "
            f"{best_delta.metrics.average_delta_vs_baseline:+.4f}, "
            f"regressions {best_delta.metrics.regressed_count}."
        ),
        (
            f"Lowest-regression guard is {best_regression.label}: "
            f"{best_regression.metrics.regressed_count}, delta "
            f"{best_regression.metrics.average_delta_vs_baseline:+.4f}."
        ),
        (
            f"{score60_label} remains the runtime-feature-only reference: "
            f"delta {score60.metrics.average_delta_vs_baseline:+.4f}, "
            f"regressions {score60.metrics.regressed_count}."
        ),
    ]
    diagnostic_candidates = [
        evaluation
        for evaluation in guard_evaluations
        if evaluation.feature_scope.startswith("diagnostic_offline_only")
    ]
    if diagnostic_candidates:
        best_diagnostic = max(
            diagnostic_candidates,
            key=lambda evaluation: (
                evaluation.metrics.average_delta_vs_baseline,
                -evaluation.metrics.regressed_count,
            ),
        )
        findings.append(
            f"Best diagnostic-only guard is {best_diagnostic.label}; it is not a "
            "runtime policy because it uses gold/citation-derived signals."
        )
    return findings


def _delta(
    evaluation: DocumentAwareGuardEvaluation,
    reference: DocumentAwareGuardEvaluation,
) -> DocumentAwareGuardDelta:
    return DocumentAwareGuardDelta(
        guard_label=evaluation.label,
        reference_label=reference.label,
        average_delta_difference=round(
            evaluation.metrics.average_delta_vs_baseline
            - reference.metrics.average_delta_vs_baseline,
            4,
        ),
        policy_average_f1_difference=round(
            evaluation.metrics.policy_average_answer_token_f1
            - reference.metrics.policy_average_answer_token_f1,
            4,
        ),
        regressed_count_difference=(
            evaluation.metrics.regressed_count - reference.metrics.regressed_count
        ),
        citation_lost_count_difference=(
            evaluation.metrics.citation_lost_count
            - reference.metrics.citation_lost_count
        ),
        gold_citation_delta_difference=(
            evaluation.metrics.gold_citation_delta
            - reference.metrics.gold_citation_delta
        ),
        replacement_count_difference=(
            evaluation.metrics.replacement_count - reference.metrics.replacement_count
        ),
    )


def _probe_cases(
    spec: DocumentAwareGuardSpec,
    cases_by_key: Mapping[str, GuardedCandidateAnswerCase],
    decisions_by_key: Mapping[str, CandidateRerankerPolicyDecision],
    row_index: Mapping[str, list[Mapping[str, Any]]],
    probe_question_ids: Sequence[str],
) -> list[DocumentAwareGuardProbe]:
    probes = []
    for question_id in probe_question_ids:
        question_key = _question_key_from_id(question_id)
        case = cases_by_key.get(question_key)
        decision = decisions_by_key.get(question_key)
        if case is None or decision is None:
            continue
        question_rows = row_index[question_key]
        final_row = _row_by_candidate_id(question_rows, decision.final_candidate_id)
        baseline_row = _row_by_candidate_id(question_rows, decision.baseline_candidate_id)
        probes.append(
            DocumentAwareGuardProbe(
                question_key=question_key,
                guard_label=spec.label,
                action=decision.action,
                decision_reasons=list(decision.decision_reasons),
                final_candidate_id=decision.final_candidate_id,
                final_candidate_rank=decision.final_candidate_rank,
                final_candidate_score=_runtime_feature_float(final_row, "candidate_score"),
                final_document_id=_document_id(final_row),
                baseline_to_final_document_transition=_document_transition(
                    from_row=baseline_row,
                    to_row=final_row,
                ),
                answer_token_f1=case.policy_answer_token_f1,
                delta_vs_baseline=case.f1_delta_vs_baseline,
                gold_cited=case.policy_gold_cited,
                citation_delta=case.citation_delta,
            )
        )
    return probes


def _changed_count(
    cases_by_key: Mapping[str, GuardedCandidateAnswerCase],
    reference_cases_by_key: Mapping[str, GuardedCandidateAnswerCase],
) -> int:
    return sum(
        case.policy_candidate_ids != reference_cases_by_key[question_key].policy_candidate_ids
        for question_key, case in cases_by_key.items()
    )


def _policy_configs_by_label() -> dict[str, Any]:
    return {label: config for label, config in default_stage39_policy_specs()}


def _evaluation_by_label(
    evaluations: Sequence[DocumentAwareGuardEvaluation],
    label: str,
) -> DocumentAwareGuardEvaluation:
    for evaluation in evaluations:
        if evaluation.label == label:
            return evaluation
    raise ValueError(f"Missing guard evaluation: {label}")


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
    row_index: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        row_index.setdefault(_question_key(row["split"], row["question_id"]), []).append(row)
    return row_index


def _cases_by_key(
    cases: Sequence[GuardedCandidateAnswerCase],
) -> dict[str, GuardedCandidateAnswerCase]:
    return {_question_key(case.split, case.question_id): case for case in cases}


def _decisions_by_key(
    decisions: Sequence[CandidateRerankerPolicyDecision],
) -> dict[str, CandidateRerankerPolicyDecision]:
    return {
        _question_key(decision.split, decision.question_id): decision
        for decision in decisions
    }


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


def _candidate_score_bucket(score: float) -> str:
    if score < 60:
        return "score_lt_60"
    if score < 90:
        return "score_60_90"
    if score < 120:
        return "score_90_120"
    return "score_120_plus"


def _question_key(split: Any, question_id: Any) -> str:
    return f"{split}::{question_id}"


def _question_key_from_id(question_id: str) -> str:
    return question_id if "::" in question_id else f"dev::{question_id}"


def _normalize_split_name(split_name: str) -> str:
    normalized = split_name.strip().lower()
    if not normalized:
        raise ValueError("split name must not be empty")
    return normalized


def _rounded_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(mean(values), 4)


def _validate_options(
    rows: Sequence[Mapping[str, Any]],
    max_answer_candidates: int,
) -> None:
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
    width = 1180
    margin_left = 420
    margin_right = 220
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
