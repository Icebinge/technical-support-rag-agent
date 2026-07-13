from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any

from ts_rag_agent.application.candidate_score_guarded_policy_split_validation import (
    CANDIDATE_SCORE_GTE_60_LABEL,
)
from ts_rag_agent.application.guarded_candidate_reranker_answer_experiment import (
    GuardedCandidateAnswerMetrics,
)
from ts_rag_agent.application.runtime_document_risk_proxy_cv import (
    RuntimeDocumentRiskProxyCVResult,
    select_runtime_document_risk_proxy_with_train_cv,
)
from ts_rag_agent.application.runtime_document_risk_proxy_search import (
    RuntimeDocumentRiskGuardEvaluation,
    VisualizationArtifact,
)


@dataclass(frozen=True)
class RiskAwareObjectiveSpec:
    """One train-CV risk constraint set for runtime proxy selection."""

    label: str
    description: str
    max_train_regressed_count: int | None
    max_train_citation_lost_count: int | None
    min_train_gold_citation_delta: int | None


@dataclass(frozen=True)
class RiskAwareObjectiveEvaluation:
    """Selection result for one risk-aware train-CV objective."""

    label: str
    description: str
    constraint_summary: str
    feasible_guard_labels: list[str]
    selected_guard_label: str
    selected_guard_description: str
    selected_train_cv_metrics: GuardedCandidateAnswerMetrics
    selected_holdout_metrics: GuardedCandidateAnswerMetrics


@dataclass(frozen=True)
class RuntimeDocumentRiskRiskAwareSelectionResult:
    """Stage 45 risk-aware train-CV objective result."""

    model_name: str
    train_split: str
    evaluation_split: str
    train_fold_count: int
    max_answer_candidates: int
    train_question_count: int
    evaluation_question_count: int
    primary_objective_label: str
    primary_selected_guard_label: str
    primary_selected_guard_description: str
    primary_train_cv_metrics: GuardedCandidateAnswerMetrics
    primary_holdout_metrics: GuardedCandidateAnswerMetrics
    objective_evaluations: list[RiskAwareObjectiveEvaluation]
    train_cv_guard_evaluations: list[RuntimeDocumentRiskGuardEvaluation]
    holdout_guard_evaluations: list[RuntimeDocumentRiskGuardEvaluation]
    findings: list[str]
    analysis_scope: str


@dataclass(frozen=True)
class RuntimeDocumentRiskRiskAwareVisualizations:
    """SVG artifacts generated for Stage 45."""

    objectives: list[VisualizationArtifact]
    guards: list[VisualizationArtifact]


PRIMARY_OBJECTIVE_LABEL = "score60_risk_parity"


def select_risk_aware_runtime_document_risk_proxy(
    stage43_report: Mapping[str, Any],
    stage44_report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    gold_answers_by_question_key: Mapping[str, str],
    model_name: str = "logistic_best_candidate",
    train_split: str = "train",
    evaluation_split: str = "dev",
    train_fold_count: int = 5,
    max_answer_candidates: int = 3,
) -> RuntimeDocumentRiskRiskAwareSelectionResult:
    """Apply risk-aware train-CV objectives to the Stage 44 guard family."""

    stage44_result = select_runtime_document_risk_proxy_with_train_cv(
        stage43_report=stage43_report,
        rows=rows,
        gold_answers_by_question_key=gold_answers_by_question_key,
        model_name=model_name,
        train_split=train_split,
        evaluation_split=evaluation_split,
        train_fold_count=train_fold_count,
        max_answer_candidates=max_answer_candidates,
    )
    _validate_stage44_report(
        report=stage44_report,
        expected=stage44_result,
    )

    objective_specs = _default_objective_specs(stage44_result.train_cv_guard_evaluations)
    objective_evaluations = [
        _evaluate_objective(
            spec=spec,
            train_cv_guard_evaluations=stage44_result.train_cv_guard_evaluations,
            holdout_guard_evaluations=stage44_result.holdout_guard_evaluations,
        )
        for spec in objective_specs
    ]
    primary_objective = _objective_by_label(
        objective_evaluations,
        PRIMARY_OBJECTIVE_LABEL,
    )

    return RuntimeDocumentRiskRiskAwareSelectionResult(
        model_name=stage44_result.model_name,
        train_split=stage44_result.train_split,
        evaluation_split=stage44_result.evaluation_split,
        train_fold_count=stage44_result.train_fold_count,
        max_answer_candidates=stage44_result.max_answer_candidates,
        train_question_count=stage44_result.train_question_count,
        evaluation_question_count=stage44_result.evaluation_question_count,
        primary_objective_label=primary_objective.label,
        primary_selected_guard_label=primary_objective.selected_guard_label,
        primary_selected_guard_description=(
            primary_objective.selected_guard_description
        ),
        primary_train_cv_metrics=primary_objective.selected_train_cv_metrics,
        primary_holdout_metrics=primary_objective.selected_holdout_metrics,
        objective_evaluations=objective_evaluations,
        train_cv_guard_evaluations=stage44_result.train_cv_guard_evaluations,
        holdout_guard_evaluations=stage44_result.holdout_guard_evaluations,
        findings=_findings(
            primary_objective=primary_objective,
            objective_evaluations=objective_evaluations,
            train_cv_guard_evaluations=stage44_result.train_cv_guard_evaluations,
            holdout_guard_evaluations=stage44_result.holdout_guard_evaluations,
        ),
        analysis_scope=(
            "Stage 45 applies risk-aware objectives to the Stage 44 runtime-only "
            "guard family. Guard selection uses train-CV constraints first, then "
            "maximizes train-CV delta among feasible guards. Dev holdout is only "
            "used for confirmation, gold/citation labels are only used for offline "
            "evaluation, held-out test data is not used, and runtime behavior is "
            "not changed."
        ),
    )


def runtime_document_risk_risk_aware_selection_to_dict(
    result: RuntimeDocumentRiskRiskAwareSelectionResult,
) -> dict[str, Any]:
    """Convert a Stage 45 risk-aware selection result to a JSON-safe dictionary."""

    return asdict(result)


def write_runtime_document_risk_risk_aware_visualizations(
    result: RuntimeDocumentRiskRiskAwareSelectionResult,
    output_dir: Path,
) -> RuntimeDocumentRiskRiskAwareVisualizations:
    """Write SVG charts for Stage 45 risk-aware runtime proxy selection."""

    output_dir.mkdir(parents=True, exist_ok=True)
    objectives = [
        VisualizationArtifact(
            name="stage45_objective_train_delta.svg",
            path=str(
                _write_svg(
                    output_dir,
                    "stage45_objective_train_delta.svg",
                    _render_objective_metric_chart(
                        title="Stage 45 selected guard train-CV delta by objective",
                        objectives=result.objective_evaluations,
                        metric_name="train-CV average F1 delta",
                        metric_value=lambda objective: (
                            objective.selected_train_cv_metrics.average_delta_vs_baseline
                        ),
                        value_label=lambda value: f"{value:+.4f}",
                    ),
                )
            ),
        ),
        VisualizationArtifact(
            name="stage45_objective_holdout_delta.svg",
            path=str(
                _write_svg(
                    output_dir,
                    "stage45_objective_holdout_delta.svg",
                    _render_objective_metric_chart(
                        title="Stage 45 selected guard dev-holdout delta by objective",
                        objectives=result.objective_evaluations,
                        metric_name="dev-holdout average F1 delta",
                        metric_value=lambda objective: (
                            objective.selected_holdout_metrics.average_delta_vs_baseline
                        ),
                        value_label=lambda value: f"{value:+.4f}",
                    ),
                )
            ),
        ),
        VisualizationArtifact(
            name="stage45_objective_holdout_regressions.svg",
            path=str(
                _write_svg(
                    output_dir,
                    "stage45_objective_holdout_regressions.svg",
                    _render_objective_metric_chart(
                        title="Stage 45 selected guard dev-holdout regressions",
                        objectives=result.objective_evaluations,
                        metric_name="dev-holdout regression cases",
                        metric_value=lambda objective: float(
                            objective.selected_holdout_metrics.regressed_count
                        ),
                        value_label=lambda value: str(int(value)),
                    ),
                )
            ),
        ),
        VisualizationArtifact(
            name="stage45_objective_feasible_guard_count.svg",
            path=str(
                _write_svg(
                    output_dir,
                    "stage45_objective_feasible_guard_count.svg",
                    _render_objective_metric_chart(
                        title="Stage 45 feasible guard count by objective",
                        objectives=result.objective_evaluations,
                        metric_name="feasible train-CV guards",
                        metric_value=lambda objective: float(
                            len(objective.feasible_guard_labels)
                        ),
                        value_label=lambda value: str(int(value)),
                    ),
                )
            ),
        ),
    ]
    guards = [
        VisualizationArtifact(
            name="stage45_train_guard_delta_under_risk_view.svg",
            path=str(
                _write_svg(
                    output_dir,
                    "stage45_train_guard_delta_under_risk_view.svg",
                    _render_guard_metric_chart(
                        title="Stage 45 train-CV guard deltas under risk view",
                        evaluations=result.train_cv_guard_evaluations,
                        metric_name="train-CV average F1 delta",
                        metric_value=lambda evaluation: (
                            evaluation.metrics.average_delta_vs_baseline
                        ),
                        value_label=lambda value: f"{value:+.4f}",
                    ),
                )
            ),
        )
    ]
    return RuntimeDocumentRiskRiskAwareVisualizations(
        objectives=objectives,
        guards=guards,
    )


def _default_objective_specs(
    train_cv_guard_evaluations: Sequence[RuntimeDocumentRiskGuardEvaluation],
) -> tuple[RiskAwareObjectiveSpec, ...]:
    score60 = _guard_evaluation_by_label(
        train_cv_guard_evaluations,
        CANDIDATE_SCORE_GTE_60_LABEL,
    )
    score60_metrics = score60.metrics
    return (
        RiskAwareObjectiveSpec(
            label=PRIMARY_OBJECTIVE_LABEL,
            description=(
                "Require train-CV risk no worse than candidate_score_gte_60 on "
                "regressions, citation loss, and gold citation delta; then maximize "
                "train-CV delta."
            ),
            max_train_regressed_count=score60_metrics.regressed_count,
            max_train_citation_lost_count=score60_metrics.citation_lost_count,
            min_train_gold_citation_delta=score60_metrics.gold_citation_delta,
        ),
        RiskAwareObjectiveSpec(
            label="no_citation_loss",
            description=(
                "Require zero train-CV citation loss and non-negative gold citation "
                "delta; then maximize train-CV delta."
            ),
            max_train_regressed_count=None,
            max_train_citation_lost_count=0,
            min_train_gold_citation_delta=0,
        ),
        RiskAwareObjectiveSpec(
            label="low_regression_no_citation_loss",
            description=(
                "Require at most two train-CV regressions, zero citation loss, and "
                "non-negative gold citation delta; then maximize train-CV delta."
            ),
            max_train_regressed_count=2,
            max_train_citation_lost_count=0,
            min_train_gold_citation_delta=0,
        ),
        RiskAwareObjectiveSpec(
            label="zero_regression_positive_gold",
            description=(
                "Require zero train-CV regressions, zero citation loss, and positive "
                "gold citation delta; then maximize train-CV delta."
            ),
            max_train_regressed_count=0,
            max_train_citation_lost_count=0,
            min_train_gold_citation_delta=1,
        ),
    )


def _evaluate_objective(
    spec: RiskAwareObjectiveSpec,
    train_cv_guard_evaluations: Sequence[RuntimeDocumentRiskGuardEvaluation],
    holdout_guard_evaluations: Sequence[RuntimeDocumentRiskGuardEvaluation],
) -> RiskAwareObjectiveEvaluation:
    feasible_evaluations = [
        evaluation
        for evaluation in train_cv_guard_evaluations
        if _satisfies_objective(evaluation, spec)
    ]
    if not feasible_evaluations:
        raise ValueError(f"No feasible guard for risk objective: {spec.label}")
    selected_train_evaluation = max(feasible_evaluations, key=_selection_key)
    selected_holdout_evaluation = _guard_evaluation_by_label(
        holdout_guard_evaluations,
        selected_train_evaluation.label,
    )
    return RiskAwareObjectiveEvaluation(
        label=spec.label,
        description=spec.description,
        constraint_summary=_constraint_summary(spec),
        feasible_guard_labels=[
            evaluation.label
            for evaluation in sorted(feasible_evaluations, key=_selection_key, reverse=True)
        ],
        selected_guard_label=selected_train_evaluation.label,
        selected_guard_description=selected_train_evaluation.description,
        selected_train_cv_metrics=selected_train_evaluation.metrics,
        selected_holdout_metrics=selected_holdout_evaluation.metrics,
    )


def _satisfies_objective(
    evaluation: RuntimeDocumentRiskGuardEvaluation,
    spec: RiskAwareObjectiveSpec,
) -> bool:
    metrics = evaluation.metrics
    if (
        spec.max_train_regressed_count is not None
        and metrics.regressed_count > spec.max_train_regressed_count
    ):
        return False
    if (
        spec.max_train_citation_lost_count is not None
        and metrics.citation_lost_count > spec.max_train_citation_lost_count
    ):
        return False
    if (
        spec.min_train_gold_citation_delta is not None
        and metrics.gold_citation_delta < spec.min_train_gold_citation_delta
    ):
        return False
    return True


def _constraint_summary(spec: RiskAwareObjectiveSpec) -> str:
    parts = []
    if spec.max_train_regressed_count is not None:
        parts.append(f"regressions <= {spec.max_train_regressed_count}")
    if spec.max_train_citation_lost_count is not None:
        parts.append(f"citation_lost <= {spec.max_train_citation_lost_count}")
    if spec.min_train_gold_citation_delta is not None:
        parts.append(f"gold_citation_delta >= {spec.min_train_gold_citation_delta}")
    return "; ".join(parts) if parts else "no explicit risk constraint"


def _selection_key(
    evaluation: RuntimeDocumentRiskGuardEvaluation,
) -> tuple[float, int, int, int, float, str]:
    metrics = evaluation.metrics
    return (
        metrics.average_delta_vs_baseline,
        -metrics.regressed_count,
        -metrics.citation_lost_count,
        metrics.gold_citation_delta,
        metrics.policy_average_answer_token_f1,
        evaluation.label,
    )


def _validate_stage44_report(
    report: Mapping[str, Any],
    expected: RuntimeDocumentRiskProxyCVResult,
) -> None:
    expected_fields = {
        "model_name": expected.model_name,
        "train_split": expected.train_split,
        "evaluation_split": expected.evaluation_split,
        "train_fold_count": expected.train_fold_count,
        "max_answer_candidates": expected.max_answer_candidates,
        "selected_guard_label": expected.selected_guard_label,
    }
    for key, expected_value in expected_fields.items():
        if report.get(key) != expected_value:
            raise ValueError(f"Stage 44 report {key} does not match Stage 45")
    _validate_report_evaluations(
        report_evaluations=report.get("train_cv_guard_evaluations"),
        expected_evaluations=expected.train_cv_guard_evaluations,
        scope="train_cv_guard_evaluations",
    )
    _validate_report_evaluations(
        report_evaluations=report.get("holdout_guard_evaluations"),
        expected_evaluations=expected.holdout_guard_evaluations,
        scope="holdout_guard_evaluations",
    )


def _validate_report_evaluations(
    report_evaluations: Any,
    expected_evaluations: Sequence[RuntimeDocumentRiskGuardEvaluation],
    scope: str,
) -> None:
    if not isinstance(report_evaluations, list):
        raise ValueError(f"Stage 44 report must contain {scope}")
    report_metrics_by_label = {
        str(evaluation["label"]): evaluation["metrics"]
        for evaluation in report_evaluations
        if isinstance(evaluation, Mapping)
        and isinstance(evaluation.get("metrics"), Mapping)
    }
    for evaluation in expected_evaluations:
        report_metrics = report_metrics_by_label.get(evaluation.label)
        if report_metrics is None:
            raise ValueError(f"Stage 44 report missing {scope}: {evaluation.label}")
        _validate_metrics_match(
            report_metrics=report_metrics,
            expected_metrics=evaluation.metrics,
            label=evaluation.label,
            scope=scope,
        )


def _validate_metrics_match(
    report_metrics: Mapping[str, Any],
    expected_metrics: GuardedCandidateAnswerMetrics,
    label: str,
    scope: str,
) -> None:
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
            raise ValueError(
                f"Stage 44 report {scope}.{label}.{key} does not match"
            )


def _findings(
    primary_objective: RiskAwareObjectiveEvaluation,
    objective_evaluations: Sequence[RiskAwareObjectiveEvaluation],
    train_cv_guard_evaluations: Sequence[RuntimeDocumentRiskGuardEvaluation],
    holdout_guard_evaluations: Sequence[RuntimeDocumentRiskGuardEvaluation],
) -> list[str]:
    train_best = max(train_cv_guard_evaluations, key=_selection_key)
    holdout_best = max(holdout_guard_evaluations, key=_selection_key)
    zero_regression = _objective_by_label(
        objective_evaluations,
        "zero_regression_positive_gold",
    )
    return [
        (
            f"Primary risk-aware objective {primary_objective.label} selects "
            f"{primary_objective.selected_guard_label}: train-CV delta "
            f"{primary_objective.selected_train_cv_metrics.average_delta_vs_baseline:+.4f}, "
            f"regressions {primary_objective.selected_train_cv_metrics.regressed_count}."
        ),
        (
            f"Primary selected guard holdout result is "
            f"{primary_objective.selected_holdout_metrics.average_delta_vs_baseline:+.4f}, "
            f"regressions {primary_objective.selected_holdout_metrics.regressed_count}, "
            f"citation loss {primary_objective.selected_holdout_metrics.citation_lost_count}."
        ),
        (
            f"Pure train-CV delta best remains {train_best.label}; risk-aware "
            f"constraints prevent choosing it when its risk exceeds the objective."
        ),
        (
            f"Dev-holdout best by the same score key is {holdout_best.label}; this is "
            "reported only as confirmation, not used for objective selection."
        ),
        (
            f"Strict zero-regression objective selects "
            f"{zero_regression.selected_guard_label}: train-CV delta "
            f"{zero_regression.selected_train_cv_metrics.average_delta_vs_baseline:+.4f}, "
            f"holdout delta "
            f"{zero_regression.selected_holdout_metrics.average_delta_vs_baseline:+.4f}."
        ),
    ]


def _objective_by_label(
    evaluations: Sequence[RiskAwareObjectiveEvaluation],
    label: str,
) -> RiskAwareObjectiveEvaluation:
    for evaluation in evaluations:
        if evaluation.label == label:
            return evaluation
    raise ValueError(f"Missing risk-aware objective: {label}")


def _guard_evaluation_by_label(
    evaluations: Sequence[RuntimeDocumentRiskGuardEvaluation],
    label: str,
) -> RuntimeDocumentRiskGuardEvaluation:
    for evaluation in evaluations:
        if evaluation.label == label:
            return evaluation
    raise ValueError(f"Missing guard evaluation: {label}")


def _write_svg(output_dir: Path, filename: str, svg: str) -> Path:
    path = output_dir / filename
    path.write_text(svg, encoding="utf-8")
    return path


def _render_objective_metric_chart(
    title: str,
    objectives: Sequence[RiskAwareObjectiveEvaluation],
    metric_name: str,
    metric_value,
    value_label,
) -> str:
    return _render_bar_chart_svg(
        title=title,
        bars=[
            _Bar(
                f"{objective.label} -> {objective.selected_guard_label}",
                float(metric_value(objective)),
                value_label(float(metric_value(objective))),
            )
            for objective in objectives
        ],
        x_label=metric_name,
    )


def _render_guard_metric_chart(
    title: str,
    evaluations: Sequence[RuntimeDocumentRiskGuardEvaluation],
    metric_name: str,
    metric_value,
    value_label,
) -> str:
    return _render_bar_chart_svg(
        title=title,
        bars=[
            _Bar(
                evaluation.label,
                float(metric_value(evaluation)),
                value_label(float(metric_value(evaluation))),
            )
            for evaluation in evaluations
        ],
        x_label=metric_name,
    )


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
    width = 1240
    margin_left = 500
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
