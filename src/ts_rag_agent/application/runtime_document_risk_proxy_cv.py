from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any

from ts_rag_agent.application.candidate_reranker_cv import (
    CandidateRerankerSelection,
    cross_validated_candidate_reranker_selections,
    split_validated_candidate_reranker_selections,
)
from ts_rag_agent.application.candidate_reranker_policy_search import (
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
    GuardedCandidateAnswerMetrics,
)
from ts_rag_agent.application.runtime_document_risk_proxy_search import (
    RuntimeDocumentRiskGuardEvaluation,
    RuntimeDocumentRiskGuardSpec,
    VisualizationArtifact,
    default_runtime_document_risk_guard_specs,
    evaluate_runtime_document_risk_guards_from_main_decisions,
)


@dataclass(frozen=True)
class RuntimeDocumentRiskProxyCVResult:
    """Stage 44 train-only CV selection for runtime document-risk proxy guards."""

    model_name: str
    train_split: str
    evaluation_split: str
    train_fold_count: int
    max_answer_candidates: int
    train_question_count: int
    evaluation_question_count: int
    train_cv_guard_evaluations: list[RuntimeDocumentRiskGuardEvaluation]
    holdout_guard_evaluations: list[RuntimeDocumentRiskGuardEvaluation]
    selected_guard_label: str
    selected_guard_description: str
    selected_train_cv_metrics: GuardedCandidateAnswerMetrics
    selected_holdout_metrics: GuardedCandidateAnswerMetrics
    train_best_guard_label: str
    holdout_best_guard_label: str
    selection_metric: str
    findings: list[str]
    analysis_scope: str


@dataclass(frozen=True)
class RuntimeDocumentRiskProxyCVVisualizations:
    """SVG artifacts generated for Stage 44."""

    train_cv: list[VisualizationArtifact]
    holdout: list[VisualizationArtifact]
    selected: list[VisualizationArtifact]


SELECTION_METRIC = (
    "max train-CV average_delta_vs_baseline, then fewer regressions, fewer "
    "citation losses, higher gold citation delta, higher policy F1, then label"
)


def select_runtime_document_risk_proxy_with_train_cv(
    stage43_report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    gold_answers_by_question_key: Mapping[str, str],
    model_name: str = "logistic_best_candidate",
    train_split: str = "train",
    evaluation_split: str = "dev",
    train_fold_count: int = 5,
    max_answer_candidates: int = 3,
    guard_specs: Sequence[RuntimeDocumentRiskGuardSpec] | None = None,
) -> RuntimeDocumentRiskProxyCVResult:
    """Select a runtime proxy guard using train-only grouped CV and confirm on dev."""

    _validate_options(
        rows=rows,
        train_fold_count=train_fold_count,
        max_answer_candidates=max_answer_candidates,
    )
    normalized_train_split = _normalize_split_name(train_split)
    normalized_evaluation_split = _normalize_split_name(evaluation_split)
    if normalized_train_split == normalized_evaluation_split:
        raise ValueError("train_split and evaluation_split must be different")

    specs = tuple(
        default_runtime_document_risk_guard_specs()
        if guard_specs is None
        else guard_specs
    )
    if not specs:
        raise ValueError("guard_specs must not be empty")

    train_rows = _rows_for_split(rows, normalized_train_split)
    evaluation_rows = _rows_for_split(rows, normalized_evaluation_split)
    train_question_count = _question_count(train_rows)
    evaluation_question_count = _question_count(evaluation_rows)

    train_cv_selections = cross_validated_candidate_reranker_selections(
        rows=train_rows,
        model_name=model_name,
        fold_count=train_fold_count,
    )
    train_cv_guard_evaluations = _guard_evaluations_for_selections(
        selections=train_cv_selections,
        rows=train_rows,
        gold_answers_by_question_key=gold_answers_by_question_key,
        max_answer_candidates=max_answer_candidates,
        guard_specs=specs,
    )

    holdout_selections = split_validated_candidate_reranker_selections(
        rows=rows,
        model_name=model_name,
        train_split=normalized_train_split,
        validation_split=normalized_evaluation_split,
    )
    holdout_guard_evaluations = _guard_evaluations_for_selections(
        selections=holdout_selections,
        rows=evaluation_rows,
        gold_answers_by_question_key=gold_answers_by_question_key,
        max_answer_candidates=max_answer_candidates,
        guard_specs=specs,
    )
    _validate_stage43_report(
        report=stage43_report,
        model_name=model_name,
        train_split=normalized_train_split,
        evaluation_split=normalized_evaluation_split,
        max_answer_candidates=max_answer_candidates,
        holdout_guard_evaluations=holdout_guard_evaluations,
    )

    selected_train_evaluation = max(
        train_cv_guard_evaluations,
        key=_selection_key,
    )
    selected_holdout_evaluation = _evaluation_by_label(
        holdout_guard_evaluations,
        selected_train_evaluation.label,
    )
    train_best_guard = max(train_cv_guard_evaluations, key=_selection_key)
    holdout_best_guard = max(holdout_guard_evaluations, key=_selection_key)

    return RuntimeDocumentRiskProxyCVResult(
        model_name=model_name,
        train_split=normalized_train_split,
        evaluation_split=normalized_evaluation_split,
        train_fold_count=train_fold_count,
        max_answer_candidates=max_answer_candidates,
        train_question_count=train_question_count,
        evaluation_question_count=evaluation_question_count,
        train_cv_guard_evaluations=train_cv_guard_evaluations,
        holdout_guard_evaluations=holdout_guard_evaluations,
        selected_guard_label=selected_train_evaluation.label,
        selected_guard_description=selected_train_evaluation.description,
        selected_train_cv_metrics=selected_train_evaluation.metrics,
        selected_holdout_metrics=selected_holdout_evaluation.metrics,
        train_best_guard_label=train_best_guard.label,
        holdout_best_guard_label=holdout_best_guard.label,
        selection_metric=SELECTION_METRIC,
        findings=_findings(
            selected_train_evaluation=selected_train_evaluation,
            selected_holdout_evaluation=selected_holdout_evaluation,
            train_cv_guard_evaluations=train_cv_guard_evaluations,
            holdout_guard_evaluations=holdout_guard_evaluations,
        ),
        analysis_scope=(
            "Stage 44 selects a runtime-only document-risk proxy guard using only "
            "train-split grouped CV. Dev holdout is used only for confirmation "
            "against the frozen Stage 43 report. Gold/citation labels are used only "
            "for offline evaluation, held-out test data is not used, and runtime "
            "behavior is not changed."
        ),
    )


def runtime_document_risk_proxy_cv_to_dict(
    result: RuntimeDocumentRiskProxyCVResult,
) -> dict[str, Any]:
    """Convert a Stage 44 CV-selection result to a JSON-safe dictionary."""

    return asdict(result)


def write_runtime_document_risk_proxy_cv_visualizations(
    result: RuntimeDocumentRiskProxyCVResult,
    output_dir: Path,
) -> RuntimeDocumentRiskProxyCVVisualizations:
    """Write SVG charts for Stage 44 runtime proxy CV selection."""

    output_dir.mkdir(parents=True, exist_ok=True)
    train_cv = [
        VisualizationArtifact(
            name="stage44_train_cv_runtime_proxy_delta.svg",
            path=str(
                _write_svg(
                    output_dir,
                    "stage44_train_cv_runtime_proxy_delta.svg",
                    _render_guard_metric_chart(
                        title="Stage 44 train-CV runtime proxy delta",
                        evaluations=result.train_cv_guard_evaluations,
                        metric_name="average answer token F1 delta vs baseline",
                        metric_value=lambda evaluation: (
                            evaluation.metrics.average_delta_vs_baseline
                        ),
                        value_label=lambda value: f"{value:+.4f}",
                    ),
                )
            ),
        ),
        VisualizationArtifact(
            name="stage44_train_cv_runtime_proxy_regressions.svg",
            path=str(
                _write_svg(
                    output_dir,
                    "stage44_train_cv_runtime_proxy_regressions.svg",
                    _render_guard_metric_chart(
                        title="Stage 44 train-CV runtime proxy regressions",
                        evaluations=result.train_cv_guard_evaluations,
                        metric_name="regression cases",
                        metric_value=lambda evaluation: float(
                            evaluation.metrics.regressed_count
                        ),
                        value_label=lambda value: str(int(value)),
                    ),
                )
            ),
        ),
    ]
    holdout = [
        VisualizationArtifact(
            name="stage44_holdout_runtime_proxy_delta.svg",
            path=str(
                _write_svg(
                    output_dir,
                    "stage44_holdout_runtime_proxy_delta.svg",
                    _render_guard_metric_chart(
                        title="Stage 44 dev-holdout runtime proxy delta",
                        evaluations=result.holdout_guard_evaluations,
                        metric_name="average answer token F1 delta vs baseline",
                        metric_value=lambda evaluation: (
                            evaluation.metrics.average_delta_vs_baseline
                        ),
                        value_label=lambda value: f"{value:+.4f}",
                    ),
                )
            ),
        ),
        VisualizationArtifact(
            name="stage44_holdout_runtime_proxy_regressions.svg",
            path=str(
                _write_svg(
                    output_dir,
                    "stage44_holdout_runtime_proxy_regressions.svg",
                    _render_guard_metric_chart(
                        title="Stage 44 dev-holdout runtime proxy regressions",
                        evaluations=result.holdout_guard_evaluations,
                        metric_name="regression cases",
                        metric_value=lambda evaluation: float(
                            evaluation.metrics.regressed_count
                        ),
                        value_label=lambda value: str(int(value)),
                    ),
                )
            ),
        ),
    ]
    selected = [
        VisualizationArtifact(
            name="stage44_selected_guard_train_vs_holdout.svg",
            path=str(
                _write_svg(
                    output_dir,
                    "stage44_selected_guard_train_vs_holdout.svg",
                    _render_bar_chart_svg(
                        title="Stage 44 selected guard train-CV vs dev-holdout",
                        bars=[
                            _Bar(
                                f"{result.selected_guard_label} train CV delta",
                                result.selected_train_cv_metrics.average_delta_vs_baseline,
                                (
                                    f"{result.selected_train_cv_metrics.average_delta_vs_baseline:+.4f}"
                                ),
                            ),
                            _Bar(
                                f"{result.selected_guard_label} holdout delta",
                                result.selected_holdout_metrics.average_delta_vs_baseline,
                                (
                                    f"{result.selected_holdout_metrics.average_delta_vs_baseline:+.4f}"
                                ),
                            ),
                            _Bar(
                                f"{result.selected_guard_label} train CV regressions",
                                float(result.selected_train_cv_metrics.regressed_count),
                                str(result.selected_train_cv_metrics.regressed_count),
                            ),
                            _Bar(
                                f"{result.selected_guard_label} holdout regressions",
                                float(result.selected_holdout_metrics.regressed_count),
                                str(result.selected_holdout_metrics.regressed_count),
                            ),
                        ],
                        x_label="delta or case count",
                    ),
                )
            ),
        )
    ]
    return RuntimeDocumentRiskProxyCVVisualizations(
        train_cv=train_cv,
        holdout=holdout,
        selected=selected,
    )


def _guard_evaluations_for_selections(
    selections: Sequence[CandidateRerankerSelection],
    rows: Sequence[Mapping[str, Any]],
    gold_answers_by_question_key: Mapping[str, str],
    max_answer_candidates: int,
    guard_specs: Sequence[RuntimeDocumentRiskGuardSpec],
) -> list[RuntimeDocumentRiskGuardEvaluation]:
    policy_configs = _policy_configs_by_label()
    main_decisions = candidate_reranker_policy_decisions_from_selections(
        config=policy_configs[STAGE39_MAIN_POLICY_LABEL],
        selections=selections,
        rows=rows,
    )
    return evaluate_runtime_document_risk_guards_from_main_decisions(
        main_decisions=main_decisions,
        rows=rows,
        gold_answers_by_question_key=gold_answers_by_question_key,
        max_answer_candidates=max_answer_candidates,
        guard_specs=guard_specs,
    )


def _validate_stage43_report(
    report: Mapping[str, Any],
    model_name: str,
    train_split: str,
    evaluation_split: str,
    max_answer_candidates: int,
    holdout_guard_evaluations: Sequence[RuntimeDocumentRiskGuardEvaluation],
) -> None:
    mode_name = f"top{max_answer_candidates}_leading_candidate_rewrite"
    if report.get("model_name") != model_name:
        raise ValueError("Stage 43 report model_name does not match Stage 44")
    if report.get("train_split") != train_split:
        raise ValueError("Stage 43 report train_split does not match Stage 44")
    if report.get("evaluation_split") != evaluation_split:
        raise ValueError("Stage 43 report evaluation_split does not match Stage 44")
    if report.get("mode_name") != mode_name:
        raise ValueError("Stage 43 report mode_name does not match Stage 44")
    report_guard_evaluations = report.get("guard_evaluations")
    if not isinstance(report_guard_evaluations, list):
        raise ValueError("Stage 43 report must contain guard_evaluations")

    report_metrics_by_label = {
        str(evaluation["label"]): evaluation["metrics"]
        for evaluation in report_guard_evaluations
        if isinstance(evaluation, Mapping)
        and isinstance(evaluation.get("metrics"), Mapping)
    }
    for evaluation in holdout_guard_evaluations:
        report_metrics = report_metrics_by_label.get(evaluation.label)
        if report_metrics is None:
            raise ValueError(f"Stage 43 report missing guard: {evaluation.label}")
        _validate_metrics_match(
            report_metrics=report_metrics,
            expected_metrics=evaluation.metrics,
            label=evaluation.label,
        )


def _validate_metrics_match(
    report_metrics: Mapping[str, Any],
    expected_metrics: GuardedCandidateAnswerMetrics,
    label: str,
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
            raise ValueError(f"Stage 43 report {label}.{key} does not match")


def _findings(
    selected_train_evaluation: RuntimeDocumentRiskGuardEvaluation,
    selected_holdout_evaluation: RuntimeDocumentRiskGuardEvaluation,
    train_cv_guard_evaluations: Sequence[RuntimeDocumentRiskGuardEvaluation],
    holdout_guard_evaluations: Sequence[RuntimeDocumentRiskGuardEvaluation],
) -> list[str]:
    train_best = max(train_cv_guard_evaluations, key=_selection_key)
    holdout_best = max(holdout_guard_evaluations, key=_selection_key)
    score60_train = _evaluation_by_label(
        train_cv_guard_evaluations,
        CANDIDATE_SCORE_GTE_60_LABEL,
    )
    score60_holdout = _evaluation_by_label(
        holdout_guard_evaluations,
        CANDIDATE_SCORE_GTE_60_LABEL,
    )
    return [
        (
            f"Train-only CV selected {selected_train_evaluation.label}: "
            f"delta {selected_train_evaluation.metrics.average_delta_vs_baseline:+.4f}, "
            f"regressions {selected_train_evaluation.metrics.regressed_count}."
        ),
        (
            f"Selected guard holdout result is "
            f"{selected_holdout_evaluation.metrics.average_delta_vs_baseline:+.4f}, "
            f"regressions {selected_holdout_evaluation.metrics.regressed_count}, "
            f"citation loss {selected_holdout_evaluation.metrics.citation_lost_count}."
        ),
        (
            f"Train-CV best is {train_best.label}; dev-holdout best by the same "
            f"metric is {holdout_best.label}."
        ),
        (
            f"candidate_score_gte_60 train-CV delta "
            f"{score60_train.metrics.average_delta_vs_baseline:+.4f}; holdout delta "
            f"{score60_holdout.metrics.average_delta_vs_baseline:+.4f}."
        ),
    ]


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


def _policy_configs_by_label() -> dict[str, Any]:
    return {label: config for label, config in default_stage39_policy_specs()}


def _evaluation_by_label(
    evaluations: Sequence[RuntimeDocumentRiskGuardEvaluation],
    label: str,
) -> RuntimeDocumentRiskGuardEvaluation:
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


def _question_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return len({f"{row['split']}::{row['question_id']}" for row in rows})


def _normalize_split_name(split_name: str) -> str:
    normalized = split_name.strip().lower()
    if not normalized:
        raise ValueError("split name must not be empty")
    return normalized


def _validate_options(
    rows: Sequence[Mapping[str, Any]],
    train_fold_count: int,
    max_answer_candidates: int,
) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    if train_fold_count < 2:
        raise ValueError("train_fold_count must be at least 2")
    if max_answer_candidates <= 0:
        raise ValueError("max_answer_candidates must be positive")


def _write_svg(output_dir: Path, filename: str, svg: str) -> Path:
    path = output_dir / filename
    path.write_text(svg, encoding="utf-8")
    return path


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
