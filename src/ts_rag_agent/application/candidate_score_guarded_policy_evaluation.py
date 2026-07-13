from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any

from ts_rag_agent.application.candidate_reranker_cv import (
    cross_validated_candidate_reranker_selections,
)
from ts_rag_agent.application.candidate_reranker_policy_search import (
    CandidateRerankerPolicyConfig,
    candidate_reranker_policy_decisions_from_selections,
)
from ts_rag_agent.application.candidate_reranker_policy_stability import (
    STAGE35_BLOCK_HOW_TO_ONLY_POLICY,
)
from ts_rag_agent.application.guarded_candidate_reranker_answer_experiment import (
    SINGLE_CANDIDATE_MODE,
    GuardedCandidateAnswerMetrics,
    GuardedCandidateAnswerMetricsBySegment,
    build_single_candidate_answer_cases_from_decisions,
    build_topk_leading_candidate_answer_cases_from_decisions,
    segment_guarded_candidate_answer_cases,
    summarize_guarded_candidate_answer_cases,
)


@dataclass(frozen=True)
class GuardedPolicyModeEvaluation:
    """One policy's metrics for one answer proxy mode."""

    mode_name: str
    metrics: GuardedCandidateAnswerMetrics
    route_metrics: list[GuardedCandidateAnswerMetricsBySegment]
    selected_rank_metrics: list[GuardedCandidateAnswerMetricsBySegment]


@dataclass(frozen=True)
class GuardedPolicyEvaluation:
    """One guarded policy evaluated across answer proxy modes."""

    label: str
    config: CandidateRerankerPolicyConfig
    mode_evaluations: list[GuardedPolicyModeEvaluation]


@dataclass(frozen=True)
class GuardedPolicyDelta:
    """Metric difference between a candidate policy and the main policy."""

    mode_name: str
    candidate_label: str
    main_label: str
    policy_average_f1_difference: float
    average_delta_difference: float
    replacement_count_difference: int
    regressed_count_difference: int
    citation_lost_count_difference: int
    citation_gained_count_difference: int
    gold_citation_delta_difference: int


@dataclass(frozen=True)
class CandidateScoreGuardedPolicyEvaluationResult:
    """Stage 39 fixed-policy evaluation result."""

    model_name: str
    fold_count: int
    max_answer_candidates: int
    policies: list[GuardedPolicyEvaluation]
    deltas_vs_main: list[GuardedPolicyDelta]
    findings: list[str]
    analysis_scope: str


@dataclass(frozen=True)
class VisualizationArtifact:
    """One generated visualization file."""

    name: str
    path: str


STAGE39_MAIN_POLICY_LABEL = "stage36_main"


def evaluate_candidate_score_guarded_policies(
    rows: Sequence[Mapping[str, Any]],
    gold_answers_by_question_key: Mapping[str, str],
    model_name: str = "logistic_best_candidate",
    fold_count: int = 5,
    max_answer_candidates: int = 3,
    policies: Sequence[tuple[str, CandidateRerankerPolicyConfig]] | None = None,
) -> CandidateScoreGuardedPolicyEvaluationResult:
    """Evaluate fixed candidate-score guarded policies over grouped-CV selections."""

    if not rows:
        raise ValueError("rows must not be empty")
    if max_answer_candidates <= 0:
        raise ValueError("max_answer_candidates must be positive")
    policy_specs = list(policies or default_stage39_policy_specs())
    if not policy_specs:
        raise ValueError("policies must not be empty")

    selections = cross_validated_candidate_reranker_selections(
        rows=rows,
        model_name=model_name,
        fold_count=fold_count,
    )
    policy_evaluations = [
        _evaluate_policy(
            label=label,
            config=config,
            selections=selections,
            rows=rows,
            gold_answers_by_question_key=gold_answers_by_question_key,
            max_answer_candidates=max_answer_candidates,
        )
        for label, config in policy_specs
    ]
    main_policy = policy_evaluations[0]
    deltas = _deltas_vs_main(main_policy=main_policy, policies=policy_evaluations[1:])
    return CandidateScoreGuardedPolicyEvaluationResult(
        model_name=model_name,
        fold_count=fold_count,
        max_answer_candidates=max_answer_candidates,
        policies=policy_evaluations,
        deltas_vs_main=deltas,
        findings=_findings(policy_evaluations, deltas),
        analysis_scope=(
            "Offline fixed-policy evaluation only. Policies use grouped-CV candidate "
            "reranker selections and runtime-available candidate features. The "
            "single-candidate mode uses candidate token-F1 labels; the top-k mode "
            "recomputes answer token F1 from local gold answers and candidate "
            "metadata sentences. Runtime behavior is not changed."
        ),
    )


def default_stage39_policy_specs() -> tuple[tuple[str, CandidateRerankerPolicyConfig], ...]:
    """Return the fixed policies compared in Stage 39."""

    return (
        (STAGE39_MAIN_POLICY_LABEL, STAGE35_BLOCK_HOW_TO_ONLY_POLICY),
        (
            "model_margin_gte_0.10",
            CandidateRerankerPolicyConfig(
                name=(
                    "rank_lte_5__margin_gte_0.10__top1_score_protect_none"
                    "__blocked_how_to_or_lookup"
                ),
                max_selected_rank=5,
                blocked_routes=("how_to_or_lookup",),
                min_score_margin_vs_top_candidate=0.10,
                protect_top1_candidate_score_min=None,
            ),
        ),
        (
            "candidate_score_gte_60",
            CandidateRerankerPolicyConfig(
                name=(
                    "rank_lte_5__margin_gte_0.05__selected_score_gte_60"
                    "__top1_score_protect_none__blocked_how_to_or_lookup"
                ),
                max_selected_rank=5,
                blocked_routes=("how_to_or_lookup",),
                min_score_margin_vs_top_candidate=0.05,
                protect_top1_candidate_score_min=None,
                min_selected_candidate_score=60.0,
            ),
        ),
        (
            "candidate_score_gte_90",
            CandidateRerankerPolicyConfig(
                name=(
                    "rank_lte_5__margin_gte_0.05__selected_score_gte_90"
                    "__top1_score_protect_none__blocked_how_to_or_lookup"
                ),
                max_selected_rank=5,
                blocked_routes=("how_to_or_lookup",),
                min_score_margin_vs_top_candidate=0.05,
                protect_top1_candidate_score_min=None,
                min_selected_candidate_score=90.0,
            ),
        ),
    )


def candidate_score_guarded_policy_evaluation_to_dict(
    result: CandidateScoreGuardedPolicyEvaluationResult,
) -> dict[str, Any]:
    """Convert a Stage 39 policy evaluation result to a JSON-safe dictionary."""

    return asdict(result)


def write_candidate_score_guarded_policy_visualizations(
    result: CandidateScoreGuardedPolicyEvaluationResult,
    output_dir: Path,
) -> list[VisualizationArtifact]:
    """Write SVG charts for Stage 39 fixed-policy evaluation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    topk_mode_name = _topk_mode_name(result.policies[0])
    topk_label = _topk_display_label(topk_mode_name)
    charts = {
        f"stage39_{topk_label}_policy_delta.svg": _render_bar_chart_svg(
            title=f"Stage 39 {topk_label} proxy delta by policy",
            bars=[
                _Bar(
                    label=policy.label,
                    value=_mode_average_delta(policy, topk_mode_name),
                    value_label=f"{_mode_average_delta(policy, topk_mode_name):+.4f}",
                )
                for policy in result.policies
            ],
            x_label="average answer token F1 delta vs baseline",
        ),
        f"stage39_{topk_label}_policy_regressions.svg": _render_bar_chart_svg(
            title=f"Stage 39 {topk_label} proxy regressions by policy",
            bars=[
                _Bar(
                    label=policy.label,
                    value=float(_mode_metrics(policy, topk_mode_name).regressed_count),
                    value_label=str(_mode_metrics(policy, topk_mode_name).regressed_count),
                )
                for policy in result.policies
            ],
            x_label="regression cases",
        ),
        f"stage39_{topk_label}_policy_citation_exchange.svg": _render_bar_chart_svg(
            title=f"Stage 39 {topk_label} proxy citation exchange by policy",
            bars=[
                bar
                for policy in result.policies
                for bar in _citation_exchange_bars(policy)
            ],
            x_label="case count",
        ),
        "stage39_single_candidate_policy_delta.svg": _render_bar_chart_svg(
            title="Stage 39 single-candidate proxy delta by policy",
            bars=[
                _Bar(
                    label=policy.label,
                    value=_mode_average_delta(policy, SINGLE_CANDIDATE_MODE),
                    value_label=f"{_mode_average_delta(policy, SINGLE_CANDIDATE_MODE):+.4f}",
                )
                for policy in result.policies
            ],
            x_label="average answer token F1 delta vs baseline",
        ),
    }

    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(VisualizationArtifact(name=filename, path=str(path)))
    return artifacts


def _evaluate_policy(
    label: str,
    config: CandidateRerankerPolicyConfig,
    selections,
    rows: Sequence[Mapping[str, Any]],
    gold_answers_by_question_key: Mapping[str, str],
    max_answer_candidates: int,
) -> GuardedPolicyEvaluation:
    decisions = candidate_reranker_policy_decisions_from_selections(
        config=config,
        selections=selections,
        rows=rows,
    )
    single_cases = build_single_candidate_answer_cases_from_decisions(
        decisions=decisions,
        rows=rows,
    )
    topk_cases = build_topk_leading_candidate_answer_cases_from_decisions(
        decisions=decisions,
        rows=rows,
        gold_answers_by_question_key=gold_answers_by_question_key,
        max_answer_candidates=max_answer_candidates,
    )
    return GuardedPolicyEvaluation(
        label=label,
        config=config,
        mode_evaluations=[
            _mode_evaluation(SINGLE_CANDIDATE_MODE, single_cases),
            _mode_evaluation(topk_cases[0].mode_name, topk_cases),
        ],
    )


def _mode_evaluation(
    mode_name: str,
    cases,
) -> GuardedPolicyModeEvaluation:
    return GuardedPolicyModeEvaluation(
        mode_name=mode_name,
        metrics=summarize_guarded_candidate_answer_cases(cases),
        route_metrics=segment_guarded_candidate_answer_cases(
            cases,
            lambda case: case.question_route,
        ),
        selected_rank_metrics=segment_guarded_candidate_answer_cases(
            cases,
            lambda case: _rank_bucket(case.policy_leading_candidate_rank),
        ),
    )


def _deltas_vs_main(
    main_policy: GuardedPolicyEvaluation,
    policies: Sequence[GuardedPolicyEvaluation],
) -> list[GuardedPolicyDelta]:
    deltas = []
    for policy in policies:
        for mode_name in _mode_names(main_policy):
            main_metrics = _mode_metrics(main_policy, mode_name)
            candidate_metrics = _mode_metrics(policy, mode_name)
            deltas.append(
                GuardedPolicyDelta(
                    mode_name=mode_name,
                    candidate_label=policy.label,
                    main_label=main_policy.label,
                    policy_average_f1_difference=round(
                        candidate_metrics.policy_average_answer_token_f1
                        - main_metrics.policy_average_answer_token_f1,
                        4,
                    ),
                    average_delta_difference=round(
                        candidate_metrics.average_delta_vs_baseline
                        - main_metrics.average_delta_vs_baseline,
                        4,
                    ),
                    replacement_count_difference=(
                        candidate_metrics.replacement_count
                        - main_metrics.replacement_count
                    ),
                    regressed_count_difference=(
                        candidate_metrics.regressed_count - main_metrics.regressed_count
                    ),
                    citation_lost_count_difference=(
                        candidate_metrics.citation_lost_count
                        - main_metrics.citation_lost_count
                    ),
                    citation_gained_count_difference=(
                        candidate_metrics.citation_gained_count
                        - main_metrics.citation_gained_count
                    ),
                    gold_citation_delta_difference=(
                        candidate_metrics.gold_citation_delta
                        - main_metrics.gold_citation_delta
                    ),
                )
            )
    return deltas


def _findings(
    policies: Sequence[GuardedPolicyEvaluation],
    deltas: Sequence[GuardedPolicyDelta],
) -> list[str]:
    topk_mode_name = _topk_mode_name(policies[0])
    topk_label = _topk_display_label(topk_mode_name)
    top3_metrics_by_policy = {
        policy.label: _mode_metrics(policy, topk_mode_name)
        for policy in policies
    }
    main_metrics = top3_metrics_by_policy[STAGE39_MAIN_POLICY_LABEL]
    best_delta_label, best_delta_metrics = max(
        top3_metrics_by_policy.items(),
        key=lambda item: (
            item[1].average_delta_vs_baseline,
            -item[1].regressed_count,
        ),
    )
    best_regression_label, best_regression_metrics = min(
        top3_metrics_by_policy.items(),
        key=lambda item: (
            item[1].regressed_count,
            -item[1].average_delta_vs_baseline,
        ),
    )
    findings = [
        (
            f"Main policy {topk_label} proxy delta is "
            f"{main_metrics.average_delta_vs_baseline:+.4f} with "
            f"{main_metrics.regressed_count} regressions."
        ),
        (
            f"Best {topk_label} average-delta policy is {best_delta_label}: "
            f"{best_delta_metrics.average_delta_vs_baseline:+.4f}, "
            f"regressions {best_delta_metrics.regressed_count}."
        ),
        (
            f"Lowest {topk_label} regression policy is {best_regression_label}: "
            f"{best_regression_metrics.regressed_count} regressions, "
            f"delta {best_regression_metrics.average_delta_vs_baseline:+.4f}."
        ),
    ]
    candidate_score_60 = top3_metrics_by_policy.get("candidate_score_gte_60")
    if candidate_score_60:
        findings.append(
            f"candidate_score_gte_60 changes {topk_label} regression count from "
            f"{main_metrics.regressed_count} to {candidate_score_60.regressed_count} "
            f"and gold citation delta from {main_metrics.gold_citation_delta:+d} "
            f"to {candidate_score_60.gold_citation_delta:+d}."
        )
    top3_deltas = [
        delta
        for delta in deltas
        if delta.mode_name == topk_mode_name and delta.average_delta_difference >= 0
    ]
    if top3_deltas:
        best_non_regressing_delta = min(
            top3_deltas,
            key=lambda delta: (
                delta.regressed_count_difference,
                -delta.average_delta_difference,
            ),
        )
        findings.append(
            f"Best non-negative {topk_label} delta versus main by regression difference is "
            f"{best_non_regressing_delta.candidate_label}: "
            f"delta diff {best_non_regressing_delta.average_delta_difference:+.4f}, "
            f"regression diff {best_non_regressing_delta.regressed_count_difference:+d}."
        )
    return findings


def _mode_names(policy: GuardedPolicyEvaluation) -> list[str]:
    return [evaluation.mode_name for evaluation in policy.mode_evaluations]


def _topk_mode_name(policy: GuardedPolicyEvaluation) -> str:
    for mode_name in _mode_names(policy):
        if mode_name != SINGLE_CANDIDATE_MODE:
            return mode_name
    raise ValueError("policy evaluation must include a top-k mode")


def _topk_display_label(mode_name: str) -> str:
    if mode_name.startswith("top") and "_" in mode_name:
        return mode_name.split("_", maxsplit=1)[0]
    return mode_name


def _mode_metrics(
    policy: GuardedPolicyEvaluation,
    mode_name: str,
) -> GuardedCandidateAnswerMetrics:
    for evaluation in policy.mode_evaluations:
        if evaluation.mode_name == mode_name:
            return evaluation.metrics
    raise ValueError(f"Missing mode evaluation: {mode_name}")


def _mode_average_delta(policy: GuardedPolicyEvaluation, mode_name: str) -> float:
    return _mode_metrics(policy, mode_name).average_delta_vs_baseline


def _citation_exchange_bars(policy: GuardedPolicyEvaluation) -> list[_Bar]:
    metrics = _mode_metrics(policy, _topk_mode_name(policy))
    return [
        _Bar(
            label=f"{policy.label} lost",
            value=float(metrics.citation_lost_count),
            value_label=str(metrics.citation_lost_count),
        ),
        _Bar(
            label=f"{policy.label} gained",
            value=float(metrics.citation_gained_count),
            value_label=str(metrics.citation_gained_count),
        ),
    ]


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
