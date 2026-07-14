from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.msqa_stage51_adapter_comparison import (
    MsqaStage51ComparisonCase,
    build_msqa_stage51_capped_adapter_cases,
    msqa_stage51_comparison_case_to_dict,
    summarize_msqa_stage51_comparison_cases,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 65"
_CREATED_AT = "2026-07-14"
_EXPECTED_STAGE64_STATUS = "msqa_stage51_capped_adapter_comparison_f1_regressed"
_DEFAULT_MODEL_NAME = "logistic_best_candidate"
_DEFAULT_TRAIN_SPLIT = "train"
_DEFAULT_MAX_ANSWER_CANDIDATES = 3
_DEFAULT_MAX_CITATION_RANK = 3


@dataclass(frozen=True)
class MsqaChangedCaseReviewVisualization:
    """One generated Stage65 visualization."""

    name: str
    path: str


def review_msqa_stage51_changed_cases(
    *,
    stage64_report_path: Path,
    split_jsonl_path: Path,
    candidate_jsonl_path: Path,
    adapter_report_path: Path,
    distribution_report_path: Path,
    candidate_reranker_dataset_path: Path,
    stage31_summary_path: Path,
    model_name: str = _DEFAULT_MODEL_NAME,
    train_split: str = _DEFAULT_TRAIN_SPLIT,
    max_answer_candidates: int = _DEFAULT_MAX_ANSWER_CANDIDATES,
    max_citation_rank: int = _DEFAULT_MAX_CITATION_RANK,
    sample_limit: int = 20,
) -> dict[str, Any]:
    """Review Stage64 changed cases and source-citation tradeoffs."""

    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")
    for path in [
        stage64_report_path,
        split_jsonl_path,
        candidate_jsonl_path,
        adapter_report_path,
        distribution_report_path,
        candidate_reranker_dataset_path,
        stage31_summary_path,
    ]:
        _ensure_file(path)

    stage64_report = _load_json(stage64_report_path)
    _validate_stage64_report(stage64_report)
    cases = build_msqa_stage51_capped_adapter_cases(
        split_jsonl_path=split_jsonl_path,
        candidate_jsonl_path=candidate_jsonl_path,
        adapter_report_path=adapter_report_path,
        distribution_report_path=distribution_report_path,
        candidate_reranker_dataset_path=candidate_reranker_dataset_path,
        stage31_summary_path=stage31_summary_path,
        model_name=model_name,
        train_split=train_split,
        max_answer_candidates=max_answer_candidates,
        max_citation_rank=max_citation_rank,
    )
    recomputed_metrics = summarize_msqa_stage51_comparison_cases(cases)
    consistency_checks = _consistency_checks(
        stage64_metrics=stage64_report["metrics"],
        recomputed_metrics=recomputed_metrics,
    )
    changed_cases = _changed_cases(cases)
    regressions = [
        case for case in cases if case.top3_f1_delta_vs_baseline < 0
    ]
    improvements = [
        case for case in cases if case.top3_f1_delta_vs_baseline > 0
    ]
    citation_gained = [case for case in cases if case.citation_delta > 0]
    citation_lost = [case for case in cases if case.citation_delta < 0]

    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_scope": (
            "Stage65 reviews the Stage64 capped MSQA Stage51 adapter comparison "
            "changed cases. It rebuilds the full Stage64 case view from the same "
            "inputs, verifies the aggregate metrics match Stage64, and analyzes "
            "route, selected-rank, score-margin, and source-transition "
            "concentration. It does not rerun a new comparison policy, does not "
            "rebuild the candidate pool, and does not change the default runtime."
        ),
        "source_files": {
            "stage64_report": _fingerprint(stage64_report_path),
            "split_jsonl": _fingerprint(split_jsonl_path),
            "candidate_jsonl": _fingerprint(candidate_jsonl_path),
            "adapter_report": _fingerprint(adapter_report_path),
            "distribution_report": _fingerprint(distribution_report_path),
            "candidate_reranker_dataset": _fingerprint(candidate_reranker_dataset_path),
            "stage31_summary": _fingerprint(stage31_summary_path),
        },
        "rebuild_contract": {
            "candidate_pool_rebuilt": False,
            "model_name": model_name,
            "train_split": train_split,
            "max_answer_candidates": max_answer_candidates,
            "max_citation_rank": max_citation_rank,
            "case_count": len(cases),
        },
        "consistency_checks": consistency_checks,
        "stage64_metrics": stage64_report["metrics"],
        "recomputed_metrics": recomputed_metrics,
        "changed_case_summary": _changed_case_summary(
            cases=cases,
            changed_cases=changed_cases,
            regressions=regressions,
            improvements=improvements,
            citation_gained=citation_gained,
            citation_lost=citation_lost,
        ),
        "cohort_summaries": {
            "changed": _cohort_summary(changed_cases),
            "top3_regressions": _cohort_summary(regressions),
            "top3_improvements": _cohort_summary(improvements),
            "citation_gained": _cohort_summary(citation_gained),
            "citation_lost": _cohort_summary(citation_lost),
        },
        "concentration": _concentration(
            changed_cases=changed_cases,
            regressions=regressions,
            improvements=improvements,
            citation_gained=citation_gained,
        ),
        "case_sets": {
            "top3_regressions": [
                _review_case(case)
                for case in sorted(
                    regressions,
                    key=lambda case: (
                        case.top3_f1_delta_vs_baseline,
                        case.query_question_id,
                    ),
                )
            ],
            "top3_improvements": [
                _review_case(case)
                for case in sorted(
                    improvements,
                    key=lambda case: (
                        case.top3_f1_delta_vs_baseline,
                        case.query_question_id,
                    ),
                    reverse=True,
                )
            ],
            "citation_gained": [
                _review_case(case)
                for case in sorted(
                    citation_gained,
                    key=lambda case: (
                        -case.top3_f1_delta_vs_baseline,
                        case.query_question_id,
                    ),
                )
            ],
            "largest_regressions_sample": [
                _review_case(case)
                for case in sorted(
                    regressions,
                    key=lambda case: (
                        case.top3_f1_delta_vs_baseline,
                        case.query_question_id,
                    ),
                )[:sample_limit]
            ],
            "changed_answer_sample": [
                _review_case(case) for case in changed_cases[:sample_limit]
            ],
        },
        "decision": _decision(
            consistency_checks=consistency_checks,
            regressions=regressions,
            improvements=improvements,
            citation_gained=citation_gained,
            citation_lost=citation_lost,
        ),
    }


def write_msqa_stage51_changed_case_review_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[MsqaChangedCaseReviewVisualization]:
    """Write SVG charts for the Stage65 changed-case review."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage65_msqa_changed_outcomes.svg": render_horizontal_bar_chart_svg(
            title="Stage65 MSQA changed-case outcomes",
            bars=_changed_outcome_bars(report),
            x_label="question count",
            margin_left=260,
        ),
        "stage65_msqa_regressions_by_route.svg": render_horizontal_bar_chart_svg(
            title="Stage65 MSQA regressions by route",
            bars=_regression_route_bars(report),
            x_label="regression count",
            margin_left=300,
        ),
        "stage65_msqa_changed_by_selected_rank.svg": render_horizontal_bar_chart_svg(
            title="Stage65 MSQA changed cases by selected rank",
            bars=_changed_selected_rank_bars(report),
            x_label="changed question count",
            margin_left=220,
        ),
        "stage65_msqa_source_transitions.svg": render_horizontal_bar_chart_svg(
            title="Stage65 MSQA source transitions",
            bars=_source_transition_bars(report),
            x_label="changed question count",
            margin_left=340,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(MsqaChangedCaseReviewVisualization(name=filename, path=str(path)))
    return artifacts


def _changed_case_summary(
    *,
    cases: Sequence[MsqaStage51ComparisonCase],
    changed_cases: Sequence[MsqaStage51ComparisonCase],
    regressions: Sequence[MsqaStage51ComparisonCase],
    improvements: Sequence[MsqaStage51ComparisonCase],
    citation_gained: Sequence[MsqaStage51ComparisonCase],
    citation_lost: Sequence[MsqaStage51ComparisonCase],
) -> dict[str, Any]:
    regression_loss_sum = sum(
        case.top3_f1_delta_vs_baseline for case in regressions
    )
    improvement_gain_sum = sum(
        case.top3_f1_delta_vs_baseline for case in improvements
    )
    return {
        "question_count": len(cases),
        "changed_answer_count": len(changed_cases),
        "changed_answer_rate": _ratio(len(changed_cases), len(cases)),
        "top3_regression_count": len(regressions),
        "top3_improvement_count": len(improvements),
        "regression_to_improvement_count_ratio": _safe_ratio(
            len(regressions),
            len(improvements),
        ),
        "regression_loss_sum": round(regression_loss_sum, 4),
        "improvement_gain_sum": round(improvement_gain_sum, 4),
        "net_top3_delta_sum": round(regression_loss_sum + improvement_gain_sum, 4),
        "citation_gained_count": len(citation_gained),
        "citation_lost_count": len(citation_lost),
        "citation_delta": sum(case.citation_delta for case in cases),
        "regressions_with_citation_gain": sum(
            case.citation_delta > 0 for case in regressions
        ),
        "improvements_with_citation_gain": sum(
            case.citation_delta > 0 for case in improvements
        ),
        "changed_without_f1_or_citation_gain": sum(
            case.top3_f1_delta_vs_baseline == 0 and case.citation_delta == 0
            for case in changed_cases
        ),
    }


def _cohort_summary(cases: Sequence[MsqaStage51ComparisonCase]) -> dict[str, Any]:
    if not cases:
        return {
            "count": 0,
            "average_top3_delta": 0.0,
            "median_top3_delta": 0.0,
            "gold_source_citation_delta": 0,
            "route_counts": {},
            "selected_rank_counts": {},
            "decision_reason_counts": {},
            "score_margin_bins": {},
            "selected_candidate_score_bins": {},
            "source_transition_counts": {},
        }
    return {
        "count": len(cases),
        "average_top3_delta": _mean(
            [case.top3_f1_delta_vs_baseline for case in cases]
        ),
        "median_top3_delta": round(
            float(statistics.median(case.top3_f1_delta_vs_baseline for case in cases)),
            4,
        ),
        "gold_source_citation_delta": sum(case.citation_delta for case in cases),
        "route_counts": _counter_dict(case.question_route for case in cases),
        "selected_rank_counts": _counter_dict(
            _rank_bucket(case.model_selected_rank) for case in cases
        ),
        "decision_reason_counts": _counter_dict(case.decision_reason for case in cases),
        "score_margin_bins": _counter_dict(
            _score_margin_bin(case.model_score_margin_vs_top_candidate)
            for case in cases
        ),
        "selected_candidate_score_bins": _counter_dict(
            _candidate_score_bin(case.model_selected_candidate_score)
            for case in cases
        ),
        "source_transition_counts": _counter_dict(
            _source_transition(case) for case in cases
        ),
    }


def _concentration(
    *,
    changed_cases: Sequence[MsqaStage51ComparisonCase],
    regressions: Sequence[MsqaStage51ComparisonCase],
    improvements: Sequence[MsqaStage51ComparisonCase],
    citation_gained: Sequence[MsqaStage51ComparisonCase],
) -> dict[str, Any]:
    return {
        "regression_route_share": _share_by_key(
            regressions,
            key_fn=lambda case: case.question_route,
        ),
        "regression_selected_rank_share": _share_by_key(
            regressions,
            key_fn=lambda case: _rank_bucket(case.model_selected_rank),
        ),
        "regression_source_transition_share": _share_by_key(
            regressions,
            key_fn=_source_transition,
        ),
        "improvement_source_transition_share": _share_by_key(
            improvements,
            key_fn=_source_transition,
        ),
        "citation_gain_source_transition_share": _share_by_key(
            citation_gained,
            key_fn=_source_transition,
        ),
        "changed_source_transition_share": _share_by_key(
            changed_cases,
            key_fn=_source_transition,
        ),
    }


def _review_case(case: MsqaStage51ComparisonCase) -> dict[str, Any]:
    base = msqa_stage51_comparison_case_to_dict(case)
    base.update(
        {
            "changed": case.baseline_candidate_ids != case.stage51_candidate_ids,
            "outcome": _outcome(case),
            "selected_rank_bucket": _rank_bucket(case.model_selected_rank),
            "score_margin_bucket": _score_margin_bin(
                case.model_score_margin_vs_top_candidate
            ),
            "selected_candidate_score_bucket": _candidate_score_bin(
                case.model_selected_candidate_score
            ),
            "source_transition": _source_transition(case),
            "baseline_source_set": sorted(set(case.baseline_source_row_ids)),
            "stage51_source_set": sorted(set(case.stage51_source_row_ids)),
            "introduced_source_row_ids": sorted(
                set(case.stage51_source_row_ids) - set(case.baseline_source_row_ids)
            ),
            "dropped_source_row_ids": sorted(
                set(case.baseline_source_row_ids) - set(case.stage51_source_row_ids)
            ),
        }
    )
    return base


def _consistency_checks(
    *,
    stage64_metrics: Mapping[str, Any],
    recomputed_metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks = []
    for key in (
        "question_count",
        "changed_answer_count",
        "replacement_count",
        "top3_improved_count",
        "top3_regressed_count",
        "gold_source_citation_delta",
        "citation_lost_count",
        "citation_gained_count",
    ):
        checks.append(
            _check(
                name=f"{key}_matches_stage64",
                passed=stage64_metrics[key] == recomputed_metrics[key],
                observed=recomputed_metrics[key],
                expected=stage64_metrics[key],
            )
        )
    for key in (
        "baseline_top3_average_answer_token_f1",
        "stage51_top3_average_answer_token_f1",
        "top3_average_delta_vs_baseline",
    ):
        checks.append(
            _check(
                name=f"{key}_matches_stage64",
                passed=abs(float(stage64_metrics[key]) - float(recomputed_metrics[key]))
                <= 0.0001,
                observed=recomputed_metrics[key],
                expected=stage64_metrics[key],
            )
        )
    return checks


def _decision(
    *,
    consistency_checks: Sequence[Mapping[str, Any]],
    regressions: Sequence[MsqaStage51ComparisonCase],
    improvements: Sequence[MsqaStage51ComparisonCase],
    citation_gained: Sequence[MsqaStage51ComparisonCase],
    citation_lost: Sequence[MsqaStage51ComparisonCase],
) -> dict[str, Any]:
    all_checks_pass = all(check["passed"] for check in consistency_checks)
    if not all_checks_pass:
        status = "msqa_stage51_changed_case_review_blocked_by_inconsistent_rebuild"
    elif regressions and len(regressions) > len(improvements):
        status = "msqa_stage51_changed_case_review_blocks_defaultization"
    else:
        status = "msqa_stage51_changed_case_review_completed"
    return {
        "status": status,
        "can_defaultize_runtime_now": False,
        "default_runtime_policy": "unchanged",
        "stage51_adapter_comparison_run_performed": False,
        "candidate_pool_rebuilt": False,
        "consistency_checks_passed": all_checks_pass,
        "regression_count_exceeds_improvement_count": (
            len(regressions) > len(improvements)
        ),
        "citation_lost_count": len(citation_lost),
        "citation_gained_count": len(citation_gained),
        "recommended_next_stage": (
            "Stage 66: choose the next evaluation route explicitly: either find "
            "another external dataset, design an MSQA-specific risk guard for a "
            "new frozen experiment, or freeze Stage51 as non-default research "
            "evidence"
        ),
        "reason": (
            "Stage64 gained source citations without citation loss, but top3 "
            "answer regressions outnumber improvements and the net answer proxy "
            "delta is negative. This blocks defaultization."
        ),
    }


def _validate_stage64_report(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 64":
        raise ValueError("Expected a Stage64 comparison report")
    if report["decision"].get("status") != _EXPECTED_STAGE64_STATUS:
        raise ValueError("Stage64 report status is not the expected F1-regressed status")
    if report["decision"].get("stage51_adapter_comparison_run_performed") is not True:
        raise ValueError("Stage64 report must have run one adapter comparison")
    if report["decision"].get("candidate_pool_reused_without_rebuild") is not True:
        raise ValueError("Stage64 report must reuse the capped candidate pool")


def _changed_cases(
    cases: Sequence[MsqaStage51ComparisonCase],
) -> list[MsqaStage51ComparisonCase]:
    return [
        case for case in cases if case.baseline_candidate_ids != case.stage51_candidate_ids
    ]


def _source_transition(case: MsqaStage51ComparisonCase) -> str:
    baseline_sources = set(case.baseline_source_row_ids)
    stage51_sources = set(case.stage51_source_row_ids)
    if case.baseline_source_row_ids == case.stage51_source_row_ids:
        if case.baseline_candidate_ids == case.stage51_candidate_ids:
            return "unchanged"
        return "same_source_sentence_rewrite"
    if case.citation_delta > 0:
        return "gold_source_added"
    if case.citation_delta < 0:
        return "gold_source_dropped"
    if case.baseline_source_row_ids[0] != case.stage51_source_row_ids[0]:
        return "leading_source_changed"
    if baseline_sources != stage51_sources:
        return "source_set_changed"
    return "source_order_changed"


def _outcome(case: MsqaStage51ComparisonCase) -> str:
    if case.top3_f1_delta_vs_baseline > 0:
        return "top3_improved"
    if case.top3_f1_delta_vs_baseline < 0:
        return "top3_regressed"
    if case.citation_delta > 0:
        return "citation_gained_only"
    if case.citation_delta < 0:
        return "citation_lost_only"
    return "tied"


def _rank_bucket(rank: int) -> str:
    if rank <= 0:
        return "missing"
    if rank == 1:
        return "rank_1"
    if rank == 2:
        return "rank_2"
    if rank == 3:
        return "rank_3"
    if rank <= 5:
        return "rank_4_5"
    if rank <= 10:
        return "rank_6_10"
    return "rank_11_15"


def _score_margin_bin(value: float) -> str:
    if value < 0.05:
        return "margin_lt_0.05"
    if value < 0.10:
        return "margin_0.05_0.10"
    if value < 0.20:
        return "margin_0.10_0.20"
    if value < 0.50:
        return "margin_0.20_0.50"
    return "margin_gte_0.50"


def _candidate_score_bin(value: float) -> str:
    if value < 60:
        return "score_lt_60"
    if value < 90:
        return "score_60_90"
    if value < 140:
        return "score_90_140"
    return "score_gte_140"


def _share_by_key(
    cases: Sequence[MsqaStage51ComparisonCase],
    key_fn,
) -> dict[str, dict[str, float | int]]:
    total = len(cases)
    counts = Counter(str(key_fn(case)) for case in cases)
    return {
        key: {
            "count": count,
            "share": _ratio(count, total),
        }
        for key, count in sorted(counts.items())
    }


def _counter_dict(values) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _check(name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "observed": observed,
        "expected": expected,
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _changed_outcome_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    summary = report["changed_case_summary"]
    return [
        _count_bar("changed", summary["changed_answer_count"]),
        _count_bar("top3_improved", summary["top3_improvement_count"]),
        _count_bar("top3_regressed", summary["top3_regression_count"]),
        _count_bar("citation_gained", summary["citation_gained_count"]),
        _count_bar("citation_lost", summary["citation_lost_count"]),
    ]


def _regression_route_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    route_counts = report["cohort_summaries"]["top3_regressions"]["route_counts"]
    return [
        BarDatum(route, float(count), str(count))
        for route, count in route_counts.items()
    ]


def _changed_selected_rank_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    rank_counts = report["cohort_summaries"]["changed"]["selected_rank_counts"]
    return [
        BarDatum(rank, float(count), str(count))
        for rank, count in rank_counts.items()
    ]


def _source_transition_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    transitions = report["cohort_summaries"]["changed"]["source_transition_counts"]
    return [
        BarDatum(transition, float(count), str(count))
        for transition, count in transitions.items()
    ]


def _count_bar(label: str, value: int) -> BarDatum:
    return BarDatum(label, float(value), str(value))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint(path: Path) -> dict[str, Any]:
    _ensure_file(path)
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
