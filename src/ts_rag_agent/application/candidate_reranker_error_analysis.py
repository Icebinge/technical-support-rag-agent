from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from ts_rag_agent.application.candidate_reranker_cv import (
    CandidateRerankerSelection,
    cross_validated_candidate_reranker_selections,
)


@dataclass(frozen=True)
class CandidateSnapshot:
    """Inspection snapshot for one candidate in an error-analysis case."""

    candidate_id: str
    candidate_rank: int
    candidate_token_f1: float
    is_gold_document: bool
    runtime_features: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CandidateRerankerErrorCase:
    """One question-level reranker outcome for offline inspection."""

    split: str
    question_id: str
    question_route: str
    outcome: str
    f1_delta_vs_top_candidate: float
    selected_model_score: float
    gold_document_candidate_count: int
    selected_missed_gold_document: bool
    selected_missed_oracle_best: bool
    selected_deep_rank: bool
    baseline: CandidateSnapshot
    selected: CandidateSnapshot
    oracle: CandidateSnapshot


@dataclass(frozen=True)
class ErrorOutcomeSummary:
    """Overall reranker outcome summary."""

    question_count: int
    improved_count: int
    regressed_count: int
    tied_count: int
    improved_rate: float
    regressed_rate: float
    tied_rate: float
    average_delta_vs_top_candidate: float
    average_regression_delta: float
    average_improvement_delta: float
    selected_missed_gold_document_count: int
    selected_missed_gold_document_rate: float
    selected_missed_oracle_best_count: int
    selected_missed_oracle_best_rate: float
    selected_deep_rank_count: int
    selected_deep_rank_rate: float


@dataclass(frozen=True)
class SegmentErrorSummary:
    """Outcome summary for one route, split, or rank segment."""

    segment_name: str
    question_count: int
    improved_count: int
    regressed_count: int
    tied_count: int
    regressed_rate: float
    average_delta_vs_top_candidate: float
    selected_missed_gold_document_count: int
    selected_missed_gold_document_rate: float
    selected_deep_rank_count: int
    selected_deep_rank_rate: float


@dataclass(frozen=True)
class FeatureContrastSummary:
    """Mean selected-minus-baseline feature contrast for improved vs regressed cases."""

    feature_name: str
    improved_mean_selected_minus_baseline: float
    regressed_mean_selected_minus_baseline: float
    regressed_minus_improved: float


@dataclass(frozen=True)
class CandidateRerankerErrorAnalysisResult:
    """Full candidate reranker regression/error analysis."""

    model_name: str
    fold_count: int
    f1_tie_margin: float
    deep_rank_min: int
    summary: ErrorOutcomeSummary
    route_summaries: list[SegmentErrorSummary]
    split_summaries: list[SegmentErrorSummary]
    selected_rank_summaries: list[SegmentErrorSummary]
    feature_contrasts: list[FeatureContrastSummary]
    sample_cases: dict[str, list[CandidateRerankerErrorCase]]
    analysis_scope: str


def analyze_candidate_reranker_errors(
    rows: Sequence[Mapping[str, Any]],
    model_name: str = "logistic_best_candidate",
    fold_count: int = 5,
    f1_tie_margin: float = 0.0,
    deep_rank_min: int = 6,
    sample_limit: int = 10,
) -> CandidateRerankerErrorAnalysisResult:
    """Analyze grouped-CV candidate reranker improvements and regressions."""

    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")
    if deep_rank_min <= 1:
        raise ValueError("deep_rank_min must be greater than 1")

    row_index = _build_row_index(rows)
    selections = cross_validated_candidate_reranker_selections(
        rows=rows,
        model_name=model_name,
        fold_count=fold_count,
        f1_tie_margin=f1_tie_margin,
    )
    cases = [
        _build_error_case(
            selection=selection,
            rows_by_question=row_index.rows_by_question,
            rows_by_candidate_id=row_index.rows_by_candidate_id,
            f1_tie_margin=f1_tie_margin,
            deep_rank_min=deep_rank_min,
        )
        for selection in selections
    ]

    return CandidateRerankerErrorAnalysisResult(
        model_name=model_name,
        fold_count=fold_count,
        f1_tie_margin=f1_tie_margin,
        deep_rank_min=deep_rank_min,
        summary=_summarize_cases(cases),
        route_summaries=_segment_summaries(
            cases,
            segment_fn=lambda case: case.question_route,
        ),
        split_summaries=_segment_summaries(cases, segment_fn=lambda case: case.split),
        selected_rank_summaries=_segment_summaries(
            cases,
            segment_fn=lambda case: _rank_bucket_label(case.selected.candidate_rank),
        ),
        feature_contrasts=_feature_contrasts(cases),
        sample_cases=_sample_cases(cases, sample_limit=sample_limit),
        analysis_scope=(
            "Offline grouped-CV error analysis only. Models are refit per validation "
            "fold using runtime_features; gold_labels and metadata are used only for "
            "labels, metrics, and human-readable inspection."
        ),
    )


def candidate_reranker_error_analysis_to_dict(
    result: CandidateRerankerErrorAnalysisResult,
) -> dict[str, Any]:
    """Convert an error-analysis result to a JSON-safe dictionary."""

    return asdict(result)


@dataclass(frozen=True)
class _RowIndex:
    rows_by_question: dict[str, list[Mapping[str, Any]]]
    rows_by_candidate_id: dict[str, Mapping[str, Any]]


def _build_row_index(rows: Sequence[Mapping[str, Any]]) -> _RowIndex:
    rows_by_question: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    rows_by_candidate_id = {}
    for row in rows:
        question_key = _question_key(row["split"], row["question_id"])
        rows_by_question[question_key].append(row)
        rows_by_candidate_id[str(row["candidate_id"])] = row
    return _RowIndex(
        rows_by_question=dict(rows_by_question),
        rows_by_candidate_id=rows_by_candidate_id,
    )


def _build_error_case(
    selection: CandidateRerankerSelection,
    rows_by_question: Mapping[str, list[Mapping[str, Any]]],
    rows_by_candidate_id: Mapping[str, Mapping[str, Any]],
    f1_tie_margin: float,
    deep_rank_min: int,
) -> CandidateRerankerErrorCase:
    question_key = _question_key(selection.split, selection.question_id)
    question_rows = rows_by_question[question_key]
    gold_document_candidate_count = sum(_is_gold_document(row) for row in question_rows)
    selected = _snapshot(rows_by_candidate_id[selection.selected_candidate_id])
    baseline = _snapshot(rows_by_candidate_id[selection.baseline_candidate_id])
    oracle = _snapshot(rows_by_candidate_id[selection.oracle_candidate_id])
    f1_delta = round(
        selection.selected_candidate_token_f1 - selection.baseline_candidate_token_f1,
        4,
    )
    return CandidateRerankerErrorCase(
        split=selection.split,
        question_id=selection.question_id,
        question_route=selection.question_route,
        outcome=_outcome_label(f1_delta, f1_tie_margin=f1_tie_margin),
        f1_delta_vs_top_candidate=f1_delta,
        selected_model_score=selection.selected_candidate_score,
        gold_document_candidate_count=gold_document_candidate_count,
        selected_missed_gold_document=(
            gold_document_candidate_count > 0 and not selection.selected_is_gold_document
        ),
        selected_missed_oracle_best=not selection.selected_is_oracle_best_f1,
        selected_deep_rank=selection.selected_candidate_rank >= deep_rank_min,
        baseline=baseline,
        selected=selected,
        oracle=oracle,
    )


def _snapshot(row: Mapping[str, Any]) -> CandidateSnapshot:
    return CandidateSnapshot(
        candidate_id=str(row["candidate_id"]),
        candidate_rank=int(row["candidate_rank"]),
        candidate_token_f1=_candidate_token_f1(row),
        is_gold_document=_is_gold_document(row),
        runtime_features=dict(_runtime_features(row)),
        metadata=dict(_metadata(row)),
    )


def _summarize_cases(cases: Sequence[CandidateRerankerErrorCase]) -> ErrorOutcomeSummary:
    question_count = len(cases)
    improved_cases = [case for case in cases if case.outcome == "improved"]
    regressed_cases = [case for case in cases if case.outcome == "regressed"]
    tied_cases = [case for case in cases if case.outcome == "tied"]
    missed_gold_count = sum(case.selected_missed_gold_document for case in cases)
    missed_oracle_count = sum(case.selected_missed_oracle_best for case in cases)
    deep_rank_count = sum(case.selected_deep_rank for case in cases)
    return ErrorOutcomeSummary(
        question_count=question_count,
        improved_count=len(improved_cases),
        regressed_count=len(regressed_cases),
        tied_count=len(tied_cases),
        improved_rate=_ratio(len(improved_cases), question_count),
        regressed_rate=_ratio(len(regressed_cases), question_count),
        tied_rate=_ratio(len(tied_cases), question_count),
        average_delta_vs_top_candidate=_rounded_mean(
            [case.f1_delta_vs_top_candidate for case in cases]
        ),
        average_regression_delta=_rounded_mean(
            [case.f1_delta_vs_top_candidate for case in regressed_cases]
        ),
        average_improvement_delta=_rounded_mean(
            [case.f1_delta_vs_top_candidate for case in improved_cases]
        ),
        selected_missed_gold_document_count=missed_gold_count,
        selected_missed_gold_document_rate=_ratio(missed_gold_count, question_count),
        selected_missed_oracle_best_count=missed_oracle_count,
        selected_missed_oracle_best_rate=_ratio(missed_oracle_count, question_count),
        selected_deep_rank_count=deep_rank_count,
        selected_deep_rank_rate=_ratio(deep_rank_count, question_count),
    )


def _segment_summaries(
    cases: Sequence[CandidateRerankerErrorCase],
    segment_fn: Callable[[CandidateRerankerErrorCase], str],
) -> list[SegmentErrorSummary]:
    cases_by_segment: dict[str, list[CandidateRerankerErrorCase]] = defaultdict(list)
    for case in cases:
        cases_by_segment[str(segment_fn(case))].append(case)

    summaries = []
    for segment_name, segment_cases in cases_by_segment.items():
        outcome_counts = Counter(case.outcome for case in segment_cases)
        missed_gold_count = sum(case.selected_missed_gold_document for case in segment_cases)
        deep_rank_count = sum(case.selected_deep_rank for case in segment_cases)
        summaries.append(
            SegmentErrorSummary(
                segment_name=segment_name,
                question_count=len(segment_cases),
                improved_count=outcome_counts["improved"],
                regressed_count=outcome_counts["regressed"],
                tied_count=outcome_counts["tied"],
                regressed_rate=_ratio(outcome_counts["regressed"], len(segment_cases)),
                average_delta_vs_top_candidate=_rounded_mean(
                    [case.f1_delta_vs_top_candidate for case in segment_cases]
                ),
                selected_missed_gold_document_count=missed_gold_count,
                selected_missed_gold_document_rate=_ratio(
                    missed_gold_count,
                    len(segment_cases),
                ),
                selected_deep_rank_count=deep_rank_count,
                selected_deep_rank_rate=_ratio(deep_rank_count, len(segment_cases)),
            )
        )
    return sorted(
        summaries,
        key=lambda summary: (
            summary.regressed_rate,
            -summary.average_delta_vs_top_candidate,
            summary.question_count,
        ),
        reverse=True,
    )


def _feature_contrasts(
    cases: Sequence[CandidateRerankerErrorCase],
    limit: int = 25,
) -> list[FeatureContrastSummary]:
    numeric_feature_names = sorted(
        {
            feature_name
            for case in cases
            for feature_name, value in case.selected.runtime_features.items()
            if _is_numeric(value)
        }
    )
    contrasts = []
    for feature_name in numeric_feature_names:
        improved_deltas = [
            _feature_delta(case, feature_name)
            for case in cases
            if case.outcome == "improved"
        ]
        regressed_deltas = [
            _feature_delta(case, feature_name)
            for case in cases
            if case.outcome == "regressed"
        ]
        improved_mean = _rounded_mean(improved_deltas)
        regressed_mean = _rounded_mean(regressed_deltas)
        contrasts.append(
            FeatureContrastSummary(
                feature_name=feature_name,
                improved_mean_selected_minus_baseline=improved_mean,
                regressed_mean_selected_minus_baseline=regressed_mean,
                regressed_minus_improved=round(regressed_mean - improved_mean, 4),
            )
        )
    return sorted(
        contrasts,
        key=lambda contrast: abs(contrast.regressed_minus_improved),
        reverse=True,
    )[:limit]


def _feature_delta(case: CandidateRerankerErrorCase, feature_name: str) -> float:
    selected_value = case.selected.runtime_features.get(feature_name, 0.0)
    baseline_value = case.baseline.runtime_features.get(feature_name, 0.0)
    if not _is_numeric(selected_value) or not _is_numeric(baseline_value):
        return 0.0
    return float(selected_value) - float(baseline_value)


def _sample_cases(
    cases: Sequence[CandidateRerankerErrorCase],
    sample_limit: int,
) -> dict[str, list[CandidateRerankerErrorCase]]:
    if sample_limit == 0:
        return {
            "largest_improvements": [],
            "largest_regressions": [],
            "how_to_or_lookup_regressions": [],
            "gold_document_missed": [],
            "deep_rank_selections": [],
        }
    return {
        "largest_improvements": sorted(
            [case for case in cases if case.outcome == "improved"],
            key=lambda case: case.f1_delta_vs_top_candidate,
            reverse=True,
        )[:sample_limit],
        "largest_regressions": sorted(
            [case for case in cases if case.outcome == "regressed"],
            key=lambda case: case.f1_delta_vs_top_candidate,
        )[:sample_limit],
        "how_to_or_lookup_regressions": sorted(
            [
                case
                for case in cases
                if case.question_route == "how_to_or_lookup"
                and case.outcome == "regressed"
            ],
            key=lambda case: case.f1_delta_vs_top_candidate,
        )[:sample_limit],
        "gold_document_missed": sorted(
            [case for case in cases if case.selected_missed_gold_document],
            key=lambda case: case.f1_delta_vs_top_candidate,
        )[:sample_limit],
        "deep_rank_selections": sorted(
            [case for case in cases if case.selected_deep_rank],
            key=lambda case: case.f1_delta_vs_top_candidate,
        )[:sample_limit],
    }


def _outcome_label(f1_delta: float, f1_tie_margin: float) -> str:
    if f1_delta > f1_tie_margin:
        return "improved"
    if f1_delta < -f1_tie_margin:
        return "regressed"
    return "tied"


def _question_key(split: Any, question_id: Any) -> str:
    return f"{split}::{question_id}"


def _runtime_features(row: Mapping[str, Any]) -> Mapping[str, Any]:
    runtime_features = row.get("runtime_features")
    if not isinstance(runtime_features, Mapping):
        raise ValueError("row runtime_features must be an object")
    return runtime_features


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("row metadata must be an object")
    return metadata


def _gold_labels(row: Mapping[str, Any]) -> Mapping[str, Any]:
    gold_labels = row.get("gold_labels")
    if not isinstance(gold_labels, Mapping):
        raise ValueError("row gold_labels must be an object")
    return gold_labels


def _candidate_token_f1(row: Mapping[str, Any]) -> float:
    return float(_gold_labels(row)["candidate_token_f1"])


def _is_gold_document(row: Mapping[str, Any]) -> bool:
    return bool(_gold_labels(row)["is_gold_document"])


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _rank_bucket_label(rank: int) -> str:
    if rank == 1:
        return "rank_1"
    if rank == 2:
        return "rank_2"
    if rank == 3:
        return "rank_3"
    if 4 <= rank <= 5:
        return "rank_4_5"
    if 6 <= rank <= 10:
        return "rank_6_10"
    return "rank_11_plus"


def _rounded_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(mean(values), 4)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)
