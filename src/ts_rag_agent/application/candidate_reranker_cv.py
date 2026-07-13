from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CandidateRerankerExample:
    """One candidate row prepared for offline reranker CV."""

    split: str
    question_id: str
    candidate_id: str
    candidate_rank: int
    question_route: str
    runtime_features: dict[str, Any]
    candidate_token_f1: float
    is_best_candidate_for_question: bool
    is_gold_document: bool

    @property
    def question_key(self) -> str:
        return f"{self.split}::{self.question_id}"


@dataclass(frozen=True)
class CandidateRerankerSelection:
    """Baseline, selected, and oracle candidates for one validation question."""

    split: str
    question_id: str
    question_route: str
    baseline_candidate_id: str
    baseline_candidate_rank: int
    baseline_candidate_token_f1: float
    selected_candidate_id: str
    selected_candidate_rank: int
    selected_candidate_token_f1: float
    selected_candidate_score: float
    selected_is_gold_document: bool
    selected_is_oracle_best_f1: bool
    oracle_candidate_id: str
    oracle_candidate_rank: int
    oracle_candidate_token_f1: float


@dataclass(frozen=True)
class CandidateRerankerEvaluationMetrics:
    """Question-level aggregate selection metrics."""

    question_count: int
    baseline_average_token_f1: float
    selected_average_token_f1: float
    oracle_average_token_f1: float
    average_delta_vs_top_candidate: float
    oracle_gap_closed_rate: float
    selected_best_candidate_count: int
    selected_best_candidate_rate: float
    selected_gold_document_candidate_count: int
    selected_gold_document_candidate_rate: float
    selected_top_rank_count: int
    selected_top_rank_rate: float
    f1_improved_count: int
    f1_regressed_count: int
    f1_tied_count: int
    selected_rank_distribution: dict[str, int]


@dataclass(frozen=True)
class CandidateRerankerFoldResult:
    """One validation fold result for one reranker model."""

    fold_index: int
    train_question_count: int
    validation_question_count: int
    metrics: CandidateRerankerEvaluationMetrics


@dataclass(frozen=True)
class CandidateRerankerSegmentMetrics:
    """Route or split metrics for one reranker model."""

    segment_name: str
    question_count: int
    baseline_average_token_f1: float
    selected_average_token_f1: float
    oracle_average_token_f1: float
    average_delta_vs_top_candidate: float
    oracle_gap_closed_rate: float
    selected_best_candidate_rate: float
    selected_gold_document_candidate_rate: float
    selected_top_rank_rate: float


@dataclass(frozen=True)
class CandidateRerankerModelCVResult:
    """Cross-validation result for one candidate reranker model."""

    model_name: str
    target_name: str
    aggregate_validation: CandidateRerankerEvaluationMetrics
    folds: list[CandidateRerankerFoldResult]
    route_metrics: list[CandidateRerankerSegmentMetrics]
    split_metrics: list[CandidateRerankerSegmentMetrics]


@dataclass(frozen=True)
class CandidateRerankerCVResult:
    """Full cross-validation result across candidate reranker baselines."""

    fold_count: int
    model_names: list[str]
    best_model_name: str
    selection_metric: str
    feature_contract: str
    f1_tie_margin: float
    models: list[CandidateRerankerModelCVResult]


@dataclass(frozen=True)
class VisualizationArtifact:
    """One generated visualization file."""

    name: str
    path: str


class CandidateRerankerScorer(ABC):
    """Trainable scorer interface for candidate-level runtime features."""

    name: str
    target_name: str

    @abstractmethod
    def fit(self, examples: Sequence[CandidateRerankerExample]) -> CandidateRerankerScorer:
        """Fit the scorer on training examples."""

    @abstractmethod
    def score(self, examples: Sequence[CandidateRerankerExample]) -> list[float]:
        """Score validation examples, higher means more likely to be selected."""


class LogisticBestCandidateScorer(CandidateRerankerScorer):
    """Logistic baseline trained to identify the best candidate in a question pool."""

    name = "logistic_best_candidate"
    target_name = "is_best_candidate_for_question"

    def __init__(self) -> None:
        self._pipeline = Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                    ),
                ),
            ]
        )

    def fit(self, examples: Sequence[CandidateRerankerExample]) -> CandidateRerankerScorer:
        labels = [int(example.is_best_candidate_for_question) for example in examples]
        if len(set(labels)) < 2:
            raise ValueError("logistic_best_candidate needs both positive and negative labels")
        self._pipeline.fit(_feature_dicts(examples), labels)
        return self

    def score(self, examples: Sequence[CandidateRerankerExample]) -> list[float]:
        probabilities = self._pipeline.predict_proba(_feature_dicts(examples))
        positive_class_index = list(self._pipeline.classes_).index(1)
        return [float(row[positive_class_index]) for row in probabilities]


class RidgeTokenF1Scorer(CandidateRerankerScorer):
    """Ridge baseline trained to predict candidate token F1."""

    name = "ridge_candidate_token_f1"
    target_name = "candidate_token_f1"

    def __init__(self) -> None:
        self._pipeline = Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )

    def fit(self, examples: Sequence[CandidateRerankerExample]) -> CandidateRerankerScorer:
        labels = [example.candidate_token_f1 for example in examples]
        self._pipeline.fit(_feature_dicts(examples), labels)
        return self

    def score(self, examples: Sequence[CandidateRerankerExample]) -> list[float]:
        return [float(score) for score in self._pipeline.predict(_feature_dicts(examples))]


SCORER_FACTORIES = {
    LogisticBestCandidateScorer.name: LogisticBestCandidateScorer,
    RidgeTokenF1Scorer.name: RidgeTokenF1Scorer,
}
DEFAULT_MODEL_NAMES = (
    LogisticBestCandidateScorer.name,
    RidgeTokenF1Scorer.name,
)


def cross_validate_candidate_rerankers(
    rows: Sequence[Mapping[str, Any]],
    fold_count: int = 5,
    model_names: Sequence[str] = DEFAULT_MODEL_NAMES,
    f1_tie_margin: float = 0.0,
) -> CandidateRerankerCVResult:
    """Cross-validate baseline candidate rerankers over grouped questions."""

    examples = candidate_reranker_rows_to_examples(rows)
    normalized_model_names = _normalize_model_names(model_names)
    _validate_cv_options(
        examples=examples,
        fold_count=fold_count,
        f1_tie_margin=f1_tie_margin,
    )

    question_groups = _group_examples_by_question(examples)
    folds = _build_deterministic_folds(list(question_groups), fold_count)
    model_results = [
        _cross_validate_model(
            model_name=model_name,
            question_groups=question_groups,
            folds=folds,
            f1_tie_margin=f1_tie_margin,
        )
        for model_name in normalized_model_names
    ]
    best_model = max(model_results, key=_model_selection_key)

    return CandidateRerankerCVResult(
        fold_count=fold_count,
        model_names=list(normalized_model_names),
        best_model_name=best_model.model_name,
        selection_metric=(
            "max aggregate average_delta_vs_top_candidate, then oracle_gap_closed_rate, "
            "then selected_best_candidate_rate"
        ),
        feature_contract="Only row.runtime_features are used for model fitting and scoring.",
        f1_tie_margin=f1_tie_margin,
        models=model_results,
    )


def candidate_reranker_rows_to_examples(
    rows: Sequence[Mapping[str, Any]],
) -> list[CandidateRerankerExample]:
    """Convert raw JSONL row dictionaries to typed examples."""

    if not rows:
        raise ValueError("rows must not be empty")

    examples = []
    for row in rows:
        runtime_features = row.get("runtime_features")
        gold_labels = row.get("gold_labels")
        if not isinstance(runtime_features, Mapping):
            raise ValueError("row runtime_features must be an object")
        if not isinstance(gold_labels, Mapping):
            raise ValueError("row gold_labels must be an object")

        examples.append(
            CandidateRerankerExample(
                split=str(row["split"]),
                question_id=str(row["question_id"]),
                candidate_id=str(row["candidate_id"]),
                candidate_rank=int(row["candidate_rank"]),
                question_route=str(runtime_features["question_route"]),
                runtime_features=dict(runtime_features),
                candidate_token_f1=float(gold_labels["candidate_token_f1"]),
                is_best_candidate_for_question=bool(
                    gold_labels["is_best_candidate_for_question"]
                ),
                is_gold_document=bool(gold_labels["is_gold_document"]),
            )
        )
    return examples


def candidate_reranker_cv_result_to_dict(
    result: CandidateRerankerCVResult,
) -> dict[str, Any]:
    """Convert a CV result to a JSON-safe dictionary."""

    return asdict(result)


def write_cv_visualizations(
    result: CandidateRerankerCVResult,
    output_dir: Path,
) -> list[VisualizationArtifact]:
    """Write compact SVG charts for the main CV result."""

    output_dir.mkdir(parents=True, exist_ok=True)
    best_model = _find_model(result, result.best_model_name)
    charts = {
        "candidate_reranker_model_delta.svg": _render_bar_chart_svg(
            title="CV average F1 delta by model",
            bars=[
                _Bar(
                    label=model.model_name,
                    value=model.aggregate_validation.average_delta_vs_top_candidate,
                    value_label=(
                        f"{model.aggregate_validation.average_delta_vs_top_candidate:+.4f}"
                    ),
                )
                for model in result.models
            ],
            x_label="average delta vs original top candidate",
        ),
        "candidate_reranker_model_gap_closed.svg": _render_bar_chart_svg(
            title="CV oracle gap closed by model",
            bars=[
                _Bar(
                    label=model.model_name,
                    value=model.aggregate_validation.oracle_gap_closed_rate,
                    value_label=f"{model.aggregate_validation.oracle_gap_closed_rate:.1%}",
                )
                for model in result.models
            ],
            x_label="oracle gap closed",
        ),
        "candidate_reranker_best_model_route_delta.svg": _render_bar_chart_svg(
            title=f"{best_model.model_name} average F1 delta by route",
            bars=[
                _Bar(
                    label=route.segment_name,
                    value=route.average_delta_vs_top_candidate,
                    value_label=(
                        f"{route.average_delta_vs_top_candidate:+.4f} "
                        f"(n={route.question_count})"
                    ),
                )
                for route in best_model.route_metrics
            ],
            x_label="average delta vs original top candidate",
        ),
        "candidate_reranker_best_model_selected_rank.svg": _render_bar_chart_svg(
            title=f"{best_model.model_name} selected rank distribution",
            bars=[
                _Bar(
                    label=rank,
                    value=float(count),
                    value_label=str(count),
                )
                for rank, count in (
                    best_model.aggregate_validation.selected_rank_distribution.items()
                )
            ],
            x_label="selected validation questions",
        ),
    }

    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(VisualizationArtifact(name=filename, path=str(path)))
    return artifacts


def _cross_validate_model(
    model_name: str,
    question_groups: Mapping[str, list[CandidateRerankerExample]],
    folds: Sequence[Sequence[str]],
    f1_tie_margin: float,
) -> CandidateRerankerModelCVResult:
    fold_results = []
    all_selections = []
    for fold_index, validation_question_keys in enumerate(folds):
        validation_key_set = set(validation_question_keys)
        train_examples = [
            example
            for question_key, examples in question_groups.items()
            if question_key not in validation_key_set
            for example in examples
        ]
        validation_groups = {
            question_key: examples
            for question_key, examples in question_groups.items()
            if question_key in validation_key_set
        }
        scorer = SCORER_FACTORIES[model_name]().fit(train_examples)
        selections = _select_validation_candidates(
            scorer=scorer,
            validation_groups=validation_groups,
        )
        all_selections.extend(selections)
        fold_results.append(
            CandidateRerankerFoldResult(
                fold_index=fold_index,
                train_question_count=len(question_groups) - len(validation_groups),
                validation_question_count=len(validation_groups),
                metrics=_evaluate_selections(selections, f1_tie_margin=f1_tie_margin),
            )
        )

    return CandidateRerankerModelCVResult(
        model_name=model_name,
        target_name=SCORER_FACTORIES[model_name].target_name,
        aggregate_validation=_evaluate_selections(
            all_selections,
            f1_tie_margin=f1_tie_margin,
        ),
        folds=fold_results,
        route_metrics=_segment_metrics(
            selections=all_selections,
            segment_fn=lambda selection: selection.question_route,
            f1_tie_margin=f1_tie_margin,
        ),
        split_metrics=_segment_metrics(
            selections=all_selections,
            segment_fn=lambda selection: selection.split,
            f1_tie_margin=f1_tie_margin,
        ),
    )


def _select_validation_candidates(
    scorer: CandidateRerankerScorer,
    validation_groups: Mapping[str, list[CandidateRerankerExample]],
) -> list[CandidateRerankerSelection]:
    validation_examples = [
        example
        for question_key in sorted(validation_groups)
        for example in validation_groups[question_key]
    ]
    scores_by_candidate_id = {
        example.candidate_id: score
        for example, score in zip(
            validation_examples,
            scorer.score(validation_examples),
            strict=True,
        )
    }

    selections = []
    for question_key in sorted(validation_groups):
        question_examples = validation_groups[question_key]
        baseline = min(question_examples, key=lambda example: example.candidate_rank)
        oracle = max(
            question_examples,
            key=lambda example: (example.candidate_token_f1, -example.candidate_rank),
        )
        selected = max(
            question_examples,
            key=lambda example: (
                scores_by_candidate_id[example.candidate_id],
                -example.candidate_rank,
            ),
        )
        selections.append(
            CandidateRerankerSelection(
                split=selected.split,
                question_id=selected.question_id,
                question_route=selected.question_route,
                baseline_candidate_id=baseline.candidate_id,
                baseline_candidate_rank=baseline.candidate_rank,
                baseline_candidate_token_f1=baseline.candidate_token_f1,
                selected_candidate_id=selected.candidate_id,
                selected_candidate_rank=selected.candidate_rank,
                selected_candidate_token_f1=selected.candidate_token_f1,
                selected_candidate_score=round(
                    scores_by_candidate_id[selected.candidate_id],
                    6,
                ),
                selected_is_gold_document=selected.is_gold_document,
                selected_is_oracle_best_f1=(
                    selected.candidate_token_f1 == oracle.candidate_token_f1
                ),
                oracle_candidate_id=oracle.candidate_id,
                oracle_candidate_rank=oracle.candidate_rank,
                oracle_candidate_token_f1=oracle.candidate_token_f1,
            )
        )
    return selections


def _evaluate_selections(
    selections: Sequence[CandidateRerankerSelection],
    f1_tie_margin: float,
) -> CandidateRerankerEvaluationMetrics:
    question_count = len(selections)
    baseline_f1_values = [
        selection.baseline_candidate_token_f1 for selection in selections
    ]
    selected_f1_values = [
        selection.selected_candidate_token_f1 for selection in selections
    ]
    oracle_f1_values = [selection.oracle_candidate_token_f1 for selection in selections]
    selected_rank_counts = Counter(
        _rank_bucket_label(selection.selected_candidate_rank)
        for selection in selections
    )
    selected_minus_baseline = [
        selected - baseline
        for selected, baseline in zip(
            selected_f1_values,
            baseline_f1_values,
            strict=True,
        )
    ]
    oracle_minus_baseline = [
        oracle - baseline
        for oracle, baseline in zip(
            oracle_f1_values,
            baseline_f1_values,
            strict=True,
        )
    ]
    improved_count = sum(delta > f1_tie_margin for delta in selected_minus_baseline)
    regressed_count = sum(delta < -f1_tie_margin for delta in selected_minus_baseline)

    return CandidateRerankerEvaluationMetrics(
        question_count=question_count,
        baseline_average_token_f1=_rounded_mean(baseline_f1_values),
        selected_average_token_f1=_rounded_mean(selected_f1_values),
        oracle_average_token_f1=_rounded_mean(oracle_f1_values),
        average_delta_vs_top_candidate=_rounded_mean(selected_minus_baseline),
        oracle_gap_closed_rate=_safe_ratio(
            sum(selected_minus_baseline),
            sum(oracle_minus_baseline),
        ),
        selected_best_candidate_count=sum(
            selection.selected_is_oracle_best_f1 for selection in selections
        ),
        selected_best_candidate_rate=_ratio(
            sum(selection.selected_is_oracle_best_f1 for selection in selections),
            question_count,
        ),
        selected_gold_document_candidate_count=sum(
            selection.selected_is_gold_document for selection in selections
        ),
        selected_gold_document_candidate_rate=_ratio(
            sum(selection.selected_is_gold_document for selection in selections),
            question_count,
        ),
        selected_top_rank_count=sum(
            selection.selected_candidate_rank == 1 for selection in selections
        ),
        selected_top_rank_rate=_ratio(
            sum(selection.selected_candidate_rank == 1 for selection in selections),
            question_count,
        ),
        f1_improved_count=improved_count,
        f1_regressed_count=regressed_count,
        f1_tied_count=question_count - improved_count - regressed_count,
        selected_rank_distribution=dict(sorted(selected_rank_counts.items())),
    )


def _segment_metrics(
    selections: Sequence[CandidateRerankerSelection],
    segment_fn,
    f1_tie_margin: float,
) -> list[CandidateRerankerSegmentMetrics]:
    selections_by_segment: dict[str, list[CandidateRerankerSelection]] = defaultdict(list)
    for selection in selections:
        selections_by_segment[str(segment_fn(selection))].append(selection)

    metrics = []
    for segment_name, segment_selections in selections_by_segment.items():
        summary = _evaluate_selections(
            segment_selections,
            f1_tie_margin=f1_tie_margin,
        )
        metrics.append(
            CandidateRerankerSegmentMetrics(
                segment_name=segment_name,
                question_count=summary.question_count,
                baseline_average_token_f1=summary.baseline_average_token_f1,
                selected_average_token_f1=summary.selected_average_token_f1,
                oracle_average_token_f1=summary.oracle_average_token_f1,
                average_delta_vs_top_candidate=summary.average_delta_vs_top_candidate,
                oracle_gap_closed_rate=summary.oracle_gap_closed_rate,
                selected_best_candidate_rate=summary.selected_best_candidate_rate,
                selected_gold_document_candidate_rate=(
                    summary.selected_gold_document_candidate_rate
                ),
                selected_top_rank_rate=summary.selected_top_rank_rate,
            )
        )
    return sorted(
        metrics,
        key=lambda metric: (
            metric.average_delta_vs_top_candidate,
            metric.question_count,
        ),
        reverse=True,
    )


def _feature_dicts(
    examples: Sequence[CandidateRerankerExample],
) -> list[dict[str, Any]]:
    return [dict(example.runtime_features) for example in examples]


def _group_examples_by_question(
    examples: Sequence[CandidateRerankerExample],
) -> dict[str, list[CandidateRerankerExample]]:
    groups: dict[str, list[CandidateRerankerExample]] = defaultdict(list)
    for example in examples:
        groups[example.question_key].append(example)
    return {
        question_key: sorted(
            question_examples,
            key=lambda example: example.candidate_rank,
        )
        for question_key, question_examples in groups.items()
    }


def _build_deterministic_folds(
    question_keys: list[str],
    fold_count: int,
) -> list[list[str]]:
    folds: list[list[str]] = [[] for _ in range(fold_count)]
    for index, question_key in enumerate(sorted(question_keys)):
        folds[index % fold_count].append(question_key)
    return folds


def _normalize_model_names(model_names: Sequence[str]) -> tuple[str, ...]:
    normalized_names = tuple(name.strip() for name in model_names if name.strip())
    if not normalized_names:
        raise ValueError("model_names must not be empty")
    unknown_names = sorted(set(normalized_names) - set(SCORER_FACTORIES))
    if unknown_names:
        raise ValueError(f"Unknown candidate reranker model(s): {', '.join(unknown_names)}")
    return normalized_names


def _validate_cv_options(
    examples: Sequence[CandidateRerankerExample],
    fold_count: int,
    f1_tie_margin: float,
) -> None:
    question_count = len({example.question_key for example in examples})
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if fold_count > question_count:
        raise ValueError("fold_count must be no larger than the number of questions")
    if f1_tie_margin < 0:
        raise ValueError("f1_tie_margin must be non-negative")
    if any(not 0.0 <= example.candidate_token_f1 <= 1.0 for example in examples):
        raise ValueError("candidate_token_f1 values must be between 0 and 1")


def _model_selection_key(
    result: CandidateRerankerModelCVResult,
) -> tuple[float, float, float, str]:
    aggregate = result.aggregate_validation
    return (
        aggregate.average_delta_vs_top_candidate,
        aggregate.oracle_gap_closed_rate,
        aggregate.selected_best_candidate_rate,
        result.model_name,
    )


def _find_model(
    result: CandidateRerankerCVResult,
    model_name: str,
) -> CandidateRerankerModelCVResult:
    for model in result.models:
        if model.model_name == model_name:
            return model
    raise ValueError(f"Unknown model in result: {model_name}")


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


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


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
    width = 980
    margin_left = 310
    margin_right = 190
    margin_top = 56
    margin_bottom = 56
    row_height = 38
    height = margin_top + margin_bottom + max(1, len(bars)) * row_height
    plot_width = width - margin_left - margin_right
    max_abs_value = max((abs(bar.value) for bar in bars), default=0.0)
    scale_denominator = max_abs_value if max_abs_value > 0 else 1.0
    zero_x = margin_left + (plot_width // 2 if any(bar.value < 0 for bar in bars) else 0)
    positive_plot_width = plot_width if zero_x == margin_left else plot_width // 2
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
