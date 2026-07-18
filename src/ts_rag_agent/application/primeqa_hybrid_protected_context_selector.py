from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CONTEXT_SELECTOR_PROTOCOL_ID = "primeqa_hybrid_protected_context_selector_v1"
CONTEXT_DEPTH = 10
CANDIDATE_POOL_DEPTH = 200
PROTECTED_PREFIX_DEPTHS = (3, 5, 7)
RUNTIME_FEATURE_NAMES = (
    "stage116_rrf_score",
    "baseline_rank_inverse",
    "current_query_overlap_count",
    "current_query_overlap_ratio",
    "current_query_overlap_combined_score",
    "route_hit_count",
    "lexical_route_hit_count",
    "dense_route_hit_count",
    "best_route_inverse_rank",
    "full_document_bm25_rank_inverse",
    "section_bm25_max_section_rollup_rank_inverse",
    "title_heading_weighted_bm25_rank_inverse",
    "special_token_boosted_bm25_rank_inverse",
    "query_title_token_overlap",
    "query_section_heading_overlap",
    "query_token_coverage",
    "query_body_token_coverage",
    "query_special_token_match_count",
    "title_special_token_match_count",
    "heading_special_token_match_count",
    "bm25_top10_indicator",
)

ModelFamily = Literal["pairwise_logistic", "pointwise_histogram_gbdt"]


@dataclass(frozen=True)
class ProtectedContextSelectorConfig:
    """One frozen Stage161 protected-prefix selector configuration."""

    config_id: str
    model_family: ModelFamily
    protected_prefix_depth: int
    context_depth: int = CONTEXT_DEPTH
    candidate_pool_depth: int = CANDIDATE_POOL_DEPTH

    def __post_init__(self) -> None:
        if self.protected_prefix_depth not in PROTECTED_PREFIX_DEPTHS:
            raise ValueError("protected prefix depth is outside the frozen Stage161 set")
        if self.context_depth != CONTEXT_DEPTH:
            raise ValueError("Stage161 context depth must remain 10")
        if self.candidate_pool_depth != CANDIDATE_POOL_DEPTH:
            raise ValueError("Stage161 candidate pool depth must remain 200")
        if self.protected_prefix_depth >= self.context_depth:
            raise ValueError("protected prefix must leave at least one learned slot")


@dataclass(frozen=True)
class ContextCandidateRecord:
    """One private in-memory runtime-visible candidate feature row."""

    sample_id: str
    fold_id: str
    document_id: str
    baseline_rank: int
    answerable: bool
    is_gold: bool
    features: Mapping[str, float]


@dataclass(frozen=True)
class ContextSelection:
    """Selected generation context plus structural safety observations."""

    selected: tuple[ContextCandidateRecord, ...]
    protected_prefix_violation_count: int
    tail_promotion_count: int


@dataclass(frozen=True)
class ScorerFitSummary:
    """Public-safe aggregate facts from one scorer fit."""

    model_family: str
    training_group_count: int
    positive_candidate_count: int
    negative_candidate_count: int
    training_example_count: int
    feature_count: int


class RuntimeCandidateScorer(Protocol):
    """Polymorphic scorer shared by training validation and future runtime work."""

    def fit(
        self,
        records: Sequence[ContextCandidateRecord],
        *,
        protected_prefix_depth: int,
    ) -> ScorerFitSummary: ...

    def score(self, records: Sequence[ContextCandidateRecord]) -> list[float]: ...


class PairwiseLogisticCandidateScorer:
    """Linear pairwise preference model with a scalar runtime utility."""

    def __init__(self, *, feature_names: Sequence[str] = RUNTIME_FEATURE_NAMES) -> None:
        self._feature_names = tuple(feature_names)
        self._pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        fit_intercept=False,
                        max_iter=2000,
                        random_state=0,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    def fit(
        self,
        records: Sequence[ContextCandidateRecord],
        *,
        protected_prefix_depth: int,
    ) -> ScorerFitSummary:
        differences: list[np.ndarray] = []
        labels: list[int] = []
        positive_count = 0
        negative_count = 0
        training_groups = 0
        for sample_records in _records_by_sample(records).values():
            eligible = [
                record for record in sample_records if record.baseline_rank > protected_prefix_depth
            ]
            positives = [record for record in eligible if record.is_gold]
            if not positives:
                continue
            positive = positives[0]
            negatives = _hard_negatives(eligible)
            if not negatives:
                continue
            positive_vector = _feature_vector(positive, self._feature_names)
            for negative in negatives:
                difference = positive_vector - _feature_vector(negative, self._feature_names)
                differences.extend((difference, -difference))
                labels.extend((1, 0))
            training_groups += 1
            positive_count += 1
            negative_count += len(negatives)
        if not differences:
            raise ValueError("pairwise logistic scorer has no recoverable training pairs")
        matrix = np.vstack(differences)
        self._pipeline.fit(matrix, np.asarray(labels, dtype=np.int8))
        return ScorerFitSummary(
            model_family="pairwise_logistic",
            training_group_count=training_groups,
            positive_candidate_count=positive_count,
            negative_candidate_count=negative_count,
            training_example_count=len(labels),
            feature_count=len(self._feature_names),
        )

    def score(self, records: Sequence[ContextCandidateRecord]) -> list[float]:
        if not records:
            return []
        matrix = np.vstack([_feature_vector(record, self._feature_names) for record in records])
        return [float(value) for value in self._pipeline.decision_function(matrix)]


class PointwiseHistogramCandidateScorer:
    """Balanced histogram GBDT trained on recoverable gold and hard negatives."""

    def __init__(self, *, feature_names: Sequence[str] = RUNTIME_FEATURE_NAMES) -> None:
        self._feature_names = tuple(feature_names)
        self._model = HistGradientBoostingClassifier(
            class_weight="balanced",
            l2_regularization=0.1,
            learning_rate=0.08,
            max_iter=120,
            max_leaf_nodes=15,
            random_state=0,
        )

    def fit(
        self,
        records: Sequence[ContextCandidateRecord],
        *,
        protected_prefix_depth: int,
    ) -> ScorerFitSummary:
        selected: list[ContextCandidateRecord] = []
        labels: list[int] = []
        positive_count = 0
        negative_count = 0
        training_groups = 0
        for sample_records in _records_by_sample(records).values():
            eligible = [
                record for record in sample_records if record.baseline_rank > protected_prefix_depth
            ]
            positives = [record for record in eligible if record.is_gold]
            if not positives:
                continue
            negatives = _hard_negatives(eligible)
            if not negatives:
                continue
            selected.append(positives[0])
            labels.append(1)
            selected.extend(negatives)
            labels.extend([0] * len(negatives))
            positive_count += 1
            negative_count += len(negatives)
            training_groups += 1
        if len(set(labels)) != 2:
            raise ValueError("histogram scorer needs recoverable gold and negative candidates")
        matrix = np.vstack([_feature_vector(record, self._feature_names) for record in selected])
        self._model.fit(matrix, np.asarray(labels, dtype=np.int8))
        return ScorerFitSummary(
            model_family="pointwise_histogram_gbdt",
            training_group_count=training_groups,
            positive_candidate_count=positive_count,
            negative_candidate_count=negative_count,
            training_example_count=len(labels),
            feature_count=len(self._feature_names),
        )

    def score(self, records: Sequence[ContextCandidateRecord]) -> list[float]:
        if not records:
            return []
        matrix = np.vstack([_feature_vector(record, self._feature_names) for record in records])
        probabilities = self._model.predict_proba(matrix)
        positive_index = list(self._model.classes_).index(1)
        return [float(row[positive_index]) for row in probabilities]


class ProtectedPrefixContextSelector:
    """Keep the original RRF prefix and learn only the remaining context slots."""

    def __init__(
        self,
        *,
        config: ProtectedContextSelectorConfig,
        scorer: RuntimeCandidateScorer,
    ) -> None:
        self._config = config
        self._scorer = scorer

    @property
    def config(self) -> ProtectedContextSelectorConfig:
        return self._config

    def select(self, records: Sequence[ContextCandidateRecord]) -> ContextSelection:
        ordered = sorted(records, key=lambda record: record.baseline_rank)
        if len(ordered) != self._config.candidate_pool_depth:
            raise ValueError("Stage161 selector requires an exact top200 candidate pool")
        protected = ordered[: self._config.protected_prefix_depth]
        remainder = ordered[self._config.protected_prefix_depth :]
        scores = self._scorer.score(remainder)
        learned_slots = self._config.context_depth - self._config.protected_prefix_depth
        selected_tail = [
            record
            for _, record in sorted(
                zip(scores, remainder, strict=True),
                key=lambda item: (-item[0], item[1].baseline_rank, item[1].document_id),
            )[:learned_slots]
        ]
        selected = tuple([*protected, *selected_tail])
        expected_prefix = tuple(record.document_id for record in protected)
        observed_prefix = tuple(
            record.document_id for record in selected[: self._config.protected_prefix_depth]
        )
        return ContextSelection(
            selected=selected,
            protected_prefix_violation_count=int(observed_prefix != expected_prefix),
            tail_promotion_count=sum(
                record.baseline_rank > self._config.context_depth for record in selected
            ),
        )


def frozen_stage161_selector_configs() -> tuple[ProtectedContextSelectorConfig, ...]:
    """Return the six user-authorized Stage161 candidate configurations."""

    configs = []
    for prefix_depth in PROTECTED_PREFIX_DEPTHS:
        configs.extend(
            (
                ProtectedContextSelectorConfig(
                    config_id=f"rrf_prefix{prefix_depth}_pairwise_logistic_top200_v1",
                    model_family="pairwise_logistic",
                    protected_prefix_depth=prefix_depth,
                ),
                ProtectedContextSelectorConfig(
                    config_id=f"rrf_prefix{prefix_depth}_histogram_gbdt_top200_v1",
                    model_family="pointwise_histogram_gbdt",
                    protected_prefix_depth=prefix_depth,
                ),
            )
        )
    return tuple(configs)


def create_candidate_scorer(model_family: ModelFamily) -> RuntimeCandidateScorer:
    """Create one deterministic scorer for a frozen model family."""

    if model_family == "pairwise_logistic":
        return PairwiseLogisticCandidateScorer()
    if model_family == "pointwise_histogram_gbdt":
        return PointwiseHistogramCandidateScorer()
    raise ValueError(f"Unsupported Stage161 model family: {model_family}")


def select_original_rrf_top10(
    records: Sequence[ContextCandidateRecord],
) -> ContextSelection:
    """Return the untouched Stage116 RRF top10 control context."""

    ordered = sorted(records, key=lambda record: record.baseline_rank)
    selected = tuple(ordered[:CONTEXT_DEPTH])
    return ContextSelection(
        selected=selected,
        protected_prefix_violation_count=0,
        tail_promotion_count=0,
    )


def select_current_query_overlap_top10(
    records: Sequence[ContextCandidateRecord],
) -> ContextSelection:
    """Reproduce the current Stage160 query-overlap Top10 context control."""

    selected = tuple(
        sorted(
            records,
            key=lambda record: (
                -record.features.get("current_query_overlap_combined_score", 0.0),
                record.baseline_rank,
                record.document_id,
            ),
        )[:CONTEXT_DEPTH]
    )
    return ContextSelection(
        selected=selected,
        protected_prefix_violation_count=0,
        tail_promotion_count=sum(record.baseline_rank > CONTEXT_DEPTH for record in selected),
    )


def records_by_sample(
    records: Sequence[ContextCandidateRecord],
) -> dict[str, list[ContextCandidateRecord]]:
    """Group private candidate rows without changing within-sample rank order."""

    return _records_by_sample(records)


def _records_by_sample(
    records: Sequence[ContextCandidateRecord],
) -> dict[str, list[ContextCandidateRecord]]:
    grouped: dict[str, list[ContextCandidateRecord]] = defaultdict(list)
    for record in records:
        grouped[record.sample_id].append(record)
    return {
        sample_id: sorted(sample_records, key=lambda item: item.baseline_rank)
        for sample_id, sample_records in grouped.items()
    }


def _hard_negatives(
    eligible: Sequence[ContextCandidateRecord],
) -> list[ContextCandidateRecord]:
    non_gold = [record for record in eligible if not record.is_gold]
    by_baseline = sorted(non_gold, key=lambda record: record.baseline_rank)[:20]
    by_overlap = sorted(
        non_gold,
        key=lambda record: (
            -record.features.get("current_query_overlap_combined_score", 0.0),
            record.baseline_rank,
            record.document_id,
        ),
    )[:20]
    selected: dict[str, ContextCandidateRecord] = {}
    for record in [*by_baseline, *by_overlap]:
        selected.setdefault(record.document_id, record)
    return list(selected.values())


def _feature_vector(
    record: ContextCandidateRecord,
    feature_names: Sequence[str],
) -> np.ndarray:
    return np.asarray(
        [float(record.features.get(feature_name, 0.0)) for feature_name in feature_names],
        dtype=np.float64,
    )
