from __future__ import annotations

import gc
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

from ts_rag_agent.application import composition_safety_constrained_lambdamart as stage194
from ts_rag_agent.application.composition_action_audit import ActionAuditRow
from ts_rag_agent.application.composition_dual_target_policy import SelectedAction
from ts_rag_agent.application.composition_f1_representation_cv import (
    build_composition_feature_indices,
)
from ts_rag_agent.application.composition_gain_sensitive_ranking import (
    _binary_metrics,
    _fit_histogram_classifier,
    _fit_logistic_classifier,
    _SafetyHead,
    build_stage182_reference_rows,
    paired_selected_action_bootstrap,
    unavailable_selected_action_bootstrap,
)
from ts_rag_agent.application.composition_joint_constraint_ranking import (
    evaluate_selected_actions,
)

FeatureRepresentation = stage194.FeatureRepresentation
SafetyEstimator = stage194.SafetyEstimator
TreeProfile = stage194.TreeProfile
ProgressSink = Callable[[Mapping[str, Any]], None]

_REPRESENTATIONS = stage194._REPRESENTATIONS
_SAFETY_ESTIMATORS = stage194._SAFETY_ESTIMATORS
_TREE_PROFILES = stage194._TREE_PROFILES
_SCALE_POS_WEIGHTS = (1.0, 2.0, 4.0)
_SAFEST_PREFIX_SIZES = (2, 4, 8, 12, 16)
_POOL_CAP = 16
_F1_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SafetyFirstFrontierPolicySpec:
    """One frozen Stage 196 safety-first frontier configuration."""

    name: str
    pool_feature_representation: FeatureRepresentation
    pool_safety_estimator: SafetyEstimator
    gain_feature_representation: FeatureRepresentation
    gain_tree_profile: TreeProfile
    risk_feature_representation: FeatureRepresentation
    risk_tree_profile: TreeProfile
    scale_pos_weight: float
    safest_prefix_size: int

    @property
    def pool_bundle_name(self) -> str:
        return f"{self.pool_feature_representation}__{self.pool_safety_estimator}"

    @property
    def gain_bundle_name(self) -> str:
        return f"{self.gain_feature_representation}__{self.gain_tree_profile}"

    @property
    def risk_bundle_name(self) -> str:
        return (
            f"{self.risk_feature_representation}__{self.risk_tree_profile}__"
            f"{self.scale_pos_weight:.1f}"
        )


@dataclass(frozen=True)
class GainPrediction:
    row: ActionAuditRow
    score: float


@dataclass(frozen=True)
class UnsafePrediction:
    row: ActionAuditRow
    score: float


@dataclass(frozen=True)
class FrontierActionPrediction:
    row: ActionAuditRow
    citation_loss_probability: float
    f1_loss_probability: float
    gain_score: float
    unsafe_score: float


@dataclass(frozen=True)
class SafetyFirstFrontierDecision:
    question_key: str
    baseline: FrontierActionPrediction
    complete_pool: tuple[FrontierActionPrediction, ...]
    frontier: tuple[FrontierActionPrediction, ...]
    winner: FrontierActionPrediction
    strict_opportunity: bool
    action_count: int
    strict_action_count: int


@dataclass(frozen=True)
class RepresentationPredictions:
    safety_by_estimator: Mapping[SafetyEstimator, tuple[stage194.SafetyPrediction, ...]]
    gain_by_profile: Mapping[TreeProfile, tuple[GainPrediction, ...]]
    risk_by_profile_and_weight: Mapping[str, tuple[UnsafePrediction, ...]]


@dataclass(frozen=True)
class RepresentationPartitionResult:
    predictions: RepresentationPredictions
    feature_count: int
    model_fit_count: int
    tree_count: int
    group_contract_validation_count: int


@dataclass(frozen=True)
class SpecPartitionResult:
    """Predictions from only the four models required by one frozen policy spec."""

    safety_predictions: tuple[stage194.SafetyPrediction, ...]
    gain_predictions: tuple[GainPrediction, ...]
    risk_predictions: tuple[UnsafePrediction, ...]
    feature_count_by_representation: Mapping[str, int]
    model_fit_count: int
    tree_count: int
    group_contract_validation_count: int


class RepresentationFitPredictor(Protocol):
    def __call__(
        self,
        training_rows: Sequence[ActionAuditRow],
        heldout_rows: Sequence[ActionAuditRow],
        feature_index: Mapping[tuple[str, str], Mapping[str, Any]],
        representation: FeatureRepresentation,
    ) -> RepresentationPartitionResult: ...


@dataclass
class _FitTotals:
    model_fit_count: int = 0
    tree_count: int = 0
    gain_ranker_fit_count: int = 0
    unsafe_head_fit_count: int = 0
    pool_safety_fit_count: int = 0
    group_contract_validation_count: int = 0

    def add(self, result: RepresentationPartitionResult) -> None:
        self.model_fit_count += result.model_fit_count
        self.tree_count += result.tree_count
        self.gain_ranker_fit_count += 2
        self.unsafe_head_fit_count += 6
        self.pool_safety_fit_count += 4
        self.group_contract_validation_count += result.group_contract_validation_count


@dataclass
class _FrontierDiagnostics:
    question_count: int = 0
    strict_opportunity_question_count: int = 0
    pool_recalled_question_count: int = 0
    frontier_recalled_question_count: int = 0
    strict_selected_question_count: int = 0
    unsafe_selected_question_count: int = 0
    safe_zero_selected_question_count: int = 0
    action_count: int = 0
    pool_action_count: int = 0
    frontier_action_count: int = 0
    strict_action_count: int = 0
    retained_strict_action_count: int = 0
    frontier_retained_strict_action_count: int = 0
    unsafe_pool_action_count: int = 0
    unsafe_frontier_action_count: int = 0
    baseline_in_pool_question_count: int = 0
    baseline_in_frontier_question_count: int = 0

    def add_decision(self, decision: SafetyFirstFrontierDecision) -> None:
        strict_in_pool = any(row.row.strict_expected for row in decision.complete_pool)
        strict_in_frontier = any(row.row.strict_expected for row in decision.frontier)
        self.question_count += 1
        self.strict_opportunity_question_count += int(decision.strict_opportunity)
        self.pool_recalled_question_count += int(decision.strict_opportunity and strict_in_pool)
        self.frontier_recalled_question_count += int(
            decision.strict_opportunity and strict_in_frontier
        )
        self.strict_selected_question_count += int(
            decision.strict_opportunity and decision.winner.row.strict_expected
        )
        self.unsafe_selected_question_count += int(stage194._is_unsafe(decision.winner.row))
        self.safe_zero_selected_question_count += int(stage194._is_safe_zero(decision.winner.row))
        self.action_count += decision.action_count
        self.pool_action_count += len(decision.complete_pool)
        self.frontier_action_count += len(decision.frontier)
        self.strict_action_count += decision.strict_action_count
        self.retained_strict_action_count += sum(
            row.row.strict_expected for row in decision.complete_pool
        )
        self.frontier_retained_strict_action_count += sum(
            row.row.strict_expected for row in decision.frontier
        )
        self.unsafe_pool_action_count += sum(
            stage194._is_unsafe(row.row) for row in decision.complete_pool
        )
        self.unsafe_frontier_action_count += sum(
            stage194._is_unsafe(row.row) for row in decision.frontier
        )
        self.baseline_in_pool_question_count += int(decision.baseline in decision.complete_pool)
        self.baseline_in_frontier_question_count += int(decision.baseline in decision.frontier)

    def add_report(self, report: Mapping[str, Any]) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + int(report[name]))

    def report(self) -> dict[str, Any]:
        return {
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
            "strict_opportunity_pool_recall": _ratio(
                self.pool_recalled_question_count,
                self.strict_opportunity_question_count,
            ),
            "strict_opportunity_frontier_recall": _ratio(
                self.frontier_recalled_question_count,
                self.strict_opportunity_question_count,
            ),
            "conditional_ranker_strict_capture": _ratio(
                self.strict_selected_question_count,
                self.pool_recalled_question_count,
            ),
            "conditional_frontier_strict_capture": _ratio(
                self.strict_selected_question_count,
                self.frontier_recalled_question_count,
            ),
            "actual_strict_opportunity_capture": _ratio(
                self.strict_selected_question_count,
                self.strict_opportunity_question_count,
            ),
            "unsafe_selection_rate": _ratio(
                self.unsafe_selected_question_count,
                self.question_count,
            ),
            "unsafe_action_retention_rate": _ratio(
                self.unsafe_frontier_action_count,
                self.unsafe_pool_action_count,
            ),
            "mean_pool_size": _ratio(self.pool_action_count, self.question_count),
            "mean_frontier_size": _ratio(self.frontier_action_count, self.question_count),
            "baseline_in_pool_rate": _ratio(
                self.baseline_in_pool_question_count,
                self.question_count,
            ),
            "baseline_in_frontier_rate": _ratio(
                self.baseline_in_frontier_question_count,
                self.question_count,
            ),
        }


def stage196_policy_specs() -> tuple[SafetyFirstFrontierPolicySpec, ...]:
    specs = []
    for pool_representation in _REPRESENTATIONS:
        for safety_estimator in _SAFETY_ESTIMATORS:
            for gain_representation in _REPRESENTATIONS:
                for gain_profile in _TREE_PROFILES:
                    for risk_representation in _REPRESENTATIONS:
                        for risk_profile in _TREE_PROFILES:
                            for weight in _SCALE_POS_WEIGHTS:
                                for prefix in _SAFEST_PREFIX_SIZES:
                                    name = (
                                        f"pool_{pool_representation}__{safety_estimator}__"
                                        f"gain_{gain_representation}__{gain_profile}__"
                                        f"risk_{risk_representation}__{risk_profile}__"
                                        f"weight_{weight:.1f}__prefix_{prefix}"
                                    )
                                    specs.append(
                                        SafetyFirstFrontierPolicySpec(
                                            name,
                                            pool_representation,
                                            safety_estimator,
                                            gain_representation,
                                            gain_profile,
                                            risk_representation,
                                            risk_profile,
                                            weight,
                                            prefix,
                                        )
                                    )
    return tuple(specs)


def fit_predict_safety_first_representation(
    training_rows: Sequence[ActionAuditRow],
    heldout_rows: Sequence[ActionAuditRow],
    feature_index: Mapping[tuple[str, str], Mapping[str, Any]],
    representation: FeatureRepresentation,
) -> RepresentationPartitionResult:
    """Fit 12 frozen models while retaining at most one unsafe head at a time."""

    import lightgbm as lgb

    training = tuple(sorted(training_rows, key=stage194._row_key))
    heldout = tuple(sorted(heldout_rows, key=stage194._row_key))
    vectorizer = DictVectorizer(sparse=True)
    train_matrix = vectorizer.fit_transform(
        [dict(feature_index[stage194._row_key(row)]) for row in training]
    ).tocsr()
    heldout_matrix = vectorizer.transform(
        [dict(feature_index[stage194._row_key(row)]) for row in heldout]
    ).tocsr()
    weights = stage194._question_balanced_weights(training)
    labels = {
        "citation_loss": np.asarray([row.citation_delta < 0 for row in training], dtype=np.int8),
        "f1_loss": np.asarray([row.f1_delta < -_F1_TOLERANCE for row in training], dtype=np.int8),
        "unsafe": np.asarray([stage194._is_unsafe(row) for row in training], dtype=np.int8),
    }
    for name, values in labels.items():
        if len(set(values.tolist())) != 2:
            raise ValueError(f"Stage196 {name} target requires both classes")

    safety_predictions: dict[SafetyEstimator, tuple[stage194.SafetyPrediction, ...]] = {}
    scaler = StandardScaler(with_mean=False)
    logistic_train = scaler.fit_transform(train_matrix).tocsr()
    logistic_heldout = scaler.transform(heldout_matrix).tocsr()
    logistic_values = {}
    for target in ("citation_loss", "f1_loss"):
        head = _SafetyHead(
            _fit_logistic_classifier(logistic_train, labels[target], weights),
            False,
        )
        logistic_values[target] = head.predict(logistic_heldout)
        del head
    safety_predictions["class_balanced_logistic"] = _safety_predictions(
        heldout,
        logistic_values["citation_loss"],
        logistic_values["f1_loss"],
    )
    del logistic_train, logistic_heldout, scaler, logistic_values

    dense_train = train_matrix.toarray()
    histogram_values = {}
    for target in ("citation_loss", "f1_loss"):
        head = _SafetyHead(
            _fit_histogram_classifier(dense_train, labels[target], weights),
            True,
        )
        histogram_values[target] = head.predict(heldout_matrix)
        del head
    safety_predictions["histogram_gradient_boosting"] = _safety_predictions(
        heldout,
        histogram_values["citation_loss"],
        histogram_values["f1_loss"],
    )
    del dense_train, histogram_values
    gc.collect()

    common = _lightgbm_common_parameters()
    relevance = np.asarray([stage194._outcome_tier(row) for row in training], dtype=np.int8)
    group_sizes = stage194._question_group_sizes(training)
    gain_predictions = {}
    risk_predictions = {}
    tree_count = 0
    for profile_name, profile in _TREE_PROFILES.items():
        parameters = {**common, **profile}
        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            lambdarank_truncation_level=4,
            lambdarank_norm=True,
            label_gain=[0, 1, 4],
            **parameters,
        )
        ranker.fit(
            train_matrix,
            relevance,
            group=group_sizes,
            sample_weight=weights,
            eval_at=[1],
        )
        values = np.asarray(ranker.predict(heldout_matrix), dtype=np.float64)
        gain_predictions[profile_name] = tuple(
            GainPrediction(row, float(score)) for row, score in zip(heldout, values, strict=True)
        )
        tree_count += int(ranker.booster_.num_trees())
        del ranker, values

        for scale_pos_weight in _SCALE_POS_WEIGHTS:
            unsafe_head = lgb.LGBMClassifier(
                objective="binary",
                metric="binary_logloss",
                class_weight=None,
                is_unbalance=False,
                scale_pos_weight=scale_pos_weight,
                **parameters,
            )
            unsafe_head.fit(train_matrix, labels["unsafe"], sample_weight=weights)
            values = np.asarray(unsafe_head.predict_proba(heldout_matrix)[:, 1], dtype=np.float64)
            key = f"{profile_name}__{scale_pos_weight:.1f}"
            risk_predictions[key] = tuple(
                UnsafePrediction(row, float(score))
                for row, score in zip(heldout, values, strict=True)
            )
            tree_count += int(unsafe_head.booster_.num_trees())
            del unsafe_head, values
            gc.collect()

    return RepresentationPartitionResult(
        predictions=RepresentationPredictions(
            safety_predictions,
            gain_predictions,
            risk_predictions,
        ),
        feature_count=len(vectorizer.feature_names_),
        model_fit_count=12,
        tree_count=tree_count,
        group_contract_validation_count=1,
    )


def fit_predict_safety_first_spec(
    training_rows: Sequence[ActionAuditRow],
    heldout_rows: Sequence[ActionAuditRow],
    feature_indices: Mapping[
        FeatureRepresentation,
        Mapping[tuple[str, str], Mapping[str, Any]],
    ],
    spec: SafetyFirstFrontierPolicySpec,
) -> SpecPartitionResult:
    """Fit only the two safety heads, gain ranker, and unsafe head used by ``spec``."""

    import lightgbm as lgb

    training = tuple(sorted(training_rows, key=stage194._row_key))
    heldout = tuple(sorted(heldout_rows, key=stage194._row_key))
    matrices: dict[str, tuple[Any, Any]] = {}
    feature_counts: dict[str, int] = {}
    for representation in {
        spec.pool_feature_representation,
        spec.gain_feature_representation,
        spec.risk_feature_representation,
    }:
        vectorizer = DictVectorizer(sparse=True)
        train_matrix = vectorizer.fit_transform(
            [dict(feature_indices[representation][stage194._row_key(row)]) for row in training]
        ).tocsr()
        heldout_matrix = vectorizer.transform(
            [dict(feature_indices[representation][stage194._row_key(row)]) for row in heldout]
        ).tocsr()
        matrices[representation] = (train_matrix, heldout_matrix)
        feature_counts[representation] = len(vectorizer.feature_names_)

    weights = stage194._question_balanced_weights(training)
    labels = {
        "citation_loss": np.asarray([row.citation_delta < 0 for row in training], dtype=np.int8),
        "f1_loss": np.asarray([row.f1_delta < -_F1_TOLERANCE for row in training], dtype=np.int8),
        "unsafe": np.asarray([stage194._is_unsafe(row) for row in training], dtype=np.int8),
    }
    for name, values in labels.items():
        if len(set(values.tolist())) != 2:
            raise ValueError(f"Stage197 {name} target requires both classes")

    safety_train, safety_heldout = matrices[spec.pool_feature_representation]
    safety_values: dict[str, np.ndarray] = {}
    if spec.pool_safety_estimator == "class_balanced_logistic":
        scaler = StandardScaler(with_mean=False)
        fitted_train = scaler.fit_transform(safety_train).tocsr()
        fitted_heldout = scaler.transform(safety_heldout).tocsr()
        for target in ("citation_loss", "f1_loss"):
            head = _SafetyHead(
                _fit_logistic_classifier(fitted_train, labels[target], weights),
                False,
            )
            safety_values[target] = head.predict(fitted_heldout)
        del scaler, fitted_train, fitted_heldout
    elif spec.pool_safety_estimator == "histogram_gradient_boosting":
        fitted_train = safety_train.toarray()
        for target in ("citation_loss", "f1_loss"):
            head = _SafetyHead(
                _fit_histogram_classifier(fitted_train, labels[target], weights),
                True,
            )
            safety_values[target] = head.predict(safety_heldout)
        del fitted_train
    else:
        raise ValueError(f"Unsupported Stage197 safety estimator: {spec.pool_safety_estimator}")
    safety_predictions = _safety_predictions(
        heldout,
        safety_values["citation_loss"],
        safety_values["f1_loss"],
    )

    common = _lightgbm_common_parameters()
    gain_train, gain_heldout = matrices[spec.gain_feature_representation]
    gain_ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        lambdarank_truncation_level=4,
        lambdarank_norm=True,
        label_gain=[0, 1, 4],
        **{**common, **_TREE_PROFILES[spec.gain_tree_profile]},
    )
    gain_ranker.fit(
        gain_train,
        np.asarray([stage194._outcome_tier(row) for row in training], dtype=np.int8),
        group=stage194._question_group_sizes(training),
        sample_weight=weights,
        eval_at=[1],
    )
    gain_values = np.asarray(gain_ranker.predict(gain_heldout), dtype=np.float64)
    gain_predictions = tuple(
        GainPrediction(row, float(score)) for row, score in zip(heldout, gain_values, strict=True)
    )
    tree_count = int(gain_ranker.booster_.num_trees())

    risk_train, risk_heldout = matrices[spec.risk_feature_representation]
    unsafe_head = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        class_weight=None,
        is_unbalance=False,
        scale_pos_weight=spec.scale_pos_weight,
        **{**common, **_TREE_PROFILES[spec.risk_tree_profile]},
    )
    unsafe_head.fit(risk_train, labels["unsafe"], sample_weight=weights)
    risk_values = np.asarray(unsafe_head.predict_proba(risk_heldout)[:, 1], dtype=np.float64)
    risk_predictions = tuple(
        UnsafePrediction(row, float(score)) for row, score in zip(heldout, risk_values, strict=True)
    )
    tree_count += int(unsafe_head.booster_.num_trees())
    del matrices, gain_ranker, unsafe_head, gain_values, risk_values, safety_values
    gc.collect()
    return SpecPartitionResult(
        safety_predictions=safety_predictions,
        gain_predictions=gain_predictions,
        risk_predictions=risk_predictions,
        feature_count_by_representation=dict(sorted(feature_counts.items())),
        model_fit_count=4,
        tree_count=tree_count,
        group_contract_validation_count=1,
    )


def build_safety_first_frontier_decisions(
    safety_predictions: Sequence[stage194.SafetyPrediction],
    gain_predictions: Sequence[GainPrediction],
    risk_predictions: Sequence[UnsafePrediction],
    spec: SafetyFirstFrontierPolicySpec,
) -> tuple[SafetyFirstFrontierDecision, ...]:
    safety_grouped = stage194._group_predictions(safety_predictions)
    gain_index = {stage194._row_key(row.row): row for row in gain_predictions}
    risk_index = {stage194._row_key(row.row): row for row in risk_predictions}
    expected = {stage194._row_key(row.row) for row in safety_predictions}
    if expected != set(gain_index) or expected != set(risk_index):
        raise ValueError("Stage196 safety, gain, and risk prediction rows differ")

    decisions = []
    for question_key, question_safety in sorted(safety_grouped.items()):
        baselines = [row for row in question_safety if row.row.action.family == "baseline"]
        if len(baselines) != 1:
            raise ValueError("Stage196 requires one baseline action per question")
        ranked_safety = sorted(question_safety, key=stage194._safety_order_key)
        pool_index = {stage194._row_key(row.row): row for row in ranked_safety[:_POOL_CAP]}
        baseline_safety = baselines[0]
        pool_index[stage194._row_key(baseline_safety.row)] = baseline_safety
        pool_safety = tuple(sorted(pool_index.values(), key=stage194._safety_order_key))
        risk_order = sorted(
            pool_safety,
            key=lambda row: (
                risk_index[stage194._row_key(row.row)].score,
                row.row.action.action_id,
            ),
        )
        frontier_index = {
            stage194._row_key(row.row): row for row in risk_order[: spec.safest_prefix_size]
        }
        frontier_index[stage194._row_key(baseline_safety.row)] = baseline_safety
        complete_pool = tuple(_combine(row, gain_index, risk_index) for row in pool_safety)
        frontier = tuple(
            _combine(row, gain_index, risk_index)
            for row in sorted(frontier_index.values(), key=stage194._safety_order_key)
        )
        winner = min(
            frontier,
            key=lambda row: (
                -row.gain_score,
                row.unsafe_score,
                row.row.action.action_id,
            ),
        )
        baseline = next(row for row in frontier if row.row.action.family == "baseline")
        decisions.append(
            SafetyFirstFrontierDecision(
                question_key,
                baseline,
                complete_pool,
                frontier,
                winner,
                any(row.row.strict_expected for row in question_safety),
                len(question_safety),
                sum(row.row.strict_expected for row in question_safety),
            )
        )
    return tuple(decisions)


def evaluate_safety_first_frontier_policy(
    safety_predictions: Sequence[stage194.SafetyPrediction],
    gain_predictions: Sequence[GainPrediction],
    risk_predictions: Sequence[UnsafePrediction],
    spec: SafetyFirstFrontierPolicySpec,
    *,
    expected_fold_ids: Sequence[str],
) -> tuple[tuple[SafetyFirstFrontierDecision, ...], dict[str, Any]]:
    decisions = build_safety_first_frontier_decisions(
        safety_predictions,
        gain_predictions,
        risk_predictions,
        spec,
    )
    aggregate = _FrontierDiagnostics()
    folds = {fold_id: _FrontierDiagnostics() for fold_id in expected_fold_ids}
    for decision in decisions:
        fold_id = decision.winner.row.fold_id
        if fold_id not in folds:
            raise ValueError(f"Stage196 observed unexpected fold {fold_id}")
        aggregate.add_decision(decision)
        folds[fold_id].add_decision(decision)
    report = aggregate.report()
    report["folds"] = {fold_id: totals.report() for fold_id, totals in folds.items()}
    report["folds_meeting_pool_recall_minimum"] = sum(
        row["strict_opportunity_pool_recall"] >= stage194._INNER_FOLD_POOL_RECALL
        for row in report["folds"].values()
    )
    report["folds_meeting_conditional_capture_minimum"] = sum(
        row["conditional_ranker_strict_capture"] >= stage194._INNER_FOLD_CONDITIONAL_CAPTURE
        for row in report["folds"].values()
    )
    report["folds_meeting_unsafe_rate_maximum"] = sum(
        row["unsafe_selection_rate"] <= stage194._INNER_FOLD_UNSAFE_RATE
        for row in report["folds"].values()
    )
    return decisions, report


def run_safety_first_frontier_nested_cv(
    *,
    action_rows: Sequence[ActionAuditRow],
    stage182_selected_actions: Sequence[SelectedAction],
    progress_sink: ProgressSink | None = None,
    representation_fit_predictor: RepresentationFitPredictor | None = None,
) -> dict[str, Any]:
    """Run the frozen Stage 196 five-by-four train-only nested CV."""

    started_at = time.perf_counter()
    rows = tuple(action_rows)
    if not rows:
        raise ValueError("Stage196 requires action rows")
    fold_ids = tuple(sorted({row.fold_id for row in rows}))
    if len(fold_ids) != 5:
        raise ValueError("Stage196 requires exactly five frozen folds")
    grouped = stage194._group_rows(rows)
    if any(
        len([row for row in question_rows if row.action.family == "baseline"]) != 1
        for question_rows in grouped.values()
    ):
        raise ValueError("Stage196 requires one baseline action per question")
    references = build_stage182_reference_rows(rows, stage182_selected_actions)
    reference_regressions = [row for row in references.values() if stage194._is_f1_regression(row)]
    base_features = build_composition_feature_indices(rows)
    feature_indices = {
        "raw_runtime": base_features["raw"],
        "question_relative_runtime": base_features["question_relative"],
    }
    specs = stage196_policy_specs()
    fit_predict = representation_fit_predictor or fit_predict_safety_first_representation
    execution = _FitTotals()
    private_prediction_count = 0
    feature_counts: dict[str, int] = {}
    selected_spec_counts: Counter[str] = Counter()
    selected_gain_profile_counts: Counter[str] = Counter()
    selected_risk_profile_counts: Counter[str] = Counter()
    selected_risk_weight_counts: Counter[str] = Counter()
    selected_prefix_counts: Counter[str] = Counter()
    outer_reports: dict[str, dict[str, Any]] = {}
    outer_rows: list[ActionAuditRow] = []
    outer_diagnostics = _FrontierDiagnostics()
    outer_safety: list[stage194.SafetyPrediction] = []
    outer_gain: list[GainPrediction] = []
    outer_risk: list[UnsafePrediction] = []
    fit_seconds = 0.0

    for outer_fold_id in fold_ids:
        outer_training = tuple(row for row in rows if row.fold_id != outer_fold_id)
        outer_heldout = tuple(row for row in rows if row.fold_id == outer_fold_id)
        inner_fold_ids = tuple(fold for fold in fold_ids if fold != outer_fold_id)
        inner_safety: dict[str, list[stage194.SafetyPrediction]] = defaultdict(list)
        inner_gain: dict[str, list[GainPrediction]] = defaultdict(list)
        inner_risk: dict[str, list[UnsafePrediction]] = defaultdict(list)
        for inner_fold_id in inner_fold_ids:
            training = tuple(row for row in outer_training if row.fold_id != inner_fold_id)
            heldout = tuple(row for row in outer_training if row.fold_id == inner_fold_id)
            for representation in _REPRESENTATIONS:
                fitted_at = time.perf_counter()
                result = fit_predict(
                    training,
                    heldout,
                    feature_indices[representation],
                    representation,
                )
                fit_seconds += time.perf_counter() - fitted_at
                execution.add(result)
                feature_counts[representation] = max(
                    feature_counts.get(representation, 0), result.feature_count
                )
                private_prediction_count += _collect_predictions(
                    result.predictions,
                    representation,
                    inner_safety,
                    inner_gain,
                    inner_risk,
                )
                del result
                gc.collect()
            _emit(
                progress_sink,
                phase="inner_partition_complete",
                outer_fold_id=outer_fold_id,
                inner_fold_id=inner_fold_id,
                cumulative_model_fit_count=execution.model_fit_count,
                cumulative_tree_count=execution.tree_count,
            )

        question_count = len({row.question_key for row in outer_training})
        candidates = _evaluate_candidates(
            specs,
            inner_safety,
            inner_gain,
            inner_risk,
            references,
            inner_fold_ids,
            question_count,
        )
        eligible = [row for row in candidates if row["eligible"]]
        ranked = sorted(candidates, key=_inner_selection_key)
        public_top = [_public_candidate(row) for row in ranked[:5]]
        if not eligible:
            outer_reports[outer_fold_id] = _no_eligible_report(question_count, public_top)
            _emit(
                progress_sink,
                phase="outer_fold_no_eligible_config",
                outer_fold_id=outer_fold_id,
                cumulative_model_fit_count=execution.model_fit_count,
            )
            continue

        selected = min(eligible, key=_inner_selection_key)
        spec = _spec_from_dict(selected["spec"])
        selected_spec_counts[spec.name] += 1
        selected_gain_profile_counts[spec.gain_tree_profile] += 1
        selected_risk_profile_counts[spec.risk_tree_profile] += 1
        selected_risk_weight_counts[f"{spec.scale_pos_weight:.1f}"] += 1
        selected_prefix_counts[str(spec.safest_prefix_size)] += 1
        heldout_safety: dict[str, list[stage194.SafetyPrediction]] = defaultdict(list)
        heldout_gain: dict[str, list[GainPrediction]] = defaultdict(list)
        heldout_risk: dict[str, list[UnsafePrediction]] = defaultdict(list)
        for representation in _REPRESENTATIONS:
            fitted_at = time.perf_counter()
            result = fit_predict(
                outer_training,
                outer_heldout,
                feature_indices[representation],
                representation,
            )
            fit_seconds += time.perf_counter() - fitted_at
            execution.add(result)
            feature_counts[representation] = max(
                feature_counts.get(representation, 0), result.feature_count
            )
            private_prediction_count += _collect_predictions(
                result.predictions,
                representation,
                heldout_safety,
                heldout_gain,
                heldout_risk,
            )
            del result
            gc.collect()
        decisions, diagnostics = evaluate_safety_first_frontier_policy(
            heldout_safety[spec.pool_bundle_name],
            heldout_gain[spec.gain_bundle_name],
            heldout_risk[spec.risk_bundle_name],
            spec,
            expected_fold_ids=(outer_fold_id,),
        )
        selected_rows = tuple(decision.winner.row for decision in decisions)
        evaluation = evaluate_selected_actions(
            selected_rows=selected_rows,
            references=references,
            expected_fold_ids=(outer_fold_id,),
        )
        outer_rows.extend(selected_rows)
        outer_diagnostics.add_report(diagnostics)
        outer_safety.extend(heldout_safety[spec.pool_bundle_name])
        outer_gain.extend(heldout_gain[spec.gain_bundle_name])
        outer_risk.extend(heldout_risk[spec.risk_bundle_name])
        outer_reports[outer_fold_id] = {
            "inner_question_count": question_count,
            "eligible_config_count": len(eligible),
            "selected_spec": _spec_dict(spec),
            "selected_inner_evaluation": selected["evaluation"],
            "selected_inner_diagnostics": selected["diagnostics"],
            "outer_evaluation": evaluation,
            "outer_diagnostics": diagnostics,
            "top_inner_candidates": public_top,
            "outer_evaluated": True,
        }
        _emit(
            progress_sink,
            phase="outer_fold_complete",
            outer_fold_id=outer_fold_id,
            selected_spec=spec.name,
            eligible_config_count=len(eligible),
            cumulative_model_fit_count=execution.model_fit_count,
        )

    eligible_outer_fold_count = sum(row["outer_evaluated"] for row in outer_reports.values())
    aggregate = evaluate_selected_actions(
        selected_rows=outer_rows,
        references=references,
        expected_fold_ids=fold_ids,
    )
    aggregate_diagnostics = outer_diagnostics.report()
    bootstrap = (
        paired_selected_action_bootstrap(outer_rows)
        if eligible_outer_fold_count == len(fold_ids)
        else unavailable_selected_action_bootstrap()
    )
    gates = stage194._advancement_gates(
        eligible_outer_fold_count,
        aggregate,
        aggregate_diagnostics,
        bootstrap,
    )
    return {
        "protocol": {
            "outer_fold_count": 5,
            "inner_fold_count": 4,
            "policy_config_count": len(specs),
            "pool_cap": _POOL_CAP,
            "scale_pos_weights": list(_SCALE_POS_WEIGHTS),
            "safest_prefix_sizes": list(_SAFEST_PREFIX_SIZES),
            "model_fits_per_partition": 24,
            "maximum_model_fit_count": 600,
            "maximum_tree_count": 120_000,
            "fallback_enabled": False,
        },
        "dataset": {
            "action_count": len(rows),
            "nonbaseline_action_count": sum(row.action.family != "baseline" for row in rows),
            "question_count": len(grouped),
            "reference_action_count": len(references),
            "reference_regression_count": len(reference_regressions),
            "fold_action_counts": {
                fold_id: sum(row.fold_id == fold_id for row in rows) for fold_id in fold_ids
            },
        },
        "outer_folds": outer_reports,
        "aggregate": aggregate,
        "aggregate_diagnostics": aggregate_diagnostics,
        "paired_bootstrap": bootstrap,
        "prediction_metrics": _prediction_metrics(outer_safety, outer_gain, outer_risk),
        "selected_spec_counts": dict(sorted(selected_spec_counts.items())),
        "selected_gain_profile_counts": dict(sorted(selected_gain_profile_counts.items())),
        "selected_risk_profile_counts": dict(sorted(selected_risk_profile_counts.items())),
        "selected_risk_weight_counts": dict(sorted(selected_risk_weight_counts.items())),
        "selected_prefix_counts": dict(sorted(selected_prefix_counts.items())),
        "advancement_gates": gates,
        "advancement_gate_pass_count": sum(row["passed"] for row in gates),
        "candidate_family_accepted": all(row["passed"] for row in gates),
        "execution": {
            "model_fit_count": execution.model_fit_count,
            "pool_safety_fit_count": execution.pool_safety_fit_count,
            "lambdamart_fit_count": execution.gain_ranker_fit_count,
            "unsafe_head_fit_count": execution.unsafe_head_fit_count,
            "tree_count": execution.tree_count,
            "group_contract_validation_count": execution.group_contract_validation_count,
            "maximum_model_fit_count": 600,
            "private_prediction_count": private_prediction_count,
            "public_training_rows_written": 0,
            "public_prediction_rows_written": 0,
            "feature_count_by_representation": dict(sorted(feature_counts.items())),
            "fit_seconds": round(fit_seconds, 6),
            "wall_seconds": round(time.perf_counter() - started_at, 6),
        },
    }


def _evaluate_candidates(
    specs: Sequence[SafetyFirstFrontierPolicySpec],
    safety: Mapping[str, Sequence[stage194.SafetyPrediction]],
    gain: Mapping[str, Sequence[GainPrediction]],
    risk: Mapping[str, Sequence[UnsafePrediction]],
    references: Mapping[str, ActionAuditRow],
    fold_ids: Sequence[str],
    question_count: int,
) -> list[dict[str, Any]]:
    candidates = []
    for spec in specs:
        decisions, diagnostics = evaluate_safety_first_frontier_policy(
            safety[spec.pool_bundle_name],
            gain[spec.gain_bundle_name],
            risk[spec.risk_bundle_name],
            spec,
            expected_fold_ids=fold_ids,
        )
        evaluation = evaluate_selected_actions(
            selected_rows=tuple(decision.winner.row for decision in decisions),
            references=references,
            expected_fold_ids=fold_ids,
        )
        candidates.append(
            {
                "spec": _spec_dict(spec),
                "eligible": stage194._inner_eligible(
                    evaluation,
                    diagnostics,
                    question_count,
                ),
                "evaluation": evaluation,
                "diagnostics": diagnostics,
            }
        )
    return candidates


def _inner_selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    evaluation = row["evaluation"]
    diagnostics = row["diagnostics"]
    return (
        -evaluation["strict_success_count"],
        -diagnostics["conditional_ranker_strict_capture"],
        -evaluation["strict_success_precision"],
        diagnostics["unsafe_selected_question_count"],
        -diagnostics["strict_opportunity_frontier_recall"],
        evaluation["f1_regression_action_count"],
        evaluation["citation_loss_action_count"],
        -evaluation["gold_citation_delta"],
        -evaluation["mean_f1_delta"],
        -evaluation["repaired_reference_regression_count"],
        row["spec"]["name"],
    )


def _collect_predictions(
    predictions: RepresentationPredictions,
    representation: str,
    safety: dict[str, list[stage194.SafetyPrediction]],
    gain: dict[str, list[GainPrediction]],
    risk: dict[str, list[UnsafePrediction]],
) -> int:
    count = 0
    for estimator, values in predictions.safety_by_estimator.items():
        safety[f"{representation}__{estimator}"].extend(values)
        count += len(values)
    for profile, values in predictions.gain_by_profile.items():
        gain[f"{representation}__{profile}"].extend(values)
        count += len(values)
    for profile_weight, values in predictions.risk_by_profile_and_weight.items():
        risk[f"{representation}__{profile_weight}"].extend(values)
        count += len(values)
    return count


def _prediction_metrics(
    safety: Sequence[stage194.SafetyPrediction],
    gain: Sequence[GainPrediction],
    risk: Sequence[UnsafePrediction],
) -> dict[str, Any]:
    return {
        "action_count": len(safety),
        "citation_loss": _binary_metrics(
            [int(row.row.citation_delta < 0) for row in safety],
            [row.citation_loss_probability for row in safety],
        )
        if safety
        else None,
        "f1_loss": _binary_metrics(
            [int(stage194._is_f1_regression(row.row)) for row in safety],
            [row.f1_loss_probability for row in safety],
        )
        if safety
        else None,
        "strict_gain": _binary_metrics(
            [int(row.row.strict_expected) for row in gain],
            [row.score for row in gain],
        )
        if gain
        else None,
        "unsafe": _binary_metrics(
            [int(stage194._is_unsafe(row.row)) for row in risk],
            [row.score for row in risk],
        )
        if risk
        else None,
    }


def _spec_dict(spec: SafetyFirstFrontierPolicySpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "pool_feature_representation": spec.pool_feature_representation,
        "pool_safety_estimator": spec.pool_safety_estimator,
        "gain_feature_representation": spec.gain_feature_representation,
        "gain_tree_profile": spec.gain_tree_profile,
        "risk_feature_representation": spec.risk_feature_representation,
        "risk_tree_profile": spec.risk_tree_profile,
        "scale_pos_weight": spec.scale_pos_weight,
        "safest_prefix_size": spec.safest_prefix_size,
    }


def _spec_from_dict(value: Mapping[str, Any]) -> SafetyFirstFrontierPolicySpec:
    return SafetyFirstFrontierPolicySpec(
        value["name"],
        value["pool_feature_representation"],
        value["pool_safety_estimator"],
        value["gain_feature_representation"],
        value["gain_tree_profile"],
        value["risk_feature_representation"],
        value["risk_tree_profile"],
        float(value["scale_pos_weight"]),
        int(value["safest_prefix_size"]),
    )


def _public_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "spec": row["spec"],
        "eligible": row["eligible"],
        "evaluation": row["evaluation"],
        "diagnostics": row["diagnostics"],
    }


def _no_eligible_report(question_count: int, top: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "inner_question_count": question_count,
        "eligible_config_count": 0,
        "selected_spec": None,
        "selected_inner_evaluation": None,
        "selected_inner_diagnostics": None,
        "outer_evaluation": None,
        "outer_diagnostics": None,
        "top_inner_candidates": list(top),
        "outer_evaluated": False,
    }


def _combine(
    safety: stage194.SafetyPrediction,
    gain_index: Mapping[tuple[str, str], GainPrediction],
    risk_index: Mapping[tuple[str, str], UnsafePrediction],
) -> FrontierActionPrediction:
    key = stage194._row_key(safety.row)
    return FrontierActionPrediction(
        safety.row,
        safety.citation_loss_probability,
        safety.f1_loss_probability,
        gain_index[key].score,
        risk_index[key].score,
    )


def _safety_predictions(
    rows: Sequence[ActionAuditRow],
    citation: Sequence[float],
    f1: Sequence[float],
) -> tuple[stage194.SafetyPrediction, ...]:
    return tuple(
        stage194.SafetyPrediction(row, float(citation_score), float(f1_score))
        for row, citation_score, f1_score in zip(rows, citation, f1, strict=True)
    )


def _lightgbm_common_parameters() -> dict[str, Any]:
    return {
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "n_estimators": 300,
        "max_bin": 63,
        "min_split_gain": 0.0,
        "reg_alpha": 0.0,
        "subsample": 1.0,
        "subsample_freq": 0,
        "colsample_bytree": 1.0,
        "random_state": 193,
        "n_jobs": 8,
        "device_type": "cpu",
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator / denominator), 6) if denominator else 0.0


def _emit(progress_sink: ProgressSink | None, **event: Any) -> None:
    if progress_sink is not None:
        progress_sink(event)
