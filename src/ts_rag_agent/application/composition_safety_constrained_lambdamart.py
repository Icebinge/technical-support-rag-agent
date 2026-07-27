from __future__ import annotations

import gc
import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

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

FeatureRepresentation = Literal["raw_runtime", "question_relative_runtime"]
SafetyEstimator = Literal["class_balanced_logistic", "histogram_gradient_boosting"]
TreeProfile = Literal["conservative", "moderate"]
ProgressSink = Callable[[Mapping[str, Any]], None]

_REPRESENTATIONS: tuple[FeatureRepresentation, ...] = (
    "raw_runtime",
    "question_relative_runtime",
)
_SAFETY_ESTIMATORS: tuple[SafetyEstimator, ...] = (
    "class_balanced_logistic",
    "histogram_gradient_boosting",
)
_TREE_PROFILES: Mapping[TreeProfile, Mapping[str, int | float]] = {
    "conservative": {
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 40,
        "reg_lambda": 2.0,
    },
    "moderate": {
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 25,
        "reg_lambda": 1.0,
    },
}
_RISK_PENALTIES = (0.25, 0.5, 1.0, 2.0)
_POOL_CAP = 16
_F1_TOLERANCE = 1e-12
_INNER_STRICT_PRECISION = 0.65
_INNER_CHANGED_RATE = 0.10
_INNER_STRICT_COUNT_RATE = 0.08
_MINIMUM_INNER_NONREGRESSING_FOLDS = 3
_INNER_POOL_RECALL = 0.95
_INNER_FOLD_POOL_RECALL = 0.90
_INNER_CONDITIONAL_CAPTURE = 0.68
_INNER_FOLD_CONDITIONAL_CAPTURE = 0.60
_INNER_UNSAFE_RATE = 0.25
_INNER_FOLD_UNSAFE_RATE = 0.35
_MINIMUM_INNER_DIAGNOSTIC_FOLDS = 3


@dataclass(frozen=True)
class SafetyConstrainedLambdaMARTPolicySpec:
    """One frozen Stage 194 policy configuration."""

    name: str
    pool_feature_representation: FeatureRepresentation
    pool_safety_estimator: SafetyEstimator
    reranker_feature_representation: FeatureRepresentation
    tree_profile: TreeProfile
    risk_penalty: float

    @property
    def pool_bundle_name(self) -> str:
        return f"{self.pool_feature_representation}__{self.pool_safety_estimator}"

    @property
    def reranker_bundle_name(self) -> str:
        return f"{self.reranker_feature_representation}__{self.tree_profile}"


@dataclass(frozen=True)
class SafetyPrediction:
    row: ActionAuditRow
    citation_loss_probability: float
    f1_loss_probability: float


@dataclass(frozen=True)
class RerankerPrediction:
    row: ActionAuditRow
    gain_score: float
    unsafe_probability: float


@dataclass(frozen=True)
class CombinedActionPrediction:
    row: ActionAuditRow
    citation_loss_probability: float
    f1_loss_probability: float
    gain_score: float
    unsafe_probability: float
    utility: float


@dataclass(frozen=True)
class SafetyConstrainedDecision:
    question_key: str
    baseline: CombinedActionPrediction
    pool: tuple[CombinedActionPrediction, ...]
    winner: CombinedActionPrediction
    strict_opportunity: bool
    action_count: int
    strict_action_count: int


@dataclass(frozen=True)
class RepresentationPredictions:
    safety_by_estimator: Mapping[SafetyEstimator, tuple[SafetyPrediction, ...]]
    reranker_by_profile: Mapping[TreeProfile, tuple[RerankerPrediction, ...]]


@dataclass(frozen=True)
class FittedSafetyConstrainedRepresentation:
    feature_representation: FeatureRepresentation
    vectorizer: DictVectorizer
    logistic_scaler: StandardScaler
    safety_heads: Mapping[SafetyEstimator, Mapping[str, _SafetyHead]]
    gain_rankers: Mapping[TreeProfile, Any]
    unsafe_heads: Mapping[TreeProfile, Any]
    diagnostics: Mapping[str, Any]

    @property
    def feature_count(self) -> int:
        return len(self.vectorizer.feature_names_)

    @property
    def model_fit_count(self) -> int:
        return 8

    def predict(
        self,
        rows: Sequence[ActionAuditRow],
        feature_index: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> RepresentationPredictions:
        ordered_rows = tuple(rows)
        matrix = self.vectorizer.transform(
            [dict(feature_index[_row_key(row)]) for row in ordered_rows]
        ).tocsr()
        logistic_matrix = self.logistic_scaler.transform(matrix).tocsr()
        safety: dict[SafetyEstimator, tuple[SafetyPrediction, ...]] = {}
        for estimator, heads in self.safety_heads.items():
            prediction_matrix = (
                matrix if estimator == "histogram_gradient_boosting" else logistic_matrix
            )
            citation = heads["citation_loss"].predict(prediction_matrix)
            f1 = heads["f1_loss"].predict(prediction_matrix)
            safety[estimator] = tuple(
                SafetyPrediction(row, float(citation_score), float(f1_score))
                for row, citation_score, f1_score in zip(ordered_rows, citation, f1, strict=True)
            )
        reranker = {}
        for profile in _TREE_PROFILES:
            gain = np.asarray(self.gain_rankers[profile].predict(matrix), dtype=np.float64)
            unsafe = np.asarray(
                self.unsafe_heads[profile].predict_proba(matrix)[:, 1], dtype=np.float64
            )
            reranker[profile] = tuple(
                RerankerPrediction(row, float(gain_score), float(unsafe_score))
                for row, gain_score, unsafe_score in zip(ordered_rows, gain, unsafe, strict=True)
            )
        return RepresentationPredictions(safety, reranker)


class RepresentationFitter(Protocol):
    def __call__(
        self,
        rows: Sequence[ActionAuditRow],
        feature_index: Mapping[tuple[str, str], Mapping[str, Any]],
        representation: FeatureRepresentation,
    ) -> FittedSafetyConstrainedRepresentation: ...


@dataclass
class _FitTotals:
    model_fit_count: int = 0
    tree_count: int = 0
    ranker_fit_count: int = 0
    unsafe_head_fit_count: int = 0
    pool_safety_fit_count: int = 0
    group_contract_validation_count: int = 0

    def add(self, fitted: FittedSafetyConstrainedRepresentation) -> None:
        self.model_fit_count += fitted.model_fit_count
        self.tree_count += int(fitted.diagnostics["tree_count"])
        self.ranker_fit_count += 2
        self.unsafe_head_fit_count += 2
        self.pool_safety_fit_count += 4
        self.group_contract_validation_count += 1


@dataclass
class _DiagnosticTotals:
    question_count: int = 0
    strict_opportunity_question_count: int = 0
    pool_recalled_question_count: int = 0
    strict_selected_question_count: int = 0
    unsafe_selected_question_count: int = 0
    safe_zero_selected_question_count: int = 0
    action_count: int = 0
    pool_action_count: int = 0
    strict_action_count: int = 0
    retained_strict_action_count: int = 0
    baseline_in_pool_question_count: int = 0

    def add_decision(self, decision: SafetyConstrainedDecision) -> None:
        self.question_count += 1
        strict_in_pool = any(row.row.strict_expected for row in decision.pool)
        self.strict_opportunity_question_count += int(decision.strict_opportunity)
        self.pool_recalled_question_count += int(decision.strict_opportunity and strict_in_pool)
        self.strict_selected_question_count += int(
            decision.strict_opportunity and decision.winner.row.strict_expected
        )
        unsafe = _is_unsafe(decision.winner.row)
        self.unsafe_selected_question_count += int(unsafe)
        self.safe_zero_selected_question_count += int(_is_safe_zero(decision.winner.row))
        self.action_count += decision.action_count
        self.pool_action_count += len(decision.pool)
        self.strict_action_count += decision.strict_action_count
        self.retained_strict_action_count += sum(row.row.strict_expected for row in decision.pool)
        self.baseline_in_pool_question_count += int(decision.baseline in decision.pool)

    def add_report(self, report: Mapping[str, Any]) -> None:
        for name in (
            "question_count",
            "strict_opportunity_question_count",
            "pool_recalled_question_count",
            "strict_selected_question_count",
            "unsafe_selected_question_count",
            "safe_zero_selected_question_count",
            "action_count",
            "pool_action_count",
            "strict_action_count",
            "retained_strict_action_count",
            "baseline_in_pool_question_count",
        ):
            setattr(self, name, getattr(self, name) + int(report[name]))

    def report(self) -> dict[str, Any]:
        return {
            "question_count": self.question_count,
            "strict_opportunity_question_count": self.strict_opportunity_question_count,
            "pool_recalled_question_count": self.pool_recalled_question_count,
            "strict_selected_question_count": self.strict_selected_question_count,
            "strict_opportunity_pool_recall": _ratio(
                self.pool_recalled_question_count,
                self.strict_opportunity_question_count,
            ),
            "conditional_ranker_strict_capture": _ratio(
                self.strict_selected_question_count,
                self.pool_recalled_question_count,
            ),
            "actual_strict_opportunity_capture": _ratio(
                self.strict_selected_question_count,
                self.strict_opportunity_question_count,
            ),
            "unsafe_selected_question_count": self.unsafe_selected_question_count,
            "unsafe_selection_rate": _ratio(
                self.unsafe_selected_question_count, self.question_count
            ),
            "safe_zero_selected_question_count": self.safe_zero_selected_question_count,
            "action_count": self.action_count,
            "pool_action_count": self.pool_action_count,
            "mean_pool_size": _ratio(self.pool_action_count, self.question_count),
            "strict_action_count": self.strict_action_count,
            "retained_strict_action_count": self.retained_strict_action_count,
            "baseline_in_pool_question_count": self.baseline_in_pool_question_count,
            "baseline_in_pool_rate": _ratio(
                self.baseline_in_pool_question_count, self.question_count
            ),
        }


def stage194_policy_specs() -> tuple[SafetyConstrainedLambdaMARTPolicySpec, ...]:
    specs = []
    for pool_representation in _REPRESENTATIONS:
        for safety_estimator in _SAFETY_ESTIMATORS:
            for reranker_representation in _REPRESENTATIONS:
                for profile in _TREE_PROFILES:
                    for penalty in _RISK_PENALTIES:
                        name = (
                            f"pool_{pool_representation}__{safety_estimator}__"
                            f"rank_{reranker_representation}__{profile}__risk_{penalty:.2f}"
                        )
                        specs.append(
                            SafetyConstrainedLambdaMARTPolicySpec(
                                name,
                                pool_representation,
                                safety_estimator,
                                reranker_representation,
                                profile,
                                penalty,
                            )
                        )
    return tuple(specs)


def fit_safety_constrained_representation(
    rows: Sequence[ActionAuditRow],
    feature_index: Mapping[tuple[str, str], Mapping[str, Any]],
    representation: FeatureRepresentation,
) -> FittedSafetyConstrainedRepresentation:
    """Fit exactly four safety heads, two rankers, and two unsafe heads."""

    import lightgbm as lgb

    ordered_rows = tuple(sorted(rows, key=_row_key))
    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(
        [dict(feature_index[_row_key(row)]) for row in ordered_rows]
    ).tocsr()
    weights = _question_balanced_weights(ordered_rows)
    citation_labels = np.asarray([row.citation_delta < 0 for row in ordered_rows], dtype=np.int8)
    f1_labels = np.asarray([row.f1_delta < -_F1_TOLERANCE for row in ordered_rows], dtype=np.int8)
    unsafe_labels = np.asarray([_is_unsafe(row) for row in ordered_rows], dtype=np.int8)
    relevance = np.asarray([_outcome_tier(row) for row in ordered_rows], dtype=np.int8)
    for name, labels in (
        ("citation_loss", citation_labels),
        ("f1_loss", f1_labels),
        ("unsafe", unsafe_labels),
    ):
        if len(set(labels.tolist())) != 2:
            raise ValueError(f"Stage194 {name} target requires both classes")

    logistic_scaler = StandardScaler(with_mean=False)
    logistic_matrix = logistic_scaler.fit_transform(matrix).tocsr()
    logistic_heads = {
        "citation_loss": _SafetyHead(
            _fit_logistic_classifier(logistic_matrix, citation_labels, weights), False
        ),
        "f1_loss": _SafetyHead(
            _fit_logistic_classifier(logistic_matrix, f1_labels, weights), False
        ),
    }
    dense_matrix = matrix.toarray()
    histogram_heads = {
        "citation_loss": _SafetyHead(
            _fit_histogram_classifier(dense_matrix, citation_labels, weights), True
        ),
        "f1_loss": _SafetyHead(_fit_histogram_classifier(dense_matrix, f1_labels, weights), True),
    }
    del dense_matrix
    gc.collect()

    group_sizes = _question_group_sizes(ordered_rows)
    common = {
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
    rankers = {}
    unsafe_heads = {}
    tree_counts = {}
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
            matrix,
            relevance,
            group=group_sizes,
            sample_weight=weights,
            eval_at=[1],
        )
        unsafe_head = lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            class_weight=None,
            scale_pos_weight=1.0,
            **parameters,
        )
        unsafe_head.fit(matrix, unsafe_labels, sample_weight=weights)
        rankers[profile_name] = ranker
        unsafe_heads[profile_name] = unsafe_head
        tree_counts[profile_name] = {
            "ranker": int(ranker.booster_.num_trees()),
            "unsafe_head": int(unsafe_head.booster_.num_trees()),
        }
    return FittedSafetyConstrainedRepresentation(
        feature_representation=representation,
        vectorizer=vectorizer,
        logistic_scaler=logistic_scaler,
        safety_heads={
            "class_balanced_logistic": logistic_heads,
            "histogram_gradient_boosting": histogram_heads,
        },
        gain_rankers=rankers,
        unsafe_heads=unsafe_heads,
        diagnostics={
            "feature_count": len(vectorizer.feature_names_),
            "training_action_count": len(ordered_rows),
            "training_question_count": len(group_sizes),
            "group_size_sum": sum(group_sizes),
            "tree_counts": tree_counts,
            "tree_count": sum(sum(row.values()) for row in tree_counts.values()),
        },
    )


def build_safety_constrained_decisions(
    safety_predictions: Sequence[SafetyPrediction],
    reranker_predictions: Sequence[RerankerPrediction],
    spec: SafetyConstrainedLambdaMARTPolicySpec,
) -> tuple[SafetyConstrainedDecision, ...]:
    safety_grouped = _group_predictions(safety_predictions)
    reranker_index = {_row_key(prediction.row): prediction for prediction in reranker_predictions}
    if {_row_key(row.row) for row in safety_predictions} != set(reranker_index):
        raise ValueError("Stage194 safety and reranker prediction rows differ")
    decisions = []
    for question_key, question_safety in sorted(safety_grouped.items()):
        baselines = [row for row in question_safety if row.row.action.family == "baseline"]
        if len(baselines) != 1:
            raise ValueError("Stage194 requires one baseline action per question")
        ranked_safety = sorted(question_safety, key=_safety_order_key)
        pool_index = {_row_key(row.row): row for row in ranked_safety[:_POOL_CAP]}
        baseline_safety = baselines[0]
        pool_index[_row_key(baseline_safety.row)] = baseline_safety
        pool_safety = tuple(sorted(pool_index.values(), key=_safety_order_key))
        combined = []
        gain_order = sorted(
            pool_safety,
            key=lambda row: (
                -reranker_index[_row_key(row.row)].gain_score,
                row.row.action.action_id,
            ),
        )
        risk_order = sorted(
            pool_safety,
            key=lambda row: (
                reranker_index[_row_key(row.row)].unsafe_probability,
                row.row.action.action_id,
            ),
        )
        gain_ranks = {_row_key(row.row): rank for rank, row in enumerate(gain_order)}
        risk_ranks = {_row_key(row.row): rank for rank, row in enumerate(risk_order)}
        denominator = len(pool_safety) - 1
        for safety in pool_safety:
            key = _row_key(safety.row)
            reranker = reranker_index[key]
            gain_fraction = gain_ranks[key] / denominator if denominator else 0.0
            risk_fraction = risk_ranks[key] / denominator if denominator else 0.0
            utility = 1.0 - gain_fraction - spec.risk_penalty * risk_fraction
            combined.append(
                CombinedActionPrediction(
                    safety.row,
                    safety.citation_loss_probability,
                    safety.f1_loss_probability,
                    reranker.gain_score,
                    reranker.unsafe_probability,
                    utility,
                )
            )
        pool = tuple(combined)
        winner = min(
            pool,
            key=lambda row: (
                -row.utility,
                row.unsafe_probability,
                -row.gain_score,
                row.row.action.action_id,
            ),
        )
        baseline = next(row for row in pool if row.row.action.family == "baseline")
        decisions.append(
            SafetyConstrainedDecision(
                question_key,
                baseline,
                pool,
                winner,
                any(row.row.strict_expected for row in question_safety),
                len(question_safety),
                sum(row.row.strict_expected for row in question_safety),
            )
        )
    return tuple(decisions)


def evaluate_safety_constrained_policy(
    safety_predictions: Sequence[SafetyPrediction],
    reranker_predictions: Sequence[RerankerPrediction],
    spec: SafetyConstrainedLambdaMARTPolicySpec,
    *,
    expected_fold_ids: Sequence[str],
) -> tuple[tuple[SafetyConstrainedDecision, ...], dict[str, Any]]:
    decisions = build_safety_constrained_decisions(safety_predictions, reranker_predictions, spec)
    aggregate = _DiagnosticTotals()
    folds = {fold_id: _DiagnosticTotals() for fold_id in expected_fold_ids}
    for decision in decisions:
        fold_id = decision.winner.row.fold_id
        if fold_id not in folds:
            raise ValueError(f"Stage194 observed unexpected fold {fold_id}")
        aggregate.add_decision(decision)
        folds[fold_id].add_decision(decision)
    report = aggregate.report()
    report["folds"] = {fold_id: totals.report() for fold_id, totals in folds.items()}
    report["folds_meeting_pool_recall_minimum"] = sum(
        row["strict_opportunity_pool_recall"] >= _INNER_FOLD_POOL_RECALL
        for row in report["folds"].values()
    )
    report["folds_meeting_conditional_capture_minimum"] = sum(
        row["conditional_ranker_strict_capture"] >= _INNER_FOLD_CONDITIONAL_CAPTURE
        for row in report["folds"].values()
    )
    report["folds_meeting_unsafe_rate_maximum"] = sum(
        row["unsafe_selection_rate"] <= _INNER_FOLD_UNSAFE_RATE for row in report["folds"].values()
    )
    return decisions, report


def run_safety_constrained_lambdamart_nested_cv(
    *,
    action_rows: Sequence[ActionAuditRow],
    stage182_selected_actions: Sequence[SelectedAction],
    progress_sink: ProgressSink | None = None,
    representation_fitter: RepresentationFitter | None = None,
) -> dict[str, Any]:
    """Run the frozen Stage 194 five-by-four train-only nested CV."""

    started_at = time.perf_counter()
    rows = tuple(action_rows)
    if not rows:
        raise ValueError("Stage194 requires action rows")
    fold_ids = tuple(sorted({row.fold_id for row in rows}))
    if len(fold_ids) != 5:
        raise ValueError("Stage194 requires exactly five frozen folds")
    grouped = _group_rows(rows)
    if any(
        len([row for row in question_rows if row.action.family == "baseline"]) != 1
        for question_rows in grouped.values()
    ):
        raise ValueError("Stage194 requires one baseline action per question")
    references = build_stage182_reference_rows(rows, stage182_selected_actions)
    reference_regressions = [row for row in references.values() if _is_f1_regression(row)]
    base_features = build_composition_feature_indices(rows)
    feature_indices = {
        "raw_runtime": base_features["raw"],
        "question_relative_runtime": base_features["question_relative"],
    }
    specs = stage194_policy_specs()
    fit_representation = representation_fitter or fit_safety_constrained_representation
    execution = _FitTotals()
    private_prediction_count = 0
    feature_counts: dict[str, int] = {}
    selected_spec_counts: Counter[str] = Counter()
    selected_profile_counts: Counter[str] = Counter()
    selected_penalty_counts: Counter[str] = Counter()
    outer_reports: dict[str, dict[str, Any]] = {}
    outer_rows: list[ActionAuditRow] = []
    outer_diagnostics = _DiagnosticTotals()
    outer_safety_predictions: list[SafetyPrediction] = []
    outer_reranker_predictions: list[RerankerPrediction] = []
    fit_seconds = 0.0

    for outer_fold_id in fold_ids:
        outer_training = tuple(row for row in rows if row.fold_id != outer_fold_id)
        outer_heldout = tuple(row for row in rows if row.fold_id == outer_fold_id)
        inner_fold_ids = tuple(fold for fold in fold_ids if fold != outer_fold_id)
        inner_safety: dict[str, list[SafetyPrediction]] = defaultdict(list)
        inner_reranker: dict[str, list[RerankerPrediction]] = defaultdict(list)
        for inner_fold_id in inner_fold_ids:
            training = tuple(row for row in outer_training if row.fold_id != inner_fold_id)
            heldout = tuple(row for row in outer_training if row.fold_id == inner_fold_id)
            for representation in _REPRESENTATIONS:
                fitted_at = time.perf_counter()
                fitted = fit_representation(
                    training, feature_indices[representation], representation
                )
                fit_seconds += time.perf_counter() - fitted_at
                execution.add(fitted)
                feature_counts[representation] = max(
                    feature_counts.get(representation, 0), fitted.feature_count
                )
                predictions = fitted.predict(heldout, feature_indices[representation])
                for estimator, values in predictions.safety_by_estimator.items():
                    inner_safety[f"{representation}__{estimator}"].extend(values)
                    private_prediction_count += len(values)
                for profile, values in predictions.reranker_by_profile.items():
                    inner_reranker[f"{representation}__{profile}"].extend(values)
                    private_prediction_count += len(values)
                del fitted, predictions
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
        candidates = []
        for spec in specs:
            decisions, diagnostics = evaluate_safety_constrained_policy(
                inner_safety[spec.pool_bundle_name],
                inner_reranker[spec.reranker_bundle_name],
                spec,
                expected_fold_ids=inner_fold_ids,
            )
            selected_rows = tuple(decision.winner.row for decision in decisions)
            evaluation = evaluate_selected_actions(
                selected_rows=selected_rows,
                references=references,
                expected_fold_ids=inner_fold_ids,
            )
            candidates.append(
                {
                    "spec": _spec_dict(spec),
                    "eligible": _inner_eligible(evaluation, diagnostics, question_count),
                    "evaluation": evaluation,
                    "diagnostics": diagnostics,
                }
            )
        eligible = [row for row in candidates if row["eligible"]]
        ranked = sorted(candidates, key=_inner_selection_key)
        public_top = [_public_candidate(row) for row in ranked[:5]]
        if not eligible:
            outer_reports[outer_fold_id] = {
                "inner_question_count": question_count,
                "eligible_config_count": 0,
                "selected_spec": None,
                "selected_inner_evaluation": None,
                "selected_inner_diagnostics": None,
                "outer_evaluation": None,
                "outer_diagnostics": None,
                "top_inner_candidates": public_top,
                "outer_evaluated": False,
            }
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
        selected_profile_counts[spec.tree_profile] += 1
        selected_penalty_counts[f"{spec.risk_penalty:.2f}"] += 1
        heldout_safety: dict[str, tuple[SafetyPrediction, ...]] = {}
        heldout_reranker: dict[str, tuple[RerankerPrediction, ...]] = {}
        for representation in _REPRESENTATIONS:
            fitted_at = time.perf_counter()
            fitted = fit_representation(
                outer_training, feature_indices[representation], representation
            )
            fit_seconds += time.perf_counter() - fitted_at
            execution.add(fitted)
            feature_counts[representation] = max(
                feature_counts.get(representation, 0), fitted.feature_count
            )
            predictions = fitted.predict(outer_heldout, feature_indices[representation])
            for estimator, values in predictions.safety_by_estimator.items():
                heldout_safety[f"{representation}__{estimator}"] = values
                private_prediction_count += len(values)
            for profile, values in predictions.reranker_by_profile.items():
                heldout_reranker[f"{representation}__{profile}"] = values
                private_prediction_count += len(values)
            del fitted, predictions
            gc.collect()
        decisions, diagnostics = evaluate_safety_constrained_policy(
            heldout_safety[spec.pool_bundle_name],
            heldout_reranker[spec.reranker_bundle_name],
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
        outer_safety_predictions.extend(heldout_safety[spec.pool_bundle_name])
        outer_reranker_predictions.extend(heldout_reranker[spec.reranker_bundle_name])
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
    gates = _advancement_gates(
        eligible_outer_fold_count, aggregate, aggregate_diagnostics, bootstrap
    )
    return {
        "protocol": {
            "outer_fold_count": 5,
            "inner_fold_count": 4,
            "policy_config_count": len(specs),
            "pool_cap": _POOL_CAP,
            "model_fits_per_partition": 16,
            "maximum_model_fit_count": 400,
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
        "prediction_metrics": _prediction_metrics(
            outer_safety_predictions, outer_reranker_predictions
        ),
        "selected_spec_counts": dict(sorted(selected_spec_counts.items())),
        "selected_profile_counts": dict(sorted(selected_profile_counts.items())),
        "selected_penalty_counts": dict(sorted(selected_penalty_counts.items())),
        "advancement_gates": gates,
        "advancement_gate_pass_count": sum(row["passed"] for row in gates),
        "candidate_family_accepted": all(row["passed"] for row in gates),
        "execution": {
            "model_fit_count": execution.model_fit_count,
            "pool_safety_fit_count": execution.pool_safety_fit_count,
            "lambdamart_fit_count": execution.ranker_fit_count,
            "unsafe_head_fit_count": execution.unsafe_head_fit_count,
            "tree_count": execution.tree_count,
            "group_contract_validation_count": execution.group_contract_validation_count,
            "maximum_model_fit_count": 400,
            "private_prediction_count": private_prediction_count,
            "public_training_rows_written": 0,
            "public_prediction_rows_written": 0,
            "feature_count_by_representation": dict(sorted(feature_counts.items())),
            "fit_seconds": round(fit_seconds, 6),
            "wall_seconds": round(time.perf_counter() - started_at, 6),
        },
    }


def _inner_eligible(
    evaluation: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    question_count: int,
) -> bool:
    return bool(
        evaluation["gold_citation_delta"] >= 0
        and evaluation["mean_f1_delta"] >= 0
        and evaluation["citation_nonregressing_fold_count"] >= _MINIMUM_INNER_NONREGRESSING_FOLDS
        and evaluation["f1_nonregressing_fold_count"] >= _MINIMUM_INNER_NONREGRESSING_FOLDS
        and evaluation["changed_question_count"] >= math.ceil(_INNER_CHANGED_RATE * question_count)
        and evaluation["strict_success_count"]
        >= math.ceil(_INNER_STRICT_COUNT_RATE * question_count)
        and evaluation["strict_success_precision"] >= _INNER_STRICT_PRECISION
        and diagnostics["strict_opportunity_pool_recall"] >= _INNER_POOL_RECALL
        and diagnostics["folds_meeting_pool_recall_minimum"] >= _MINIMUM_INNER_DIAGNOSTIC_FOLDS
        and diagnostics["conditional_ranker_strict_capture"] >= _INNER_CONDITIONAL_CAPTURE
        and diagnostics["folds_meeting_conditional_capture_minimum"]
        >= _MINIMUM_INNER_DIAGNOSTIC_FOLDS
        and diagnostics["unsafe_selection_rate"] <= _INNER_UNSAFE_RATE
        and diagnostics["folds_meeting_unsafe_rate_maximum"] >= _MINIMUM_INNER_DIAGNOSTIC_FOLDS
    )


def _inner_selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    evaluation = row["evaluation"]
    diagnostics = row["diagnostics"]
    return (
        -evaluation["strict_success_count"],
        -diagnostics["conditional_ranker_strict_capture"],
        -evaluation["strict_success_precision"],
        diagnostics["unsafe_selected_question_count"],
        evaluation["f1_regression_action_count"],
        evaluation["citation_loss_action_count"],
        -evaluation["gold_citation_delta"],
        -evaluation["mean_f1_delta"],
        -evaluation["repaired_reference_regression_count"],
        row["spec"]["name"],
    )


def _advancement_gates(
    eligible_outer_fold_count: int,
    aggregate: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> list[dict[str, Any]]:
    citation_bootstrap = bootstrap["gold_citation_delta"] or {}
    f1_bootstrap = bootstrap["mean_f1_delta"] or {}
    checks = (
        ("outer_folds_with_inner_eligible_config_at_least_5", eligible_outer_fold_count >= 5),
        ("gold_citation_delta_at_least_5", aggregate["gold_citation_delta"] >= 5),
        ("mean_f1_delta_at_least_0_005249", aggregate["mean_f1_delta"] >= 0.005249),
        (
            "citation_bootstrap_ci95_lower_nonnegative",
            citation_bootstrap.get("ci95_lower", -math.inf) >= 0,
        ),
        ("f1_bootstrap_ci95_lower_nonnegative", f1_bootstrap.get("ci95_lower", -math.inf) >= 0),
        (
            "citation_nonregressing_outer_folds_at_least_4",
            aggregate["citation_nonregressing_fold_count"] >= 4,
        ),
        ("f1_nonregressing_outer_folds_at_least_4", aggregate["f1_nonregressing_fold_count"] >= 4),
        ("strict_success_count_at_least_37", aggregate["strict_success_count"] >= 37),
        ("strict_success_precision_at_least_0_65", aggregate["strict_success_precision"] >= 0.65),
        ("citation_loss_action_count_at_most_4", aggregate["citation_loss_action_count"] <= 4),
        ("f1_regression_action_count_at_most_27", aggregate["f1_regression_action_count"] <= 27),
        (
            "stage182_regression_repair_rate_at_least_0_50",
            aggregate["stage182_regression_repair_rate"] >= 0.50,
        ),
        ("new_f1_regression_rate_at_most_0_02", aggregate["new_f1_regression_rate"] <= 0.02),
        ("changed_question_count_at_least_37", aggregate["changed_question_count"] >= 37),
        (
            "strict_opportunity_pool_recall_at_least_0_95",
            diagnostics["strict_opportunity_pool_recall"] >= 0.95,
        ),
        (
            "conditional_ranker_strict_capture_at_least_0_68",
            diagnostics["conditional_ranker_strict_capture"] >= 0.68,
        ),
        ("unsafe_selection_rate_at_most_0_25", diagnostics["unsafe_selection_rate"] <= 0.25),
    )
    return [{"name": name, "passed": bool(passed)} for name, passed in checks]


def _prediction_metrics(
    safety: Sequence[SafetyPrediction], reranker: Sequence[RerankerPrediction]
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
            [int(_is_f1_regression(row.row)) for row in safety],
            [row.f1_loss_probability for row in safety],
        )
        if safety
        else None,
        "strict_gain": _binary_metrics(
            [int(row.row.strict_expected) for row in reranker],
            [row.gain_score for row in reranker],
        )
        if reranker
        else None,
        "unsafe": _binary_metrics(
            [int(_is_unsafe(row.row)) for row in reranker],
            [row.unsafe_probability for row in reranker],
        )
        if reranker
        else None,
    }


def _spec_dict(spec: SafetyConstrainedLambdaMARTPolicySpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "pool_feature_representation": spec.pool_feature_representation,
        "pool_safety_estimator": spec.pool_safety_estimator,
        "reranker_feature_representation": spec.reranker_feature_representation,
        "tree_profile": spec.tree_profile,
        "risk_penalty": spec.risk_penalty,
    }


def _spec_from_dict(value: Mapping[str, Any]) -> SafetyConstrainedLambdaMARTPolicySpec:
    return SafetyConstrainedLambdaMARTPolicySpec(
        value["name"],
        value["pool_feature_representation"],
        value["pool_safety_estimator"],
        value["reranker_feature_representation"],
        value["tree_profile"],
        float(value["risk_penalty"]),
    )


def _public_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "spec": row["spec"],
        "eligible": row["eligible"],
        "evaluation": row["evaluation"],
        "diagnostics": row["diagnostics"],
    }


def _outcome_tier(row: ActionAuditRow) -> int:
    if row.strict_expected:
        return 2
    if _is_safe_zero(row):
        return 1
    return 0


def _is_unsafe(row: ActionAuditRow) -> bool:
    return row.citation_delta < 0 or _is_f1_regression(row)


def _is_f1_regression(row: ActionAuditRow) -> bool:
    return row.f1_delta < -_F1_TOLERANCE


def _is_safe_zero(row: ActionAuditRow) -> bool:
    return row.citation_delta == 0 and abs(row.f1_delta) <= _F1_TOLERANCE


def _safety_order_key(row: SafetyPrediction) -> tuple[float, float, str]:
    return (
        max(row.citation_loss_probability, row.f1_loss_probability),
        row.citation_loss_probability + row.f1_loss_probability,
        row.row.action.action_id,
    )


def _question_group_sizes(rows: Sequence[ActionAuditRow]) -> list[int]:
    sizes = []
    previous = None
    for row in rows:
        if row.question_key != previous:
            sizes.append(0)
            previous = row.question_key
        sizes[-1] += 1
    if sum(sizes) != len(rows):
        raise ValueError("Stage194 group sizes do not sum to action rows")
    return sizes


def _question_balanced_weights(rows: Sequence[ActionAuditRow]) -> np.ndarray:
    counts = Counter(row.question_key for row in rows)
    return np.asarray([1.0 / counts[row.question_key] for row in rows], dtype=np.float64)


def _group_rows(rows: Sequence[ActionAuditRow]) -> dict[str, list[ActionAuditRow]]:
    grouped: dict[str, list[ActionAuditRow]] = defaultdict(list)
    for row in rows:
        grouped[row.question_key].append(row)
    return grouped


def _group_predictions(
    rows: Sequence[SafetyPrediction],
) -> dict[str, list[SafetyPrediction]]:
    grouped: dict[str, list[SafetyPrediction]] = defaultdict(list)
    for row in rows:
        grouped[row.row.question_key].append(row)
    return grouped


def _row_key(row: ActionAuditRow) -> tuple[str, str]:
    return row.question_key, row.action.action_id


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator / denominator), 6) if denominator else 0.0


def _emit(progress_sink: ProgressSink | None, **event: Any) -> None:
    if progress_sink is not None:
        progress_sink(event)
