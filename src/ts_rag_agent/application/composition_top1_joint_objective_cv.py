from __future__ import annotations

import gc
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from sklearn.feature_extraction import DictVectorizer

from ts_rag_agent.application import composition_joint_risk_winner_cv as stage199
from ts_rag_agent.application import composition_safety_constrained_lambdamart as stage194
from ts_rag_agent.application import composition_safety_first_frontier as stage196
from ts_rag_agent.application.composition_action_audit import ActionAuditRow
from ts_rag_agent.application.composition_dual_target_policy import SelectedAction
from ts_rag_agent.application.composition_f1_representation_cv import (
    build_composition_feature_indices,
)
from ts_rag_agent.application.composition_gain_sensitive_ranking import (
    build_stage182_reference_rows,
    paired_selected_action_bootstrap,
    unavailable_selected_action_bootstrap,
)
from ts_rag_agent.application.composition_joint_constraint_ranking import (
    evaluate_selected_actions,
)
from ts_rag_agent.application.composition_safety_first_frontier import (
    FrontierActionPrediction,
    SafetyFirstFrontierDecision,
    SafetyFirstFrontierPolicySpec,
)

ProgressSink = Callable[[Mapping[str, Any]], None]

_SAFETY_WEIGHTS = (0.0, 0.5, 1.0, 2.0)
_PRECISION_WEIGHTS = (0.0, 0.5, 1.0, 2.0)
_POOL_CAP = 16
_HESSIAN_FLOOR = 1e-6
_CONTROL_NAME = "stage196_exact_control"
_STRICT_ONLY_NAME = "top1_safety_0.00__precision_0.00"

_UNSAFE_LABEL = 0
_SAFE_ZERO_LABEL = 1
_BASELINE_LABEL = 2
_STRICT_LABEL = 3


@dataclass(frozen=True)
class Top1ObjectiveSpec:
    name: str
    safety_weight: float
    precision_weight: float
    ablation_family: str


@dataclass(frozen=True)
class Top1ScorePrediction:
    row: ActionAuditRow
    score: float


@dataclass(frozen=True)
class Top1PartitionResult:
    safety_predictions: tuple[stage194.SafetyPrediction, ...]
    gain_predictions: tuple[stage196.GainPrediction, ...]
    risk_predictions: tuple[stage196.UnsafePrediction, ...]
    objective_predictions: Mapping[str, tuple[Top1ScorePrediction, ...]]
    feature_count_by_representation: Mapping[str, int]
    model_fit_count: int
    source_model_fit_count: int
    custom_objective_fit_count: int
    tree_count: int
    source_tree_count: int
    custom_objective_tree_count: int
    group_contract_validation_count: int
    objective_callback_call_count: int


@dataclass(frozen=True)
class Top1ObjectiveCandidateSnapshot:
    """Read-only private evidence for one Stage203 cell and outer context."""

    spec: Mapping[str, Any]
    eligible: bool
    evaluation: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    paired_vs_control: Mapping[str, Any]
    paired_vs_strict_only: Mapping[str, Any]
    decisions: tuple[SafetyFirstFrontierDecision, ...]


@dataclass(frozen=True)
class Top1ObjectiveDiagnosticSnapshot:
    """Private Stage204 stream item; consumers must aggregate and discard it."""

    outer_fold_id: str
    inner_fold_ids: tuple[str, ...]
    inner_question_count: int
    candidates: tuple[Top1ObjectiveCandidateSnapshot, ...]


DiagnosticSink = Callable[[Top1ObjectiveDiagnosticSnapshot], None]


class Top1PartitionFitPredictor(Protocol):
    def __call__(
        self,
        training_rows: Sequence[ActionAuditRow],
        heldout_rows: Sequence[ActionAuditRow],
        feature_indices: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
        source_spec: SafetyFirstFrontierPolicySpec,
        objective_specs: Sequence[Top1ObjectiveSpec],
    ) -> Top1PartitionResult: ...


@dataclass
class _FitTotals:
    model_fit_count: int = 0
    source_model_fit_count: int = 0
    pool_safety_fit_count: int = 0
    gain_ranker_fit_count: int = 0
    classifier_risk_fit_count: int = 0
    custom_objective_fit_count: int = 0
    tree_count: int = 0
    source_tree_count: int = 0
    custom_objective_tree_count: int = 0
    group_contract_validation_count: int = 0
    objective_callback_call_count: int = 0

    def add(self, result: Top1PartitionResult) -> None:
        self.model_fit_count += result.model_fit_count
        self.source_model_fit_count += result.source_model_fit_count
        self.pool_safety_fit_count += 2
        self.gain_ranker_fit_count += 1
        self.classifier_risk_fit_count += 1
        self.custom_objective_fit_count += result.custom_objective_fit_count
        self.tree_count += result.tree_count
        self.source_tree_count += result.source_tree_count
        self.custom_objective_tree_count += result.custom_objective_tree_count
        self.group_contract_validation_count += result.group_contract_validation_count
        self.objective_callback_call_count += result.objective_callback_call_count


class GroupedTop1Objective:
    """Vectorized grouped cross-entropy objective with frozen target mixtures."""

    def __init__(
        self,
        *,
        labels: Sequence[int] | np.ndarray,
        group_sizes: Sequence[int] | np.ndarray,
        sample_weights: Sequence[float] | np.ndarray,
        spec: Top1ObjectiveSpec,
    ) -> None:
        self._labels = np.asarray(labels, dtype=np.int8)
        self._groups = np.asarray(group_sizes, dtype=np.int32)
        self._weights = np.asarray(sample_weights, dtype=np.float64)
        _validate_group_contract(self._labels, self._groups, self._weights)
        self._starts = np.concatenate(
            (np.asarray([0], dtype=np.int64), np.cumsum(self._groups[:-1], dtype=np.int64))
        )
        self._repeated_group_weights = np.repeat(
            np.add.reduceat(self._weights, self._starts), self._groups
        )
        self._target = build_grouped_top1_target(
            labels=self._labels,
            group_sizes=self._groups,
            safety_weight=spec.safety_weight,
            precision_weight=spec.precision_weight,
        )
        self._target_sum_max_error = float(
            np.max(np.abs(np.add.reduceat(self._target, self._starts) - 1.0))
        )
        self.call_count = 0

    def __call__(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        weights: np.ndarray | None,
        group_sizes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        actual_labels = np.asarray(labels, dtype=np.int8)
        actual_groups = np.asarray(group_sizes, dtype=np.int32)
        if not np.array_equal(actual_labels, self._labels):
            raise ValueError("Stage203 objective labels changed after target construction")
        if not np.array_equal(actual_groups, self._groups):
            raise ValueError("Stage203 objective group sizes changed after target construction")
        if weights is None or not np.allclose(weights, self._weights, rtol=0.0, atol=1e-7):
            raise ValueError("Stage203 objective sample weights changed after target construction")
        scores = np.asarray(predictions, dtype=np.float64)
        if scores.shape != self._labels.shape or not np.all(np.isfinite(scores)):
            raise ValueError("Stage203 objective received invalid predictions")
        group_maximum = np.maximum.reduceat(scores, self._starts)
        exponentials = np.exp(scores - np.repeat(group_maximum, self._groups))
        denominators = np.add.reduceat(exponentials, self._starts)
        probabilities = exponentials / np.repeat(denominators, self._groups)
        gradients = (probabilities - self._target) * self._repeated_group_weights
        hessians = (
            np.maximum(probabilities * (1.0 - probabilities), _HESSIAN_FLOOR)
            * self._repeated_group_weights
        )
        if not np.all(np.isfinite(gradients)) or not np.all(np.isfinite(hessians)):
            raise ValueError("Stage203 objective produced non-finite derivatives")
        if not np.all(hessians > 0.0):
            raise ValueError("Stage203 objective produced non-positive Hessians")
        self.call_count += 1
        return gradients, hessians

    def diagnostics(self) -> dict[str, Any]:
        return {
            "group_count": len(self._groups),
            "row_count": len(self._labels),
            "group_size_sum": int(self._groups.sum()),
            "target_sum_max_error": round(self._target_sum_max_error, 12),
            "minimum_target_probability": round(float(self._target.min()), 12),
            "maximum_target_probability": round(float(self._target.max()), 12),
            "callback_call_count": self.call_count,
        }


def stage203_objective_specs() -> tuple[Top1ObjectiveSpec, ...]:
    return tuple(
        Top1ObjectiveSpec(
            name=f"top1_safety_{safety:.2f}__precision_{precision:.2f}",
            safety_weight=safety,
            precision_weight=precision,
            ablation_family=(
                "strict_only"
                if safety == 0.0 and precision == 0.0
                else "safety_only"
                if safety > 0.0 and precision == 0.0
                else "precision_only"
                if safety == 0.0 and precision > 0.0
                else "full_joint"
            ),
        )
        for safety in _SAFETY_WEIGHTS
        for precision in _PRECISION_WEIGHTS
    )


def build_grouped_top1_target(
    *,
    labels: Sequence[int] | np.ndarray,
    group_sizes: Sequence[int] | np.ndarray,
    safety_weight: float,
    precision_weight: float,
) -> np.ndarray:
    encoded = np.asarray(labels, dtype=np.int8)
    groups = np.asarray(group_sizes, dtype=np.int32)
    if safety_weight < 0.0 or precision_weight < 0.0:
        raise ValueError("Stage203 objective weights must be nonnegative")
    balanced_weights = np.repeat(1.0 / groups.astype(np.float64), groups)
    _validate_group_contract(encoded, groups, balanced_weights)
    group_ids = np.repeat(np.arange(len(groups), dtype=np.int32), groups)
    strict = encoded == _STRICT_LABEL
    baseline = encoded == _BASELINE_LABEL
    safe = encoded != _UNSAFE_LABEL
    precision_safe = strict | baseline
    strict_counts = np.bincount(group_ids, weights=strict, minlength=len(groups))
    safe_counts = np.bincount(group_ids, weights=safe, minlength=len(groups))
    precision_counts = np.bincount(
        group_ids,
        weights=precision_safe,
        minlength=len(groups),
    )
    capture = np.where(
        strict_counts[group_ids] > 0,
        strict / np.maximum(strict_counts[group_ids], 1.0),
        baseline.astype(np.float64),
    )
    safety = safe / safe_counts[group_ids]
    precision = precision_safe / precision_counts[group_ids]
    target = (capture + safety_weight * safety + precision_weight * precision) / (
        1.0 + safety_weight + precision_weight
    )
    starts = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(groups[:-1], dtype=np.int64))
    )
    if not np.allclose(np.add.reduceat(target, starts), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("Stage203 objective target distributions do not sum to one")
    if not np.all(np.isfinite(target)) or np.any(target < 0.0):
        raise ValueError("Stage203 objective target contains invalid probabilities")
    return target.astype(np.float64, copy=False)


def fit_predict_top1_partition(
    training_rows: Sequence[ActionAuditRow],
    heldout_rows: Sequence[ActionAuditRow],
    feature_indices: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
    source_spec: SafetyFirstFrontierPolicySpec,
    objective_specs: Sequence[Top1ObjectiveSpec],
) -> Top1PartitionResult:
    """Fit the exact four-model source and the requested grouped Top-1 rankers."""

    import lightgbm as lgb

    specs = tuple(objective_specs)
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Stage203 objective specs must be unique")
    source = stage196.fit_predict_safety_first_spec(
        training_rows,
        heldout_rows,
        feature_indices,
        source_spec,
    )
    training = tuple(sorted(training_rows, key=stage194._row_key))
    heldout = tuple(sorted(heldout_rows, key=stage194._row_key))
    representation = source_spec.gain_feature_representation
    vectorizer = DictVectorizer(sparse=True)
    train_matrix = vectorizer.fit_transform(
        [dict(feature_indices[representation][stage194._row_key(row)]) for row in training]
    ).tocsr()
    heldout_matrix = vectorizer.transform(
        [dict(feature_indices[representation][stage194._row_key(row)]) for row in heldout]
    ).tocsr()
    labels = np.asarray([_outcome_label(row) for row in training], dtype=np.int8)
    group_sizes = np.asarray(stage194._question_group_sizes(training), dtype=np.int32)
    weights = stage194._question_balanced_weights(training)
    _validate_group_contract(labels, group_sizes, weights)
    predictions: dict[str, tuple[Top1ScorePrediction, ...]] = {}
    tree_count = source.tree_count
    custom_tree_count = 0
    callback_count = 0
    for spec in specs:
        objective = GroupedTop1Objective(
            labels=labels,
            group_sizes=group_sizes,
            sample_weights=weights,
            spec=spec,
        )
        objective_function = _objective_callable(objective)
        ranker = lgb.LGBMRanker(
            objective=objective_function,
            metric="None",
            **{
                **stage196._lightgbm_common_parameters(),
                **stage196._TREE_PROFILES[source_spec.gain_tree_profile],
            },
        )
        ranker.fit(
            train_matrix,
            labels,
            group=group_sizes,
            sample_weight=weights,
        )
        values = np.asarray(ranker.predict(heldout_matrix), dtype=np.float64)
        if values.shape != (len(heldout),) or not np.all(np.isfinite(values)):
            raise ValueError(f"Stage203 {spec.name} produced invalid heldout scores")
        diagnostics = objective.diagnostics()
        if diagnostics["target_sum_max_error"] > 1e-12:
            raise ValueError(f"Stage203 {spec.name} target normalization drifted")
        if not 1 <= diagnostics["callback_call_count"] <= 300:
            raise ValueError(f"Stage203 {spec.name} callback count is outside 1..300")
        predictions[spec.name] = tuple(
            Top1ScorePrediction(row, float(score))
            for row, score in zip(heldout, values, strict=True)
        )
        actual_tree_count = int(ranker.booster_.num_trees())
        tree_count += actual_tree_count
        custom_tree_count += actual_tree_count
        callback_count += objective.call_count
        del ranker, objective, objective_function, values
        gc.collect()
    feature_counts = dict(source.feature_count_by_representation)
    feature_counts[representation] = max(
        feature_counts.get(representation, 0), len(vectorizer.feature_names_)
    )
    del train_matrix, heldout_matrix, vectorizer
    gc.collect()
    return Top1PartitionResult(
        safety_predictions=source.safety_predictions,
        gain_predictions=source.gain_predictions,
        risk_predictions=source.risk_predictions,
        objective_predictions=dict(sorted(predictions.items())),
        feature_count_by_representation=dict(sorted(feature_counts.items())),
        model_fit_count=source.model_fit_count + len(specs),
        source_model_fit_count=source.model_fit_count,
        custom_objective_fit_count=len(specs),
        tree_count=tree_count,
        source_tree_count=source.tree_count,
        custom_objective_tree_count=custom_tree_count,
        group_contract_validation_count=source.group_contract_validation_count + len(specs),
        objective_callback_call_count=callback_count,
    )


def build_top1_decisions(
    safety_predictions: Sequence[stage194.SafetyPrediction],
    score_predictions: Sequence[Top1ScorePrediction],
) -> tuple[SafetyFirstFrontierDecision, ...]:
    safety_grouped = stage194._group_predictions(safety_predictions)
    score_index = {stage194._row_key(row.row): row.score for row in score_predictions}
    expected = {stage194._row_key(row.row) for row in safety_predictions}
    if expected != set(score_index):
        raise ValueError("Stage203 safety and objective prediction rows differ")
    decisions = []
    for question_key, question_safety in sorted(safety_grouped.items()):
        baselines = [row for row in question_safety if row.row.action.family == "baseline"]
        if len(baselines) != 1:
            raise ValueError("Stage203 requires one baseline action per question")
        ranked_safety = sorted(question_safety, key=stage194._safety_order_key)
        pool_index = {stage194._row_key(row.row): row for row in ranked_safety[:_POOL_CAP]}
        baseline_safety = baselines[0]
        pool_index[stage194._row_key(baseline_safety.row)] = baseline_safety
        pool_safety = tuple(sorted(pool_index.values(), key=stage194._safety_order_key))
        complete_pool = tuple(
            FrontierActionPrediction(
                row.row,
                row.citation_loss_probability,
                row.f1_loss_probability,
                score_index[stage194._row_key(row.row)],
                max(row.citation_loss_probability, row.f1_loss_probability),
            )
            for row in pool_safety
        )
        winner = min(
            complete_pool,
            key=lambda row: (-row.gain_score, row.unsafe_score, row.row.action.action_id),
        )
        baseline = next(row for row in complete_pool if row.row.action.family == "baseline")
        decisions.append(
            SafetyFirstFrontierDecision(
                question_key=question_key,
                baseline=baseline,
                complete_pool=complete_pool,
                frontier=complete_pool,
                winner=winner,
                strict_opportunity=any(row.row.strict_expected for row in question_safety),
                action_count=len(question_safety),
                strict_action_count=sum(row.row.strict_expected for row in question_safety),
            )
        )
    return tuple(decisions)


def evaluate_top1_decisions(
    decisions: Sequence[SafetyFirstFrontierDecision],
    *,
    expected_fold_ids: Sequence[str],
) -> dict[str, Any]:
    aggregate = stage196._FrontierDiagnostics()
    folds = {fold_id: stage196._FrontierDiagnostics() for fold_id in expected_fold_ids}
    for decision in decisions:
        fold_id = decision.winner.row.fold_id
        if fold_id not in folds:
            raise ValueError(f"Stage203 observed unexpected fold {fold_id}")
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
    return report


def run_top1_joint_objective_nested_cv(
    *,
    action_rows: Sequence[ActionAuditRow],
    stage182_selected_actions: Sequence[SelectedAction],
    stage202_protocol: Mapping[str, Any],
    stage199_report: Mapping[str, Any],
    progress_sink: ProgressSink | None = None,
    partition_fit_predictor: Top1PartitionFitPredictor | None = None,
    diagnostic_sink: DiagnosticSink | None = None,
) -> dict[str, Any]:
    """Run the frozen Stage 203 five-by-four train-only nested CV."""

    started_at = time.perf_counter()
    rows = tuple(action_rows)
    fold_ids = tuple(sorted({row.fold_id for row in rows}))
    if len(fold_ids) != 5:
        raise ValueError("Stage203 requires exactly five frozen folds")
    grouped = stage194._group_rows(rows)
    references = build_stage182_reference_rows(rows, stage182_selected_actions)
    base_features = build_composition_feature_indices(rows)
    feature_indices = {
        "raw_runtime": base_features["raw"],
        "question_relative_runtime": base_features["question_relative"],
    }
    trajectories = {
        row["outer_context"]: stage196._spec_from_dict(row["source_spec"])
        for row in stage202_protocol["frozen_protocol"]["source_trajectory_contract"][
            "trajectories"
        ]
    }
    source_evidence = stage199_report["joint_risk_winner_nested_cv"]["outer_contexts"]
    objective_specs = stage203_objective_specs()
    fit_predict = partition_fit_predictor or fit_predict_top1_partition
    execution = _FitTotals()
    feature_counts: dict[str, int] = {}
    private_prediction_count = 0
    fit_seconds = 0.0
    outer_reports: dict[str, Any] = {}
    outer_rows: list[ActionAuditRow] = []
    outer_diagnostics = stage196._FrontierDiagnostics()
    cell_reports: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_specs: Counter[str] = Counter()
    selected_ablation_families: Counter[str] = Counter()
    control_reproduction_count = 0
    outer_custom_refit_count = 0

    for outer_fold_id in fold_ids:
        source_spec = trajectories[outer_fold_id]
        outer_training = tuple(row for row in rows if row.fold_id != outer_fold_id)
        outer_heldout = tuple(row for row in rows if row.fold_id == outer_fold_id)
        inner_fold_ids = tuple(fold for fold in fold_ids if fold != outer_fold_id)
        safety_predictions: list[stage194.SafetyPrediction] = []
        gain_predictions: list[stage196.GainPrediction] = []
        risk_predictions: list[stage196.UnsafePrediction] = []
        objective_predictions: dict[str, list[Top1ScorePrediction]] = {
            spec.name: [] for spec in objective_specs
        }
        for inner_fold_id in inner_fold_ids:
            training = tuple(row for row in outer_training if row.fold_id != inner_fold_id)
            heldout = tuple(row for row in outer_training if row.fold_id == inner_fold_id)
            fitted_at = time.perf_counter()
            result = fit_predict(
                training,
                heldout,
                feature_indices,
                source_spec,
                objective_specs,
            )
            fit_seconds += time.perf_counter() - fitted_at
            execution.add(result)
            private_prediction_count += (4 + len(objective_specs)) * len(heldout)
            safety_predictions.extend(result.safety_predictions)
            gain_predictions.extend(result.gain_predictions)
            risk_predictions.extend(result.risk_predictions)
            for spec in objective_specs:
                objective_predictions[spec.name].extend(result.objective_predictions[spec.name])
            for name, count in result.feature_count_by_representation.items():
                feature_counts[name] = max(feature_counts.get(name, 0), count)
            _emit(
                progress_sink,
                phase="stage203_inner_partition_complete",
                outer_fold_id=outer_fold_id,
                inner_fold_id=inner_fold_id,
                cumulative_model_fit_count=execution.model_fit_count,
                cumulative_tree_count=execution.tree_count,
            )

        question_count = len({row.question_key for row in outer_training})
        control_decisions = stage196.build_safety_first_frontier_decisions(
            safety_predictions,
            gain_predictions,
            risk_predictions,
            source_spec,
        )
        decisions_by_spec: dict[str, tuple[SafetyFirstFrontierDecision, ...]] = {
            _CONTROL_NAME: control_decisions
        }
        control = _candidate_report(
            spec=_control_spec_dict(),
            decisions=control_decisions,
            references=references,
            expected_fold_ids=inner_fold_ids,
            question_count=question_count,
        )
        formal_control = _stage199_control_evidence(source_evidence[outer_fold_id])
        if not stage199._nested_close(
            control["evaluation"], formal_control["evaluation"]
        ) or not stage199._nested_close(control["diagnostics"], formal_control["diagnostics"]):
            raise ValueError(f"Stage203 control did not reproduce {outer_fold_id}")
        control_reproduction_count += 1
        candidates = [control]
        for spec in objective_specs:
            decisions = build_top1_decisions(
                safety_predictions,
                objective_predictions[spec.name],
            )
            decisions_by_spec[spec.name] = decisions
            candidates.append(
                _candidate_report(
                    spec=_objective_spec_dict(spec),
                    decisions=decisions,
                    references=references,
                    expected_fold_ids=inner_fold_ids,
                    question_count=question_count,
                )
            )
        strict_only = next(row for row in candidates if row["spec"]["name"] == _STRICT_ONLY_NAME)
        for candidate in candidates:
            candidate["paired_vs_control"] = stage199._paired_delta(candidate, control)
            candidate["paired_vs_strict_only"] = stage199._paired_delta(candidate, strict_only)
            cell_reports[candidate["spec"]["name"]].append(candidate)

        if diagnostic_sink is not None:
            diagnostic_sink(
                _diagnostic_snapshot(
                    outer_fold_id=outer_fold_id,
                    inner_fold_ids=inner_fold_ids,
                    inner_question_count=question_count,
                    candidates=candidates,
                    decisions_by_spec=decisions_by_spec,
                )
            )

        eligible = [row for row in candidates if row["eligible"]]
        ranked = sorted(candidates, key=stage196._inner_selection_key)
        public_top = [_public_candidate(row) for row in ranked[:5]]
        if not eligible:
            outer_reports[outer_fold_id] = {
                "source_spec": stage196._spec_dict(source_spec),
                "inner_question_count": question_count,
                "control_reproduction_exact": True,
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
                phase="stage203_outer_context_no_eligible_config",
                outer_fold_id=outer_fold_id,
            )
            continue

        selected = min(eligible, key=stage196._inner_selection_key)
        selected_spec = selected["spec"]
        selected_specs[str(selected_spec["name"])] += 1
        selected_ablation_families[str(selected_spec["ablation_family"])] += 1
        selected_objectives = (
            ()
            if selected_spec["ablation_family"] == "exact_control"
            else (_objective_spec_from_dict(selected_spec),)
        )
        outer_custom_refit_count += len(selected_objectives)
        fitted_at = time.perf_counter()
        heldout_result = fit_predict(
            outer_training,
            outer_heldout,
            feature_indices,
            source_spec,
            selected_objectives,
        )
        fit_seconds += time.perf_counter() - fitted_at
        execution.add(heldout_result)
        private_prediction_count += (4 + len(selected_objectives)) * len(outer_heldout)
        for name, count in heldout_result.feature_count_by_representation.items():
            feature_counts[name] = max(feature_counts.get(name, 0), count)
        if selected_spec["ablation_family"] == "exact_control":
            decisions = stage196.build_safety_first_frontier_decisions(
                heldout_result.safety_predictions,
                heldout_result.gain_predictions,
                heldout_result.risk_predictions,
                source_spec,
            )
        else:
            decisions = build_top1_decisions(
                heldout_result.safety_predictions,
                heldout_result.objective_predictions[str(selected_spec["name"])],
            )
        diagnostics = evaluate_top1_decisions(
            decisions,
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
        outer_reports[outer_fold_id] = {
            "source_spec": stage196._spec_dict(source_spec),
            "inner_question_count": question_count,
            "control_reproduction_exact": True,
            "eligible_config_count": len(eligible),
            "selected_spec": selected_spec,
            "selected_inner_evaluation": selected["evaluation"],
            "selected_inner_diagnostics": selected["diagnostics"],
            "outer_evaluation": evaluation,
            "outer_diagnostics": diagnostics,
            "top_inner_candidates": public_top,
            "outer_evaluated": True,
        }
        del heldout_result
        gc.collect()
        _emit(
            progress_sink,
            phase="stage203_outer_context_complete",
            outer_fold_id=outer_fold_id,
            selected_spec=selected_spec["name"],
            eligible_config_count=len(eligible),
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
    cell_aggregates = {
        name: _aggregate_cell_reports(reports) for name, reports in sorted(cell_reports.items())
    }
    return {
        "protocol": {
            "outer_fold_count": 5,
            "inner_fold_count": 4,
            "custom_objective_count": len(objective_specs),
            "exact_control_count": 1,
            "candidate_config_count_per_outer_context": len(objective_specs) + 1,
            "pool_cap": _POOL_CAP,
            "model_fits_per_inner_partition": 20,
            "maximum_model_fit_count": 425,
            "maximum_lightgbm_tree_count": 112_500,
            "fallback_enabled": False,
        },
        "dataset": {
            "action_count": len(rows),
            "question_count": len(grouped),
            "reference_action_count": len(references),
            "fold_action_counts": {
                fold_id: sum(row.fold_id == fold_id for row in rows) for fold_id in fold_ids
            },
        },
        "outer_contexts": outer_reports,
        "aggregate": aggregate,
        "aggregate_diagnostics": aggregate_diagnostics,
        "paired_bootstrap": bootstrap,
        "cell_aggregates": cell_aggregates,
        "ablation_family_aggregates": _factor_aggregates(
            cell_aggregates,
            "ablation_family",
        ),
        "safety_weight_aggregates": _numeric_factor_aggregates(
            cell_aggregates,
            "safety_weight",
        ),
        "precision_weight_aggregates": _numeric_factor_aggregates(
            cell_aggregates,
            "precision_weight",
        ),
        "directional_penalty_response": _directional_penalty_response(cell_aggregates),
        "selected_spec_counts": dict(sorted(selected_specs.items())),
        "selected_ablation_family_counts": dict(sorted(selected_ablation_families.items())),
        "advancement_gates": gates,
        "advancement_gate_pass_count": sum(row["passed"] for row in gates),
        "candidate_family_accepted": all(row["passed"] for row in gates),
        "execution": {
            "model_fit_count": execution.model_fit_count,
            "source_model_fit_count": execution.source_model_fit_count,
            "pool_safety_fit_count": execution.pool_safety_fit_count,
            "gain_ranker_fit_count": execution.gain_ranker_fit_count,
            "classifier_risk_fit_count": execution.classifier_risk_fit_count,
            "custom_objective_fit_count": execution.custom_objective_fit_count,
            "outer_custom_objective_refit_count": outer_custom_refit_count,
            "tree_count": execution.tree_count,
            "source_tree_count": execution.source_tree_count,
            "custom_objective_tree_count": execution.custom_objective_tree_count,
            "group_contract_validation_count": execution.group_contract_validation_count,
            "objective_callback_call_count": execution.objective_callback_call_count,
            "control_reproduction_count": control_reproduction_count,
            "all_controls_reproduced_exactly": control_reproduction_count == 5,
            "private_prediction_count": private_prediction_count,
            "public_training_rows_written": 0,
            "public_prediction_rows_written": 0,
            "feature_count_by_representation": dict(sorted(feature_counts.items())),
            "fit_seconds": round(fit_seconds, 6),
            "wall_seconds": round(time.perf_counter() - started_at, 6),
        },
    }


def _candidate_report(
    *,
    spec: Mapping[str, Any],
    decisions: Sequence[SafetyFirstFrontierDecision],
    references: Mapping[str, ActionAuditRow],
    expected_fold_ids: Sequence[str],
    question_count: int,
) -> dict[str, Any]:
    diagnostics = evaluate_top1_decisions(decisions, expected_fold_ids=expected_fold_ids)
    evaluation = evaluate_selected_actions(
        selected_rows=tuple(decision.winner.row for decision in decisions),
        references=references,
        expected_fold_ids=expected_fold_ids,
    )
    return {
        "spec": dict(spec),
        "eligible": stage194._inner_eligible(evaluation, diagnostics, question_count),
        "evaluation": evaluation,
        "diagnostics": diagnostics,
    }


def _aggregate_cell_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = stage199._aggregate_cell_reports(reports)
    aggregate["paired_vs_strict_only"] = {
        name: round(sum(row["paired_vs_strict_only"][name] for row in reports), 6)
        for name in reports[0]["paired_vs_strict_only"]
    }
    return aggregate


def _factor_aggregates(
    cells: Mapping[str, Mapping[str, Any]],
    dimension: str,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells.values():
        grouped[str(cell["spec"][dimension])].append(cell)
    return {value: _factor_row(rows) for value, rows in sorted(grouped.items())}


def _numeric_factor_aggregates(
    cells: Mapping[str, Mapping[str, Any]],
    dimension: str,
) -> dict[str, Any]:
    grouped: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells.values():
        value = cell["spec"][dimension]
        if value is not None:
            grouped[float(value)].append(cell)
    return {f"{value:.2f}": _factor_row(rows) for value, rows in sorted(grouped.items())}


def _factor_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "cell_count": len(rows),
        "mean_unsafe_selection_rate": _mean(row["unsafe_selection_rate"] for row in rows),
        "mean_conditional_capture": _mean(row["conditional_ranker_strict_capture"] for row in rows),
        "mean_strict_success_precision": _mean(row["strict_success_precision"] for row in rows),
        "mean_gold_citation_delta": _mean(row["gold_citation_delta"] for row in rows),
        "mean_f1_delta": _mean(row["mean_f1_delta"] for row in rows),
        "best_cell": min(rows, key=stage199._aggregate_selection_key)["spec"]["name"],
    }


def _directional_penalty_response(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    custom = [row for row in cells.values() if row["spec"]["safety_weight"] is not None]
    safety_rows = []
    for precision in _PRECISION_WEIGHTS:
        ordered = sorted(
            (row for row in custom if row["spec"]["precision_weight"] == precision),
            key=lambda row: row["spec"]["safety_weight"],
        )
        safety_rows.extend(
            {
                "precision_weight": precision,
                "lower_safety_weight": left["spec"]["safety_weight"],
                "higher_safety_weight": right["spec"]["safety_weight"],
                "unsafe_rate_delta": round(
                    right["unsafe_selection_rate"] - left["unsafe_selection_rate"], 6
                ),
                "nonincreasing_unsafe_rate": (
                    right["unsafe_selection_rate"] <= left["unsafe_selection_rate"]
                ),
            }
            for left, right in zip(ordered, ordered[1:], strict=False)
        )
    precision_rows = []
    for safety in _SAFETY_WEIGHTS:
        ordered = sorted(
            (row for row in custom if row["spec"]["safety_weight"] == safety),
            key=lambda row: row["spec"]["precision_weight"],
        )
        precision_rows.extend(
            {
                "safety_weight": safety,
                "lower_precision_weight": left["spec"]["precision_weight"],
                "higher_precision_weight": right["spec"]["precision_weight"],
                "strict_precision_delta": round(
                    right["strict_success_precision"] - left["strict_success_precision"], 6
                ),
                "nondecreasing_strict_precision": (
                    right["strict_success_precision"] >= left["strict_success_precision"]
                ),
            }
            for left, right in zip(ordered, ordered[1:], strict=False)
        )
    return {
        "safety_adjacent_comparison_count": len(safety_rows),
        "safety_nonincreasing_unsafe_count": sum(
            row["nonincreasing_unsafe_rate"] for row in safety_rows
        ),
        "safety_rows": safety_rows,
        "precision_adjacent_comparison_count": len(precision_rows),
        "precision_nondecreasing_strict_precision_count": sum(
            row["nondecreasing_strict_precision"] for row in precision_rows
        ),
        "precision_rows": precision_rows,
        "directional_response_is_acceptance_gate": False,
    }


def _control_spec_dict() -> dict[str, Any]:
    return {
        "name": _CONTROL_NAME,
        "safety_weight": None,
        "precision_weight": None,
        "ablation_family": "exact_control",
    }


def _stage199_control_evidence(outer_context: Mapping[str, Any]) -> Mapping[str, Any]:
    if outer_context.get("control_reproduction_exact") is not True:
        raise ValueError("Stage203 requires exact Stage199 control reproduction")
    candidates = outer_context.get("top_inner_candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("Stage203 requires Stage199 top-inner candidate evidence")
    controls = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and isinstance(candidate.get("spec"), Mapping)
        and candidate["spec"].get("risk_signal") == "source_weighted_classifier"
        and candidate["spec"].get("winner_rule") == "gain_only"
    ]
    if len(controls) != 1:
        raise ValueError("Stage203 requires exactly one published Stage199 control candidate")
    control = controls[0]
    if not isinstance(control.get("evaluation"), Mapping) or not isinstance(
        control.get("diagnostics"), Mapping
    ):
        raise ValueError("Stage203 requires complete Stage199 control evidence")
    return control


def _objective_spec_dict(spec: Top1ObjectiveSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "safety_weight": spec.safety_weight,
        "precision_weight": spec.precision_weight,
        "ablation_family": spec.ablation_family,
    }


def _objective_spec_from_dict(value: Mapping[str, Any]) -> Top1ObjectiveSpec:
    return Top1ObjectiveSpec(
        name=str(value["name"]),
        safety_weight=float(value["safety_weight"]),
        precision_weight=float(value["precision_weight"]),
        ablation_family=str(value["ablation_family"]),
    )


def _public_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "spec": row["spec"],
        "eligible": row["eligible"],
        "evaluation": row["evaluation"],
        "diagnostics": row["diagnostics"],
        "paired_vs_control": row["paired_vs_control"],
        "paired_vs_strict_only": row["paired_vs_strict_only"],
    }


def _diagnostic_snapshot(
    *,
    outer_fold_id: str,
    inner_fold_ids: Sequence[str],
    inner_question_count: int,
    candidates: Sequence[Mapping[str, Any]],
    decisions_by_spec: Mapping[str, tuple[SafetyFirstFrontierDecision, ...]],
) -> Top1ObjectiveDiagnosticSnapshot:
    return Top1ObjectiveDiagnosticSnapshot(
        outer_fold_id=outer_fold_id,
        inner_fold_ids=tuple(inner_fold_ids),
        inner_question_count=inner_question_count,
        candidates=tuple(
            Top1ObjectiveCandidateSnapshot(
                spec=_deep_freeze(candidate["spec"]),
                eligible=bool(candidate["eligible"]),
                evaluation=_deep_freeze(candidate["evaluation"]),
                diagnostics=_deep_freeze(candidate["diagnostics"]),
                paired_vs_control=_deep_freeze(candidate["paired_vs_control"]),
                paired_vs_strict_only=_deep_freeze(candidate["paired_vs_strict_only"]),
                decisions=tuple(decisions_by_spec[str(candidate["spec"]["name"])]),
            )
            for candidate in candidates
        ),
    )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(child) for key, child in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_deep_freeze(child) for child in value)
    return value


def _outcome_label(row: ActionAuditRow) -> int:
    if row.action.family == "baseline":
        return _BASELINE_LABEL
    if row.strict_expected:
        return _STRICT_LABEL
    if stage194._is_unsafe(row):
        return _UNSAFE_LABEL
    if stage194._is_safe_zero(row):
        return _SAFE_ZERO_LABEL
    raise ValueError("Stage203 row is outside the frozen four-way outcome encoding")


def _objective_callable(
    objective: GroupedTop1Objective,
) -> Callable[
    [np.ndarray, np.ndarray, np.ndarray | None, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    # LightGBM deep-copies callable objects; a closure preserves the audited instance state.
    def grouped_objective(
        labels: np.ndarray,
        predictions: np.ndarray,
        weights: np.ndarray | None,
        group_sizes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return objective(labels, predictions, weights, group_sizes)

    return grouped_objective


def _validate_group_contract(
    labels: np.ndarray,
    group_sizes: np.ndarray,
    weights: np.ndarray,
) -> None:
    if labels.ndim != 1 or group_sizes.ndim != 1 or weights.ndim != 1:
        raise ValueError("Stage203 grouped arrays must be one-dimensional")
    if len(labels) == 0 or len(group_sizes) == 0 or int(group_sizes.sum()) != len(labels):
        raise ValueError("Stage203 group sizes must sum to nonempty labels")
    if len(weights) != len(labels) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Stage203 sample weights must be finite and positive")
    if np.any(group_sizes <= 0):
        raise ValueError("Stage203 groups must be nonempty")
    if not set(np.unique(labels)).issubset(
        {_UNSAFE_LABEL, _SAFE_ZERO_LABEL, _BASELINE_LABEL, _STRICT_LABEL}
    ):
        raise ValueError("Stage203 labels use an unknown outcome encoding")
    starts = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(group_sizes[:-1], dtype=np.int64))
    )
    baseline_counts = np.add.reduceat(labels == _BASELINE_LABEL, starts)
    if not np.all(baseline_counts == 1):
        raise ValueError("Stage203 requires exactly one baseline label per question")
    group_weight_sums = np.add.reduceat(weights, starts)
    if not np.allclose(group_weight_sums, 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("Stage203 question-balanced weights must sum to one per group")


def _mean(values: Sequence[float] | Any) -> float:
    rows = tuple(values)
    return round(float(sum(rows) / len(rows)), 6) if rows else 0.0


def _emit(progress_sink: ProgressSink | None, **event: Any) -> None:
    if progress_sink is not None:
        progress_sink(event)
