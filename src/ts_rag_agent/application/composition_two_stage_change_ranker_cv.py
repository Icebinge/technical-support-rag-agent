from __future__ import annotations

import gc
import hashlib
import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ts_rag_agent.application import composition_joint_risk_winner_cv as stage199
from ts_rag_agent.application import composition_safety_constrained_lambdamart as stage194
from ts_rag_agent.application import composition_safety_first_frontier as stage196
from ts_rag_agent.application import composition_top1_joint_objective_cv as stage203
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

_POOL_CAP = 16
_GATE_CROSSFIT_FOLDS = 4
_TARGET_CHANGE_COVERAGES = (0.25, 0.40, 0.55, 0.70, 0.85)
_CONTROL_NAME = "stage196_exact_control"
_CROSSFIT_SEED = 205


@dataclass(frozen=True)
class TwoStagePolicySpec:
    name: str
    ranker_family: str
    target_change_coverage: float


@dataclass(frozen=True)
class ConditionalRankerSpec:
    name: str
    labels: Mapping[str, int]
    label_gain: tuple[int, ...]


@dataclass(frozen=True)
class RankerScorePrediction:
    row: ActionAuditRow
    score: float


@dataclass(frozen=True)
class _QuestionCandidate:
    question_key: str
    candidate: FrontierActionPrediction
    baseline: FrontierActionPrediction
    complete_pool: tuple[FrontierActionPrediction, ...]
    gate_features: Mapping[str, Any]
    gate_label: int
    strict_opportunity: bool
    action_count: int
    strict_action_count: int


@dataclass(frozen=True)
class TwoStagePartitionResult:
    safety_predictions: tuple[stage194.SafetyPrediction, ...]
    gain_predictions: tuple[stage196.GainPrediction, ...]
    risk_predictions: tuple[stage196.UnsafePrediction, ...]
    policy_decisions: Mapping[str, tuple[SafetyFirstFrontierDecision, ...]]
    gate_diagnostics: Mapping[str, Mapping[str, Any]]
    feature_count_by_representation: Mapping[str, int]
    model_fit_count: int
    source_model_fit_count: int
    source_safety_crossfit_fit_count: int
    conditional_ranker_fit_count: int
    gate_fit_count: int
    lightgbm_model_fit_count: int
    tree_count: int
    source_tree_count: int
    conditional_ranker_tree_count: int
    gate_tree_count: int
    source_group_contract_validation_count: int
    ranker_group_contract_validation_count: int
    source_safety_oof_prediction_count: int
    ranker_oof_prediction_count: int
    gate_training_question_count: int
    private_prediction_count: int


class TwoStagePartitionFitPredictor(Protocol):
    def __call__(
        self,
        training_rows: Sequence[ActionAuditRow],
        heldout_rows: Sequence[ActionAuditRow],
        feature_indices: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
        source_spec: SafetyFirstFrontierPolicySpec,
        policy_specs: Sequence[TwoStagePolicySpec],
        outer_context: str,
        heldout_context: str,
    ) -> TwoStagePartitionResult: ...


@dataclass
class _FitTotals:
    model_fit_count: int = 0
    source_model_fit_count: int = 0
    source_safety_crossfit_fit_count: int = 0
    conditional_ranker_fit_count: int = 0
    gate_fit_count: int = 0
    lightgbm_model_fit_count: int = 0
    tree_count: int = 0
    source_tree_count: int = 0
    conditional_ranker_tree_count: int = 0
    gate_tree_count: int = 0
    source_group_contract_validation_count: int = 0
    ranker_group_contract_validation_count: int = 0
    source_safety_oof_prediction_count: int = 0
    ranker_oof_prediction_count: int = 0
    gate_training_question_count: int = 0
    private_prediction_count: int = 0

    def add(self, result: TwoStagePartitionResult) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(result, name))


def stage206_policy_specs() -> tuple[TwoStagePolicySpec, ...]:
    return tuple(
        TwoStagePolicySpec(
            name=f"{family}__change_c{int(round(coverage * 100)):02d}",
            ranker_family=family,
            target_change_coverage=coverage,
        )
        for family in ("strict_binary", "strict_safety_graded")
        for coverage in _TARGET_CHANGE_COVERAGES
    )


def conditional_ranker_specs() -> Mapping[str, ConditionalRankerSpec]:
    return {
        "strict_binary": ConditionalRankerSpec(
            "strict_binary",
            {"unsafe": 0, "safe_zero": 0, "strict_success": 1},
            (0, 1),
        ),
        "strict_safety_graded": ConditionalRankerSpec(
            "strict_safety_graded",
            {"unsafe": 0, "safe_zero": 1, "strict_success": 2},
            (0, 1, 4),
        ),
    }


def gate_crossfit_index(
    question_key: str,
    *,
    outer_context: str,
    heldout_context: str,
) -> int:
    value = f"{outer_context}|{heldout_context}|{question_key}|{_CROSSFIT_SEED}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % _GATE_CROSSFIT_FOLDS


def fit_predict_two_stage_partition(
    training_rows: Sequence[ActionAuditRow],
    heldout_rows: Sequence[ActionAuditRow],
    feature_indices: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
    source_spec: SafetyFirstFrontierPolicySpec,
    policy_specs: Sequence[TwoStagePolicySpec],
    outer_context: str,
    heldout_context: str,
) -> TwoStagePartitionResult:
    """Fit one leakage-resistant Stage206 partition and predict its heldout rows."""

    policies = tuple(policy_specs)
    if len({spec.name for spec in policies}) != len(policies):
        raise ValueError("Stage206 policy specs must be unique")
    families = tuple(sorted({spec.ranker_family for spec in policies}))
    ranker_specs = conditional_ranker_specs()
    if not set(families).issubset(ranker_specs):
        raise ValueError("Stage206 received an unknown conditional ranker family")
    training = tuple(sorted(training_rows, key=stage194._row_key))
    heldout = tuple(sorted(heldout_rows, key=stage194._row_key))
    _validate_question_partition(training, heldout)
    source = stage196.fit_predict_safety_first_spec(
        training,
        heldout,
        feature_indices,
        source_spec,
    )
    feature_counts = dict(source.feature_count_by_representation)
    if not policies:
        return TwoStagePartitionResult(
            safety_predictions=source.safety_predictions,
            gain_predictions=source.gain_predictions,
            risk_predictions=source.risk_predictions,
            policy_decisions={},
            gate_diagnostics={},
            feature_count_by_representation=feature_counts,
            model_fit_count=4,
            source_model_fit_count=4,
            source_safety_crossfit_fit_count=0,
            conditional_ranker_fit_count=0,
            gate_fit_count=0,
            lightgbm_model_fit_count=2,
            tree_count=source.tree_count,
            source_tree_count=source.tree_count,
            conditional_ranker_tree_count=0,
            gate_tree_count=0,
            source_group_contract_validation_count=1,
            ranker_group_contract_validation_count=0,
            source_safety_oof_prediction_count=0,
            ranker_oof_prediction_count=0,
            gate_training_question_count=0,
            private_prediction_count=4 * len(heldout),
        )

    assignments = {
        question_key: gate_crossfit_index(
            question_key,
            outer_context=outer_context,
            heldout_context=heldout_context,
        )
        for question_key in sorted({row.question_key for row in training})
    }
    if set(assignments.values()) != set(range(_GATE_CROSSFIT_FOLDS)):
        raise ValueError("Stage206 gate crossfit assignment must populate all four folds")
    oof_safety: list[stage194.SafetyPrediction] = []
    oof_ranker: dict[str, list[RankerScorePrediction]] = {family: [] for family in families}
    ranker_tree_count = 0
    ranker_fit_count = 0
    ranker_validation_count = 0
    source_safety_fit_count = 0
    source_safety_oof_prediction_count = 0
    ranker_oof_prediction_count = 0
    for crossfit_id in range(_GATE_CROSSFIT_FOLDS):
        crossfit_training = tuple(
            row for row in training if assignments[row.question_key] != crossfit_id
        )
        crossfit_heldout = tuple(
            row for row in training if assignments[row.question_key] == crossfit_id
        )
        if not crossfit_training or not crossfit_heldout:
            raise ValueError("Stage206 gate crossfit fold must have train and heldout rows")
        safety, safety_feature_count = _fit_predict_source_safety_heads(
            crossfit_training,
            crossfit_heldout,
            feature_indices[source_spec.pool_feature_representation],
            source_spec.pool_safety_estimator,
        )
        oof_safety.extend(safety)
        source_safety_fit_count += 2
        source_safety_oof_prediction_count += 2 * len(crossfit_heldout)
        feature_counts[source_spec.pool_feature_representation] = max(
            feature_counts.get(source_spec.pool_feature_representation, 0),
            safety_feature_count,
        )
        for family in families:
            predictions, trees, feature_count = _fit_predict_conditional_ranker(
                crossfit_training,
                crossfit_heldout,
                feature_indices[source_spec.gain_feature_representation],
                source_spec.gain_tree_profile,
                ranker_specs[family],
            )
            oof_ranker[family].extend(predictions)
            ranker_tree_count += trees
            ranker_fit_count += 1
            ranker_validation_count += 1
            ranker_oof_prediction_count += len(predictions)
            feature_counts[source_spec.gain_feature_representation] = max(
                feature_counts.get(source_spec.gain_feature_representation, 0),
                feature_count,
            )

    expected_training_keys = {stage194._row_key(row) for row in training}
    oof_safety_keys = [stage194._row_key(row.row) for row in oof_safety]
    if len(oof_safety_keys) != len(expected_training_keys) or set(oof_safety_keys) != (
        expected_training_keys
    ):
        raise ValueError("Stage206 source-safety OOF predictions do not cover training rows once")
    for family in families:
        expected_ranker_keys = {
            stage194._row_key(row) for row in training if row.action.family != "baseline"
        }
        family_oof_keys = [stage194._row_key(row.row) for row in oof_ranker[family]]
        if len(family_oof_keys) != len(expected_ranker_keys) or set(family_oof_keys) != (
            expected_ranker_keys
        ):
            raise ValueError(
                f"Stage206 {family} OOF ranker predictions do not cover nonbaseline rows once"
            )
    policy_decisions: dict[str, tuple[SafetyFirstFrontierDecision, ...]] = {}
    gate_diagnostics: dict[str, Mapping[str, Any]] = {}
    gate_tree_count = 0
    gate_fit_count = 0
    private_prediction_count = (
        4 * len(heldout) + source_safety_oof_prediction_count + ranker_oof_prediction_count
    )
    for family in families:
        training_candidates = _build_question_candidates(
            oof_safety,
            oof_ranker[family],
            feature_indices,
        )
        gate_vectorizer, gate_model, gate_scores, train_metrics = _fit_gate(
            training_candidates,
            source_spec.risk_tree_profile,
        )
        actual_gate_trees = int(gate_model.booster_.num_trees())
        gate_tree_count += actual_gate_trees
        gate_fit_count += 1
        private_prediction_count += len(training_candidates)
        heldout_ranker, trees, feature_count = _fit_predict_conditional_ranker(
            training,
            heldout,
            feature_indices[source_spec.gain_feature_representation],
            source_spec.gain_tree_profile,
            ranker_specs[family],
        )
        ranker_tree_count += trees
        ranker_fit_count += 1
        ranker_validation_count += 1
        private_prediction_count += len(heldout_ranker)
        feature_counts[source_spec.gain_feature_representation] = max(
            feature_counts.get(source_spec.gain_feature_representation, 0),
            feature_count,
        )
        heldout_candidates = _build_question_candidates(
            source.safety_predictions,
            heldout_ranker,
            feature_indices,
        )
        heldout_matrix = gate_vectorizer.transform(
            [dict(row.gate_features) for row in heldout_candidates]
        )
        heldout_scores = np.asarray(
            gate_model.predict_proba(heldout_matrix)[:, 1], dtype=np.float64
        )
        private_prediction_count += len(heldout_candidates)
        heldout_metrics = _binary_metrics(
            [row.gate_label for row in heldout_candidates], heldout_scores
        )
        for policy in (row for row in policies if row.ranker_family == family):
            threshold = _coverage_threshold(gate_scores, policy.target_change_coverage)
            decisions = _build_gated_decisions(
                heldout_candidates,
                heldout_scores,
                threshold=threshold,
            )
            policy_decisions[policy.name] = decisions
            changed = sum(decision.winner.row.action.family != "baseline" for decision in decisions)
            pre_gate_strict = sum(row.candidate.row.strict_expected for row in heldout_candidates)
            pre_gate_unsafe = sum(
                stage194._is_unsafe(row.candidate.row) for row in heldout_candidates
            )
            gate_diagnostics[policy.name] = {
                "ranker_family": family,
                "target_change_coverage": policy.target_change_coverage,
                "learned_gate_threshold": round(threshold, 12),
                "training_gate": dict(train_metrics),
                "heldout_gate": heldout_metrics,
                "heldout_question_count": len(heldout_candidates),
                "realized_change_count": changed,
                "realized_change_coverage": _ratio(changed, len(heldout_candidates)),
                "pre_gate_ranker_strict_count": pre_gate_strict,
                "pre_gate_ranker_strict_rate": _ratio(pre_gate_strict, len(heldout_candidates)),
                "pre_gate_ranker_unsafe_count": pre_gate_unsafe,
                "pre_gate_ranker_unsafe_rate": _ratio(pre_gate_unsafe, len(heldout_candidates)),
            }
        del gate_vectorizer, gate_model, heldout_matrix, heldout_scores
        gc.collect()

    source_tree_count = source.tree_count
    model_fit_count = 4 + source_safety_fit_count + ranker_fit_count + gate_fit_count
    return TwoStagePartitionResult(
        safety_predictions=source.safety_predictions,
        gain_predictions=source.gain_predictions,
        risk_predictions=source.risk_predictions,
        policy_decisions=dict(sorted(policy_decisions.items())),
        gate_diagnostics=dict(sorted(gate_diagnostics.items())),
        feature_count_by_representation=dict(sorted(feature_counts.items())),
        model_fit_count=model_fit_count,
        source_model_fit_count=4,
        source_safety_crossfit_fit_count=source_safety_fit_count,
        conditional_ranker_fit_count=ranker_fit_count,
        gate_fit_count=gate_fit_count,
        lightgbm_model_fit_count=2 + ranker_fit_count + gate_fit_count,
        tree_count=source_tree_count + ranker_tree_count + gate_tree_count,
        source_tree_count=source_tree_count,
        conditional_ranker_tree_count=ranker_tree_count,
        gate_tree_count=gate_tree_count,
        source_group_contract_validation_count=1,
        ranker_group_contract_validation_count=ranker_validation_count,
        source_safety_oof_prediction_count=source_safety_oof_prediction_count,
        ranker_oof_prediction_count=ranker_oof_prediction_count,
        gate_training_question_count=len(assignments) * len(families),
        private_prediction_count=private_prediction_count,
    )


def _fit_predict_source_safety_heads(
    training_rows: Sequence[ActionAuditRow],
    heldout_rows: Sequence[ActionAuditRow],
    feature_index: Mapping[tuple[str, str], Mapping[str, Any]],
    estimator: str,
) -> tuple[tuple[stage194.SafetyPrediction, ...], int]:
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
        "f1_loss": np.asarray([stage194._is_f1_regression(row) for row in training], dtype=np.int8),
    }
    for name, values in labels.items():
        if len(set(values.tolist())) != 2:
            raise ValueError(f"Stage206 source safety {name} target requires both classes")
    predictions: dict[str, np.ndarray] = {}
    if estimator == "class_balanced_logistic":
        scaler = StandardScaler(with_mean=False)
        fitted_train = scaler.fit_transform(train_matrix).tocsr()
        fitted_heldout = scaler.transform(heldout_matrix).tocsr()
        for target, values in labels.items():
            head = stage196._SafetyHead(
                stage196._fit_logistic_classifier(fitted_train, values, weights), False
            )
            predictions[target] = head.predict(fitted_heldout)
    elif estimator == "histogram_gradient_boosting":
        fitted_train = train_matrix.toarray()
        for target, values in labels.items():
            head = stage196._SafetyHead(
                stage196._fit_histogram_classifier(fitted_train, values, weights), True
            )
            predictions[target] = head.predict(heldout_matrix)
    else:
        raise ValueError(f"Stage206 unsupported source safety estimator: {estimator}")
    result = tuple(
        stage194.SafetyPrediction(row, float(citation), float(f1))
        for row, citation, f1 in zip(
            heldout,
            predictions["citation_loss"],
            predictions["f1_loss"],
            strict=True,
        )
    )
    return result, len(vectorizer.feature_names_)


def _fit_predict_conditional_ranker(
    training_rows: Sequence[ActionAuditRow],
    heldout_rows: Sequence[ActionAuditRow],
    feature_index: Mapping[tuple[str, str], Mapping[str, Any]],
    tree_profile: str,
    spec: ConditionalRankerSpec,
) -> tuple[tuple[RankerScorePrediction, ...], int, int]:
    import lightgbm as lgb

    training = tuple(
        sorted(
            (row for row in training_rows if row.action.family != "baseline"),
            key=stage194._row_key,
        )
    )
    heldout = tuple(
        sorted(
            (row for row in heldout_rows if row.action.family != "baseline"),
            key=stage194._row_key,
        )
    )
    if not training or not heldout:
        raise ValueError("Stage206 conditional ranker requires nonbaseline train and heldout rows")
    vectorizer = DictVectorizer(sparse=True)
    train_matrix = vectorizer.fit_transform(
        [dict(feature_index[stage194._row_key(row)]) for row in training]
    ).tocsr()
    heldout_matrix = vectorizer.transform(
        [dict(feature_index[stage194._row_key(row)]) for row in heldout]
    ).tocsr()
    labels = np.asarray([_ranker_label(row, spec) for row in training], dtype=np.int8)
    if len(set(labels.tolist())) < 2:
        raise ValueError(f"Stage206 {spec.name} ranker target requires at least two classes")
    groups = stage194._question_group_sizes(training)
    if len(groups) != len({row.question_key for row in training}):
        raise ValueError("Stage206 conditional ranker group count drifted")
    weights = stage194._question_balanced_weights(training)
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        lambdarank_truncation_level=_POOL_CAP,
        lambdarank_norm=True,
        label_gain=list(spec.label_gain),
        **{
            **stage196._lightgbm_common_parameters(),
            **stage196._TREE_PROFILES[tree_profile],
        },
    )
    ranker.fit(train_matrix, labels, group=groups, sample_weight=weights, eval_at=[1])
    scores = np.asarray(ranker.predict(heldout_matrix), dtype=np.float64)
    if scores.shape != (len(heldout),) or not np.all(np.isfinite(scores)):
        raise ValueError(f"Stage206 {spec.name} produced invalid ranker scores")
    result = tuple(
        RankerScorePrediction(row, float(score)) for row, score in zip(heldout, scores, strict=True)
    )
    return result, int(ranker.booster_.num_trees()), len(vectorizer.feature_names_)


def _ranker_label(row: ActionAuditRow, spec: ConditionalRankerSpec) -> int:
    if row.strict_expected:
        return spec.labels["strict_success"]
    if stage194._is_safe_zero(row):
        return spec.labels["safe_zero"]
    if stage194._is_unsafe(row):
        return spec.labels["unsafe"]
    raise ValueError("Stage206 row is outside strict/safe-zero/unsafe taxonomy")


def _build_question_candidates(
    safety_predictions: Sequence[stage194.SafetyPrediction],
    ranker_predictions: Sequence[RankerScorePrediction],
    feature_indices: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
) -> tuple[_QuestionCandidate, ...]:
    safety_grouped = stage194._group_predictions(safety_predictions)
    ranker_index = {stage194._row_key(row.row): row.score for row in ranker_predictions}
    expected_nonbaseline = {
        stage194._row_key(row.row)
        for row in safety_predictions
        if row.row.action.family != "baseline"
    }
    if expected_nonbaseline != set(ranker_index):
        raise ValueError("Stage206 safety and conditional-ranker prediction rows differ")
    candidates = []
    for question_key, question_safety in sorted(safety_grouped.items()):
        baselines = [row for row in question_safety if row.row.action.family == "baseline"]
        if len(baselines) != 1:
            raise ValueError("Stage206 requires exactly one baseline per question")
        ranked_safety = sorted(question_safety, key=stage194._safety_order_key)
        pool_index = {stage194._row_key(row.row): row for row in ranked_safety[:_POOL_CAP]}
        baseline_safety = baselines[0]
        pool_index[stage194._row_key(baseline_safety.row)] = baseline_safety
        pool_safety = tuple(sorted(pool_index.values(), key=stage194._safety_order_key))
        nonbaseline = [row for row in pool_safety if row.row.action.family != "baseline"]
        if not nonbaseline:
            raise ValueError("Stage206 fixed pool requires a nonbaseline conditional action")
        winner_safety = min(
            nonbaseline,
            key=lambda row: (
                -ranker_index[stage194._row_key(row.row)],
                *stage194._safety_order_key(row),
            ),
        )
        score_values = [ranker_index[stage194._row_key(row.row)] for row in nonbaseline]
        minimum = min(score_values)
        maximum = max(score_values)
        ordered_scores = sorted(score_values, reverse=True)
        second = ordered_scores[1] if len(ordered_scores) > 1 else ordered_scores[0]
        normalized_margin = (
            (ordered_scores[0] - second) / (maximum - minimum) if maximum > minimum else 0.0
        )
        complete_pool = tuple(
            FrontierActionPrediction(
                row.row,
                row.citation_loss_probability,
                row.f1_loss_probability,
                ranker_index.get(stage194._row_key(row.row), 0.0),
                max(row.citation_loss_probability, row.f1_loss_probability),
            )
            for row in pool_safety
        )
        winner_key = stage194._row_key(winner_safety.row)
        winner = next(row for row in complete_pool if stage194._row_key(row.row) == winner_key)
        baseline = next(row for row in complete_pool if row.row.action.family == "baseline")
        candidates.append(
            _QuestionCandidate(
                question_key=question_key,
                candidate=winner,
                baseline=baseline,
                complete_pool=complete_pool,
                gate_features=_gate_features(
                    winner_safety,
                    baseline_safety,
                    pool_safety,
                    normalized_margin,
                    feature_indices,
                ),
                gate_label=int(winner.row.strict_expected),
                strict_opportunity=any(row.row.strict_expected for row in question_safety),
                action_count=len(question_safety),
                strict_action_count=sum(row.row.strict_expected for row in question_safety),
            )
        )
    return tuple(candidates)


def _gate_features(
    winner: stage194.SafetyPrediction,
    baseline: stage194.SafetyPrediction,
    pool: Sequence[stage194.SafetyPrediction],
    normalized_margin: float,
    feature_indices: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"ranker_normalized_top1_top2_margin": normalized_margin}
    for representation in ("raw_runtime", "question_relative_runtime"):
        winner_features = feature_indices[representation][stage194._row_key(winner.row)]
        baseline_features = feature_indices[representation][stage194._row_key(baseline.row)]
        for key, value in winner_features.items():
            result[f"winner::{representation}::{key}"] = value
        for key, value in baseline_features.items():
            result[f"baseline::{representation}::{key}"] = value
        for key in sorted(set(winner_features) & set(baseline_features)):
            left = winner_features[key]
            right = baseline_features[key]
            if isinstance(left, (int, float, bool)) and isinstance(right, (int, float, bool)):
                result[f"delta::{representation}::{key}"] = float(left) - float(right)
    result["winner_citation_loss_probability"] = winner.citation_loss_probability
    result["winner_f1_loss_probability"] = winner.f1_loss_probability
    result["winner_max_loss_probability"] = max(
        winner.citation_loss_probability, winner.f1_loss_probability
    )
    result["baseline_citation_loss_probability"] = baseline.citation_loss_probability
    result["baseline_f1_loss_probability"] = baseline.f1_loss_probability
    denominator = max(1, len(pool) - 1)
    for name, key in (
        ("max_loss", lambda row: max(row.citation_loss_probability, row.f1_loss_probability)),
        ("citation_loss", lambda row: row.citation_loss_probability),
        ("f1_loss", lambda row: row.f1_loss_probability),
    ):
        ordered = sorted(pool, key=lambda row: (key(row), row.row.action.action_id))
        winner_key = stage194._row_key(winner.row)
        rank = next(
            index for index, row in enumerate(ordered) if stage194._row_key(row.row) == winner_key
        )
        result[f"winner_{name}_rank_fraction"] = rank / denominator
    return result


def _fit_gate(
    candidates: Sequence[_QuestionCandidate],
    tree_profile: str,
) -> tuple[DictVectorizer, Any, np.ndarray, Mapping[str, Any]]:
    import lightgbm as lgb

    labels = np.asarray([row.gate_label for row in candidates], dtype=np.int8)
    if len(set(labels.tolist())) != 2:
        raise ValueError("Stage206 gate target requires strict and non-strict winners")
    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform([dict(row.gate_features) for row in candidates]).tocsr()
    gate = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        class_weight="balanced",
        **{
            **stage196._lightgbm_common_parameters(),
            **stage196._TREE_PROFILES[tree_profile],
        },
    )
    gate.fit(matrix, labels)
    scores = np.asarray(gate.predict_proba(matrix)[:, 1], dtype=np.float64)
    return vectorizer, gate, scores, _binary_metrics(labels, scores)


def _coverage_threshold(scores: Sequence[float] | np.ndarray, coverage: float) -> float:
    values = sorted((float(value) for value in scores), reverse=True)
    if not values or not 0.0 < coverage <= 1.0:
        raise ValueError("Stage206 gate coverage requires nonempty scores and coverage in (0,1]")
    target = max(1, math.ceil(len(values) * coverage))
    return values[target - 1]


def _build_gated_decisions(
    candidates: Sequence[_QuestionCandidate],
    scores: Sequence[float] | np.ndarray,
    *,
    threshold: float,
) -> tuple[SafetyFirstFrontierDecision, ...]:
    decisions = []
    for row, score in zip(candidates, scores, strict=True):
        winner = row.candidate if float(score) >= threshold else row.baseline
        decisions.append(
            SafetyFirstFrontierDecision(
                question_key=row.question_key,
                baseline=row.baseline,
                complete_pool=row.complete_pool,
                frontier=row.complete_pool,
                winner=winner,
                strict_opportunity=row.strict_opportunity,
                action_count=row.action_count,
                strict_action_count=row.strict_action_count,
            )
        )
    return tuple(decisions)


def _binary_metrics(
    labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray
) -> dict[str, Any]:
    actual = np.asarray(labels, dtype=np.int8)
    predicted = np.asarray(scores, dtype=np.float64)
    positives = int(actual.sum())
    return {
        "question_count": len(actual),
        "positive_count": positives,
        "positive_prevalence": _ratio(positives, len(actual)),
        "roc_auc": round(float(roc_auc_score(actual, predicted)), 6)
        if len(set(actual.tolist())) == 2
        else None,
        "average_precision": round(float(average_precision_score(actual, predicted)), 6)
        if positives
        else None,
    }


def run_two_stage_change_ranker_nested_cv(
    *,
    action_rows: Sequence[ActionAuditRow],
    stage182_selected_actions: Sequence[SelectedAction],
    stage205_protocol: Mapping[str, Any],
    stage199_report: Mapping[str, Any],
    progress_sink: ProgressSink | None = None,
    partition_fit_predictor: TwoStagePartitionFitPredictor | None = None,
) -> dict[str, Any]:
    """Run the frozen Stage206 five-by-four train-only nested CV."""

    started_at = time.perf_counter()
    rows = tuple(action_rows)
    fold_ids = tuple(sorted({row.fold_id for row in rows}))
    if len(fold_ids) != 5:
        raise ValueError("Stage206 requires exactly five frozen folds")
    grouped = stage194._group_rows(rows)
    references = build_stage182_reference_rows(rows, stage182_selected_actions)
    base_features = build_composition_feature_indices(rows)
    feature_indices = {
        "raw_runtime": base_features["raw"],
        "question_relative_runtime": base_features["question_relative"],
    }
    trajectories = {
        row["outer_context"]: stage196._spec_from_dict(row["source_spec"])
        for row in stage205_protocol["frozen_protocol"]["source_trajectory_contract"][
            "trajectories"
        ]
    }
    source_evidence = stage199_report["joint_risk_winner_nested_cv"]["outer_contexts"]
    policies = stage206_policy_specs()
    fit_predict = partition_fit_predictor or fit_predict_two_stage_partition
    execution = _FitTotals()
    feature_counts: dict[str, int] = {}
    fit_seconds = 0.0
    outer_reports: dict[str, Any] = {}
    outer_rows: list[ActionAuditRow] = []
    outer_diagnostics = stage196._FrontierDiagnostics()
    cell_reports: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_specs: Counter[str] = Counter()
    selected_ranker_families: Counter[str] = Counter()
    control_reproduction_count = 0
    gate_diagnostic_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)

    for outer_fold_id in fold_ids:
        source_spec = trajectories[outer_fold_id]
        outer_training = tuple(row for row in rows if row.fold_id != outer_fold_id)
        outer_heldout = tuple(row for row in rows if row.fold_id == outer_fold_id)
        inner_fold_ids = tuple(fold for fold in fold_ids if fold != outer_fold_id)
        safety_predictions: list[stage194.SafetyPrediction] = []
        gain_predictions: list[stage196.GainPrediction] = []
        risk_predictions: list[stage196.UnsafePrediction] = []
        policy_decisions: dict[str, list[SafetyFirstFrontierDecision]] = {
            spec.name: [] for spec in policies
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
                policies,
                outer_fold_id,
                inner_fold_id,
            )
            fit_seconds += time.perf_counter() - fitted_at
            execution.add(result)
            safety_predictions.extend(result.safety_predictions)
            gain_predictions.extend(result.gain_predictions)
            risk_predictions.extend(result.risk_predictions)
            for policy in policies:
                policy_decisions[policy.name].extend(result.policy_decisions[policy.name])
                gate_diagnostic_rows[policy.name].append(result.gate_diagnostics[policy.name])
            for name, count in result.feature_count_by_representation.items():
                feature_counts[name] = max(feature_counts.get(name, 0), count)
            _emit(
                progress_sink,
                phase="stage206_inner_partition_complete",
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
        control = stage203._candidate_report(
            spec=_control_spec_dict(),
            decisions=control_decisions,
            references=references,
            expected_fold_ids=inner_fold_ids,
            question_count=question_count,
        )
        formal_control = stage203._stage199_control_evidence(source_evidence[outer_fold_id])
        if not stage199._nested_close(
            control["evaluation"], formal_control["evaluation"]
        ) or not stage199._nested_close(control["diagnostics"], formal_control["diagnostics"]):
            raise ValueError(f"Stage206 control did not reproduce {outer_fold_id}")
        control_reproduction_count += 1
        candidates = [control]
        for policy in policies:
            candidate = stage203._candidate_report(
                spec=_policy_spec_dict(policy),
                decisions=policy_decisions[policy.name],
                references=references,
                expected_fold_ids=inner_fold_ids,
                question_count=question_count,
            )
            candidate["gate_diagnostics"] = _aggregate_gate_diagnostics(
                gate_diagnostic_rows[policy.name][-4:]
            )
            candidates.append(candidate)
        for candidate in candidates:
            candidate["paired_vs_control"] = stage199._paired_delta(candidate, control)
            cell_reports[candidate["spec"]["name"]].append(candidate)
        eligible = [row for row in candidates if row["eligible"]]
        ranked = sorted(candidates, key=stage196._inner_selection_key)
        public_top = [_public_candidate(row) for row in ranked[:5]]
        if not eligible:
            outer_reports[outer_fold_id] = _outer_report(
                source_spec=source_spec,
                question_count=question_count,
                eligible=(),
                selected=None,
                public_top=public_top,
            )
            _emit(
                progress_sink,
                phase="stage206_outer_context_no_eligible_config",
                outer_fold_id=outer_fold_id,
            )
            continue
        selected = min(eligible, key=stage196._inner_selection_key)
        selected_spec = selected["spec"]
        selected_specs[str(selected_spec["name"])] += 1
        selected_ranker_families[str(selected_spec["ranker_family"])] += 1
        selected_policies = (
            ()
            if selected_spec["ranker_family"] == "exact_control"
            else (_policy_spec_from_dict(selected_spec),)
        )
        fitted_at = time.perf_counter()
        heldout_result = fit_predict(
            outer_training,
            outer_heldout,
            feature_indices,
            source_spec,
            selected_policies,
            outer_fold_id,
            outer_fold_id,
        )
        fit_seconds += time.perf_counter() - fitted_at
        execution.add(heldout_result)
        for name, count in heldout_result.feature_count_by_representation.items():
            feature_counts[name] = max(feature_counts.get(name, 0), count)
        if not selected_policies:
            decisions = stage196.build_safety_first_frontier_decisions(
                heldout_result.safety_predictions,
                heldout_result.gain_predictions,
                heldout_result.risk_predictions,
                source_spec,
            )
            outer_gate = None
        else:
            decisions = heldout_result.policy_decisions[str(selected_spec["name"])]
            outer_gate = heldout_result.gate_diagnostics[str(selected_spec["name"])]
        diagnostics = stage203.evaluate_top1_decisions(
            decisions, expected_fold_ids=(outer_fold_id,)
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
            "selected_inner_gate_diagnostics": selected.get("gate_diagnostics"),
            "outer_evaluation": evaluation,
            "outer_diagnostics": diagnostics,
            "outer_gate_diagnostics": outer_gate,
            "top_inner_candidates": public_top,
            "outer_evaluated": True,
        }
        del heldout_result
        gc.collect()
        _emit(
            progress_sink,
            phase="stage206_outer_context_complete",
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
    cell_aggregates = {}
    for name, reports in sorted(cell_reports.items()):
        aggregate_row = stage199._aggregate_cell_reports(reports)
        gate_rows = [row["gate_diagnostics"] for row in reports if "gate_diagnostics" in row]
        if gate_rows:
            aggregate_row["gate_diagnostics"] = _aggregate_outer_gate_diagnostics(gate_rows)
        cell_aggregates[name] = aggregate_row
    return {
        "protocol": {
            "outer_fold_count": 5,
            "inner_fold_count": 4,
            "gate_crossfit_fold_count": 4,
            "ranker_family_count": 2,
            "target_change_coverage_count": 5,
            "two_stage_policy_count": 10,
            "exact_control_count": 1,
            "candidate_config_count_per_outer_context": 11,
            "pool_cap": _POOL_CAP,
            "model_fits_per_inner_partition": 24,
            "maximum_model_fit_count": 570,
            "maximum_lightgbm_tree_count": 96_000,
            "source_safety_gate_features_oof": True,
            "fallback_enabled": False,
        },
        "dataset": {
            "action_count": len(rows),
            "question_count": len(grouped),
            "reference_action_count": len(references),
            "fold_action_counts": {
                fold_id: sum(row.fold_id == fold_id for row in rows) for fold_id in fold_ids
            },
            "fold_nonbaseline_action_counts": {
                fold_id: sum(
                    row.fold_id == fold_id and row.action.family != "baseline" for row in rows
                )
                for fold_id in fold_ids
            },
            "fold_question_counts": {
                fold_id: len({row.question_key for row in rows if row.fold_id == fold_id})
                for fold_id in fold_ids
            },
        },
        "outer_contexts": outer_reports,
        "aggregate": aggregate,
        "aggregate_diagnostics": aggregate_diagnostics,
        "paired_bootstrap": bootstrap,
        "cell_aggregates": cell_aggregates,
        "ranker_family_aggregates": _factor_aggregates(cell_aggregates, "ranker_family"),
        "coverage_aggregates": _coverage_aggregates(cell_aggregates),
        "selected_spec_counts": dict(sorted(selected_specs.items())),
        "selected_ranker_family_counts": dict(sorted(selected_ranker_families.items())),
        "advancement_gates": gates,
        "advancement_gate_pass_count": sum(row["passed"] for row in gates),
        "candidate_family_accepted": all(row["passed"] for row in gates),
        "execution": {
            **{name: getattr(execution, name) for name in execution.__dataclass_fields__},
            "control_reproduction_count": control_reproduction_count,
            "all_controls_reproduced_exactly": control_reproduction_count == 5,
            "public_training_rows_written": 0,
            "public_prediction_rows_written": 0,
            "feature_count_by_representation": dict(sorted(feature_counts.items())),
            "fit_seconds": round(fit_seconds, 6),
            "wall_seconds": round(time.perf_counter() - started_at, 6),
        },
    }


def _outer_report(
    *,
    source_spec: SafetyFirstFrontierPolicySpec,
    question_count: int,
    eligible: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any] | None,
    public_top: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "source_spec": stage196._spec_dict(source_spec),
        "inner_question_count": question_count,
        "control_reproduction_exact": True,
        "eligible_config_count": len(eligible),
        "selected_spec": selected["spec"] if selected else None,
        "selected_inner_evaluation": selected["evaluation"] if selected else None,
        "selected_inner_diagnostics": selected["diagnostics"] if selected else None,
        "selected_inner_gate_diagnostics": selected.get("gate_diagnostics") if selected else None,
        "outer_evaluation": None,
        "outer_diagnostics": None,
        "outer_gate_diagnostics": None,
        "top_inner_candidates": list(public_top),
        "outer_evaluated": False,
    }


def _control_spec_dict() -> dict[str, Any]:
    return {
        "name": _CONTROL_NAME,
        "ranker_family": "exact_control",
        "target_change_coverage": None,
        "ablation_family": "exact_control",
    }


def _policy_spec_dict(spec: TwoStagePolicySpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "ranker_family": spec.ranker_family,
        "target_change_coverage": spec.target_change_coverage,
        "ablation_family": "two_stage",
    }


def _policy_spec_from_dict(value: Mapping[str, Any]) -> TwoStagePolicySpec:
    return TwoStagePolicySpec(
        str(value["name"]),
        str(value["ranker_family"]),
        float(value["target_change_coverage"]),
    )


def _public_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "spec": row["spec"],
        "eligible": row["eligible"],
        "evaluation": row["evaluation"],
        "diagnostics": row["diagnostics"],
        "gate_diagnostics": row.get("gate_diagnostics"),
    }


def _aggregate_gate_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    question_count = sum(int(row["heldout_question_count"]) for row in rows)
    changed = sum(int(row["realized_change_count"]) for row in rows)
    pre_gate_strict = sum(int(row["pre_gate_ranker_strict_count"]) for row in rows)
    pre_gate_unsafe = sum(int(row["pre_gate_ranker_unsafe_count"]) for row in rows)
    return {
        "partition_count": len(rows),
        "ranker_family": rows[0]["ranker_family"],
        "target_change_coverage": rows[0]["target_change_coverage"],
        "heldout_question_count": question_count,
        "realized_change_count": changed,
        "realized_change_coverage": _ratio(changed, question_count),
        "pre_gate_ranker_strict_count": pre_gate_strict,
        "pre_gate_ranker_strict_rate": _ratio(pre_gate_strict, question_count),
        "pre_gate_ranker_unsafe_count": pre_gate_unsafe,
        "pre_gate_ranker_unsafe_rate": _ratio(pre_gate_unsafe, question_count),
        "mean_training_gate_roc_auc": _mean(row["training_gate"]["roc_auc"] for row in rows),
        "mean_training_gate_average_precision": _mean(
            row["training_gate"]["average_precision"] for row in rows
        ),
        "mean_heldout_gate_roc_auc": _mean(row["heldout_gate"]["roc_auc"] for row in rows),
        "mean_heldout_gate_average_precision": _mean(
            row["heldout_gate"]["average_precision"] for row in rows
        ),
    }


def _aggregate_outer_gate_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    question_count = sum(int(row["heldout_question_count"]) for row in rows)
    changed = sum(int(row["realized_change_count"]) for row in rows)
    pre_gate_strict = sum(int(row["pre_gate_ranker_strict_count"]) for row in rows)
    pre_gate_unsafe = sum(int(row["pre_gate_ranker_unsafe_count"]) for row in rows)
    return {
        "outer_context_count": len(rows),
        "heldout_question_count": question_count,
        "realized_change_count": changed,
        "realized_change_coverage": _ratio(changed, question_count),
        "pre_gate_ranker_strict_count": pre_gate_strict,
        "pre_gate_ranker_strict_rate": _ratio(pre_gate_strict, question_count),
        "pre_gate_ranker_unsafe_count": pre_gate_unsafe,
        "pre_gate_ranker_unsafe_rate": _ratio(pre_gate_unsafe, question_count),
        "mean_training_gate_roc_auc": _mean(row["mean_training_gate_roc_auc"] for row in rows),
        "mean_training_gate_average_precision": _mean(
            row["mean_training_gate_average_precision"] for row in rows
        ),
        "mean_heldout_gate_roc_auc": _mean(row["mean_heldout_gate_roc_auc"] for row in rows),
        "mean_heldout_gate_average_precision": _mean(
            row["mean_heldout_gate_average_precision"] for row in rows
        ),
    }


def _factor_aggregates(cells: Mapping[str, Mapping[str, Any]], dimension: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells.values():
        grouped[str(cell["spec"][dimension])].append(cell)
    result = {}
    for value, rows in sorted(grouped.items()):
        aggregate = {
            "cell_count": len(rows),
            "mean_unsafe_selection_rate": _mean(row["unsafe_selection_rate"] for row in rows),
            "mean_conditional_capture": _mean(
                row["conditional_ranker_strict_capture"] for row in rows
            ),
            "mean_strict_success_precision": _mean(row["strict_success_precision"] for row in rows),
            "mean_gold_citation_delta": _mean(row["gold_citation_delta"] for row in rows),
            "mean_f1_delta": _mean(row["mean_f1_delta"] for row in rows),
            "best_cell": min(rows, key=stage199._aggregate_selection_key)["spec"]["name"],
        }
        gate_rows = [row["gate_diagnostics"] for row in rows if "gate_diagnostics" in row]
        if gate_rows:
            aggregate.update(
                {
                    "mean_realized_change_coverage": _mean(
                        row["realized_change_coverage"] for row in gate_rows
                    ),
                    "mean_heldout_gate_roc_auc": _mean(
                        row["mean_heldout_gate_roc_auc"] for row in gate_rows
                    ),
                    "mean_heldout_gate_average_precision": _mean(
                        row["mean_heldout_gate_average_precision"] for row in gate_rows
                    ),
                    "mean_pre_gate_ranker_strict_rate": _mean(
                        row["pre_gate_ranker_strict_rate"] for row in gate_rows
                    ),
                    "mean_pre_gate_ranker_unsafe_rate": _mean(
                        row["pre_gate_ranker_unsafe_rate"] for row in gate_rows
                    ),
                }
            )
        result[value] = aggregate
    return result


def _coverage_aggregates(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    custom = {
        name: row
        for name, row in cells.items()
        if row["spec"]["target_change_coverage"] is not None
    }
    return _factor_aggregates(custom, "target_change_coverage")


def _validate_question_partition(
    training: Sequence[ActionAuditRow], heldout: Sequence[ActionAuditRow]
) -> None:
    training_questions = {row.question_key for row in training}
    heldout_questions = {row.question_key for row in heldout}
    if not training_questions or not heldout_questions or training_questions & heldout_questions:
        raise ValueError("Stage206 train and heldout question groups must be nonempty and disjoint")
    for rows in (training, heldout):
        grouped = stage194._group_rows(rows)
        if any(
            sum(row.action.family == "baseline" for row in values) != 1
            for values in grouped.values()
        ):
            raise ValueError("Stage206 requires one baseline in every question group")
        if any(
            not any(row.action.family != "baseline" for row in values)
            for values in grouped.values()
        ):
            raise ValueError("Stage206 requires nonbaseline actions in every question group")


def _mean(values: Sequence[float] | Any) -> float:
    rows = [float(value) for value in values if value is not None]
    return round(sum(rows) / len(rows), 6) if rows else 0.0


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator / denominator), 6) if denominator else 0.0


def _emit(progress_sink: ProgressSink | None, **event: Any) -> None:
    if progress_sink is not None:
        progress_sink(event)
