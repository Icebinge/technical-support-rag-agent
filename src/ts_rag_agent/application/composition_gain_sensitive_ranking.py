from __future__ import annotations

import gc
import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ts_rag_agent.application.composition_action_audit import ActionAuditRow
from ts_rag_agent.application.composition_dual_target_policy import SelectedAction
from ts_rag_agent.application.composition_f1_representation_cv import (
    build_composition_feature_indices,
)
from ts_rag_agent.application.composition_joint_constraint_ranking import (
    evaluate_selected_actions,
)

FeatureRepresentation = Literal["raw_runtime", "question_relative_runtime"]
SafetyEstimator = Literal["class_balanced_logistic", "histogram_gradient_boosting"]
GainRankerKind = Literal["pairwise_pareto_logistic", "linear_listnet_top_frontier"]
ProgressSink = Callable[[Mapping[str, Any]], None]

_F1_TOLERANCE = 1e-12
_INNER_STRICT_PRECISION = 0.60
_INNER_CHANGED_RATE = 0.10
_INNER_STRICT_COUNT_RATE = 0.08
_MINIMUM_INNER_NONREGRESSING_FOLDS = 3
_LISTNET_LEARNING_RATE = 0.05
_LISTNET_L2 = 0.001
_LISTNET_MAX_ITERATIONS = 400
_LISTNET_GRADIENT_TOLERANCE = 1e-7
_LISTNET_PATIENCE = 20
_LISTNET_IMPROVEMENT_TOLERANCE = 1e-12
_BOOTSTRAP_REPLICATES = 2_000
_BOOTSTRAP_SEED = 187


@dataclass(frozen=True)
class GainSensitivePolicySpec:
    """One frozen Stage 188 gain-sensitive policy configuration."""

    name: str
    feature_representation: FeatureRepresentation
    safety_estimator: SafetyEstimator
    gain_ranker: GainRankerKind
    safety_frontier_margin: float

    @property
    def bundle_name(self) -> str:
        return f"{self.feature_representation}__{self.safety_estimator}__{self.gain_ranker}"


@dataclass(frozen=True)
class GainSensitivePrediction:
    """One private held-out Stage 188 action prediction."""

    row: ActionAuditRow
    citation_loss_probability: float
    f1_loss_probability: float
    gain_score: float


@dataclass(frozen=True)
class PairwiseTrainingData:
    """Sparse comparable-pair training data and public-safe counts."""

    matrix: csr_matrix
    labels: np.ndarray
    weights: np.ndarray
    comparable_pair_count: int
    omitted_incomparable_pair_count: int
    question_count_with_pairs: int


@dataclass(frozen=True)
class ListNetTrainingTarget:
    """Complete-list target distribution and grouped row positions."""

    probabilities: np.ndarray
    question_positions: tuple[np.ndarray, ...]
    frontier_action_count: int


@dataclass(frozen=True)
class _SafetyHead:
    model: Any
    dense: bool

    def predict(self, sparse_matrix: csr_matrix) -> np.ndarray:
        matrix: Any = sparse_matrix.toarray() if self.dense else sparse_matrix
        return np.asarray(self.model.predict_proba(matrix)[:, 1], dtype=np.float64)


class _GainRanker(Protocol):
    diagnostics: Mapping[str, Any]

    def score(self, action_matrix: csr_matrix) -> np.ndarray: ...


@dataclass(frozen=True)
class _PairwiseLogisticRanker:
    scaler: StandardScaler
    model: LogisticRegression
    diagnostics: Mapping[str, Any]

    def score(self, action_matrix: csr_matrix) -> np.ndarray:
        matrix = self.scaler.transform(action_matrix)
        return np.asarray(self.model.decision_function(matrix), dtype=np.float64)


@dataclass(frozen=True)
class _LinearListNetRanker:
    scaler: StandardScaler
    coefficients: np.ndarray
    diagnostics: Mapping[str, Any]

    def score(self, action_matrix: csr_matrix) -> np.ndarray:
        matrix = self.scaler.transform(action_matrix)
        return np.asarray(matrix @ self.coefficients, dtype=np.float64).reshape(-1)


@dataclass(frozen=True)
class FittedGainSensitiveRepresentation:
    """Shared Stage 188 encoder with safety heads and gain rankers."""

    feature_representation: FeatureRepresentation
    vectorizer: DictVectorizer
    logistic_scaler: StandardScaler
    safety_heads: Mapping[SafetyEstimator, Mapping[str, _SafetyHead]]
    gain_rankers: Mapping[GainRankerKind, _GainRanker]
    diagnostics: Mapping[str, Any]

    @property
    def feature_count(self) -> int:
        return len(self.vectorizer.feature_names_)

    @property
    def model_fit_count(self) -> int:
        return 6

    def predict(
        self,
        rows: Sequence[ActionAuditRow],
        feature_index: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> Mapping[str, tuple[GainSensitivePrediction, ...]]:
        features = [dict(feature_index[_row_key(row)]) for row in rows]
        action_matrix = self.vectorizer.transform(features).tocsr()
        logistic_matrix = self.logistic_scaler.transform(action_matrix).tocsr()
        safety_scores: dict[SafetyEstimator, tuple[np.ndarray, np.ndarray]] = {}
        for estimator, heads in self.safety_heads.items():
            matrix = (
                action_matrix if estimator == "histogram_gradient_boosting" else logistic_matrix
            )
            safety_scores[estimator] = (
                heads["citation_loss"].predict(matrix),
                heads["f1_loss"].predict(matrix),
            )
        gain_scores = {
            name: ranker.score(action_matrix) for name, ranker in self.gain_rankers.items()
        }
        predictions = {}
        for estimator, (citation_scores, f1_scores) in safety_scores.items():
            for gain_name, scores in gain_scores.items():
                bundle_name = f"{self.feature_representation}__{estimator}__{gain_name}"
                predictions[bundle_name] = tuple(
                    GainSensitivePrediction(
                        row=row,
                        citation_loss_probability=float(citation),
                        f1_loss_probability=float(f1),
                        gain_score=float(gain),
                    )
                    for row, citation, f1, gain in zip(
                        rows,
                        citation_scores,
                        f1_scores,
                        scores,
                        strict=True,
                    )
                )
        return predictions


class RepresentationFitter(Protocol):
    def __call__(
        self,
        rows: Sequence[ActionAuditRow],
        feature_indices: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
    ) -> Mapping[str, FittedGainSensitiveRepresentation]: ...


def stage188_policy_specs() -> tuple[GainSensitivePolicySpec, ...]:
    """Return the complete frozen 32-policy Stage 188 grid."""

    specs = []
    for representation in ("raw_runtime", "question_relative_runtime"):
        for estimator in ("class_balanced_logistic", "histogram_gradient_boosting"):
            for ranker in ("pairwise_pareto_logistic", "linear_listnet_top_frontier"):
                for margin in (0.0, 0.02, 0.05, 0.10):
                    specs.append(
                        GainSensitivePolicySpec(
                            name=(
                                f"{representation}__{estimator}__{ranker}__frontier_{margin:.2f}"
                            ),
                            feature_representation=representation,
                            safety_estimator=estimator,
                            gain_ranker=ranker,
                            safety_frontier_margin=margin,
                        )
                    )
    return tuple(specs)


def run_gain_sensitive_nested_cv(
    *,
    action_rows: Sequence[ActionAuditRow],
    stage182_selected_actions: Sequence[SelectedAction],
    progress_sink: ProgressSink | None = None,
    representation_fitter: RepresentationFitter | None = None,
) -> dict[str, Any]:
    """Run the frozen Stage 188 five-by-four nested ranking experiment."""

    started_at = time.perf_counter()
    rows = tuple(action_rows)
    if not rows:
        raise ValueError("Stage188 requires action rows")
    fold_ids = tuple(sorted({row.fold_id for row in rows}))
    if len(fold_ids) != 5:
        raise ValueError("Stage188 requires exactly five frozen folds")
    grouped = _group_rows(rows)
    question_count = len(grouped)
    if any(
        len([row for row in question_rows if row.action.family == "baseline"]) != 1
        for question_rows in grouped.values()
    ):
        raise ValueError("Stage188 requires one original baseline action per question")

    references = _reference_rows(rows, stage182_selected_actions)
    reference_regressions = [row for row in references.values() if row.f1_delta < -_F1_TOLERANCE]
    base_feature_indices = build_composition_feature_indices(rows)
    feature_indices = {
        "raw_runtime": base_feature_indices["raw"],
        "question_relative_runtime": base_feature_indices["question_relative"],
    }
    specs = stage188_policy_specs()
    fit_representations = representation_fitter or fit_gain_sensitive_representations
    model_fit_count = 0
    private_prediction_count = 0
    comparable_pair_count = 0
    omitted_pair_count = 0
    listwise_question_fit_count = 0
    listnet_iteration_count = 0
    outer_rows: list[ActionAuditRow] = []
    outer_predictions_for_metrics: list[GainSensitivePrediction] = []
    outer_reports: dict[str, dict[str, Any]] = {}
    selected_spec_counts: Counter[str] = Counter()
    selected_ranker_counts: Counter[str] = Counter()
    feature_counts: dict[str, int] = {}
    fit_seconds = 0.0

    for outer_fold_id in fold_ids:
        outer_training = tuple(row for row in rows if row.fold_id != outer_fold_id)
        outer_heldout = tuple(row for row in rows if row.fold_id == outer_fold_id)
        inner_fold_ids = tuple(fold_id for fold_id in fold_ids if fold_id != outer_fold_id)
        inner_predictions: dict[str, list[GainSensitivePrediction]] = defaultdict(list)

        for inner_fold_id in inner_fold_ids:
            inner_training = tuple(row for row in outer_training if row.fold_id != inner_fold_id)
            inner_heldout = tuple(row for row in outer_training if row.fold_id == inner_fold_id)
            fitted_at = time.perf_counter()
            representations = fit_representations(inner_training, feature_indices)
            fit_seconds += time.perf_counter() - fitted_at
            (
                model_fit_count,
                comparable_pair_count,
                omitted_pair_count,
                listwise_question_fit_count,
                listnet_iteration_count,
            ) = _accumulate_fit_diagnostics(
                representations,
                model_fit_count=model_fit_count,
                comparable_pair_count=comparable_pair_count,
                omitted_pair_count=omitted_pair_count,
                listwise_question_fit_count=listwise_question_fit_count,
                listnet_iteration_count=listnet_iteration_count,
            )
            for representation in representations.values():
                feature_counts[representation.feature_representation] = max(
                    feature_counts.get(representation.feature_representation, 0),
                    representation.feature_count,
                )
                predicted = representation.predict(
                    inner_heldout,
                    feature_indices[representation.feature_representation],
                )
                for bundle_name, bundle_predictions in predicted.items():
                    inner_predictions[bundle_name].extend(bundle_predictions)
                    private_prediction_count += len(bundle_predictions)
            _emit(
                progress_sink,
                phase="inner_partition_complete",
                outer_fold_id=outer_fold_id,
                inner_fold_id=inner_fold_id,
                training_action_count=len(inner_training),
                heldout_action_count=len(inner_heldout),
                cumulative_model_fit_count=model_fit_count,
                cumulative_comparable_pair_count=comparable_pair_count,
                cumulative_listnet_iteration_count=listnet_iteration_count,
            )

        candidate_reports = []
        inner_question_count = len({row.question_key for row in outer_training})
        for spec in specs:
            selected_rows = select_gain_sensitive_actions(
                inner_predictions[spec.bundle_name],
                spec,
            )
            evaluation = evaluate_selected_actions(
                selected_rows=selected_rows,
                references=references,
                expected_fold_ids=inner_fold_ids,
            )
            eligible = _inner_eligible(evaluation, inner_question_count)
            candidate_reports.append(
                {
                    "spec": _spec_dict(spec),
                    "eligible": eligible,
                    "evaluation": evaluation,
                }
            )

        eligible_reports = [row for row in candidate_reports if row["eligible"]]
        public_top_candidates = [
            _public_candidate(row)
            for row in sorted(candidate_reports, key=_inner_selection_key)[:5]
        ]
        if not eligible_reports:
            outer_reports[outer_fold_id] = {
                "inner_question_count": inner_question_count,
                "eligible_config_count": 0,
                "selected_spec": None,
                "selected_inner_evaluation": None,
                "outer_evaluation": None,
                "top_inner_candidates": public_top_candidates,
                "outer_evaluated": False,
            }
            _emit(
                progress_sink,
                phase="outer_fold_no_eligible_config",
                outer_fold_id=outer_fold_id,
                cumulative_model_fit_count=model_fit_count,
            )
            continue

        selected_report = min(eligible_reports, key=_inner_selection_key)
        selected_spec = _spec_from_dict(selected_report["spec"])
        selected_spec_counts[selected_spec.name] += 1
        selected_ranker_counts[selected_spec.gain_ranker] += 1

        fitted_at = time.perf_counter()
        outer_representations = fit_representations(outer_training, feature_indices)
        fit_seconds += time.perf_counter() - fitted_at
        (
            model_fit_count,
            comparable_pair_count,
            omitted_pair_count,
            listwise_question_fit_count,
            listnet_iteration_count,
        ) = _accumulate_fit_diagnostics(
            outer_representations,
            model_fit_count=model_fit_count,
            comparable_pair_count=comparable_pair_count,
            omitted_pair_count=omitted_pair_count,
            listwise_question_fit_count=listwise_question_fit_count,
            listnet_iteration_count=listnet_iteration_count,
        )
        for representation in outer_representations.values():
            feature_counts[representation.feature_representation] = max(
                feature_counts.get(representation.feature_representation, 0),
                representation.feature_count,
            )
        selected_representation = outer_representations[selected_spec.feature_representation]
        heldout_by_bundle = selected_representation.predict(
            outer_heldout,
            feature_indices[selected_spec.feature_representation],
        )
        heldout_predictions = heldout_by_bundle[selected_spec.bundle_name]
        private_prediction_count += len(heldout_predictions)
        selected_outer_rows = select_gain_sensitive_actions(
            heldout_predictions,
            selected_spec,
        )
        outer_evaluation = evaluate_selected_actions(
            selected_rows=selected_outer_rows,
            references=references,
            expected_fold_ids=(outer_fold_id,),
        )
        outer_rows.extend(selected_outer_rows)
        outer_predictions_for_metrics.extend(heldout_predictions)
        outer_reports[outer_fold_id] = {
            "inner_question_count": inner_question_count,
            "eligible_config_count": len(eligible_reports),
            "selected_spec": _spec_dict(selected_spec),
            "selected_inner_evaluation": selected_report["evaluation"],
            "outer_evaluation": outer_evaluation,
            "top_inner_candidates": public_top_candidates,
            "outer_evaluated": True,
        }
        _emit(
            progress_sink,
            phase="outer_fold_complete",
            outer_fold_id=outer_fold_id,
            selected_spec=selected_spec.name,
            eligible_config_count=len(eligible_reports),
            cumulative_model_fit_count=model_fit_count,
        )
        del outer_representations
        gc.collect()

    eligible_outer_fold_count = sum(row["outer_evaluated"] for row in outer_reports.values())
    aggregate = evaluate_selected_actions(
        selected_rows=outer_rows,
        references=references,
        expected_fold_ids=fold_ids,
    )
    bootstrap = (
        _paired_bootstrap(outer_rows)
        if eligible_outer_fold_count == len(fold_ids)
        else _unavailable_bootstrap()
    )
    gates = _advancement_gates(
        eligible_outer_fold_count=eligible_outer_fold_count,
        aggregate=aggregate,
        bootstrap=bootstrap,
    )
    return {
        "protocol": {
            "outer_fold_count": 5,
            "inner_fold_count": 4,
            "policy_config_count": len(specs),
            "representation_count": 2,
            "safety_estimators_per_representation": 2,
            "gain_rankers_per_representation": 2,
            "model_fits_per_partition": 12,
            "maximum_model_fit_count": 300,
            "listnet_scaling": "StandardScaler(with_mean=False)",
            "listnet_patience_improvement_tolerance": _LISTNET_IMPROVEMENT_TOLERANCE,
            "user_confirmed_scaling_choice": "A",
            "inner_selection": (
                "citation/F1 aggregate and fold nonregression, nontrivial changed and "
                "strict-success counts, strict precision >= 0.60, then frozen "
                "gain-first lexicographic objective"
            ),
            "no_eligible_behavior": (
                "record no-eligible and do not evaluate a weaker outer configuration"
            ),
            "gold_scope": "training targets and offline evaluation only",
            "pair_sampling": False,
            "list_sampling": False,
            "fallback_enabled": False,
        },
        "dataset": {
            "action_count": len(rows),
            "nonbaseline_action_count": sum(row.action.family != "baseline" for row in rows),
            "question_count": question_count,
            "reference_action_count": len(references),
            "reference_regression_count": len(reference_regressions),
            "fold_action_counts": {
                fold_id: sum(row.fold_id == fold_id for row in rows) for fold_id in fold_ids
            },
            "raw_runtime_feature_count": len(
                {name for features in feature_indices["raw_runtime"].values() for name in features}
            ),
            "relative_runtime_feature_count": len(
                {
                    name
                    for features in feature_indices["question_relative_runtime"].values()
                    for name in features
                }
            ),
        },
        "outer_folds": outer_reports,
        "aggregate": aggregate,
        "paired_bootstrap": bootstrap,
        "prediction_metrics": _prediction_metrics(outer_predictions_for_metrics),
        "selected_spec_counts": dict(sorted(selected_spec_counts.items())),
        "selected_ranker_counts": dict(sorted(selected_ranker_counts.items())),
        "advancement_gates": gates,
        "advancement_gate_pass_count": sum(row["passed"] for row in gates),
        "candidate_family_accepted": all(row["passed"] for row in gates),
        "execution": {
            "model_fit_count": model_fit_count,
            "maximum_model_fit_count": 300,
            "comparable_pair_count_across_fits": comparable_pair_count,
            "omitted_incomparable_pair_count_across_fits": omitted_pair_count,
            "listwise_question_fit_count": listwise_question_fit_count,
            "listnet_iteration_count": listnet_iteration_count,
            "private_prediction_count": private_prediction_count,
            "public_pair_rows_written": 0,
            "public_listwise_targets_written": 0,
            "public_prediction_rows_written": 0,
            "feature_count_by_representation": dict(sorted(feature_counts.items())),
            "fit_seconds": round(fit_seconds, 6),
            "wall_seconds": round(time.perf_counter() - started_at, 6),
        },
    }


def fit_gain_sensitive_representations(
    rows: Sequence[ActionAuditRow],
    feature_indices: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
) -> Mapping[str, FittedGainSensitiveRepresentation]:
    """Fit both frozen feature representations for one nested-CV partition."""

    return {
        representation: _fit_representation(
            rows,
            feature_indices[representation],
            representation=representation,
        )
        for representation in ("raw_runtime", "question_relative_runtime")
    }


def select_gain_sensitive_actions(
    predictions: Sequence[GainSensitivePrediction],
    spec: GainSensitivePolicySpec,
) -> tuple[ActionAuditRow, ...]:
    """Select one action per question from the guaranteed-nonempty safety frontier."""

    grouped: dict[str, list[GainSensitivePrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.row.question_key].append(prediction)
    selected = []
    for question_predictions in grouped.values():
        citation_minimum = min(row.citation_loss_probability for row in question_predictions)
        f1_minimum = min(row.f1_loss_probability for row in question_predictions)
        ranked_with_excess = [
            (
                row,
                max(
                    row.citation_loss_probability - citation_minimum,
                    row.f1_loss_probability - f1_minimum,
                ),
            )
            for row in question_predictions
        ]
        minimum_joint_excess = min(excess for _, excess in ranked_with_excess)
        frontier = [
            (row, excess)
            for row, excess in ranked_with_excess
            if excess <= minimum_joint_excess + spec.safety_frontier_margin + _F1_TOLERANCE
        ]
        if not frontier:
            raise AssertionError("Stage188 relative safety frontier must be nonempty")
        winner, _ = min(
            frontier,
            key=lambda item: (
                -item[0].gain_score,
                item[1],
                item[0].row.action.action_id,
            ),
        )
        selected.append(winner.row)
    return tuple(sorted(selected, key=lambda row: row.question_key))


def build_pairwise_training_data(
    rows: Sequence[ActionAuditRow],
    action_matrix: csr_matrix,
) -> PairwiseTrainingData:
    """Build all within-question comparable pair differences without sampling."""

    ordered_rows = tuple(rows)
    row_positions = {_row_key(row): index for index, row in enumerate(ordered_rows)}
    left_indices: list[int] = []
    right_indices: list[int] = []
    base_labels: list[int] = []
    base_weights: list[float] = []
    comparable_pair_count = 0
    omitted_incomparable_pair_count = 0
    question_count_with_pairs = 0
    for question_rows in _group_rows(rows).values():
        comparable = []
        omitted = 0
        for left_index, left in enumerate(question_rows):
            for right in question_rows[left_index + 1 :]:
                preference = pairwise_preference(left, right)
                if preference == 0:
                    omitted += 1
                    continue
                comparable.append((left, right, preference))
        omitted_incomparable_pair_count += omitted
        if not comparable:
            continue
        question_count_with_pairs += 1
        pair_weight = 1.0 / (2.0 * len(comparable))
        for left, right, preference in comparable:
            left_indices.append(row_positions[_row_key(left)])
            right_indices.append(row_positions[_row_key(right)])
            base_labels.append(int(preference > 0))
            base_weights.append(pair_weight)
        comparable_pair_count += len(comparable)
    if not left_indices:
        raise ValueError("Stage188 pairwise target produced no comparable pairs")
    differences = action_matrix[left_indices] - action_matrix[right_indices]
    matrix = vstack((differences, -differences), format="csr")
    labels = np.asarray(
        [*base_labels, *(1 - label for label in base_labels)],
        dtype=np.int8,
    )
    weights = np.asarray([*base_weights, *base_weights], dtype=np.float64)
    if len(set(labels.tolist())) != 2:
        raise ValueError("Stage188 pairwise target requires both classes")
    return PairwiseTrainingData(
        matrix=matrix,
        labels=labels,
        weights=weights,
        comparable_pair_count=comparable_pair_count,
        omitted_incomparable_pair_count=omitted_incomparable_pair_count,
        question_count_with_pairs=question_count_with_pairs,
    )


def pairwise_preference(left: ActionAuditRow, right: ActionAuditRow) -> int:
    """Return 1 when left is preferred, -1 when right is preferred, else 0."""

    left_tier = _outcome_tier(left)
    right_tier = _outcome_tier(right)
    if left_tier != right_tier:
        return 1 if left_tier > right_tier else -1
    if left_tier == 0:
        return 0
    if _pareto_dominates(left, right):
        return 1
    if _pareto_dominates(right, left):
        return -1
    return 0


def build_listnet_training_target(
    rows: Sequence[ActionAuditRow],
) -> ListNetTrainingTarget:
    """Build full-list top-frontier target distributions for every question."""

    ordered_rows = tuple(rows)
    positions = {_row_key(row): index for index, row in enumerate(ordered_rows)}
    probabilities = np.zeros(len(rows), dtype=np.float64)
    question_positions = []
    frontier_action_count = 0
    for question_rows in _group_rows(rows).values():
        tiers = [_outcome_tier(row) for row in question_rows]
        if max(tiers) < 1:
            raise ValueError("Stage188 requires a safe baseline tier for every question")
        highest_tier = max(tiers)
        top_tier = [
            row for row, tier in zip(question_rows, tiers, strict=True) if tier == highest_tier
        ]
        frontier = [
            row
            for row in top_tier
            if not any(
                _pareto_dominates(candidate, row)
                for candidate in top_tier
                if candidate.action.action_id != row.action.action_id
            )
        ]
        if not frontier:
            raise AssertionError("Stage188 ListNet top frontier must be nonempty")
        probability = 1.0 / len(frontier)
        for row in frontier:
            probabilities[positions[_row_key(row)]] = probability
        question_positions.append(
            np.asarray(
                [positions[_row_key(row)] for row in question_rows],
                dtype=np.int64,
            )
        )
        frontier_action_count += len(frontier)
    return ListNetTrainingTarget(
        probabilities=probabilities,
        question_positions=tuple(question_positions),
        frontier_action_count=frontier_action_count,
    )


def _fit_representation(
    rows: Sequence[ActionAuditRow],
    feature_index: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    representation: FeatureRepresentation,
) -> FittedGainSensitiveRepresentation:
    ordered_rows = tuple(rows)
    vectorizer = DictVectorizer(sparse=True)
    action_matrix = vectorizer.fit_transform(
        [dict(feature_index[_row_key(row)]) for row in ordered_rows]
    ).tocsr()
    weights = _question_balanced_weights(rows)
    labels = {
        "citation_loss": np.asarray(
            [int(row.citation_delta < 0) for row in ordered_rows],
            dtype=np.int8,
        ),
        "f1_loss": np.asarray(
            [int(row.f1_delta < -_F1_TOLERANCE) for row in ordered_rows],
            dtype=np.int8,
        ),
    }
    for name, values in labels.items():
        if len(set(values.tolist())) != 2:
            raise ValueError(f"Stage188 {name} target requires both classes")

    logistic_scaler = StandardScaler(with_mean=False)
    logistic_matrix = logistic_scaler.fit_transform(action_matrix).tocsr()
    logistic_heads = {
        target: _SafetyHead(
            model=_fit_logistic_classifier(logistic_matrix, target_labels, weights),
            dense=False,
        )
        for target, target_labels in labels.items()
    }

    dense_matrix = action_matrix.toarray()
    histogram_heads = {
        target: _SafetyHead(
            model=_fit_histogram_classifier(dense_matrix, target_labels, weights),
            dense=True,
        )
        for target, target_labels in labels.items()
    }
    del dense_matrix
    gc.collect()

    pair_data = build_pairwise_training_data(ordered_rows, action_matrix)
    pair_scaler = StandardScaler(with_mean=False, copy=False)
    pair_matrix = pair_scaler.fit_transform(pair_data.matrix).tocsr()
    pair_model = LogisticRegression(
        class_weight="balanced",
        max_iter=2_000,
        random_state=188,
        solver="liblinear",
    )
    pair_model.fit(
        pair_matrix,
        pair_data.labels,
        sample_weight=pair_data.weights,
    )
    pair_ranker = _PairwiseLogisticRanker(
        scaler=pair_scaler,
        model=pair_model,
        diagnostics={
            "comparable_pair_count": pair_data.comparable_pair_count,
            "omitted_incomparable_pair_count": pair_data.omitted_incomparable_pair_count,
            "question_count_with_pairs": pair_data.question_count_with_pairs,
            "training_row_count_with_orientations": int(pair_data.matrix.shape[0]),
        },
    )
    del pair_matrix, pair_data
    gc.collect()

    listnet_target = build_listnet_training_target(ordered_rows)
    listnet_coefficients, listnet_diagnostics = _fit_linear_listnet(
        logistic_matrix,
        listnet_target,
    )
    listnet_ranker = _LinearListNetRanker(
        scaler=logistic_scaler,
        coefficients=listnet_coefficients,
        diagnostics=listnet_diagnostics,
    )
    return FittedGainSensitiveRepresentation(
        feature_representation=representation,
        vectorizer=vectorizer,
        logistic_scaler=logistic_scaler,
        safety_heads={
            "class_balanced_logistic": logistic_heads,
            "histogram_gradient_boosting": histogram_heads,
        },
        gain_rankers={
            "pairwise_pareto_logistic": pair_ranker,
            "linear_listnet_top_frontier": listnet_ranker,
        },
        diagnostics={
            "feature_count": len(vectorizer.feature_names_),
            "pairwise": dict(pair_ranker.diagnostics),
            "listnet": dict(listnet_ranker.diagnostics),
        },
    )


def _fit_linear_listnet(
    matrix: csr_matrix,
    target: ListNetTrainingTarget,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    feature_count = matrix.shape[1]
    coefficients = np.zeros(feature_count, dtype=np.float64)
    first_moment = np.zeros_like(coefficients)
    second_moment = np.zeros_like(coefficients)
    best_coefficients = coefficients.copy()
    best_loss = math.inf
    stale_iterations = 0
    converged = False
    final_gradient_norm = math.inf
    completed_iterations = 0
    question_count = len(target.question_positions)
    if question_count == 0:
        raise ValueError("Stage188 ListNet requires at least one question")

    for iteration in range(1, _LISTNET_MAX_ITERATIONS + 1):
        scores = np.asarray(matrix @ coefficients, dtype=np.float64).reshape(-1)
        residual = np.zeros_like(scores)
        cross_entropy = 0.0
        for positions in target.question_positions:
            question_scores = scores[positions]
            shifted = question_scores - float(np.max(question_scores))
            exponentials = np.exp(shifted)
            predicted = exponentials / float(np.sum(exponentials))
            expected = target.probabilities[positions]
            residual[positions] = (predicted - expected) / question_count
            positive = expected > 0
            cross_entropy -= (
                float(np.sum(expected[positive] * np.log(predicted[positive] + 1e-300)))
                / question_count
            )
        loss = cross_entropy + 0.5 * _LISTNET_L2 * float(np.dot(coefficients, coefficients))
        gradient = np.asarray(matrix.T @ residual, dtype=np.float64).reshape(-1)
        gradient += _LISTNET_L2 * coefficients
        final_gradient_norm = float(np.linalg.norm(gradient))
        completed_iterations = iteration

        if loss < best_loss - _LISTNET_IMPROVEMENT_TOLERANCE:
            best_loss = loss
            best_coefficients = coefficients.copy()
            stale_iterations = 0
        else:
            stale_iterations += 1
        if final_gradient_norm <= _LISTNET_GRADIENT_TOLERANCE:
            converged = True
            break
        if stale_iterations >= _LISTNET_PATIENCE:
            break

        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * np.square(gradient)
        first_unbiased = first_moment / (1.0 - 0.9**iteration)
        second_unbiased = second_moment / (1.0 - 0.999**iteration)
        coefficients -= _LISTNET_LEARNING_RATE * first_unbiased / (np.sqrt(second_unbiased) + 1e-8)

    return best_coefficients, {
        "question_count": question_count,
        "frontier_action_count": target.frontier_action_count,
        "completed_iterations": completed_iterations,
        "converged_by_gradient": converged,
        "best_objective": round(float(best_loss), 9),
        "final_gradient_norm": round(final_gradient_norm, 9),
        "scaler": "StandardScaler(with_mean=False)",
        "patience_improvement_tolerance": _LISTNET_IMPROVEMENT_TOLERANCE,
    }


def _fit_logistic_classifier(
    matrix: csr_matrix,
    labels: np.ndarray,
    weights: np.ndarray,
) -> LogisticRegression:
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2_000,
        random_state=186,
        solver="liblinear",
    )
    model.fit(matrix, labels, sample_weight=weights)
    return model


def _fit_histogram_classifier(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        class_weight="balanced",
        learning_rate=0.06,
        l2_regularization=1.0,
        max_iter=120,
        max_leaf_nodes=15,
        random_state=186,
    )
    model.fit(matrix, labels, sample_weight=weights)
    return model


def _reference_rows(
    rows: Sequence[ActionAuditRow],
    stage182_selected_actions: Sequence[SelectedAction],
) -> dict[str, ActionAuditRow]:
    grouped = _group_rows(rows)
    selected = {item.row.question_key: item.row for item in stage182_selected_actions}
    references = {}
    for question_key, question_rows in grouped.items():
        if question_key in selected:
            references[question_key] = selected[question_key]
            continue
        baseline_rows = [row for row in question_rows if row.action.family == "baseline"]
        if len(baseline_rows) != 1:
            raise ValueError("Stage188 requires one baseline action for unselected questions")
        references[question_key] = baseline_rows[0]
    return references


def _outcome_tier(row: ActionAuditRow) -> int:
    if (
        row.citation_delta >= 0
        and row.f1_delta >= -_F1_TOLERANCE
        and (row.citation_delta > 0 or row.f1_delta > _F1_TOLERANCE)
    ):
        return 2
    if row.citation_delta == 0 and abs(row.f1_delta) <= _F1_TOLERANCE:
        return 1
    return 0


def _pareto_dominates(left: ActionAuditRow, right: ActionAuditRow) -> bool:
    citation_no_worse = left.citation_delta >= right.citation_delta
    f1_no_worse = left.f1_delta >= right.f1_delta - _F1_TOLERANCE
    strictly_better = (
        left.citation_delta > right.citation_delta or left.f1_delta > right.f1_delta + _F1_TOLERANCE
    )
    return bool(citation_no_worse and f1_no_worse and strictly_better)


def _inner_eligible(evaluation: Mapping[str, Any], question_count: int) -> bool:
    return bool(
        evaluation["gold_citation_delta"] >= 0
        and evaluation["mean_f1_delta"] >= 0
        and evaluation["citation_nonregressing_fold_count"] >= _MINIMUM_INNER_NONREGRESSING_FOLDS
        and evaluation["f1_nonregressing_fold_count"] >= _MINIMUM_INNER_NONREGRESSING_FOLDS
        and evaluation["changed_question_count"] >= math.ceil(_INNER_CHANGED_RATE * question_count)
        and evaluation["strict_success_count"]
        >= math.ceil(_INNER_STRICT_COUNT_RATE * question_count)
        and evaluation["strict_success_precision"] >= _INNER_STRICT_PRECISION
    )


def _inner_selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    evaluation = row["evaluation"]
    return (
        -evaluation["strict_success_count"],
        -evaluation["strict_success_precision"],
        evaluation["f1_regression_action_count"],
        evaluation["citation_loss_action_count"],
        -evaluation["gold_citation_delta"],
        -evaluation["mean_f1_delta"],
        -evaluation["repaired_reference_regression_count"],
        row["spec"]["name"],
    )


def _prediction_metrics(
    predictions: Sequence[GainSensitivePrediction],
) -> dict[str, Any]:
    if not predictions:
        return {
            "action_count": 0,
            "citation_loss": None,
            "f1_loss": None,
            "strict_gain": None,
            "comparable_pair_accuracy": None,
            "question_gain_top1_strict_rate": None,
        }
    citation_labels = [int(row.row.citation_delta < 0) for row in predictions]
    f1_labels = [int(row.row.f1_delta < -_F1_TOLERANCE) for row in predictions]
    gain_labels = [int(_outcome_tier(row.row) == 2) for row in predictions]
    grouped: dict[str, list[GainSensitivePrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.row.question_key].append(prediction)
    comparable_correct = 0
    comparable_count = 0
    top1_strict = 0
    for question_rows in grouped.values():
        top1 = min(
            question_rows,
            key=lambda row: (-row.gain_score, row.row.action.action_id),
        )
        top1_strict += _outcome_tier(top1.row) == 2
        for left_index, left in enumerate(question_rows):
            for right in question_rows[left_index + 1 :]:
                preference = pairwise_preference(left.row, right.row)
                if preference == 0:
                    continue
                comparable_count += 1
                predicted = (
                    1
                    if left.gain_score > right.gain_score
                    else -1
                    if right.gain_score > left.gain_score
                    else 0
                )
                comparable_correct += predicted == preference
    return {
        "action_count": len(predictions),
        "citation_loss": _binary_metrics(
            citation_labels,
            [row.citation_loss_probability for row in predictions],
        ),
        "f1_loss": _binary_metrics(
            f1_labels,
            [row.f1_loss_probability for row in predictions],
        ),
        "strict_gain": _binary_metrics(
            gain_labels,
            [row.gain_score for row in predictions],
        ),
        "comparable_pair_count": comparable_count,
        "comparable_pair_accuracy": _ratio(comparable_correct, comparable_count),
        "question_gain_top1_strict_rate": _ratio(top1_strict, len(grouped)),
    }


def _binary_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    positive_count = sum(labels)
    return {
        "positive_count": positive_count,
        "prevalence": _ratio(positive_count, len(labels)),
        "roc_auc": (
            round(float(roc_auc_score(labels, scores)), 6) if len(set(labels)) == 2 else None
        ),
        "average_precision": (
            round(float(average_precision_score(labels, scores)), 6) if positive_count else None
        ),
    }


def _paired_bootstrap(rows: Sequence[ActionAuditRow]) -> dict[str, Any]:
    citation = np.asarray([row.citation_delta for row in rows], dtype=np.float64)
    f1 = np.asarray([row.f1_delta for row in rows], dtype=np.float64)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    citation_values = []
    f1_values = []
    for _ in range(_BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, len(rows), size=len(rows))
        citation_values.append(float(np.sum(citation[indices])))
        f1_values.append(float(np.mean(f1[indices])))
    return {
        "available": True,
        "replicates": _BOOTSTRAP_REPLICATES,
        "seed": _BOOTSTRAP_SEED,
        "gold_citation_delta": _bootstrap_summary(citation_values),
        "mean_f1_delta": _bootstrap_summary(f1_values),
    }


def _unavailable_bootstrap() -> dict[str, Any]:
    return {
        "available": False,
        "reason": "not all outer folds had an inner-eligible configuration",
        "replicates": 0,
        "seed": _BOOTSTRAP_SEED,
        "gold_citation_delta": None,
        "mean_f1_delta": None,
    }


def _bootstrap_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": round(float(np.mean(values)), 6),
        "ci95_lower": round(float(np.quantile(values, 0.025)), 6),
        "ci95_upper": round(float(np.quantile(values, 0.975)), 6),
    }


def _advancement_gates(
    *,
    eligible_outer_fold_count: int,
    aggregate: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> list[dict[str, Any]]:
    citation_bootstrap = bootstrap["gold_citation_delta"] or {}
    f1_bootstrap = bootstrap["mean_f1_delta"] or {}
    return [
        _gate("outer_folds_with_inner_eligible_config_at_least_5", eligible_outer_fold_count >= 5),
        _gate("gold_citation_delta_at_least_5", aggregate["gold_citation_delta"] >= 5),
        _gate("mean_f1_delta_at_least_0_005249", aggregate["mean_f1_delta"] >= 0.005249),
        _gate(
            "citation_bootstrap_ci95_lower_nonnegative",
            citation_bootstrap.get("ci95_lower", -math.inf) >= 0,
        ),
        _gate(
            "f1_bootstrap_ci95_lower_nonnegative",
            f1_bootstrap.get("ci95_lower", -math.inf) >= 0,
        ),
        _gate(
            "citation_nonregressing_outer_folds_at_least_4",
            aggregate["citation_nonregressing_fold_count"] >= 4,
        ),
        _gate(
            "f1_nonregressing_outer_folds_at_least_4",
            aggregate["f1_nonregressing_fold_count"] >= 4,
        ),
        _gate("strict_success_count_at_least_37", aggregate["strict_success_count"] >= 37),
        _gate(
            "strict_success_precision_at_least_0_65",
            aggregate["strict_success_precision"] >= 0.65,
        ),
        _gate(
            "citation_loss_action_count_at_most_4",
            aggregate["citation_loss_action_count"] <= 4,
        ),
        _gate(
            "f1_regression_action_count_at_most_27",
            aggregate["f1_regression_action_count"] <= 27,
        ),
        _gate(
            "stage182_regression_repair_rate_at_least_0_50",
            aggregate["stage182_regression_repair_rate"] >= 0.50,
        ),
        _gate(
            "new_f1_regression_rate_at_most_0_02",
            aggregate["new_f1_regression_rate"] <= 0.02,
        ),
        _gate(
            "changed_question_count_at_least_37",
            aggregate["changed_question_count"] >= 37,
        ),
    ]


def _accumulate_fit_diagnostics(
    representations: Mapping[str, FittedGainSensitiveRepresentation],
    *,
    model_fit_count: int,
    comparable_pair_count: int,
    omitted_pair_count: int,
    listwise_question_fit_count: int,
    listnet_iteration_count: int,
) -> tuple[int, int, int, int, int]:
    for representation in representations.values():
        model_fit_count += representation.model_fit_count
        pairwise = representation.diagnostics["pairwise"]
        listnet = representation.diagnostics["listnet"]
        comparable_pair_count += int(pairwise["comparable_pair_count"])
        omitted_pair_count += int(pairwise["omitted_incomparable_pair_count"])
        listwise_question_fit_count += int(listnet["question_count"])
        listnet_iteration_count += int(listnet["completed_iterations"])
    return (
        model_fit_count,
        comparable_pair_count,
        omitted_pair_count,
        listwise_question_fit_count,
        listnet_iteration_count,
    )


def _spec_dict(spec: GainSensitivePolicySpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "feature_representation": spec.feature_representation,
        "safety_estimator": spec.safety_estimator,
        "gain_ranker": spec.gain_ranker,
        "safety_frontier_margin": spec.safety_frontier_margin,
    }


def _spec_from_dict(value: Mapping[str, Any]) -> GainSensitivePolicySpec:
    return GainSensitivePolicySpec(
        name=value["name"],
        feature_representation=value["feature_representation"],
        safety_estimator=value["safety_estimator"],
        gain_ranker=value["gain_ranker"],
        safety_frontier_margin=float(value["safety_frontier_margin"]),
    )


def _public_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "spec": row["spec"],
        "eligible": row["eligible"],
        "evaluation": row["evaluation"],
    }


def _group_rows(rows: Sequence[ActionAuditRow]) -> dict[str, list[ActionAuditRow]]:
    grouped: dict[str, list[ActionAuditRow]] = defaultdict(list)
    for row in rows:
        grouped[row.question_key].append(row)
    return grouped


def _question_balanced_weights(rows: Sequence[ActionAuditRow]) -> np.ndarray:
    counts = Counter(row.question_key for row in rows)
    return np.asarray([1.0 / counts[row.question_key] for row in rows], dtype=np.float64)


def _row_key(row: ActionAuditRow) -> tuple[str, str]:
    return row.question_key, row.action.action_id


def _mean(values: Sequence[float] | Any) -> float:
    materialized = list(values)
    return round(float(sum(materialized) / len(materialized)), 6) if materialized else 0.0


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator / denominator), 6) if denominator else 0.0


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _emit(progress_sink: ProgressSink | None, **event: Any) -> None:
    if progress_sink is not None:
        progress_sink(event)
