from __future__ import annotations

import math
import random
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

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

FeatureRepresentation = Literal["raw_runtime", "question_relative_runtime"]
EstimatorFamily = Literal["class_balanced_logistic", "histogram_gradient_boosting"]
RankingRule = Literal[
    "max_safety_risk_lexicographic",
    "citation_first_lexicographic",
    "pareto_constraint_dominance",
]
ProgressSink = Callable[[Mapping[str, Any]], None]

_F1_TOLERANCE = 1e-12
_BOOTSTRAP_REPLICATES = 2_000
_BOOTSTRAP_SEED = 185
_MINIMUM_INNER_NONREGRESSING_FOLDS = 3
_MINIMUM_OUTER_NONREGRESSING_FOLDS = 4
_MINIMUM_REPAIR_RATE = 0.50
_MAXIMUM_NEW_REGRESSION_RATE = 0.02
_MAXIMUM_CITATION_LOSS_ACTIONS = 4
_MINIMUM_STRICT_PRECISION = 0.65
_MINIMUM_CHANGED_QUESTIONS = 37


@dataclass(frozen=True)
class JointConstraintPolicySpec:
    """One frozen Stage 186 joint-constraint ranking configuration."""

    name: str
    feature_representation: FeatureRepresentation
    estimator_family: EstimatorFamily
    ranking_rule: RankingRule
    safety_dominance_margin: float
    strict_gain_margin: float

    @property
    def bundle_name(self) -> str:
        return f"{self.feature_representation}__{self.estimator_family}"


@dataclass(frozen=True)
class JointConstraintPrediction:
    """Three runtime-safe probabilities attached to one private action row."""

    row: ActionAuditRow
    citation_loss_probability: float
    f1_loss_probability: float
    strict_gain_probability: float


@dataclass(frozen=True)
class _FittedBundle:
    feature_representation: FeatureRepresentation
    estimator_family: EstimatorFamily
    vectorizer: DictVectorizer
    scaler: StandardScaler | None
    citation_loss_model: Any
    f1_loss_model: Any
    strict_gain_model: Any

    @property
    def name(self) -> str:
        return f"{self.feature_representation}__{self.estimator_family}"

    @property
    def feature_count(self) -> int:
        return len(self.vectorizer.feature_names_)

    def predict(
        self,
        rows: Sequence[ActionAuditRow],
        feature_index: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> tuple[JointConstraintPrediction, ...]:
        matrix = self.vectorizer.transform([dict(feature_index[_row_key(row)]) for row in rows])
        if self.scaler is not None:
            matrix = self.scaler.transform(matrix)
        else:
            matrix = matrix.toarray()
        probabilities = [
            model.predict_proba(matrix)[:, 1]
            for model in (
                self.citation_loss_model,
                self.f1_loss_model,
                self.strict_gain_model,
            )
        ]
        return tuple(
            JointConstraintPrediction(
                row=row,
                citation_loss_probability=float(probabilities[0][index]),
                f1_loss_probability=float(probabilities[1][index]),
                strict_gain_probability=float(probabilities[2][index]),
            )
            for index, row in enumerate(rows)
        )


class BundleFitter(Protocol):
    def __call__(
        self,
        rows: Sequence[ActionAuditRow],
        feature_indices: Mapping[
            str,
            Mapping[tuple[str, str], Mapping[str, Any]],
        ],
    ) -> Mapping[str, _FittedBundle]: ...


def stage186_policy_specs() -> tuple[JointConstraintPolicySpec, ...]:
    """Return the 72 frozen Stage 186 policy configurations."""

    specs = []
    for feature_representation in ("raw_runtime", "question_relative_runtime"):
        for estimator_family in (
            "class_balanced_logistic",
            "histogram_gradient_boosting",
        ):
            for ranking_rule in (
                "max_safety_risk_lexicographic",
                "citation_first_lexicographic",
                "pareto_constraint_dominance",
            ):
                for safety_margin in (0.0, 0.02, 0.05):
                    for gain_margin in (0.0, 0.05):
                        name = (
                            f"{feature_representation}__{estimator_family}__"
                            f"{ranking_rule}__safety_{safety_margin:.2f}__"
                            f"gain_{gain_margin:.2f}"
                        )
                        specs.append(
                            JointConstraintPolicySpec(
                                name=name,
                                feature_representation=feature_representation,
                                estimator_family=estimator_family,
                                ranking_rule=ranking_rule,
                                safety_dominance_margin=safety_margin,
                                strict_gain_margin=gain_margin,
                            )
                        )
    return tuple(specs)


def run_joint_constraint_nested_cv(
    *,
    action_rows: Sequence[ActionAuditRow],
    stage182_selected_actions: Sequence[SelectedAction],
    progress_sink: ProgressSink | None = None,
    bundle_fitter: BundleFitter | None = None,
) -> dict[str, Any]:
    """Run the frozen Stage 186 five-by-four nested ranking experiment."""

    started_at = time.perf_counter()
    rows = tuple(action_rows)
    fold_ids = tuple(sorted({row.fold_id for row in rows}))
    if len(fold_ids) != 5:
        raise ValueError("Stage186 requires five frozen question-grouped folds")
    if len({_row_key(row) for row in rows}) != len(rows):
        raise ValueError("Stage186 action rows must be unique by question and action")
    question_count = len({row.question_key for row in rows})
    if question_count != 370:
        raise ValueError("Stage186 requires the frozen 370 answerable train questions")

    references = _reference_rows(rows, stage182_selected_actions)
    if len(references) != question_count:
        raise ValueError("Stage186 requires one reference action per question")
    reference_regressions = {
        key for key, row in references.items() if row.f1_delta < -_F1_TOLERANCE
    }
    if len(reference_regressions) != 55:
        raise ValueError("Stage186 requires the frozen 55 Stage182 F1 regressions")

    feature_indices = build_composition_feature_indices(rows)
    feature_space_by_name = {
        "raw_runtime": feature_indices["raw"],
        "question_relative_runtime": feature_indices["question_relative"],
    }
    specs = stage186_policy_specs()
    fit_bundles = bundle_fitter or _fit_all_bundles
    model_fit_count = 0
    private_prediction_count = 0
    outer_rows: list[ActionAuditRow] = []
    outer_predictions_for_metrics: list[JointConstraintPrediction] = []
    outer_reports = {}
    selected_spec_counts: Counter[str] = Counter()
    feature_counts: dict[str, int] = {}
    fit_seconds = 0.0

    for outer_fold_id in fold_ids:
        outer_training = tuple(row for row in rows if row.fold_id != outer_fold_id)
        outer_heldout = tuple(row for row in rows if row.fold_id == outer_fold_id)
        inner_fold_ids = tuple(fold_id for fold_id in fold_ids if fold_id != outer_fold_id)
        inner_predictions: dict[str, list[JointConstraintPrediction]] = defaultdict(list)

        for inner_fold_id in inner_fold_ids:
            inner_training = tuple(row for row in outer_training if row.fold_id != inner_fold_id)
            inner_heldout = tuple(row for row in outer_training if row.fold_id == inner_fold_id)
            fitted_at = time.perf_counter()
            bundles = fit_bundles(inner_training, feature_space_by_name)
            fit_seconds += time.perf_counter() - fitted_at
            model_fit_count += len(bundles) * 3
            for bundle_name, bundle in bundles.items():
                feature_counts[bundle_name] = max(
                    feature_counts.get(bundle_name, 0),
                    bundle.feature_count,
                )
                predictions = bundle.predict(
                    inner_heldout,
                    feature_space_by_name[bundle.feature_representation],
                )
                inner_predictions[bundle_name].extend(predictions)
                private_prediction_count += len(predictions)
            _emit(
                progress_sink,
                phase="inner_partition_complete",
                outer_fold_id=outer_fold_id,
                inner_fold_id=inner_fold_id,
                training_action_count=len(inner_training),
                heldout_action_count=len(inner_heldout),
                cumulative_model_head_fit_count=model_fit_count,
            )

        candidate_reports = []
        inner_question_count = len({row.question_key for row in outer_training})
        for spec in specs:
            selected_rows = select_actions(
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
                cumulative_model_head_fit_count=model_fit_count,
            )
            continue

        selected_report = min(eligible_reports, key=_inner_selection_key)
        selected_spec = _spec_from_dict(selected_report["spec"])
        selected_spec_counts[selected_spec.name] += 1

        fitted_at = time.perf_counter()
        outer_bundles = fit_bundles(outer_training, feature_space_by_name)
        fit_seconds += time.perf_counter() - fitted_at
        model_fit_count += len(outer_bundles) * 3
        for bundle_name, bundle in outer_bundles.items():
            feature_counts[bundle_name] = max(
                feature_counts.get(bundle_name, 0),
                bundle.feature_count,
            )
        selected_bundle = outer_bundles[selected_spec.bundle_name]
        heldout_predictions = selected_bundle.predict(
            outer_heldout,
            feature_space_by_name[selected_bundle.feature_representation],
        )
        private_prediction_count += len(heldout_predictions)
        selected_outer_rows = select_actions(heldout_predictions, selected_spec)
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
            cumulative_model_head_fit_count=model_fit_count,
        )

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
            "bundle_count": 4,
            "model_targets_per_bundle": 3,
            "maximum_model_head_fit_count": 300,
            "inner_selection": (
                "citation/F1 aggregate nonregression, at least 3/4 fold "
                "nonregression, at least 10% changed, then frozen lexicographic objective"
            ),
            "no_eligible_behavior": (
                "record no-eligible and do not evaluate a weaker outer configuration"
            ),
            "gold_scope": "training targets and offline evaluation only",
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
                {name for features in feature_indices["raw"].values() for name in features}
            ),
            "relative_runtime_feature_count": len(
                {
                    name
                    for features in feature_indices["question_relative"].values()
                    for name in features
                }
            ),
        },
        "outer_folds": outer_reports,
        "aggregate": aggregate,
        "paired_bootstrap": bootstrap,
        "head_metrics": _head_metrics(outer_predictions_for_metrics),
        "selected_spec_counts": dict(sorted(selected_spec_counts.items())),
        "advancement_gates": gates,
        "advancement_gate_pass_count": sum(row["passed"] for row in gates),
        "candidate_family_accepted": all(row["passed"] for row in gates),
        "execution": {
            "model_head_fit_count": model_fit_count,
            "maximum_model_head_fit_count": 300,
            "private_prediction_count": private_prediction_count,
            "public_prediction_rows_written": 0,
            "feature_count_by_bundle": dict(sorted(feature_counts.items())),
            "fit_seconds": round(fit_seconds, 6),
            "wall_seconds": round(time.perf_counter() - started_at, 6),
        },
    }


def select_actions(
    predictions: Sequence[JointConstraintPrediction],
    spec: JointConstraintPolicySpec,
) -> tuple[ActionAuditRow, ...]:
    """Select exactly one action per question with the frozen ranking semantics."""

    grouped: dict[str, list[JointConstraintPrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.row.question_key].append(prediction)
    selected = []
    for question_predictions in grouped.values():
        if spec.ranking_rule == "pareto_constraint_dominance":
            ranked = sorted(
                question_predictions,
                key=lambda row: _pareto_key(row, question_predictions, spec),
            )
        else:
            ranked = sorted(
                question_predictions,
                key=lambda row: _lexicographic_key(row, spec),
            )
        selected.append(ranked[0].row)
    return tuple(sorted(selected, key=lambda row: row.question_key))


def evaluate_selected_actions(
    *,
    selected_rows: Sequence[ActionAuditRow],
    references: Mapping[str, ActionAuditRow],
    expected_fold_ids: Sequence[str],
) -> dict[str, Any]:
    """Evaluate selected actions against baseline outcomes and Stage 182 references."""

    selected = {row.question_key: row for row in selected_rows}
    changed_rows = [
        row
        for key, row in selected.items()
        if row.action.action_id != references[key].action.action_id
    ]
    reference_regression_keys = {
        key for key in selected if references[key].f1_delta < -_F1_TOLERANCE
    }
    reference_safe_keys = set(selected) - reference_regression_keys
    repaired = sum(
        selected[key].f1_delta >= -_F1_TOLERANCE
        and selected[key].citation_delta >= references[key].citation_delta
        for key in reference_regression_keys
    )
    new_regressions = sum(selected[key].f1_delta < -_F1_TOLERANCE for key in reference_safe_keys)
    fold_metrics = {}
    for fold_id in expected_fold_ids:
        fold_rows = [row for row in selected.values() if row.fold_id == fold_id]
        fold_refs = [references[row.question_key] for row in fold_rows]
        fold_metrics[fold_id] = {
            "question_count": len(fold_rows),
            "changed_question_count": sum(
                row.action.action_id != reference.action.action_id
                for row, reference in zip(fold_rows, fold_refs, strict=True)
            ),
            "gold_citation_delta": sum(row.citation_delta for row in fold_rows),
            "mean_f1_delta": _mean(row.f1_delta for row in fold_rows),
            "citation_delta_vs_reference": sum(
                row.citation_delta - reference.citation_delta
                for row, reference in zip(fold_rows, fold_refs, strict=True)
            ),
            "mean_f1_delta_vs_reference": _mean(
                row.f1_delta - reference.f1_delta
                for row, reference in zip(fold_rows, fold_refs, strict=True)
            ),
        }
    citation_delta = sum(row.citation_delta for row in selected.values())
    mean_f1_delta = _mean(row.f1_delta for row in selected.values())
    return {
        "question_count": len(selected),
        "changed_question_count": len(changed_rows),
        "changed_question_rate": _ratio(len(changed_rows), len(selected)),
        "strict_success_count": sum(row.strict_expected for row in changed_rows),
        "strict_success_precision": _ratio(
            sum(row.strict_expected for row in changed_rows),
            len(changed_rows),
        ),
        "citation_gain_action_count": sum(row.citation_delta > 0 for row in selected.values()),
        "citation_loss_action_count": sum(row.citation_delta < 0 for row in selected.values()),
        "f1_regression_action_count": sum(
            row.f1_delta < -_F1_TOLERANCE for row in selected.values()
        ),
        "gold_citation_delta": citation_delta,
        "mean_f1_delta": mean_f1_delta,
        "citation_delta_vs_reference": sum(
            row.citation_delta - references[key].citation_delta for key, row in selected.items()
        ),
        "mean_f1_delta_vs_reference": _mean(
            row.f1_delta - references[key].f1_delta for key, row in selected.items()
        ),
        "reference_regression_count": len(reference_regression_keys),
        "repaired_reference_regression_count": repaired,
        "stage182_regression_repair_rate": _ratio(
            repaired,
            len(reference_regression_keys),
        ),
        "reference_safe_count": len(reference_safe_keys),
        "new_f1_regression_count": new_regressions,
        "new_f1_regression_rate": _ratio(new_regressions, len(reference_safe_keys)),
        "citation_nonregressing_fold_count": sum(
            row["question_count"] > 0 and row["gold_citation_delta"] >= 0
            for row in fold_metrics.values()
        ),
        "f1_nonregressing_fold_count": sum(
            row["question_count"] > 0 and row["mean_f1_delta"] >= 0 for row in fold_metrics.values()
        ),
        "folds": fold_metrics,
        "strict_aggregate_pass": bool(
            citation_delta >= 0 and mean_f1_delta >= 0 and (citation_delta > 0 or mean_f1_delta > 0)
        ),
    }


def _fit_all_bundles(
    rows: Sequence[ActionAuditRow],
    feature_indices: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
) -> Mapping[str, _FittedBundle]:
    weights = _question_balanced_weights(rows)
    labels = {
        "citation_loss": [int(row.citation_delta < 0) for row in rows],
        "f1_loss": [int(row.f1_delta < -_F1_TOLERANCE) for row in rows],
        "strict_gain": [int(row.strict_expected) for row in rows],
    }
    for name, values in labels.items():
        if len(set(values)) != 2:
            raise ValueError(f"Stage186 {name} target requires both classes")

    bundles: dict[str, _FittedBundle] = {}
    for representation, feature_key in (
        ("raw_runtime", "raw_runtime"),
        ("question_relative_runtime", "question_relative_runtime"),
    ):
        vectorizer = DictVectorizer(sparse=True)
        matrix = vectorizer.fit_transform(
            [dict(feature_indices[feature_key][_row_key(row)]) for row in rows]
        )

        scaler = StandardScaler(with_mean=False)
        logistic_matrix = scaler.fit_transform(matrix)
        logistic_models = [
            _fit_logistic(logistic_matrix, labels[target], weights) for target in labels
        ]
        logistic_bundle = _FittedBundle(
            feature_representation=representation,
            estimator_family="class_balanced_logistic",
            vectorizer=vectorizer,
            scaler=scaler,
            citation_loss_model=logistic_models[0],
            f1_loss_model=logistic_models[1],
            strict_gain_model=logistic_models[2],
        )
        bundles[logistic_bundle.name] = logistic_bundle

        dense_matrix = matrix.toarray()
        hist_models = [_fit_histogram(dense_matrix, labels[target], weights) for target in labels]
        hist_bundle = _FittedBundle(
            feature_representation=representation,
            estimator_family="histogram_gradient_boosting",
            vectorizer=vectorizer,
            scaler=None,
            citation_loss_model=hist_models[0],
            f1_loss_model=hist_models[1],
            strict_gain_model=hist_models[2],
        )
        bundles[hist_bundle.name] = hist_bundle
    return bundles


def _fit_logistic(matrix: Any, labels: Sequence[int], weights: Sequence[float]) -> Any:
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2_000,
        random_state=186,
        solver="liblinear",
    )
    model.fit(matrix, labels, sample_weight=weights)
    return model


def _fit_histogram(matrix: Any, labels: Sequence[int], weights: Sequence[float]) -> Any:
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
            raise ValueError("Stage186 requires one baseline action for unselected questions")
        references[question_key] = baseline_rows[0]
    return references


def _lexicographic_key(
    row: JointConstraintPrediction,
    spec: JointConstraintPolicySpec,
) -> tuple[Any, ...]:
    citation_risk = _bucket(row.citation_loss_probability, spec.safety_dominance_margin)
    f1_risk = _bucket(row.f1_loss_probability, spec.safety_dominance_margin)
    gain = _bucket(row.strict_gain_probability, spec.strict_gain_margin)
    if spec.ranking_rule == "max_safety_risk_lexicographic":
        return (
            max(citation_risk, f1_risk),
            citation_risk + f1_risk,
            -gain,
            row.row.action.action_id,
        )
    if spec.ranking_rule == "citation_first_lexicographic":
        return (citation_risk, f1_risk, -gain, row.row.action.action_id)
    raise ValueError(f"unsupported lexicographic rule: {spec.ranking_rule}")


def _pareto_key(
    row: JointConstraintPrediction,
    question_rows: Sequence[JointConstraintPrediction],
    spec: JointConstraintPolicySpec,
) -> tuple[Any, ...]:
    dominator_count = sum(
        _dominates(candidate, row, spec)
        for candidate in question_rows
        if candidate.row.action.action_id != row.row.action.action_id
    )
    citation_risk = _bucket(row.citation_loss_probability, spec.safety_dominance_margin)
    f1_risk = _bucket(row.f1_loss_probability, spec.safety_dominance_margin)
    gain = _bucket(row.strict_gain_probability, spec.strict_gain_margin)
    return (
        dominator_count,
        max(citation_risk, f1_risk),
        citation_risk + f1_risk,
        -gain,
        row.row.action.action_id,
    )


def _dominates(
    left: JointConstraintPrediction,
    right: JointConstraintPrediction,
    spec: JointConstraintPolicySpec,
) -> bool:
    safety = spec.safety_dominance_margin
    gain = spec.strict_gain_margin
    citation_better = left.citation_loss_probability <= right.citation_loss_probability - safety
    f1_better = left.f1_loss_probability <= right.f1_loss_probability - safety
    gain_better = left.strict_gain_probability >= right.strict_gain_probability + gain
    if not (citation_better and f1_better and gain_better):
        return False
    return bool(
        left.citation_loss_probability < right.citation_loss_probability
        or left.f1_loss_probability < right.f1_loss_probability
        or left.strict_gain_probability > right.strict_gain_probability
    )


def _bucket(value: float, width: float) -> float | int:
    if width <= 0:
        return value
    return math.floor((value + 1e-12) / width)


def _inner_eligible(evaluation: Mapping[str, Any], question_count: int) -> bool:
    return bool(
        evaluation["gold_citation_delta"] >= 0
        and evaluation["mean_f1_delta"] >= 0
        and evaluation["citation_nonregressing_fold_count"] >= _MINIMUM_INNER_NONREGRESSING_FOLDS
        and evaluation["f1_nonregressing_fold_count"] >= _MINIMUM_INNER_NONREGRESSING_FOLDS
        and evaluation["changed_question_count"] >= math.ceil(0.10 * question_count)
    )


def _inner_selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    evaluation = row["evaluation"]
    return (
        -evaluation["repaired_reference_regression_count"],
        evaluation["new_f1_regression_count"],
        -evaluation["strict_success_precision"],
        -evaluation["gold_citation_delta"],
        -evaluation["mean_f1_delta"],
        row["spec"]["name"],
    )


def _public_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "spec": row["spec"],
        "eligible": row["eligible"],
        "evaluation": row["evaluation"],
    }


def _spec_dict(spec: JointConstraintPolicySpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "feature_representation": spec.feature_representation,
        "estimator_family": spec.estimator_family,
        "ranking_rule": spec.ranking_rule,
        "safety_dominance_margin": spec.safety_dominance_margin,
        "strict_gain_margin": spec.strict_gain_margin,
    }


def _spec_from_dict(value: Mapping[str, Any]) -> JointConstraintPolicySpec:
    return JointConstraintPolicySpec(
        name=value["name"],
        feature_representation=value["feature_representation"],
        estimator_family=value["estimator_family"],
        ranking_rule=value["ranking_rule"],
        safety_dominance_margin=value["safety_dominance_margin"],
        strict_gain_margin=value["strict_gain_margin"],
    )


def _advancement_gates(
    *,
    eligible_outer_fold_count: int,
    aggregate: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> list[dict[str, Any]]:
    available = bootstrap["available"]
    gates = [
        _gate("outer_folds_with_inner_eligible_config_at_least_5", eligible_outer_fold_count >= 5),
        _gate("gold_citation_delta_nonnegative", aggregate["gold_citation_delta"] >= 0),
        _gate("mean_f1_delta_nonnegative", aggregate["mean_f1_delta"] >= 0),
        _gate(
            "citation_bootstrap_ci95_lower_nonnegative",
            available and bootstrap["gold_citation_delta"]["ci95_lower"] >= 0,
        ),
        _gate(
            "f1_bootstrap_ci95_lower_nonnegative",
            available and bootstrap["mean_f1_delta"]["ci95_lower"] >= 0,
        ),
        _gate(
            "citation_nonregressing_outer_folds_at_least_4",
            aggregate["citation_nonregressing_fold_count"] >= _MINIMUM_OUTER_NONREGRESSING_FOLDS,
        ),
        _gate(
            "f1_nonregressing_outer_folds_at_least_4",
            aggregate["f1_nonregressing_fold_count"] >= _MINIMUM_OUTER_NONREGRESSING_FOLDS,
        ),
        _gate(
            "stage182_regression_repair_rate_at_least_0_50",
            aggregate["stage182_regression_repair_rate"] >= _MINIMUM_REPAIR_RATE,
        ),
        _gate(
            "new_f1_regression_rate_at_most_0_02",
            aggregate["new_f1_regression_rate"] <= _MAXIMUM_NEW_REGRESSION_RATE,
        ),
        _gate(
            "citation_loss_action_count_at_most_4",
            aggregate["citation_loss_action_count"] <= _MAXIMUM_CITATION_LOSS_ACTIONS,
        ),
        _gate(
            "strict_success_precision_at_least_0_65",
            aggregate["strict_success_precision"] >= _MINIMUM_STRICT_PRECISION,
        ),
        _gate(
            "changed_question_count_at_least_37",
            aggregate["changed_question_count"] >= _MINIMUM_CHANGED_QUESTIONS,
        ),
    ]
    return gates


def _head_metrics(
    predictions: Sequence[JointConstraintPrediction],
) -> dict[str, Any]:
    if not predictions:
        return {
            "action_count": 0,
            "citation_loss": None,
            "f1_loss": None,
            "strict_gain": None,
        }
    targets = (
        (
            "citation_loss",
            [int(row.row.citation_delta < 0) for row in predictions],
            [row.citation_loss_probability for row in predictions],
        ),
        (
            "f1_loss",
            [int(row.row.f1_delta < -_F1_TOLERANCE) for row in predictions],
            [row.f1_loss_probability for row in predictions],
        ),
        (
            "strict_gain",
            [int(row.row.strict_expected) for row in predictions],
            [row.strict_gain_probability for row in predictions],
        ),
    )
    return {
        "action_count": len(predictions),
        **{name: _binary_metrics(labels, scores) for name, labels, scores in targets},
    }


def _binary_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    positives = sum(labels)
    return {
        "positive_count": positives,
        "prevalence": _ratio(positives, len(labels)),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 6)
        if len(set(labels)) == 2
        else None,
        "average_precision": round(float(average_precision_score(labels, scores)), 6)
        if positives
        else None,
    }


def _paired_bootstrap(rows: Sequence[ActionAuditRow]) -> dict[str, Any]:
    citation = [row.citation_delta for row in rows]
    f1 = [row.f1_delta for row in rows]
    rng = random.Random(_BOOTSTRAP_SEED)
    citation_replicates = []
    f1_replicates = []
    for _ in range(_BOOTSTRAP_REPLICATES):
        indices = [rng.randrange(len(rows)) for _ in rows]
        citation_replicates.append(sum(citation[index] for index in indices))
        f1_replicates.append(sum(f1[index] for index in indices) / len(rows))
    return {
        "available": True,
        "replicates": _BOOTSTRAP_REPLICATES,
        "seed": _BOOTSTRAP_SEED,
        "gold_citation_delta": _bootstrap_summary(citation_replicates),
        "mean_f1_delta": _bootstrap_summary(f1_replicates),
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
    ordered = sorted(values)
    lower_index = math.floor(0.025 * len(ordered))
    upper_index = min(len(ordered) - 1, math.ceil(0.975 * len(ordered)) - 1)
    return {
        "mean": round(float(statistics.fmean(ordered)), 6),
        "ci95_lower": round(float(ordered[lower_index]), 6),
        "ci95_upper": round(float(ordered[upper_index]), 6),
    }


def _group_rows(rows: Sequence[ActionAuditRow]) -> dict[str, list[ActionAuditRow]]:
    grouped: dict[str, list[ActionAuditRow]] = defaultdict(list)
    for row in rows:
        grouped[row.question_key].append(row)
    for question_rows in grouped.values():
        question_rows.sort(key=lambda row: row.action.action_id)
    return grouped


def _question_balanced_weights(rows: Sequence[ActionAuditRow]) -> list[float]:
    counts = Counter(row.question_key for row in rows)
    return [1.0 / counts[row.question_key] for row in rows]


def _row_key(row: ActionAuditRow) -> tuple[str, str]:
    return row.question_key, row.action.action_id


def _mean(values: Any) -> float:
    materialized = list(values)
    return round(float(statistics.fmean(materialized)), 6) if materialized else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _emit(progress_sink: ProgressSink | None, **event: Any) -> None:
    if progress_sink is not None:
        progress_sink(event)
