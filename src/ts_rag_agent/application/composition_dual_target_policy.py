from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ts_rag_agent.application.composition_action_audit import ActionAuditRow

ModelFamily = Literal["logistic", "hist_gradient_boosting"]
ScoreMode = Literal[
    "citation_only",
    "safe_product",
    "citation_minus_half_risk",
    "citation_minus_risk",
]

_F1_EQUALITY_TOLERANCE = 1e-12
_BOOTSTRAP_REPLICATES = 2_000
_BOOTSTRAP_SEED = 182


@dataclass(frozen=True)
class DualTargetPolicySpec:
    """One fixed model, utility, and learned-coverage policy candidate."""

    name: str
    model_family: ModelFamily
    score_mode: ScoreMode
    target_coverage: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("policy name must not be empty")
        if self.model_family not in {"logistic", "hist_gradient_boosting"}:
            raise ValueError(f"unsupported model family: {self.model_family}")
        if self.score_mode not in {
            "citation_only",
            "safe_product",
            "citation_minus_half_risk",
            "citation_minus_risk",
        }:
            raise ValueError(f"unsupported score mode: {self.score_mode}")
        if not 0.0 < self.target_coverage <= 1.0:
            raise ValueError("target coverage must be in (0, 1]")


@dataclass(frozen=True)
class DualTargetPrediction:
    """Runtime-safe dual-head probabilities attached to one offline row."""

    row: ActionAuditRow
    citation_gain_probability: float
    f1_regression_probability: float


@dataclass(frozen=True)
class SelectedAction:
    """One selected action and its policy confidence."""

    row: ActionAuditRow
    utility: float
    citation_gain_probability: float
    f1_regression_probability: float


@dataclass(frozen=True)
class _FittedBinaryHead:
    vectorizer: DictVectorizer
    scaler: StandardScaler | None
    model: Any

    def predict(self, rows: Sequence[ActionAuditRow]) -> list[float]:
        matrix = self.vectorizer.transform([dict(row.runtime_features) for row in rows])
        if self.scaler is not None:
            matrix = self.scaler.transform(matrix)
        elif self.model.__class__.__name__ == "HistGradientBoostingClassifier":
            matrix = matrix.toarray()
        return [float(value) for value in self.model.predict_proba(matrix)[:, 1]]


@dataclass(frozen=True)
class _FittedDualTargetHeads:
    citation_head: _FittedBinaryHead
    f1_risk_head: _FittedBinaryHead

    def predict(self, rows: Sequence[ActionAuditRow]) -> tuple[DualTargetPrediction, ...]:
        citation = self.citation_head.predict(rows)
        risk = self.f1_risk_head.predict(rows)
        return tuple(
            DualTargetPrediction(
                row=row,
                citation_gain_probability=citation_probability,
                f1_regression_probability=risk_probability,
            )
            for row, citation_probability, risk_probability in zip(
                rows, citation, risk, strict=True
            )
        )


def stage182_policy_specs() -> tuple[DualTargetPolicySpec, ...]:
    """Return the frozen Stage 182 model/utility/coverage grid."""

    specs = []
    for model_family in ("logistic", "hist_gradient_boosting"):
        for score_mode in (
            "citation_only",
            "safe_product",
            "citation_minus_half_risk",
            "citation_minus_risk",
        ):
            for coverage in (0.10, 0.25, 0.50, 1.00):
                coverage_name = f"c{int(coverage * 100):03d}"
                specs.append(
                    DualTargetPolicySpec(
                        name=f"{model_family}_{score_mode}_{coverage_name}",
                        model_family=model_family,
                        score_mode=score_mode,
                        target_coverage=coverage,
                    )
                )
    return tuple(specs)


def run_nested_dual_target_selection(
    rows: Sequence[ActionAuditRow],
    *,
    specs: Sequence[DualTargetPolicySpec] | None = None,
    total_question_count: int | None = None,
) -> dict[str, Any]:
    """Select policies in inner OOF and evaluate once in each outer fold."""

    candidates = tuple(row for row in rows if row.action.family != "baseline")
    fold_ids = tuple(sorted({row.fold_id for row in candidates}))
    if len(fold_ids) != 5:
        raise ValueError("Stage182 nested selection requires five frozen folds")
    if not candidates:
        raise ValueError("Stage182 requires nonbaseline action rows")
    policy_specs = tuple(specs or stage182_policy_specs())
    if not policy_specs:
        raise ValueError("Stage182 requires policy candidates")
    model_families = tuple(sorted({spec.model_family for spec in policy_specs}))
    question_count = total_question_count or len({row.question_key for row in rows})

    outer_selected: list[SelectedAction] = []
    outer_reports: dict[str, Any] = {}
    model_head_fit_count = 0
    for outer_fold in fold_ids:
        outer_train = tuple(row for row in candidates if row.fold_id != outer_fold)
        outer_heldout = tuple(row for row in candidates if row.fold_id == outer_fold)
        inner_fold_ids = tuple(sorted({row.fold_id for row in outer_train}))
        inner_predictions: dict[str, tuple[DualTargetPrediction, ...]] = {}
        inner_head_metrics: dict[str, Any] = {}
        for family in model_families:
            family_predictions = []
            fold_metrics = {}
            for inner_fold in inner_fold_ids:
                training = tuple(row for row in outer_train if row.fold_id != inner_fold)
                heldout = tuple(row for row in outer_train if row.fold_id == inner_fold)
                heads = _fit_dual_target_heads(training, model_family=family)
                model_head_fit_count += 2
                predictions = heads.predict(heldout)
                family_predictions.extend(predictions)
                fold_metrics[inner_fold] = _head_metrics(predictions)
            inner_predictions[family] = tuple(family_predictions)
            inner_head_metrics[family] = {
                "aggregate": _head_metrics(family_predictions),
                "folds": fold_metrics,
            }

        evaluated_specs = []
        for spec in policy_specs:
            predictions = inner_predictions[spec.model_family]
            threshold = _learn_coverage_threshold(predictions, spec=spec)
            evaluation = evaluate_dual_target_policy(
                predictions,
                spec=spec,
                utility_threshold=threshold,
                total_question_count=len({row.question_key for row in outer_train}),
                expected_fold_ids=inner_fold_ids,
            )
            evaluated_specs.append(
                {
                    "spec": spec,
                    "utility_threshold": threshold,
                    "evaluation": evaluation,
                    "eligible": _inner_policy_eligible(evaluation),
                }
            )
        eligible = [row for row in evaluated_specs if row["eligible"]]
        selected = max(eligible, key=_inner_selection_key) if eligible else None
        heldout_selected: tuple[SelectedAction, ...] = ()
        if selected is not None:
            selected_spec = selected["spec"]
            outer_heads = _fit_dual_target_heads(
                outer_train,
                model_family=selected_spec.model_family,
            )
            model_head_fit_count += 2
            heldout_predictions = outer_heads.predict(outer_heldout)
            heldout_selected = select_dual_target_actions(
                heldout_predictions,
                spec=selected_spec,
                utility_threshold=selected["utility_threshold"],
            )
            outer_selected.extend(heldout_selected)
            heldout_head_metrics = _head_metrics(heldout_predictions)
            selected_name = selected_spec.name
            selected_inner_evaluation = selected["evaluation"]
            utility_threshold = selected["utility_threshold"]
        else:
            heldout_head_metrics = None
            selected_name = None
            selected_inner_evaluation = None
            utility_threshold = None
        outer_reports[outer_fold] = {
            "training_question_count": len({row.question_key for row in outer_train}),
            "heldout_question_count": len({row.question_key for row in outer_heldout}),
            "eligible_spec_count": len(eligible),
            "selected_spec": selected_name,
            "selected_utility_threshold": utility_threshold,
            "selected_inner_evaluation": selected_inner_evaluation,
            "heldout_head_metrics": heldout_head_metrics,
            "heldout_policy_evaluation": _selected_action_metrics(
                heldout_selected,
                total_question_count=len({row.question_key for row in outer_heldout}),
                expected_fold_ids=(outer_fold,),
                fold_question_counts={outer_fold: len({row.question_key for row in outer_heldout})},
            ),
            "inner_head_metrics": inner_head_metrics,
            "candidate_leaderboard": _public_candidate_leaderboard(evaluated_specs),
        }

    aggregate = _selected_action_metrics(
        outer_selected,
        total_question_count=question_count,
        expected_fold_ids=fold_ids,
        fold_question_counts={
            fold_id: len({row.question_key for row in candidates if row.fold_id == fold_id})
            for fold_id in fold_ids
        },
    )
    bootstrap = _paired_bootstrap(outer_selected, total_question_count=question_count)
    return {
        "protocol": {
            "outer_fold_count": len(fold_ids),
            "inner_fold_count_per_outer": len(fold_ids) - 1,
            "policy_candidate_count": len(policy_specs),
            "model_families": list(model_families),
            "target_labels": ["citation_gain", "f1_regression"],
            "utility_modes": sorted({spec.score_mode for spec in policy_specs}),
            "target_coverages": sorted({spec.target_coverage for spec in policy_specs}),
            "inner_fold_nonregression_requirement": "4/4 for citation and F1",
            "no_action_is_explicit_abstention": True,
            "fallback_enabled": False,
        },
        "model_head_fit_count": model_head_fit_count,
        "outer_folds": outer_reports,
        "aggregate": aggregate,
        "paired_bootstrap": bootstrap,
        "selected_spec_counts": dict(
            sorted(
                Counter(
                    report["selected_spec"] or "no_eligible_policy"
                    for report in outer_reports.values()
                ).items()
            )
        ),
        "selected_actions": tuple(outer_selected),
    }


def evaluate_dual_target_policy(
    predictions: Sequence[DualTargetPrediction],
    *,
    spec: DualTargetPolicySpec,
    utility_threshold: float | None,
    total_question_count: int,
    expected_fold_ids: Sequence[str],
) -> dict[str, Any]:
    selected = select_dual_target_actions(
        predictions,
        spec=spec,
        utility_threshold=utility_threshold,
    )
    return _selected_action_metrics(
        selected,
        total_question_count=total_question_count,
        expected_fold_ids=expected_fold_ids,
        fold_question_counts={
            fold_id: len(
                {
                    prediction.row.question_key
                    for prediction in predictions
                    if prediction.row.fold_id == fold_id
                }
            )
            for fold_id in expected_fold_ids
        },
    )


def select_dual_target_actions(
    predictions: Sequence[DualTargetPrediction],
    *,
    spec: DualTargetPolicySpec,
    utility_threshold: float | None,
) -> tuple[SelectedAction, ...]:
    """Choose at most one action per question and deliberately abstain below threshold."""

    grouped: dict[str, list[DualTargetPrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.row.question_key].append(prediction)
    selected = []
    for question_key in sorted(grouped):
        best = max(
            grouped[question_key],
            key=lambda row: (
                _policy_utility(row, spec.score_mode),
                row.citation_gain_probability,
                -row.f1_regression_probability,
                row.row.action.action_id,
            ),
        )
        utility = _policy_utility(best, spec.score_mode)
        if utility_threshold is not None and utility < utility_threshold:
            continue
        selected.append(
            SelectedAction(
                row=best.row,
                utility=utility,
                citation_gain_probability=best.citation_gain_probability,
                f1_regression_probability=best.f1_regression_probability,
            )
        )
    return tuple(selected)


def _fit_dual_target_heads(
    rows: Sequence[ActionAuditRow],
    *,
    model_family: ModelFamily,
) -> _FittedDualTargetHeads:
    citation_labels = [int(row.citation_delta > 0) for row in rows]
    f1_risk_labels = [int(row.f1_delta < -_F1_EQUALITY_TOLERANCE) for row in rows]
    if len(set(citation_labels)) != 2:
        raise ValueError("citation-gain training labels require both classes")
    if len(set(f1_risk_labels)) != 2:
        raise ValueError("F1-regression training labels require both classes")
    return _FittedDualTargetHeads(
        citation_head=_fit_binary_head(rows, citation_labels, model_family=model_family),
        f1_risk_head=_fit_binary_head(rows, f1_risk_labels, model_family=model_family),
    )


def _fit_binary_head(
    rows: Sequence[ActionAuditRow],
    labels: Sequence[int],
    *,
    model_family: ModelFamily,
) -> _FittedBinaryHead:
    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform([dict(row.runtime_features) for row in rows])
    weights = _question_balanced_weights(rows)
    if model_family == "logistic":
        scaler = StandardScaler(with_mean=False)
        matrix = scaler.fit_transform(matrix)
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=182,
            solver="liblinear",
        )
    elif model_family == "hist_gradient_boosting":
        scaler = None
        matrix = matrix.toarray()
        model = HistGradientBoostingClassifier(
            class_weight="balanced",
            learning_rate=0.08,
            l2_regularization=1.0,
            max_iter=100,
            max_leaf_nodes=15,
            random_state=182,
        )
    else:
        raise ValueError(f"unsupported model family: {model_family}")
    model.fit(matrix, labels, sample_weight=weights)
    return _FittedBinaryHead(vectorizer=vectorizer, scaler=scaler, model=model)


def _learn_coverage_threshold(
    predictions: Sequence[DualTargetPrediction],
    *,
    spec: DualTargetPolicySpec,
) -> float | None:
    if spec.target_coverage >= 1.0:
        return None
    grouped: dict[str, list[DualTargetPrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.row.question_key].append(prediction)
    top_utilities = sorted(
        (
            max(_policy_utility(row, spec.score_mode) for row in question_rows)
            for question_rows in grouped.values()
        ),
        reverse=True,
    )
    target = max(1, math.ceil(len(top_utilities) * spec.target_coverage))
    return float(top_utilities[target - 1])


def _policy_utility(prediction: DualTargetPrediction, mode: ScoreMode) -> float:
    citation = prediction.citation_gain_probability
    risk = prediction.f1_regression_probability
    if mode == "citation_only":
        return citation
    if mode == "safe_product":
        return citation * (1.0 - risk)
    if mode == "citation_minus_half_risk":
        return citation - 0.5 * risk
    if mode == "citation_minus_risk":
        return citation - risk
    raise ValueError(f"unsupported score mode: {mode}")


def _head_metrics(predictions: Sequence[DualTargetPrediction]) -> dict[str, Any]:
    citation_labels = [int(row.row.citation_delta > 0) for row in predictions]
    citation_scores = [row.citation_gain_probability for row in predictions]
    risk_labels = [int(row.row.f1_delta < -_F1_EQUALITY_TOLERANCE) for row in predictions]
    risk_scores = [row.f1_regression_probability for row in predictions]
    return {
        "action_count": len(predictions),
        "citation_gain": _binary_metrics(citation_labels, citation_scores),
        "f1_regression": _binary_metrics(risk_labels, risk_scores),
    }


def _binary_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    positives = sum(labels)
    return {
        "positive_count": positives,
        "prevalence": _ratio(positives, len(labels)),
        "roc_auc": round(roc_auc_score(labels, scores), 6) if len(set(labels)) == 2 else None,
        "average_precision": round(average_precision_score(labels, scores), 6)
        if positives
        else None,
    }


def _selected_action_metrics(
    selected: Sequence[SelectedAction],
    *,
    total_question_count: int,
    expected_fold_ids: Sequence[str],
    fold_question_counts: Mapping[str, int],
) -> dict[str, Any]:
    fold_rows = {
        fold_id: [selection.row for selection in selected if selection.row.fold_id == fold_id]
        for fold_id in expected_fold_ids
    }
    fold_metrics = {
        fold_id: {
            "selected_question_count": len(rows),
            "gold_citation_delta": sum(row.citation_delta for row in rows),
            "mean_f1_delta_all_fold_questions": round(
                sum(row.f1_delta for row in rows) / max(1, fold_question_counts[fold_id]),
                6,
            ),
            "selected_action_mean_f1_delta": _mean(row.f1_delta for row in rows),
        }
        for fold_id, rows in fold_rows.items()
    }
    rows = [selection.row for selection in selected]
    citation_delta = sum(row.citation_delta for row in rows)
    f1_delta = round(sum(row.f1_delta for row in rows) / total_question_count, 6)
    return {
        "total_question_count": total_question_count,
        "selected_question_count": len(selected),
        "question_coverage": _ratio(len(selected), total_question_count),
        "strict_expected_count": sum(row.strict_expected for row in rows),
        "strict_expected_precision": _ratio(sum(row.strict_expected for row in rows), len(rows)),
        "citation_gain_action_count": sum(row.citation_delta > 0 for row in rows),
        "citation_loss_action_count": sum(row.citation_delta < 0 for row in rows),
        "f1_regression_action_count": sum(row.f1_delta < -_F1_EQUALITY_TOLERANCE for row in rows),
        "gold_citation_delta": citation_delta,
        "mean_f1_delta_all_questions": f1_delta,
        "selected_action_mean_f1_delta": _mean(row.f1_delta for row in rows),
        "citation_nonregressing_fold_count": sum(
            metrics["gold_citation_delta"] >= 0 for metrics in fold_metrics.values()
        ),
        "f1_nonregressing_fold_count": sum(
            metrics["mean_f1_delta_all_fold_questions"] >= 0 for metrics in fold_metrics.values()
        ),
        "folds": fold_metrics,
        "strict_aggregate_pass": bool(
            citation_delta >= 0 and f1_delta >= 0 and (citation_delta > 0 or f1_delta > 0)
        ),
    }


def _inner_policy_eligible(evaluation: Mapping[str, Any]) -> bool:
    fold_count = len(evaluation["folds"])
    return bool(
        evaluation["strict_aggregate_pass"]
        and evaluation["citation_nonregressing_fold_count"] == fold_count
        and evaluation["f1_nonregressing_fold_count"] == fold_count
    )


def _inner_selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    evaluation = row["evaluation"]
    spec = row["spec"]
    return (
        evaluation["gold_citation_delta"],
        evaluation["mean_f1_delta_all_questions"],
        evaluation["strict_expected_precision"],
        -evaluation["selected_question_count"],
        spec.name,
    )


def _public_candidate_leaderboard(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=_inner_selection_key, reverse=True)
    return [
        {
            "name": row["spec"].name,
            "model_family": row["spec"].model_family,
            "score_mode": row["spec"].score_mode,
            "target_coverage": row["spec"].target_coverage,
            "utility_threshold": row["utility_threshold"],
            "eligible": row["eligible"],
            "evaluation": row["evaluation"],
        }
        for row in ordered
    ]


def _paired_bootstrap(
    selected: Sequence[SelectedAction],
    *,
    total_question_count: int,
) -> dict[str, Any]:
    by_question = {selection.row.question_key: selection.row for selection in selected}
    observed_keys = sorted(by_question)
    padded_citation = [by_question[key].citation_delta for key in observed_keys]
    padded_f1 = [by_question[key].f1_delta for key in observed_keys]
    missing = total_question_count - len(observed_keys)
    padded_citation.extend([0] * missing)
    padded_f1.extend([0.0] * missing)
    rng = random.Random(_BOOTSTRAP_SEED)
    citation_replicates = []
    f1_replicates = []
    for _ in range(_BOOTSTRAP_REPLICATES):
        indices = [rng.randrange(total_question_count) for _ in range(total_question_count)]
        citation_replicates.append(sum(padded_citation[index] for index in indices))
        f1_replicates.append(sum(padded_f1[index] for index in indices) / total_question_count)
    return {
        "replicates": _BOOTSTRAP_REPLICATES,
        "seed": _BOOTSTRAP_SEED,
        "gold_citation_delta": _bootstrap_summary(citation_replicates),
        "mean_f1_delta": _bootstrap_summary(f1_replicates),
    }


def _bootstrap_summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    lower_index = math.floor(0.025 * len(ordered))
    upper_index = min(len(ordered) - 1, math.ceil(0.975 * len(ordered)) - 1)
    return {
        "mean": round(statistics.fmean(ordered), 6),
        "ci95_lower": round(float(ordered[lower_index]), 6),
        "ci95_upper": round(float(ordered[upper_index]), 6),
    }


def _question_balanced_weights(rows: Sequence[ActionAuditRow]) -> list[float]:
    counts = Counter(row.question_key for row in rows)
    return [1.0 / counts[row.question_key] for row in rows]


def _mean(values: Any) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return round(statistics.fmean(materialized), 6)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)
