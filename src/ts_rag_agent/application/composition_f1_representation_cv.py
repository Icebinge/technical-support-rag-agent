from __future__ import annotations

import math
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from scipy.sparse import vstack
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ts_rag_agent.application.composition_action_audit import ActionAuditRow
from ts_rag_agent.application.composition_dual_target_policy import SelectedAction

FeatureSpace = Literal["raw", "question_relative"]
TargetKind = Literal["binary", "ordinal", "quantile", "pairwise"]
ModelFamily = Literal["logistic", "hist_gradient_boosting"]
ProgressSink = Callable[[Mapping[str, Any]], None]

_F1_TOLERANCE = 1e-12
_ORDINAL_THRESHOLDS = (-_F1_TOLERANCE, -0.01, -0.05)
_MINIMUM_RISK_AUC = 0.62
_MINIMUM_AUC_GAIN_VS_RAW = 0.03
_MINIMUM_NONREGRESSING_FOLDS = 4
_MINIMUM_SAFE_TOP3_RATE = 0.70
_MINIMUM_SAFE_TOP5_RATE = 0.85


@dataclass(frozen=True)
class F1RepresentationSpec:
    """One frozen runtime-visible F1 representation candidate."""

    name: str
    feature_space: FeatureSpace
    target_kind: TargetKind
    model_family: ModelFamily


@dataclass(frozen=True)
class F1RepresentationPrediction:
    """One private held-out F1-risk and F1-safety score."""

    row: ActionAuditRow
    risk_score: float
    safety_score: float


class _Predictor(Protocol):
    feature_count: int
    fit_count: int

    def predict(
        self,
        rows: Sequence[ActionAuditRow],
        feature_index: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> tuple[F1RepresentationPrediction, ...]: ...


@dataclass(frozen=True)
class _BinaryHead:
    vectorizer: DictVectorizer
    scaler: StandardScaler | None
    model: Any

    @property
    def feature_count(self) -> int:
        return len(self.vectorizer.feature_names_)

    def predict(self, features: Sequence[Mapping[str, Any]]) -> list[float]:
        matrix = self.vectorizer.transform([dict(row) for row in features])
        if self.scaler is not None:
            matrix = self.scaler.transform(matrix)
        else:
            matrix = matrix.toarray()
        return [float(value) for value in self.model.predict_proba(matrix)[:, 1]]


@dataclass(frozen=True)
class _BinaryPredictor:
    head: _BinaryHead

    @property
    def feature_count(self) -> int:
        return self.head.feature_count

    @property
    def fit_count(self) -> int:
        return 1

    def predict(
        self,
        rows: Sequence[ActionAuditRow],
        feature_index: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> tuple[F1RepresentationPrediction, ...]:
        risks = self.head.predict(_features_for(rows, feature_index))
        return tuple(
            F1RepresentationPrediction(row=row, risk_score=risk, safety_score=1.0 - risk)
            for row, risk in zip(rows, risks, strict=True)
        )


@dataclass(frozen=True)
class _OrdinalPredictor:
    heads: tuple[_BinaryHead, ...]

    @property
    def feature_count(self) -> int:
        return max(head.feature_count for head in self.heads)

    @property
    def fit_count(self) -> int:
        return len(self.heads)

    def predict(
        self,
        rows: Sequence[ActionAuditRow],
        feature_index: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> tuple[F1RepresentationPrediction, ...]:
        features = _features_for(rows, feature_index)
        probabilities = [head.predict(features) for head in self.heads]
        risks = [
            sum((index + 1) * values[row_index] for index, values in enumerate(probabilities)) / 6.0
            for row_index in range(len(rows))
        ]
        return tuple(
            F1RepresentationPrediction(row=row, risk_score=risk, safety_score=1.0 - risk)
            for row, risk in zip(rows, risks, strict=True)
        )


@dataclass(frozen=True)
class _QuantilePredictor:
    vectorizer: DictVectorizer
    model: HistGradientBoostingRegressor

    @property
    def feature_count(self) -> int:
        return len(self.vectorizer.feature_names_)

    @property
    def fit_count(self) -> int:
        return 1

    def predict(
        self,
        rows: Sequence[ActionAuditRow],
        feature_index: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> tuple[F1RepresentationPrediction, ...]:
        matrix = self.vectorizer.transform(_features_for(rows, feature_index)).toarray()
        predicted_deltas = [float(value) for value in self.model.predict(matrix)]
        return tuple(
            F1RepresentationPrediction(
                row=row,
                risk_score=-predicted_delta,
                safety_score=predicted_delta,
            )
            for row, predicted_delta in zip(rows, predicted_deltas, strict=True)
        )


@dataclass(frozen=True)
class _PairwisePredictor:
    vectorizer: DictVectorizer
    scaler: StandardScaler
    model: LogisticRegression

    @property
    def feature_count(self) -> int:
        return len(self.vectorizer.feature_names_)

    @property
    def fit_count(self) -> int:
        return 1

    def predict(
        self,
        rows: Sequence[ActionAuditRow],
        feature_index: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> tuple[F1RepresentationPrediction, ...]:
        grouped = _group_rows(rows)
        scores: dict[tuple[str, str], float] = {}
        for question_rows in grouped.values():
            if len(question_rows) == 1:
                scores[_row_key(question_rows[0])] = 0.5
                continue
            action_matrix = self.vectorizer.transform(_features_for(question_rows, feature_index))
            left_indices = []
            right_indices = []
            left_keys = []
            for left_index, left in enumerate(question_rows):
                for right_index, _right in enumerate(question_rows):
                    if left_index == right_index:
                        continue
                    left_indices.append(left_index)
                    right_indices.append(right_index)
                    left_keys.append(_row_key(left))
            differences = action_matrix[left_indices] - action_matrix[right_indices]
            differences = self.scaler.transform(differences)
            probabilities = self.model.predict_proba(differences)[:, 1]
            sums: dict[tuple[str, str], float] = defaultdict(float)
            counts: Counter[tuple[str, str]] = Counter()
            for key, probability in zip(left_keys, probabilities, strict=True):
                sums[key] += float(probability)
                counts[key] += 1
            for key in sums:
                scores[key] = sums[key] / counts[key]
        return tuple(
            F1RepresentationPrediction(
                row=row,
                risk_score=1.0 - scores[_row_key(row)],
                safety_score=scores[_row_key(row)],
            )
            for row in rows
        )


def stage184_representation_specs() -> tuple[F1RepresentationSpec, ...]:
    """Return the frozen Stage 184 representation grid."""

    return (
        F1RepresentationSpec("raw_logistic_binary", "raw", "binary", "logistic"),
        F1RepresentationSpec("raw_hist_binary", "raw", "binary", "hist_gradient_boosting"),
        F1RepresentationSpec("relative_logistic_binary", "question_relative", "binary", "logistic"),
        F1RepresentationSpec(
            "relative_hist_binary",
            "question_relative",
            "binary",
            "hist_gradient_boosting",
        ),
        F1RepresentationSpec(
            "relative_logistic_ordinal", "question_relative", "ordinal", "logistic"
        ),
        F1RepresentationSpec(
            "relative_hist_ordinal",
            "question_relative",
            "ordinal",
            "hist_gradient_boosting",
        ),
        F1RepresentationSpec(
            "relative_hist_quantile_p25",
            "question_relative",
            "quantile",
            "hist_gradient_boosting",
        ),
        F1RepresentationSpec(
            "relative_pairwise_logistic", "question_relative", "pairwise", "logistic"
        ),
    )


def run_f1_representation_cv(
    *,
    action_rows: Sequence[ActionAuditRow],
    stage182_selected_actions: Sequence[SelectedAction],
    progress_sink: ProgressSink | None = None,
) -> dict[str, Any]:
    """Compare runtime-visible F1 representations in frozen five-fold OOF."""

    started_at = time.perf_counter()
    rows = tuple(row for row in action_rows if row.action.family != "baseline")
    fold_ids = tuple(sorted({row.fold_id for row in rows}))
    if len(fold_ids) != 5:
        raise ValueError("Stage184 requires five frozen folds")
    if len({_row_key(row) for row in rows}) != len(rows):
        raise ValueError("Stage184 action rows must be unique by question and action")
    selected_regressions = tuple(
        row for row in stage182_selected_actions if row.row.f1_delta < -_F1_TOLERANCE
    )
    if not selected_regressions:
        raise ValueError("Stage184 requires Stage182 selected F1 regressions")

    feature_indices = {
        "raw": {_row_key(row): dict(row.runtime_features) for row in rows},
        "question_relative": _question_relative_feature_index(rows),
    }
    specs = stage184_representation_specs()
    predictions_by_spec: dict[str, list[F1RepresentationPrediction]] = {
        spec.name: [] for spec in specs
    }
    fold_metrics_by_spec: dict[str, dict[str, Any]] = {spec.name: {} for spec in specs}
    fit_counts = Counter()
    feature_counts: dict[str, int] = {}
    fit_seconds = Counter()
    for fold_id in fold_ids:
        training = tuple(row for row in rows if row.fold_id != fold_id)
        heldout = tuple(row for row in rows if row.fold_id == fold_id)
        for spec in specs:
            fitted_at = time.perf_counter()
            predictor = _fit_predictor(
                spec,
                training,
                feature_indices[spec.feature_space],
            )
            predictions = predictor.predict(heldout, feature_indices[spec.feature_space])
            elapsed = time.perf_counter() - fitted_at
            predictions_by_spec[spec.name].extend(predictions)
            fold_metrics_by_spec[spec.name][fold_id] = _prediction_metrics(predictions)
            fit_counts[spec.name] += predictor.fit_count
            feature_counts[spec.name] = max(
                feature_counts.get(spec.name, 0), predictor.feature_count
            )
            fit_seconds[spec.name] += elapsed
            _emit(
                progress_sink,
                phase="representation_fold_complete",
                fold_id=fold_id,
                representation=spec.name,
                heldout_action_count=len(heldout),
                elapsed_seconds=round(elapsed, 6),
            )

    reports = {}
    for spec in specs:
        predictions = tuple(predictions_by_spec[spec.name])
        reports[spec.name] = {
            "spec": {
                "feature_space": spec.feature_space,
                "target_kind": spec.target_kind,
                "model_family": spec.model_family,
            },
            "aggregate": _prediction_metrics(predictions),
            "folds": fold_metrics_by_spec[spec.name],
            "stage182_regression_headroom": _selected_regression_headroom(
                predictions=predictions,
                selected_regressions=selected_regressions,
            ),
            "feature_count": feature_counts[spec.name],
            "model_fit_count": fit_counts[spec.name],
            "fit_and_predict_seconds": round(fit_seconds[spec.name], 6),
        }

    selection = _select_representation(reports)
    return {
        "protocol": {
            "fold_count": len(fold_ids),
            "representation_candidate_count": len(specs),
            "selection_metric": (
                "maximum aggregate F1-regression ROC AUC, then average precision, "
                "Stage182-regression safe top3/top1 rate, then name"
            ),
            "ordinal_thresholds": list(_ORDINAL_THRESHOLDS),
            "quantile": 0.25,
            "pairwise_training": (
                "all within-question unequal-F1 pairs in both orientations with "
                "question-balanced weights"
            ),
            "runtime_feature_scope": (
                "raw action features plus label-free within-question deltas, z-scores, "
                "percentiles, extrema, and added-versus-selected contrasts"
            ),
            "gold_scope": "training targets and offline evaluation only",
            "minimum_risk_auc": _MINIMUM_RISK_AUC,
            "minimum_auc_gain_vs_raw": _MINIMUM_AUC_GAIN_VS_RAW,
            "minimum_nonregressing_folds": _MINIMUM_NONREGRESSING_FOLDS,
            "minimum_safe_top3_rate": _MINIMUM_SAFE_TOP3_RATE,
            "minimum_safe_top5_rate": _MINIMUM_SAFE_TOP5_RATE,
            "replacement_policy_selection_enabled": False,
            "development_and_test_closed": True,
            "fallback_enabled": False,
        },
        "dataset": {
            "action_count": len(rows),
            "question_count": len({row.question_key for row in rows}),
            "fold_action_counts": {
                fold_id: sum(row.fold_id == fold_id for row in rows) for fold_id in fold_ids
            },
            "stage182_selected_regression_count": len(selected_regressions),
            "raw_runtime_feature_count": len(
                {name for row in rows for name in row.runtime_features}
            ),
            "relative_runtime_feature_count": len(
                {
                    name
                    for features in feature_indices["question_relative"].values()
                    for name in features
                }
            ),
        },
        "representations": reports,
        "selection": selection,
        "execution": {
            "model_fit_count": sum(fit_counts.values()),
            "fit_count_by_representation": dict(sorted(fit_counts.items())),
            "wall_seconds": round(time.perf_counter() - started_at, 6),
            "private_prediction_count": sum(len(rows) for _ in specs),
            "public_prediction_rows_written": 0,
        },
    }


def _fit_predictor(
    spec: F1RepresentationSpec,
    rows: Sequence[ActionAuditRow],
    feature_index: Mapping[tuple[str, str], Mapping[str, Any]],
) -> _Predictor:
    features = _features_for(rows, feature_index)
    weights = _question_balanced_weights(rows)
    if spec.target_kind == "binary":
        labels = [int(row.f1_delta < -_F1_TOLERANCE) for row in rows]
        return _BinaryPredictor(
            _fit_binary_head(features, labels, weights, model_family=spec.model_family)
        )
    if spec.target_kind == "ordinal":
        return _OrdinalPredictor(
            tuple(
                _fit_binary_head(
                    features,
                    [int(row.f1_delta < threshold) for row in rows],
                    weights,
                    model_family=spec.model_family,
                )
                for threshold in _ORDINAL_THRESHOLDS
            )
        )
    if spec.target_kind == "quantile":
        vectorizer = DictVectorizer(sparse=True)
        matrix = vectorizer.fit_transform([dict(row) for row in features]).toarray()
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.25,
            learning_rate=0.06,
            l2_regularization=1.0,
            max_iter=120,
            max_leaf_nodes=15,
            random_state=184,
        )
        model.fit(matrix, [row.f1_delta for row in rows], sample_weight=weights)
        return _QuantilePredictor(vectorizer=vectorizer, model=model)
    if spec.target_kind == "pairwise":
        return _fit_pairwise_predictor(rows, feature_index)
    raise ValueError(f"unsupported target kind: {spec.target_kind}")


def _fit_binary_head(
    features: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    weights: Sequence[float],
    *,
    model_family: ModelFamily,
) -> _BinaryHead:
    if len(set(labels)) != 2:
        raise ValueError("binary representation target requires both classes")
    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform([dict(row) for row in features])
    if model_family == "logistic":
        scaler = StandardScaler(with_mean=False)
        matrix = scaler.fit_transform(matrix)
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=184,
            solver="liblinear",
        )
    elif model_family == "hist_gradient_boosting":
        scaler = None
        matrix = matrix.toarray()
        model = HistGradientBoostingClassifier(
            class_weight="balanced",
            learning_rate=0.06,
            l2_regularization=1.0,
            max_iter=120,
            max_leaf_nodes=15,
            random_state=184,
        )
    else:
        raise ValueError(f"unsupported model family: {model_family}")
    model.fit(matrix, labels, sample_weight=weights)
    return _BinaryHead(vectorizer=vectorizer, scaler=scaler, model=model)


def _fit_pairwise_predictor(
    rows: Sequence[ActionAuditRow],
    feature_index: Mapping[tuple[str, str], Mapping[str, Any]],
) -> _PairwisePredictor:
    ordered_rows = tuple(rows)
    row_positions = {_row_key(row): index for index, row in enumerate(ordered_rows)}
    vectorizer = DictVectorizer(sparse=True)
    action_matrix = vectorizer.fit_transform(_features_for(ordered_rows, feature_index))
    left_indices = []
    right_indices = []
    base_labels = []
    base_weights = []
    for question_rows in _group_rows(rows).values():
        unequal_pairs = [
            (left, right)
            for left_index, left in enumerate(question_rows)
            for right in question_rows[left_index + 1 :]
            if abs(left.f1_delta - right.f1_delta) > _F1_TOLERANCE
        ]
        if not unequal_pairs:
            continue
        pair_weight = 1.0 / (2.0 * len(unequal_pairs))
        for left, right in unequal_pairs:
            left_indices.append(row_positions[_row_key(left)])
            right_indices.append(row_positions[_row_key(right)])
            base_labels.append(int(left.f1_delta > right.f1_delta))
            base_weights.append(pair_weight)
    differences = action_matrix[left_indices] - action_matrix[right_indices]
    matrix = vstack((differences, -differences), format="csr")
    labels = [*base_labels, *(1 - label for label in base_labels)]
    weights = [*base_weights, *base_weights]
    if len(set(labels)) != 2:
        raise ValueError("pairwise representation target requires both classes")
    scaler = StandardScaler(with_mean=False)
    matrix = scaler.fit_transform(matrix)
    model = LogisticRegression(
        max_iter=2_000,
        random_state=184,
        solver="liblinear",
    )
    model.fit(matrix, labels, sample_weight=weights)
    return _PairwisePredictor(vectorizer=vectorizer, scaler=scaler, model=model)


def _question_relative_feature_index(
    rows: Sequence[ActionAuditRow],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for question_rows in _group_rows(rows).values():
        numeric_names = sorted(
            {
                str(name)
                for row in question_rows
                for name, value in row.runtime_features.items()
                if isinstance(value, (bool, int, float))
            }
        )
        distributions = {
            name: [float(row.runtime_features.get(name, 0.0)) for row in question_rows]
            for name in numeric_names
        }
        for row in question_rows:
            features = dict(row.runtime_features)
            for name, values in distributions.items():
                value = float(row.runtime_features.get(name, 0.0))
                mean = statistics.fmean(values)
                deviation = statistics.pstdev(values)
                features[f"relative_delta_mean__{name}"] = value - mean
                features[f"relative_zscore__{name}"] = (
                    (value - mean) / deviation if deviation > 0 else 0.0
                )
                features[f"relative_percentile__{name}"] = sum(
                    candidate <= value for candidate in values
                ) / len(values)
                features[f"relative_is_min__{name}"] = value == min(values)
                features[f"relative_is_max__{name}"] = value == max(values)
            for name, value in tuple(row.runtime_features.items()):
                if not name.startswith("added_") or not isinstance(value, (bool, int, float)):
                    continue
                selected_name = f"selected_{name.removeprefix('added_')}"
                selected_value = row.runtime_features.get(selected_name)
                if isinstance(selected_value, (bool, int, float)):
                    features[f"added_minus_selected__{name.removeprefix('added_')}"] = float(
                        value
                    ) - float(selected_value)
            result[_row_key(row)] = features
    return result


def _prediction_metrics(
    predictions: Sequence[F1RepresentationPrediction],
) -> dict[str, Any]:
    labels = [int(row.row.f1_delta < -_F1_TOLERANCE) for row in predictions]
    risks = [row.risk_score for row in predictions]
    grouped: dict[str, list[F1RepresentationPrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.row.question_key].append(prediction)
    eligible_questions = 0
    safe_top1 = 0
    safe_top3 = 0
    top1_rows = []
    for question_rows in grouped.values():
        safe = [row for row in question_rows if row.row.f1_delta >= -_F1_TOLERANCE]
        if not safe:
            continue
        eligible_questions += 1
        ranked = sorted(
            question_rows, key=lambda row: (-row.safety_score, row.row.action.action_id)
        )
        safe_top1 += ranked[0].row.f1_delta >= -_F1_TOLERANCE
        safe_top3 += any(row.row.f1_delta >= -_F1_TOLERANCE for row in ranked[:3])
        top1_rows.append(ranked[0].row)
    return {
        "action_count": len(predictions),
        "f1_regression_count": sum(labels),
        "f1_regression_prevalence": _ratio(sum(labels), len(labels)),
        "roc_auc": round(float(roc_auc_score(labels, risks)), 6) if len(set(labels)) == 2 else None,
        "average_precision": round(float(average_precision_score(labels, risks)), 6)
        if sum(labels)
        else None,
        "questions_with_safe_action": eligible_questions,
        "safety_rank_top1_nonregression_rate": _ratio(safe_top1, eligible_questions),
        "safety_rank_top3_nonregression_rate": _ratio(safe_top3, eligible_questions),
        "safety_rank_top1_mean_f1_delta": _mean(row.f1_delta for row in top1_rows),
        "safety_rank_top1_gold_citation_delta": sum(row.citation_delta for row in top1_rows),
    }


def _selected_regression_headroom(
    *,
    predictions: Sequence[F1RepresentationPrediction],
    selected_regressions: Sequence[SelectedAction],
) -> dict[str, Any]:
    grouped: dict[str, list[F1RepresentationPrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.row.question_key].append(prediction)
    top_counts = {1: 0, 3: 0, 5: 0}
    best_safe_ranks = []
    for selected in selected_regressions:
        ranked = sorted(
            grouped[selected.row.question_key],
            key=lambda row: (-row.safety_score, row.row.action.action_id),
        )
        safe_ids = {
            row.row.action.action_id
            for row in ranked
            if row.row.f1_delta >= -_F1_TOLERANCE
            and row.row.citation_delta >= selected.row.citation_delta
        }
        safe_ranks = [
            index
            for index, row in enumerate(ranked, start=1)
            if row.row.action.action_id in safe_ids
        ]
        if not safe_ranks:
            continue
        best_rank = min(safe_ranks)
        best_safe_ranks.append(float(best_rank))
        for depth in top_counts:
            top_counts[depth] += best_rank <= depth
    count = len(selected_regressions)
    return {
        "selected_regression_count": count,
        "safe_alternative_top1_count": top_counts[1],
        "safe_alternative_top1_rate": _ratio(top_counts[1], count),
        "safe_alternative_top3_count": top_counts[3],
        "safe_alternative_top3_rate": _ratio(top_counts[3], count),
        "safe_alternative_top5_count": top_counts[5],
        "safe_alternative_top5_rate": _ratio(top_counts[5], count),
        "best_safe_rank": _distribution(best_safe_ranks),
    }


def _select_representation(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    raw_names = [name for name in reports if name.startswith("raw_")]
    candidate_names = [name for name in reports if not name.startswith("raw_")]
    best_raw = max(raw_names, key=lambda name: _representation_key(name, reports[name]))
    selected = max(
        candidate_names,
        key=lambda name: _representation_key(name, reports[name]),
    )
    raw_report = reports[best_raw]
    selected_report = reports[selected]
    raw_auc = float(raw_report["aggregate"]["roc_auc"])
    selected_auc = float(selected_report["aggregate"]["roc_auc"])
    fold_nonregression = sum(
        float(selected_report["folds"][fold_id]["roc_auc"])
        >= float(raw_report["folds"][fold_id]["roc_auc"])
        for fold_id in raw_report["folds"]
    )
    headroom = selected_report["stage182_regression_headroom"]
    gates = [
        _gate("risk_auc_at_least_0_62", selected_auc >= _MINIMUM_RISK_AUC),
        _gate(
            "risk_auc_gain_vs_best_raw_at_least_0_03",
            selected_auc - raw_auc >= _MINIMUM_AUC_GAIN_VS_RAW,
        ),
        _gate(
            "fold_auc_nonregression_at_least_4_of_5",
            fold_nonregression >= _MINIMUM_NONREGRESSING_FOLDS,
        ),
        _gate(
            "stage182_safe_alternative_top3_at_least_0_70",
            headroom["safe_alternative_top3_rate"] >= _MINIMUM_SAFE_TOP3_RATE,
        ),
        _gate(
            "stage182_safe_alternative_top5_at_least_0_85",
            headroom["safe_alternative_top5_rate"] >= _MINIMUM_SAFE_TOP5_RATE,
        ),
    ]
    return {
        "best_raw_reference": best_raw,
        "selected_candidate": selected,
        "best_raw_roc_auc": raw_auc,
        "selected_roc_auc": selected_auc,
        "roc_auc_gain_vs_best_raw": round(selected_auc - raw_auc, 6),
        "fold_auc_nonregression_count": fold_nonregression,
        "quality_gates": gates,
        "quality_gate_pass_count": sum(row["passed"] for row in gates),
        "candidate_accepted_for_nested_policy_experiment": all(row["passed"] for row in gates),
    }


def _representation_key(name: str, report: Mapping[str, Any]) -> tuple[Any, ...]:
    aggregate = report["aggregate"]
    headroom = report["stage182_regression_headroom"]
    return (
        float(aggregate["roc_auc"]),
        float(aggregate["average_precision"]),
        headroom["safe_alternative_top3_rate"],
        headroom["safe_alternative_top1_rate"],
        name,
    )


def _features_for(
    rows: Sequence[ActionAuditRow],
    feature_index: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [feature_index[_row_key(row)] for row in rows]


def _group_rows(rows: Sequence[ActionAuditRow]) -> dict[str, list[ActionAuditRow]]:
    grouped: dict[str, list[ActionAuditRow]] = defaultdict(list)
    for row in rows:
        grouped[row.question_key].append(row)
    for question_rows in grouped.values():
        question_rows.sort(key=lambda row: row.action.action_id)
    return grouped


def _row_key(row: ActionAuditRow) -> tuple[str, str]:
    return row.question_key, row.action.action_id


def _question_balanced_weights(rows: Sequence[ActionAuditRow]) -> list[float]:
    counts = Counter(row.question_key for row in rows)
    return [1.0 / counts[row.question_key] for row in rows]


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "median": None, "p95": None, "maximum": None, "mean": None}
    ordered = sorted(values)
    return {
        "minimum": round(ordered[0], 6),
        "median": round(float(statistics.median(ordered)), 6),
        "p95": round(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)], 6),
        "maximum": round(ordered[-1], 6),
        "mean": round(float(statistics.fmean(ordered)), 6),
    }


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
