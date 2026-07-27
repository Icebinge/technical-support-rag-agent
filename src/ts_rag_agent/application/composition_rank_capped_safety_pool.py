from __future__ import annotations

import gc
import math
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ts_rag_agent.application.composition_action_audit import ActionAuditRow
from ts_rag_agent.application.composition_dual_target_policy import SelectedAction
from ts_rag_agent.application.composition_f1_representation_cv import (
    build_composition_feature_indices,
)
from ts_rag_agent.application.composition_gain_sensitive_ranking import (
    FittedGainSensitiveRepresentation,
    GainRankerKind,
    GainSensitivePrediction,
    ProgressSink,
    RepresentationFitter,
    SafetyEstimator,
    build_stage182_reference_rows,
    fit_gain_sensitive_representations,
    gain_sensitive_prediction_metrics,
    paired_selected_action_bootstrap,
    unavailable_selected_action_bootstrap,
)
from ts_rag_agent.application.composition_joint_constraint_ranking import (
    evaluate_selected_actions,
)

FeatureRepresentation = Literal["raw_runtime", "question_relative_runtime"]
PoolCap = Literal[4, 8, 16, "all"]

_F1_TOLERANCE = 1e-12
_INNER_STRICT_PRECISION = 0.60
_INNER_CHANGED_RATE = 0.10
_INNER_STRICT_COUNT_RATE = 0.08
_MINIMUM_INNER_NONREGRESSING_FOLDS = 3
_INNER_POOL_RECALL = 0.80
_INNER_FOLD_POOL_RECALL = 0.70
_MINIMUM_INNER_POOL_RECALL_FOLDS = 3


@dataclass(frozen=True)
class RankCappedSafetyPoolPolicySpec:
    """One frozen Stage 191 rank-capped safety-pool policy."""

    name: str
    feature_representation: FeatureRepresentation
    safety_estimator: SafetyEstimator
    gain_ranker: GainRankerKind
    pool_cap: PoolCap

    @property
    def bundle_name(self) -> str:
        return f"{self.feature_representation}__{self.safety_estimator}__{self.gain_ranker}"


@dataclass(frozen=True)
class RankCappedSafetyPoolDecision:
    """One question's deterministic safety pool and selected action."""

    question_key: str
    baseline: GainSensitivePrediction
    pool: tuple[GainSensitivePrediction, ...]
    winner: GainSensitivePrediction


@dataclass
class _FitExecutionTotals:
    model_fit_count: int = 0
    comparable_pair_count: int = 0
    omitted_pair_count: int = 0
    listwise_question_fit_count: int = 0
    listnet_iteration_count: int = 0

    def add(
        self,
        representations: Mapping[str, FittedGainSensitiveRepresentation],
    ) -> None:
        for representation in representations.values():
            self.model_fit_count += representation.model_fit_count
            pairwise = representation.diagnostics["pairwise"]
            listnet = representation.diagnostics["listnet"]
            self.comparable_pair_count += int(pairwise["comparable_pair_count"])
            self.omitted_pair_count += int(pairwise["omitted_incomparable_pair_count"])
            self.listwise_question_fit_count += int(listnet["question_count"])
            self.listnet_iteration_count += int(listnet["completed_iterations"])


@dataclass
class _PoolRecallTotals:
    question_count: int = 0
    strict_opportunity_question_count: int = 0
    recalled_strict_opportunity_question_count: int = 0
    action_count: int = 0
    strict_action_count: int = 0
    pool_action_count: int = 0
    retained_strict_action_count: int = 0
    baseline_in_pool_question_count: int = 0

    def add(self, metrics: Mapping[str, Any]) -> None:
        for name in (
            "question_count",
            "strict_opportunity_question_count",
            "recalled_strict_opportunity_question_count",
            "action_count",
            "strict_action_count",
            "pool_action_count",
            "retained_strict_action_count",
            "baseline_in_pool_question_count",
        ):
            setattr(self, name, getattr(self, name) + int(metrics[name]))

    def report(self) -> dict[str, Any]:
        return {
            "question_count": self.question_count,
            "strict_opportunity_question_count": self.strict_opportunity_question_count,
            "recalled_strict_opportunity_question_count": (
                self.recalled_strict_opportunity_question_count
            ),
            "strict_opportunity_pool_recall": _ratio(
                self.recalled_strict_opportunity_question_count,
                self.strict_opportunity_question_count,
            ),
            "action_count": self.action_count,
            "strict_action_count": self.strict_action_count,
            "pool_action_count": self.pool_action_count,
            "retained_strict_action_count": self.retained_strict_action_count,
            "strict_action_retention_rate": _ratio(
                self.retained_strict_action_count,
                self.strict_action_count,
            ),
            "mean_pool_size": _ratio(self.pool_action_count, self.question_count),
            "baseline_in_pool_question_count": self.baseline_in_pool_question_count,
            "baseline_in_pool_rate": _ratio(
                self.baseline_in_pool_question_count,
                self.question_count,
            ),
        }


def stage191_policy_specs() -> tuple[RankCappedSafetyPoolPolicySpec, ...]:
    """Return the complete frozen 32-policy Stage 191 grid."""

    specs = []
    for representation in ("raw_runtime", "question_relative_runtime"):
        for estimator in ("class_balanced_logistic", "histogram_gradient_boosting"):
            for ranker in ("pairwise_pareto_logistic", "linear_listnet_top_frontier"):
                for pool_cap in (4, 8, 16, "all"):
                    specs.append(
                        RankCappedSafetyPoolPolicySpec(
                            name=(f"{representation}__{estimator}__{ranker}__pool_{pool_cap}"),
                            feature_representation=representation,
                            safety_estimator=estimator,
                            gain_ranker=ranker,
                            pool_cap=pool_cap,
                        )
                    )
    return tuple(specs)


def build_rank_capped_safety_pool_decisions(
    predictions: Sequence[GainSensitivePrediction],
    spec: RankCappedSafetyPoolPolicySpec,
) -> tuple[RankCappedSafetyPoolDecision, ...]:
    """Build the frozen safety-ranked pool and select one action per question."""

    grouped: dict[str, list[GainSensitivePrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.row.question_key].append(prediction)
    decisions = []
    for question_key, question_predictions in sorted(grouped.items()):
        baselines = [row for row in question_predictions if row.row.action.family == "baseline"]
        if len(baselines) != 1:
            raise ValueError("Stage191 requires one original baseline action per question")
        safety_ranked = sorted(question_predictions, key=_safety_order_key)
        capped = safety_ranked if spec.pool_cap == "all" else safety_ranked[: spec.pool_cap]
        pool_by_action = {row.row.action.action_id: row for row in capped}
        baseline = baselines[0]
        pool_by_action[baseline.row.action.action_id] = baseline
        pool = tuple(sorted(pool_by_action.values(), key=_safety_order_key))
        winner = min(
            pool,
            key=lambda row: (
                -row.gain_score,
                _joint_safety_risk(row),
                row.row.action.action_id,
            ),
        )
        decisions.append(
            RankCappedSafetyPoolDecision(
                question_key=question_key,
                baseline=baseline,
                pool=pool,
                winner=winner,
            )
        )
    return tuple(decisions)


def select_rank_capped_safety_pool_actions(
    predictions: Sequence[GainSensitivePrediction],
    spec: RankCappedSafetyPoolPolicySpec,
) -> tuple[ActionAuditRow, ...]:
    """Select one Stage 191 action per question."""

    return tuple(
        decision.winner.row
        for decision in build_rank_capped_safety_pool_decisions(predictions, spec)
    )


def evaluate_rank_capped_safety_pool(
    predictions: Sequence[GainSensitivePrediction],
    spec: RankCappedSafetyPoolPolicySpec,
    *,
    expected_fold_ids: Sequence[str],
) -> dict[str, Any]:
    """Measure question-level strict-opportunity recall for one private pool."""

    decisions = build_rank_capped_safety_pool_decisions(predictions, spec)
    predictions_by_question: dict[str, list[GainSensitivePrediction]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_question[prediction.row.question_key].append(prediction)
    fold_totals = {fold_id: _PoolRecallTotals() for fold_id in expected_fold_ids}
    aggregate = _PoolRecallTotals()
    for decision in decisions:
        fold_id = decision.winner.row.fold_id
        if fold_id not in fold_totals:
            raise ValueError(f"Stage191 observed unexpected fold {fold_id}")
        metrics = _question_pool_counts(
            decision,
            predictions_by_question[decision.question_key],
        )
        aggregate.add(metrics)
        fold_totals[fold_id].add(metrics)
    folds = {fold_id: totals.report() for fold_id, totals in fold_totals.items()}
    report = aggregate.report()
    report["folds"] = folds
    report["folds_meeting_recall_minimum"] = sum(
        row["strict_opportunity_pool_recall"] >= _INNER_FOLD_POOL_RECALL for row in folds.values()
    )
    return report


def run_rank_capped_safety_pool_nested_cv(
    *,
    action_rows: Sequence[ActionAuditRow],
    stage182_selected_actions: Sequence[SelectedAction],
    progress_sink: ProgressSink | None = None,
    representation_fitter: RepresentationFitter | None = None,
) -> dict[str, Any]:
    """Run the frozen Stage 191 five-by-four train-only nested CV."""

    started_at = time.perf_counter()
    rows = tuple(action_rows)
    if not rows:
        raise ValueError("Stage191 requires action rows")
    fold_ids = tuple(sorted({row.fold_id for row in rows}))
    if len(fold_ids) != 5:
        raise ValueError("Stage191 requires exactly five frozen folds")
    grouped = _group_rows(rows)
    if any(
        len([row for row in question_rows if row.action.family == "baseline"]) != 1
        for question_rows in grouped.values()
    ):
        raise ValueError("Stage191 requires one original baseline action per question")

    references = build_stage182_reference_rows(rows, stage182_selected_actions)
    reference_regressions = [row for row in references.values() if row.f1_delta < -_F1_TOLERANCE]
    base_features = build_composition_feature_indices(rows)
    feature_indices = {
        "raw_runtime": base_features["raw"],
        "question_relative_runtime": base_features["question_relative"],
    }
    specs = stage191_policy_specs()
    fit_representations = representation_fitter or fit_gain_sensitive_representations
    execution = _FitExecutionTotals()
    private_prediction_count = 0
    outer_rows: list[ActionAuditRow] = []
    outer_predictions: list[GainSensitivePrediction] = []
    outer_pool_totals = _PoolRecallTotals()
    outer_reports: dict[str, dict[str, Any]] = {}
    selected_spec_counts: Counter[str] = Counter()
    selected_pool_cap_counts: Counter[str] = Counter()
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
            execution.add(representations)
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
                cumulative_model_fit_count=execution.model_fit_count,
            )

        candidate_reports = []
        inner_question_count = len({row.question_key for row in outer_training})
        for spec in specs:
            predictions = inner_predictions[spec.bundle_name]
            selected_rows = select_rank_capped_safety_pool_actions(predictions, spec)
            evaluation = evaluate_selected_actions(
                selected_rows=selected_rows,
                references=references,
                expected_fold_ids=inner_fold_ids,
            )
            pool_metrics = evaluate_rank_capped_safety_pool(
                predictions,
                spec,
                expected_fold_ids=inner_fold_ids,
            )
            candidate_reports.append(
                {
                    "spec": _spec_dict(spec),
                    "eligible": _inner_eligible(
                        evaluation,
                        pool_metrics,
                        inner_question_count,
                    ),
                    "evaluation": evaluation,
                    "pool_metrics": pool_metrics,
                }
            )

        eligible_reports = [row for row in candidate_reports if row["eligible"]]
        ranked_reports = sorted(candidate_reports, key=_inner_selection_key)
        public_top_candidates = [_public_candidate(row) for row in ranked_reports[:5]]
        if not eligible_reports:
            outer_reports[outer_fold_id] = {
                "inner_question_count": inner_question_count,
                "eligible_config_count": 0,
                "selected_spec": None,
                "selected_inner_evaluation": None,
                "selected_inner_pool_metrics": None,
                "outer_evaluation": None,
                "outer_pool_metrics": None,
                "top_inner_candidates": public_top_candidates,
                "outer_evaluated": False,
            }
            _emit(
                progress_sink,
                phase="outer_fold_no_eligible_config",
                outer_fold_id=outer_fold_id,
                cumulative_model_fit_count=execution.model_fit_count,
            )
            continue

        selected_report = min(eligible_reports, key=_inner_selection_key)
        selected_spec = _spec_from_dict(selected_report["spec"])
        selected_spec_counts[selected_spec.name] += 1
        selected_pool_cap_counts[str(selected_spec.pool_cap)] += 1
        selected_ranker_counts[selected_spec.gain_ranker] += 1

        fitted_at = time.perf_counter()
        outer_representations = fit_representations(outer_training, feature_indices)
        fit_seconds += time.perf_counter() - fitted_at
        execution.add(outer_representations)
        for representation in outer_representations.values():
            feature_counts[representation.feature_representation] = max(
                feature_counts.get(representation.feature_representation, 0),
                representation.feature_count,
            )
        representation = outer_representations[selected_spec.feature_representation]
        heldout_predictions = representation.predict(
            outer_heldout,
            feature_indices[selected_spec.feature_representation],
        )[selected_spec.bundle_name]
        private_prediction_count += len(heldout_predictions)
        selected_rows = select_rank_capped_safety_pool_actions(
            heldout_predictions,
            selected_spec,
        )
        outer_evaluation = evaluate_selected_actions(
            selected_rows=selected_rows,
            references=references,
            expected_fold_ids=(outer_fold_id,),
        )
        outer_pool_metrics = evaluate_rank_capped_safety_pool(
            heldout_predictions,
            selected_spec,
            expected_fold_ids=(outer_fold_id,),
        )
        outer_pool_totals.add(outer_pool_metrics)
        outer_rows.extend(selected_rows)
        outer_predictions.extend(heldout_predictions)
        outer_reports[outer_fold_id] = {
            "inner_question_count": inner_question_count,
            "eligible_config_count": len(eligible_reports),
            "selected_spec": _spec_dict(selected_spec),
            "selected_inner_evaluation": selected_report["evaluation"],
            "selected_inner_pool_metrics": selected_report["pool_metrics"],
            "outer_evaluation": outer_evaluation,
            "outer_pool_metrics": outer_pool_metrics,
            "top_inner_candidates": public_top_candidates,
            "outer_evaluated": True,
        }
        _emit(
            progress_sink,
            phase="outer_fold_complete",
            outer_fold_id=outer_fold_id,
            selected_spec=selected_spec.name,
            eligible_config_count=len(eligible_reports),
            cumulative_model_fit_count=execution.model_fit_count,
        )
        del outer_representations
        gc.collect()

    eligible_outer_fold_count = sum(row["outer_evaluated"] for row in outer_reports.values())
    aggregate = evaluate_selected_actions(
        selected_rows=outer_rows,
        references=references,
        expected_fold_ids=fold_ids,
    )
    aggregate_pool_metrics = outer_pool_totals.report()
    bootstrap = (
        paired_selected_action_bootstrap(outer_rows)
        if eligible_outer_fold_count == len(fold_ids)
        else unavailable_selected_action_bootstrap()
    )
    gates = _advancement_gates(
        eligible_outer_fold_count=eligible_outer_fold_count,
        aggregate=aggregate,
        aggregate_pool_metrics=aggregate_pool_metrics,
        bootstrap=bootstrap,
    )
    return {
        "protocol": {
            "outer_fold_count": 5,
            "inner_fold_count": 4,
            "policy_config_count": len(specs),
            "pool_caps": [4, 8, 16, "all"],
            "model_fits_per_partition": 12,
            "maximum_model_fit_count": 300,
            "inner_aggregate_pool_recall_minimum": _INNER_POOL_RECALL,
            "inner_per_fold_pool_recall_minimum": _INNER_FOLD_POOL_RECALL,
            "inner_folds_meeting_pool_recall_minimum": (_MINIMUM_INNER_POOL_RECALL_FOLDS),
            "pair_sampling": False,
            "list_sampling": False,
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
        "aggregate_pool_metrics": aggregate_pool_metrics,
        "paired_bootstrap": bootstrap,
        "prediction_metrics": gain_sensitive_prediction_metrics(outer_predictions),
        "selected_spec_counts": dict(sorted(selected_spec_counts.items())),
        "selected_pool_cap_counts": dict(sorted(selected_pool_cap_counts.items())),
        "selected_ranker_counts": dict(sorted(selected_ranker_counts.items())),
        "advancement_gates": gates,
        "advancement_gate_pass_count": sum(row["passed"] for row in gates),
        "candidate_family_accepted": all(row["passed"] for row in gates),
        "execution": {
            "model_fit_count": execution.model_fit_count,
            "maximum_model_fit_count": 300,
            "comparable_pair_count_across_fits": execution.comparable_pair_count,
            "omitted_incomparable_pair_count_across_fits": execution.omitted_pair_count,
            "listwise_question_fit_count": execution.listwise_question_fit_count,
            "listnet_iteration_count": execution.listnet_iteration_count,
            "private_prediction_count": private_prediction_count,
            "public_pair_rows_written": 0,
            "public_listwise_targets_written": 0,
            "public_prediction_rows_written": 0,
            "feature_count_by_representation": dict(sorted(feature_counts.items())),
            "fit_seconds": round(fit_seconds, 6),
            "wall_seconds": round(time.perf_counter() - started_at, 6),
        },
    }


def _question_pool_counts(
    decision: RankCappedSafetyPoolDecision,
    predictions: Sequence[GainSensitivePrediction],
) -> dict[str, int]:
    strict_actions = [row for row in predictions if row.row.strict_expected]
    retained_strict = [row for row in decision.pool if row.row.strict_expected]
    return {
        "question_count": 1,
        "strict_opportunity_question_count": int(bool(strict_actions)),
        "recalled_strict_opportunity_question_count": int(bool(retained_strict)),
        "action_count": len(predictions),
        "strict_action_count": len(strict_actions),
        "pool_action_count": len(decision.pool),
        "retained_strict_action_count": len(retained_strict),
        "baseline_in_pool_question_count": int(decision.baseline in decision.pool),
    }


def _inner_eligible(
    evaluation: Mapping[str, Any],
    pool_metrics: Mapping[str, Any],
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
        and pool_metrics["strict_opportunity_pool_recall"] >= _INNER_POOL_RECALL
        and pool_metrics["folds_meeting_recall_minimum"] >= _MINIMUM_INNER_POOL_RECALL_FOLDS
    )


def _inner_selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    evaluation = row["evaluation"]
    pool_metrics = row["pool_metrics"]
    return (
        -evaluation["strict_success_count"],
        -evaluation["strict_success_precision"],
        -pool_metrics["strict_opportunity_pool_recall"],
        evaluation["f1_regression_action_count"],
        evaluation["citation_loss_action_count"],
        -evaluation["gold_citation_delta"],
        -evaluation["mean_f1_delta"],
        -evaluation["repaired_reference_regression_count"],
        row["spec"]["name"],
    )


def _advancement_gates(
    *,
    eligible_outer_fold_count: int,
    aggregate: Mapping[str, Any],
    aggregate_pool_metrics: Mapping[str, Any],
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
        _gate("changed_question_count_at_least_37", aggregate["changed_question_count"] >= 37),
        _gate(
            "strict_opportunity_pool_recall_at_least_0_80",
            aggregate_pool_metrics["strict_opportunity_pool_recall"] >= 0.80,
        ),
    ]


def _safety_order_key(row: GainSensitivePrediction) -> tuple[float, float, str]:
    return (
        _joint_safety_risk(row),
        row.citation_loss_probability + row.f1_loss_probability,
        row.row.action.action_id,
    )


def _joint_safety_risk(row: GainSensitivePrediction) -> float:
    return max(row.citation_loss_probability, row.f1_loss_probability)


def _spec_dict(spec: RankCappedSafetyPoolPolicySpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "feature_representation": spec.feature_representation,
        "safety_estimator": spec.safety_estimator,
        "gain_ranker": spec.gain_ranker,
        "pool_cap": spec.pool_cap,
    }


def _spec_from_dict(value: Mapping[str, Any]) -> RankCappedSafetyPoolPolicySpec:
    pool_cap = value["pool_cap"]
    if pool_cap not in (4, 8, 16, "all"):
        raise ValueError(f"unsupported Stage191 pool cap: {pool_cap}")
    return RankCappedSafetyPoolPolicySpec(
        name=value["name"],
        feature_representation=value["feature_representation"],
        safety_estimator=value["safety_estimator"],
        gain_ranker=value["gain_ranker"],
        pool_cap=pool_cap,
    )


def _public_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "spec": row["spec"],
        "eligible": row["eligible"],
        "evaluation": row["evaluation"],
        "pool_metrics": row["pool_metrics"],
    }


def _group_rows(rows: Sequence[ActionAuditRow]) -> dict[str, list[ActionAuditRow]]:
    grouped: dict[str, list[ActionAuditRow]] = defaultdict(list)
    for row in rows:
        grouped[row.question_key].append(row)
    return grouped


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator / denominator), 6) if denominator else 0.0


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _emit(progress_sink: ProgressSink | None, **event: Any) -> None:
    if progress_sink is not None:
        progress_sink(event)
