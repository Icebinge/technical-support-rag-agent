from __future__ import annotations

import gc
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol

import numpy as np
from sklearn.feature_extraction import DictVectorizer

from ts_rag_agent.application import composition_safety_constrained_lambdamart as stage194
from ts_rag_agent.application import composition_safety_first_frontier as stage196
from ts_rag_agent.application.composition_action_audit import ActionAuditRow
from ts_rag_agent.application.composition_dual_target_policy import SelectedAction
from ts_rag_agent.application.composition_f1_representation_cv import (
    build_composition_feature_indices,
)
from ts_rag_agent.application.composition_gain_sensitive_ranking import (
    _binary_metrics,
    build_stage182_reference_rows,
    paired_selected_action_bootstrap,
    unavailable_selected_action_bootstrap,
)
from ts_rag_agent.application.composition_joint_constraint_ranking import (
    evaluate_selected_actions,
)
from ts_rag_agent.application.composition_safety_first_frontier import (
    FrontierActionPrediction,
    GainPrediction,
    SafetyFirstFrontierDecision,
    SafetyFirstFrontierPolicySpec,
    UnsafePrediction,
    fit_predict_safety_first_spec,
)

RiskSignal = Literal[
    "source_weighted_classifier",
    "decomposed_loss_risk",
    "pairwise_safety_ranker",
    "decomposed_pairwise_rank_fusion",
]
WinnerFamily = Literal["control", "rank_utility", "gain_shortlist_then_risk"]
ProgressSink = Callable[[Mapping[str, Any]], None]

_RISK_SIGNALS: tuple[RiskSignal, ...] = (
    "source_weighted_classifier",
    "decomposed_loss_risk",
    "pairwise_safety_ranker",
    "decomposed_pairwise_rank_fusion",
)
_RISK_PENALTIES = (0.25, 0.5, 1.0, 2.0)
_SHORTLIST_SIZES = (2, 4)
_POOL_CAP = 16


@dataclass(frozen=True)
class PairwiseSafetyPrediction:
    row: ActionAuditRow
    safety_score: float


@dataclass(frozen=True)
class JointPartitionResult:
    safety_predictions: tuple[stage194.SafetyPrediction, ...]
    gain_predictions: tuple[GainPrediction, ...]
    classifier_risk_predictions: tuple[UnsafePrediction, ...]
    pairwise_safety_predictions: tuple[PairwiseSafetyPrediction, ...]
    feature_count_by_representation: Mapping[str, int]
    model_fit_count: int
    tree_count: int
    group_contract_validation_count: int


@dataclass(frozen=True)
class WinnerRuleSpec:
    name: str
    family: WinnerFamily
    risk_penalty: float | None = None
    shortlist_size: int | None = None


@dataclass(frozen=True)
class JointRiskWinnerPolicySpec:
    name: str
    risk_signal: RiskSignal
    winner_rule: WinnerRuleSpec


@dataclass(frozen=True)
class JointRiskWinnerCandidateSnapshot:
    """Read-only private evidence for one policy cell in one outer context."""

    spec: Mapping[str, Any]
    eligible: bool
    evaluation: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    paired_vs_control: Mapping[str, Any]
    decisions: tuple[SafetyFirstFrontierDecision, ...]


@dataclass(frozen=True)
class JointRiskWinnerDiagnosticSnapshot:
    """Read-only Stage201 stream item; callers must aggregate and discard it."""

    outer_fold_id: str
    inner_fold_ids: tuple[str, ...]
    inner_question_count: int
    candidates: tuple[JointRiskWinnerCandidateSnapshot, ...]


DiagnosticSink = Callable[[JointRiskWinnerDiagnosticSnapshot], None]


class JointPartitionFitPredictor(Protocol):
    def __call__(
        self,
        training_rows: Sequence[ActionAuditRow],
        heldout_rows: Sequence[ActionAuditRow],
        feature_indices: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
        source_spec: SafetyFirstFrontierPolicySpec,
    ) -> JointPartitionResult: ...


@dataclass
class _FitTotals:
    model_fit_count: int = 0
    pool_safety_fit_count: int = 0
    gain_ranker_fit_count: int = 0
    classifier_risk_fit_count: int = 0
    pairwise_safety_fit_count: int = 0
    tree_count: int = 0
    group_contract_validation_count: int = 0

    def add(self, result: JointPartitionResult) -> None:
        self.model_fit_count += result.model_fit_count
        self.pool_safety_fit_count += 2
        self.gain_ranker_fit_count += 1
        self.classifier_risk_fit_count += 1
        self.pairwise_safety_fit_count += 1
        self.tree_count += result.tree_count
        self.group_contract_validation_count += result.group_contract_validation_count


def stage199_policy_specs() -> tuple[JointRiskWinnerPolicySpec, ...]:
    winner_rules = [WinnerRuleSpec("gain_only", "control")]
    winner_rules.extend(
        WinnerRuleSpec(f"rank_utility_{penalty:.2f}", "rank_utility", penalty)
        for penalty in _RISK_PENALTIES
    )
    winner_rules.extend(
        WinnerRuleSpec(
            f"gain_shortlist_{size}_then_risk",
            "gain_shortlist_then_risk",
            shortlist_size=size,
        )
        for size in _SHORTLIST_SIZES
    )
    return tuple(
        JointRiskWinnerPolicySpec(
            f"risk_{risk_signal}__winner_{winner_rule.name}",
            risk_signal,
            winner_rule,
        )
        for risk_signal in _RISK_SIGNALS
        for winner_rule in winner_rules
    )


def fit_predict_joint_partition(
    training_rows: Sequence[ActionAuditRow],
    heldout_rows: Sequence[ActionAuditRow],
    feature_indices: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
    source_spec: SafetyFirstFrontierPolicySpec,
) -> JointPartitionResult:
    """Fit the four source models and one question-grouped pairwise safety ranker."""

    import lightgbm as lgb

    source = fit_predict_safety_first_spec(
        training_rows,
        heldout_rows,
        feature_indices,
        source_spec,
    )
    training = tuple(sorted(training_rows, key=stage194._row_key))
    heldout = tuple(sorted(heldout_rows, key=stage194._row_key))
    representation = source_spec.risk_feature_representation
    vectorizer = DictVectorizer(sparse=True)
    train_matrix = vectorizer.fit_transform(
        [dict(feature_indices[representation][stage194._row_key(row)]) for row in training]
    ).tocsr()
    heldout_matrix = vectorizer.transform(
        [dict(feature_indices[representation][stage194._row_key(row)]) for row in heldout]
    ).tocsr()
    labels = np.asarray([not stage194._is_unsafe(row) for row in training], dtype=np.int8)
    if len(set(labels.tolist())) != 2:
        raise ValueError("Stage199 pairwise safety target requires both classes")
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        lambdarank_truncation_level=16,
        lambdarank_norm=True,
        label_gain=[0, 1],
        **{
            **stage196._lightgbm_common_parameters(),
            **stage196._TREE_PROFILES[source_spec.risk_tree_profile],
        },
    )
    ranker.fit(
        train_matrix,
        labels,
        group=stage194._question_group_sizes(training),
        sample_weight=stage194._question_balanced_weights(training),
        eval_at=[1, 4, 8, 16],
    )
    values = np.asarray(ranker.predict(heldout_matrix), dtype=np.float64)
    pairwise = tuple(
        PairwiseSafetyPrediction(row, float(score))
        for row, score in zip(heldout, values, strict=True)
    )
    feature_counts = dict(source.feature_count_by_representation)
    feature_counts[representation] = max(
        feature_counts.get(representation, 0), len(vectorizer.feature_names_)
    )
    tree_count = source.tree_count + int(ranker.booster_.num_trees())
    del ranker, values, train_matrix, heldout_matrix, vectorizer
    gc.collect()
    return JointPartitionResult(
        safety_predictions=source.safety_predictions,
        gain_predictions=source.gain_predictions,
        classifier_risk_predictions=source.risk_predictions,
        pairwise_safety_predictions=pairwise,
        feature_count_by_representation=dict(sorted(feature_counts.items())),
        model_fit_count=5,
        tree_count=tree_count,
        group_contract_validation_count=2,
    )


def build_joint_risk_winner_decisions(
    safety_predictions: Sequence[stage194.SafetyPrediction],
    gain_predictions: Sequence[GainPrediction],
    classifier_risk_predictions: Sequence[UnsafePrediction],
    pairwise_safety_predictions: Sequence[PairwiseSafetyPrediction],
    source_spec: SafetyFirstFrontierPolicySpec,
    policy_spec: JointRiskWinnerPolicySpec,
) -> tuple[SafetyFirstFrontierDecision, ...]:
    safety_grouped = stage194._group_predictions(safety_predictions)
    gain_index = {stage194._row_key(row.row): row.score for row in gain_predictions}
    classifier_index = {
        stage194._row_key(row.row): row.score for row in classifier_risk_predictions
    }
    pairwise_index = {
        stage194._row_key(row.row): row.safety_score for row in pairwise_safety_predictions
    }
    expected = {stage194._row_key(row.row) for row in safety_predictions}
    if expected != set(gain_index) or expected != set(classifier_index):
        raise ValueError("Stage199 source prediction rows differ")
    if expected != set(pairwise_index):
        raise ValueError("Stage199 pairwise prediction rows differ")

    decisions = []
    for question_key, question_safety in sorted(safety_grouped.items()):
        baselines = [row for row in question_safety if row.row.action.family == "baseline"]
        if len(baselines) != 1:
            raise ValueError("Stage199 requires one baseline action per question")
        ranked_safety = sorted(question_safety, key=stage194._safety_order_key)
        pool_index = {stage194._row_key(row.row): row for row in ranked_safety[:_POOL_CAP]}
        baseline_safety = baselines[0]
        pool_index[stage194._row_key(baseline_safety.row)] = baseline_safety
        pool_safety = tuple(sorted(pool_index.values(), key=stage194._safety_order_key))
        risk_values = _risk_values(
            pool_safety,
            classifier_index,
            pairwise_index,
            policy_spec.risk_signal,
        )
        risk_order = sorted(
            pool_safety,
            key=lambda row: (
                risk_values[stage194._row_key(row.row)],
                row.row.action.action_id,
            ),
        )
        frontier_index = {
            stage194._row_key(row.row): row for row in risk_order[: source_spec.safest_prefix_size]
        }
        frontier_index[stage194._row_key(baseline_safety.row)] = baseline_safety
        complete_pool = tuple(_combine(row, gain_index, risk_values) for row in pool_safety)
        frontier = tuple(
            _combine(row, gain_index, risk_values)
            for row in sorted(frontier_index.values(), key=stage194._safety_order_key)
        )
        winner = _select_winner(frontier, policy_spec.winner_rule)
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


def evaluate_joint_risk_winner_policy(
    safety_predictions: Sequence[stage194.SafetyPrediction],
    gain_predictions: Sequence[GainPrediction],
    classifier_risk_predictions: Sequence[UnsafePrediction],
    pairwise_safety_predictions: Sequence[PairwiseSafetyPrediction],
    source_spec: SafetyFirstFrontierPolicySpec,
    policy_spec: JointRiskWinnerPolicySpec,
    *,
    expected_fold_ids: Sequence[str],
) -> tuple[tuple[SafetyFirstFrontierDecision, ...], dict[str, Any]]:
    decisions = build_joint_risk_winner_decisions(
        safety_predictions,
        gain_predictions,
        classifier_risk_predictions,
        pairwise_safety_predictions,
        source_spec,
        policy_spec,
    )
    aggregate = stage196._FrontierDiagnostics()
    folds = {fold_id: stage196._FrontierDiagnostics() for fold_id in expected_fold_ids}
    for decision in decisions:
        fold_id = decision.winner.row.fold_id
        if fold_id not in folds:
            raise ValueError(f"Stage199 observed unexpected fold {fold_id}")
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


def run_joint_risk_winner_nested_cv(
    *,
    action_rows: Sequence[ActionAuditRow],
    stage182_selected_actions: Sequence[SelectedAction],
    stage198_protocol: Mapping[str, Any],
    stage197_report: Mapping[str, Any],
    progress_sink: ProgressSink | None = None,
    partition_fit_predictor: JointPartitionFitPredictor | None = None,
    diagnostic_sink: DiagnosticSink | None = None,
) -> dict[str, Any]:
    """Run the frozen Stage 199 five-by-four train-only nested CV."""

    started_at = time.perf_counter()
    rows = tuple(action_rows)
    fold_ids = tuple(sorted({row.fold_id for row in rows}))
    if len(fold_ids) != 5:
        raise ValueError("Stage199 requires exactly five frozen folds")
    grouped = stage194._group_rows(rows)
    references = build_stage182_reference_rows(rows, stage182_selected_actions)
    base_features = build_composition_feature_indices(rows)
    feature_indices = {
        "raw_runtime": base_features["raw"],
        "question_relative_runtime": base_features["question_relative"],
    }
    source_trajectories = {
        row["outer_context"]: stage196._spec_from_dict(row["source_spec"])
        for row in stage198_protocol["frozen_protocol"]["source_trajectory_contract"][
            "trajectories"
        ]
    }
    source_evidence = stage197_report["surviving_unsafe_winner_attribution"]["outer_contexts"]
    specs = stage199_policy_specs()
    fit_predict = partition_fit_predictor or fit_predict_joint_partition
    execution = _FitTotals()
    feature_counts: dict[str, int] = {}
    private_prediction_count = 0
    fit_seconds = 0.0
    outer_reports: dict[str, Any] = {}
    outer_rows: list[ActionAuditRow] = []
    outer_diagnostics = stage196._FrontierDiagnostics()
    cell_reports: dict[str, list[dict[str, Any]]] = defaultdict(list)
    risk_metric_labels: dict[str, list[int]] = defaultdict(list)
    risk_metric_scores: dict[str, list[float]] = defaultdict(list)
    selected_risk_signals: Counter[str] = Counter()
    selected_winner_rules: Counter[str] = Counter()
    control_reproduction_count = 0

    for outer_fold_id in fold_ids:
        source_spec = source_trajectories[outer_fold_id]
        outer_training = tuple(row for row in rows if row.fold_id != outer_fold_id)
        outer_heldout = tuple(row for row in rows if row.fold_id == outer_fold_id)
        inner_fold_ids = tuple(fold for fold in fold_ids if fold != outer_fold_id)
        safety_predictions: list[stage194.SafetyPrediction] = []
        gain_predictions: list[GainPrediction] = []
        classifier_predictions: list[UnsafePrediction] = []
        pairwise_predictions: list[PairwiseSafetyPrediction] = []
        for inner_fold_id in inner_fold_ids:
            training = tuple(row for row in outer_training if row.fold_id != inner_fold_id)
            heldout = tuple(row for row in outer_training if row.fold_id == inner_fold_id)
            fitted_at = time.perf_counter()
            result = fit_predict(training, heldout, feature_indices, source_spec)
            fit_seconds += time.perf_counter() - fitted_at
            execution.add(result)
            private_prediction_count += 5 * len(heldout)
            safety_predictions.extend(result.safety_predictions)
            gain_predictions.extend(result.gain_predictions)
            classifier_predictions.extend(result.classifier_risk_predictions)
            pairwise_predictions.extend(result.pairwise_safety_predictions)
            for name, count in result.feature_count_by_representation.items():
                feature_counts[name] = max(feature_counts.get(name, 0), count)
            _emit(
                progress_sink,
                phase="stage199_inner_partition_complete",
                outer_fold_id=outer_fold_id,
                inner_fold_id=inner_fold_id,
                cumulative_model_fit_count=execution.model_fit_count,
                cumulative_tree_count=execution.tree_count,
            )

        question_count = len({row.question_key for row in outer_training})
        candidates = []
        decisions_by_spec: dict[str, tuple[SafetyFirstFrontierDecision, ...]] = {}
        for policy_spec in specs:
            decisions, diagnostics = evaluate_joint_risk_winner_policy(
                safety_predictions,
                gain_predictions,
                classifier_predictions,
                pairwise_predictions,
                source_spec,
                policy_spec,
                expected_fold_ids=inner_fold_ids,
            )
            evaluation = evaluate_selected_actions(
                selected_rows=tuple(decision.winner.row for decision in decisions),
                references=references,
                expected_fold_ids=inner_fold_ids,
            )
            candidate = {
                "spec": _policy_spec_dict(policy_spec),
                "eligible": stage194._inner_eligible(evaluation, diagnostics, question_count),
                "evaluation": evaluation,
                "diagnostics": diagnostics,
            }
            candidates.append(candidate)
            decisions_by_spec[policy_spec.name] = decisions
            cell_reports[policy_spec.name].append(candidate)

        control = next(
            row
            for row in candidates
            if row["spec"]["risk_signal"] == "source_weighted_classifier"
            and row["spec"]["winner_rule"] == "gain_only"
        )
        source_formal = source_evidence[outer_fold_id]
        control_exact = _nested_close(
            control["evaluation"], source_formal["top_inner_evaluation"]
        ) and _nested_close(control["diagnostics"], source_formal["top_inner_diagnostics"])
        if not control_exact:
            raise ValueError(f"Stage199 control did not reproduce {outer_fold_id}")
        control_reproduction_count += 1
        for candidate in candidates:
            candidate["paired_vs_control"] = _paired_delta(candidate, control)

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

        for risk_signal in _RISK_SIGNALS:
            policy_name = f"risk_{risk_signal}__winner_gain_only"
            for decision in decisions_by_spec[policy_name]:
                for action in decision.complete_pool:
                    risk_metric_labels[risk_signal].append(int(stage194._is_unsafe(action.row)))
                    risk_metric_scores[risk_signal].append(action.unsafe_score)

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
                phase="stage199_outer_context_no_eligible_config",
                outer_fold_id=outer_fold_id,
            )
            continue

        selected = min(eligible, key=stage196._inner_selection_key)
        selected_spec = _policy_spec_from_dict(selected["spec"])
        selected_risk_signals[selected_spec.risk_signal] += 1
        selected_winner_rules[selected_spec.winner_rule.name] += 1
        fitted_at = time.perf_counter()
        heldout_result = fit_predict(
            outer_training,
            outer_heldout,
            feature_indices,
            source_spec,
        )
        fit_seconds += time.perf_counter() - fitted_at
        execution.add(heldout_result)
        private_prediction_count += 5 * len(outer_heldout)
        for name, count in heldout_result.feature_count_by_representation.items():
            feature_counts[name] = max(feature_counts.get(name, 0), count)
        decisions, diagnostics = evaluate_joint_risk_winner_policy(
            heldout_result.safety_predictions,
            heldout_result.gain_predictions,
            heldout_result.classifier_risk_predictions,
            heldout_result.pairwise_safety_predictions,
            source_spec,
            selected_spec,
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
            "selected_spec": selected["spec"],
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
            phase="stage199_outer_context_complete",
            outer_fold_id=outer_fold_id,
            selected_spec=selected_spec.name,
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
            "risk_signal_count": len(_RISK_SIGNALS),
            "winner_rule_count": 7,
            "policy_config_count_per_outer_context": len(specs),
            "model_fits_per_partition": 5,
            "maximum_model_fit_count": 125,
            "maximum_lightgbm_tree_count": 22_500,
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
        "risk_signal_factor_aggregates": _factor_aggregates(cell_aggregates, "risk_signal"),
        "winner_rule_factor_aggregates": _factor_aggregates(cell_aggregates, "winner_rule"),
        "complete_pool_risk_metrics": {
            signal: {
                "action_context_count": len(risk_metric_labels[signal]),
                **_binary_metrics(risk_metric_labels[signal], risk_metric_scores[signal]),
            }
            for signal in _RISK_SIGNALS
        },
        "selected_risk_signal_counts": dict(sorted(selected_risk_signals.items())),
        "selected_winner_rule_counts": dict(sorted(selected_winner_rules.items())),
        "advancement_gates": gates,
        "advancement_gate_pass_count": sum(row["passed"] for row in gates),
        "candidate_family_accepted": all(row["passed"] for row in gates),
        "execution": {
            "model_fit_count": execution.model_fit_count,
            "pool_safety_fit_count": execution.pool_safety_fit_count,
            "gain_ranker_fit_count": execution.gain_ranker_fit_count,
            "classifier_risk_fit_count": execution.classifier_risk_fit_count,
            "pairwise_safety_fit_count": execution.pairwise_safety_fit_count,
            "tree_count": execution.tree_count,
            "group_contract_validation_count": execution.group_contract_validation_count,
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


def _risk_values(
    pool_safety: Sequence[stage194.SafetyPrediction],
    classifier_index: Mapping[tuple[str, str], float],
    pairwise_index: Mapping[tuple[str, str], float],
    risk_signal: RiskSignal,
) -> dict[tuple[str, str], float]:
    decomposed = {
        stage194._row_key(row.row): max(row.citation_loss_probability, row.f1_loss_probability)
        for row in pool_safety
    }
    pairwise_risk = {
        stage194._row_key(row.row): -pairwise_index[stage194._row_key(row.row)]
        for row in pool_safety
    }
    if risk_signal == "source_weighted_classifier":
        return {
            stage194._row_key(row.row): classifier_index[stage194._row_key(row.row)]
            for row in pool_safety
        }
    if risk_signal == "decomposed_loss_risk":
        return decomposed
    if risk_signal == "pairwise_safety_ranker":
        return pairwise_risk
    decomposed_ranks = _normalized_ranks(pool_safety, decomposed)
    pairwise_ranks = _normalized_ranks(pool_safety, pairwise_risk)
    return {key: (decomposed_ranks[key] + pairwise_ranks[key]) / 2.0 for key in decomposed_ranks}


def _normalized_ranks(
    rows: Sequence[stage194.SafetyPrediction],
    values: Mapping[tuple[str, str], float],
) -> dict[tuple[str, str], float]:
    ordered = sorted(
        rows,
        key=lambda row: (
            values[stage194._row_key(row.row)],
            row.row.action.action_id,
        ),
    )
    denominator = max(1, len(ordered) - 1)
    return {stage194._row_key(row.row): index / denominator for index, row in enumerate(ordered)}


def _combine(
    safety: stage194.SafetyPrediction,
    gain_index: Mapping[tuple[str, str], float],
    risk_values: Mapping[tuple[str, str], float],
) -> FrontierActionPrediction:
    key = stage194._row_key(safety.row)
    return FrontierActionPrediction(
        safety.row,
        safety.citation_loss_probability,
        safety.f1_loss_probability,
        gain_index[key],
        risk_values[key],
    )


def _select_winner(
    frontier: Sequence[FrontierActionPrediction],
    rule: WinnerRuleSpec,
) -> FrontierActionPrediction:
    gain_order = sorted(
        frontier,
        key=lambda row: (-row.gain_score, row.unsafe_score, row.row.action.action_id),
    )
    if rule.family == "control":
        return gain_order[0]
    if rule.family == "gain_shortlist_then_risk":
        if rule.shortlist_size is None:
            raise ValueError("Stage199 shortlist rule requires a size")
        shortlist = gain_order[: rule.shortlist_size]
        return min(
            shortlist,
            key=lambda row: (row.unsafe_score, -row.gain_score, row.row.action.action_id),
        )
    if rule.risk_penalty is None:
        raise ValueError("Stage199 rank utility requires a risk penalty")
    risk_order = sorted(
        frontier,
        key=lambda row: (row.unsafe_score, -row.gain_score, row.row.action.action_id),
    )
    denominator = max(1, len(frontier) - 1)
    gain_rank = {
        stage194._row_key(row.row): index / denominator for index, row in enumerate(gain_order)
    }
    risk_rank = {
        stage194._row_key(row.row): index / denominator for index, row in enumerate(risk_order)
    }
    return min(
        frontier,
        key=lambda row: (
            gain_rank[stage194._row_key(row.row)]
            + rule.risk_penalty * risk_rank[stage194._row_key(row.row)],
            risk_rank[stage194._row_key(row.row)],
            gain_rank[stage194._row_key(row.row)],
            row.row.action.action_id,
        ),
    )


def _policy_spec_dict(spec: JointRiskWinnerPolicySpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "risk_signal": spec.risk_signal,
        "winner_rule": spec.winner_rule.name,
        "winner_family": spec.winner_rule.family,
        "risk_penalty": spec.winner_rule.risk_penalty,
        "shortlist_size": spec.winner_rule.shortlist_size,
    }


def _policy_spec_from_dict(value: Mapping[str, Any]) -> JointRiskWinnerPolicySpec:
    return JointRiskWinnerPolicySpec(
        value["name"],
        value["risk_signal"],
        WinnerRuleSpec(
            value["winner_rule"],
            value["winner_family"],
            value["risk_penalty"],
            value["shortlist_size"],
        ),
    )


def _public_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "spec": row["spec"],
        "eligible": row["eligible"],
        "evaluation": row["evaluation"],
        "diagnostics": row["diagnostics"],
        "paired_vs_control": row["paired_vs_control"],
    }


def _diagnostic_snapshot(
    *,
    outer_fold_id: str,
    inner_fold_ids: Sequence[str],
    inner_question_count: int,
    candidates: Sequence[Mapping[str, Any]],
    decisions_by_spec: Mapping[str, tuple[SafetyFirstFrontierDecision, ...]],
) -> JointRiskWinnerDiagnosticSnapshot:
    return JointRiskWinnerDiagnosticSnapshot(
        outer_fold_id=outer_fold_id,
        inner_fold_ids=tuple(inner_fold_ids),
        inner_question_count=inner_question_count,
        candidates=tuple(
            JointRiskWinnerCandidateSnapshot(
                spec=_deep_freeze(candidate["spec"]),
                eligible=bool(candidate["eligible"]),
                evaluation=_deep_freeze(candidate["evaluation"]),
                diagnostics=_deep_freeze(candidate["diagnostics"]),
                paired_vs_control=_deep_freeze(candidate["paired_vs_control"]),
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


def _paired_delta(candidate: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = candidate["evaluation"]
    control_evaluation = control["evaluation"]
    diagnostics = candidate["diagnostics"]
    control_diagnostics = control["diagnostics"]
    return {
        "strict_success_count_delta": (
            evaluation["strict_success_count"] - control_evaluation["strict_success_count"]
        ),
        "gold_citation_delta_delta": (
            evaluation["gold_citation_delta"] - control_evaluation["gold_citation_delta"]
        ),
        "mean_f1_delta_delta": round(
            evaluation["mean_f1_delta"] - control_evaluation["mean_f1_delta"], 6
        ),
        "unsafe_selected_count_delta": (
            diagnostics["unsafe_selected_question_count"]
            - control_diagnostics["unsafe_selected_question_count"]
        ),
        "unsafe_selection_rate_delta": round(
            diagnostics["unsafe_selection_rate"] - control_diagnostics["unsafe_selection_rate"],
            6,
        ),
        "conditional_capture_delta": round(
            diagnostics["conditional_ranker_strict_capture"]
            - control_diagnostics["conditional_ranker_strict_capture"],
            6,
        ),
    }


def _aggregate_cell_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    question_count = sum(row["evaluation"]["question_count"] for row in reports)
    strict_success = sum(row["evaluation"]["strict_success_count"] for row in reports)
    changed = sum(row["evaluation"]["changed_question_count"] for row in reports)
    strict_opportunity = sum(
        row["diagnostics"]["strict_opportunity_question_count"] for row in reports
    )
    pool_recalled = sum(row["diagnostics"]["pool_recalled_question_count"] for row in reports)
    frontier_recalled = sum(
        row["diagnostics"]["frontier_recalled_question_count"] for row in reports
    )
    unsafe = sum(row["diagnostics"]["unsafe_selected_question_count"] for row in reports)
    spec = reports[0]["spec"]
    return {
        "spec": spec,
        "outer_context_count": len(reports),
        "question_context_count": question_count,
        "eligible_outer_context_count": sum(row["eligible"] for row in reports),
        "strict_success_count": strict_success,
        "strict_success_precision": _ratio(strict_success, changed),
        "strict_opportunity_pool_recall": _ratio(pool_recalled, strict_opportunity),
        "strict_opportunity_frontier_recall": _ratio(frontier_recalled, strict_opportunity),
        "conditional_ranker_strict_capture": _ratio(strict_success, pool_recalled),
        "unsafe_selected_question_count": unsafe,
        "unsafe_selection_rate": _ratio(unsafe, question_count),
        "gold_citation_delta": sum(row["evaluation"]["gold_citation_delta"] for row in reports),
        "mean_f1_delta": round(
            sum(
                row["evaluation"]["mean_f1_delta"] * row["evaluation"]["question_count"]
                for row in reports
            )
            / question_count,
            6,
        ),
        "f1_regression_action_count": sum(
            row["evaluation"]["f1_regression_action_count"] for row in reports
        ),
        "citation_loss_action_count": sum(
            row["evaluation"]["citation_loss_action_count"] for row in reports
        ),
        "paired_vs_control": {
            name: round(sum(row["paired_vs_control"][name] for row in reports), 6)
            for name in reports[0]["paired_vs_control"]
        },
    }


def _factor_aggregates(cells: Mapping[str, Mapping[str, Any]], dimension: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells.values():
        grouped[str(cell["spec"][dimension])].append(cell)
    result = {}
    for value, rows in sorted(grouped.items()):
        result[value] = {
            "cell_count": len(rows),
            "mean_unsafe_selection_rate": _mean(row["unsafe_selection_rate"] for row in rows),
            "mean_conditional_capture": _mean(
                row["conditional_ranker_strict_capture"] for row in rows
            ),
            "mean_strict_success_precision": _mean(row["strict_success_precision"] for row in rows),
            "mean_gold_citation_delta": _mean(row["gold_citation_delta"] for row in rows),
            "mean_f1_delta": _mean(row["mean_f1_delta"] for row in rows),
            "mean_unsafe_count_delta_vs_control": _mean(
                row["paired_vs_control"]["unsafe_selected_count_delta"] for row in rows
            ),
            "best_cell": min(rows, key=_aggregate_selection_key)["spec"]["name"],
        }
    return result


def _aggregate_selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -row["strict_success_count"],
        -row["conditional_ranker_strict_capture"],
        -row["strict_success_precision"],
        row["unsafe_selected_question_count"],
        row["spec"]["name"],
    )


def _nested_close(actual: Any, expected: Any) -> bool:
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(
            _nested_close(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, Sequence) and not isinstance(actual, (str, bytes)):
        return (
            isinstance(expected, Sequence)
            and not isinstance(expected, (str, bytes))
            and len(actual) == len(expected)
            and all(
                _nested_close(left, right) for left, right in zip(actual, expected, strict=True)
            )
        )
    if isinstance(actual, float) or isinstance(expected, float):
        return abs(float(actual) - float(expected)) <= 1e-6
    return actual == expected


def _mean(values: Sequence[float] | Any) -> float:
    rows = tuple(values)
    return round(float(sum(rows) / len(rows)), 6) if rows else 0.0


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator / denominator), 6) if denominator else 0.0


def _emit(progress_sink: ProgressSink | None, **event: Any) -> None:
    if progress_sink is not None:
        progress_sink(event)
