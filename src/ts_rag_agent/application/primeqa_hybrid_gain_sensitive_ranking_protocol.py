from __future__ import annotations

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

STAGE = "Stage 187"
CREATED_AT = "2026-07-23"
PROTOCOL_ID = "primeqa_hybrid_gain_sensitive_ranking_protocol_v1"
NEXT_STAGE = "Stage 188"
EXPECTED_SOURCE_SHA256 = {
    "stage181_report": "a9c557d7346eb2b4958cddd2505937eba828556c7671d7e936bf883d80cfe88b",
    "stage182_report": "c3dfbe7484a604b8491bed0531fc82b20bd092016fd7ddf303955b7c7c89044a",
    "stage183_report": "4dd611c9a759fd791288886c638bd9ec36b7564328eb53f2fef1742544540f1a",
    "stage184_report": "bdbb49bf31a0f889a431924ee1630c7593ec485fb1bf283def44048776a29eea",
    "stage185_report": "742ea385e76faa950677941d760f321c834cb23cbfb054458a66ed17807b837e",
    "stage186_report": "a3aee4190aca1f71f2cd3c611675a8b69090e41eee00fdae0515bce55edf02f4",
}
EXPECTED_SOURCE_STATUS = {
    "stage181_report": "stage181_counterfactual_action_audit_complete",
    "stage182_report": "stage182_dual_target_nested_cv_insufficient",
    "stage183_report": "stage183_f1_risk_failure_attribution_complete",
    "stage184_report": "stage184_f1_representation_cv_insufficient",
    "stage185_report": "stage185_joint_constraint_ranking_protocol_frozen",
    "stage186_report": "stage186_joint_constraint_ranking_insufficient",
}
FEATURE_REPRESENTATIONS = ("raw_runtime", "question_relative_runtime")
SAFETY_ESTIMATORS = ("class_balanced_logistic", "histogram_gradient_boosting")
SAFETY_TARGETS = ("citation_loss", "f1_loss")
GAIN_RANKERS = ("pairwise_pareto_logistic", "linear_listnet_top_frontier")
SAFETY_FRONTIER_MARGINS = (0.0, 0.02, 0.05, 0.10)
OUTER_FOLD_COUNT = 5
INNER_FOLD_COUNT = 4
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 187
STAGE182_CITATION_DELTA = 5
STAGE182_MEAN_F1_DELTA = 0.005249
STAGE182_F1_REGRESSION_COUNT = 55
FORBIDDEN_RUNTIME_FEATURES = (
    "answer_doc_id",
    "citation_delta",
    "f1_delta",
    "gold_answer",
    "gold_document_id",
    "outcome_class",
    "question_id",
    "split_membership",
    "strict_expected",
)
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "action_id",
        "answer",
        "answer_doc_id",
        "candidate_actions",
        "citation_delta",
        "document_id",
        "document_text",
        "f1_delta",
        "feature_rows",
        "gold_answer",
        "gold_document_id",
        "outcome_class",
        "predictions",
        "question_id",
        "question_text",
        "strict_expected",
    }
)


@dataclass(frozen=True)
class Stage187Visualization:
    """One public-safe Stage 187 protocol visualization."""

    name: str
    path: str


def freeze_gain_sensitive_ranking_protocol(
    *,
    stage181_report_path: Path,
    stage182_report_path: Path,
    stage183_report_path: Path,
    stage184_report_path: Path,
    stage185_report_path: Path,
    stage186_report_path: Path,
    user_confirmed: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the Stage 188 train-only gain-sensitive ranking protocol."""

    started_at = time.perf_counter()
    source_paths = {
        "stage181_report": stage181_report_path,
        "stage182_report": stage182_report_path,
        "stage183_report": stage183_report_path,
        "stage184_report": stage184_report_path,
        "stage185_report": stage185_report_path,
        "stage186_report": stage186_report_path,
    }
    source_reports = {name: _load_json_object(path) for name, path in source_paths.items()}
    source_files = {name: _fingerprint(path) for name, path in source_paths.items()}
    loaded_at = time.perf_counter()

    evidence = _evidence_summary(source_reports)
    protocol = _frozen_protocol()
    preliminary = {
        "stage": STAGE,
        "created_at": CREATED_AT,
        "protocol_id": PROTOCOL_ID,
        "protocol_scope": (
            "Train-only protocol freeze for within-question pairwise and listwise "
            "gain-sensitive action ranking under separately predicted citation/F1 "
            "constraints. This stage reads only aggregate public-safe Stage 181-186 "
            "reports, loads no split rows or documents, fits no model, evaluates no "
            "policy, keeps development and test closed, adds no fallback, and changes "
            "no runtime default."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed),
            "confirmation_note": confirmation_note,
        },
        "source_files": source_files,
        "evidence_summary": evidence,
        "frozen_protocol": protocol,
        "execution_boundaries": {
            "loaded_public_reports_only": True,
            "train_rows_loaded": False,
            "development_loaded": False,
            "test_loaded": False,
            "model_fit_count": 0,
            "pair_rows_materialized": 0,
            "listwise_questions_materialized": 0,
            "policy_evaluation_run": False,
            "replacement_policy_selected": False,
            "runtime_e2e_run": False,
            "runtime_registered_as_default": False,
            "stage178b_run": False,
            "retry_action_count": 0,
            "fallback_action_count": 0,
        },
    }
    guard_checks = _guard_checks(
        preliminary,
        source_reports=source_reports,
        source_files=source_files,
    )
    checked_at = time.perf_counter()
    decision = _decision(guard_checks)
    report = {
        **preliminary,
        "guard_checks": guard_checks,
        "decision": decision,
        "timing_seconds": {
            "load_public_reports": round(loaded_at - started_at, 6),
            "freeze_and_guard": round(checked_at - loaded_at, 6),
            "total": round(checked_at - started_at, 6),
        },
    }
    forbidden = sorted(_forbidden_keys_found(report))
    return {
        **report,
        "public_safe_contract": {
            "forbidden_public_keys": sorted(FORBIDDEN_PUBLIC_KEYS),
            "forbidden_keys_found": forbidden,
            "private_action_rows_persisted": False,
            "private_pair_rows_persisted": False,
            "private_listwise_targets_persisted": False,
            "private_predictions_persisted": False,
            "public_report_safe": not forbidden,
        },
    }


def write_stage187_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage187Visualization, ...]:
    """Write and XML-validate aggregate Stage 187 protocol charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = report["evidence_summary"]
    protocol = report["frozen_protocol"]
    stage186 = evidence["stage186"]
    charts = {
        "stage187_strict_gain_opportunity.svg": _chart(
            "Stage 187 strict-gain opportunity",
            (
                BarDatum(
                    "questions with a strict action",
                    evidence["stage181"]["questions_with_strict_action"],
                    str(evidence["stage181"]["questions_with_strict_action"]),
                ),
                BarDatum(
                    "Stage 182 selected strict actions",
                    evidence["stage182"]["strict_success_count"],
                    str(evidence["stage182"]["strict_success_count"]),
                ),
                BarDatum(
                    "Stage 186 selected strict actions",
                    stage186["strict_success_count"],
                    str(stage186["strict_success_count"]),
                ),
            ),
            "question or selected-action count",
        ),
        "stage187_stage186_outcomes.svg": _chart(
            "Stage 186 conservative-collapse outcomes",
            tuple(
                BarDatum(name, value, str(value))
                for name, value in (
                    ("changed questions", stage186["changed_question_count"]),
                    ("repaired regressions", stage186["repaired_reference_regression_count"]),
                    ("strict-success actions", stage186["strict_success_count"]),
                    ("citation-loss actions", stage186["citation_loss_action_count"]),
                    ("F1-regression actions", stage186["f1_regression_action_count"]),
                )
            ),
            "question or selected-action count",
        ),
        "stage187_stage186_head_auc.svg": _chart(
            "Stage 186 held-out head ROC AUC",
            tuple(
                BarDatum(
                    target,
                    stage186["head_metrics"][target]["roc_auc"],
                    f"{stage186['head_metrics'][target]['roc_auc']:.3f}",
                )
                for target in ("citation_loss", "f1_loss", "strict_gain")
            ),
            "ROC AUC",
        ),
        "stage187_stage186_gates.svg": _chart(
            "Stage 186 advancement gates",
            tuple(
                BarDatum(row["name"], float(row["passed"]), "pass" if row["passed"] else "fail")
                for row in stage186["advancement_gates"]
            ),
            "1 means passed",
            margin_left=900,
        ),
        "stage187_candidate_grid.svg": _chart(
            "Stage 188 frozen candidate grid",
            tuple(
                BarDatum(name, value, str(value))
                for name, value in (
                    ("feature representations", len(FEATURE_REPRESENTATIONS)),
                    ("safety estimators", len(SAFETY_ESTIMATORS)),
                    ("gain rankers", len(GAIN_RANKERS)),
                    ("frontier margins", len(SAFETY_FRONTIER_MARGINS)),
                    ("policy configurations", protocol["candidate_grid"]["policy_config_count"]),
                )
            ),
            "count",
        ),
        "stage187_fit_budget.svg": _chart(
            "Stage 188 frozen fit budget",
            tuple(
                BarDatum(name, value, str(value))
                for name, value in (
                    (
                        "inner partitions",
                        protocol["cross_validation"]["inner_partition_count"],
                    ),
                    ("outer refits", protocol["cross_validation"]["outer_refit_count"]),
                    (
                        "fits per partition",
                        protocol["cross_validation"]["model_fits_per_partition"],
                    ),
                    (
                        "maximum model fits",
                        protocol["cross_validation"]["maximum_model_fit_count"],
                    ),
                )
            ),
            "count",
        ),
        "stage187_decision_flags.svg": _chart(
            "Stage 187 protocol decision flags",
            tuple(
                BarDatum(name, float(value), "true" if value else "false")
                for name, value in report["decision"].items()
                if isinstance(value, bool)
            ),
            "1 means true",
            margin_left=820,
        ),
        "stage187_guard_checks.svg": _chart(
            "Stage 187 protocol guard checks",
            tuple(
                BarDatum(row["name"], float(row["passed"]), "pass" if row["passed"] else "fail")
                for row in report["guard_checks"]
            ),
            "1 means passed",
            margin_left=920,
        ),
    }
    written = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        written.append(Stage187Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _frozen_protocol() -> dict[str, Any]:
    policy_config_count = (
        len(FEATURE_REPRESENTATIONS)
        * len(SAFETY_ESTIMATORS)
        * len(GAIN_RANKERS)
        * len(SAFETY_FRONTIER_MARGINS)
    )
    model_fits_per_representation = len(SAFETY_ESTIMATORS) * len(SAFETY_TARGETS) + len(GAIN_RANKERS)
    model_fits_per_partition = len(FEATURE_REPRESENTATIONS) * model_fits_per_representation
    inner_partition_count = OUTER_FOLD_COUNT * INNER_FOLD_COUNT
    outer_refit_count = OUTER_FOLD_COUNT
    return {
        "experiment_name": "train_only_gain_sensitive_within_question_ranking_nested_cv",
        "split_contract": {
            "selection_split": "train",
            "frozen_question_grouped_outer_folds": OUTER_FOLD_COUNT,
            "inner_folds_per_outer_fold": INNER_FOLD_COUNT,
            "development_opened": False,
            "test_opened": False,
            "all_actions_for_one_question_remain_in_one_fold": True,
        },
        "action_contract": {
            "candidate_set": (
                "all unique runtime-generatable Stage 181 counterfactual actions, "
                "including the unique original baseline action"
            ),
            "stage182_action_used_for_regression_repair_measurement_only": True,
            "reference_receives_no_selection_preference": True,
            "empty_candidate_set_possible": False,
            "fallback_enabled": False,
            "deterministic_tie_break": "canonical runtime action generation order",
        },
        "outcome_tiers": {
            "strict_gain": (
                "citation_delta >= 0 and f1_delta >= -1e-12 and at least one delta "
                "is strictly positive"
            ),
            "safe_zero": "citation_delta == 0 and abs(f1_delta) <= 1e-12",
            "unsafe": "citation_delta < 0 or f1_delta < -1e-12",
            "gold_outcomes_used_for_training_and_offline_evaluation_only": True,
            "gold_outcomes_available_to_runtime": False,
        },
        "feature_contract": {
            "feature_representations": list(FEATURE_REPRESENTATIONS),
            "raw_source": "Stage 181 frozen runtime-visible action features",
            "relative_source": "Stage 184 label-free within-question transforms",
            "forbidden_runtime_features": list(FORBIDDEN_RUNTIME_FEATURES),
            "feature_difference_uses_runtime_visible_values_only": True,
            "second_retrieval_required": False,
            "second_model_call_required": False,
        },
        "safety_model_contract": {
            "targets": {
                "citation_loss": "citation_delta < 0",
                "f1_loss": "f1_delta < -1e-12",
            },
            "estimators": list(SAFETY_ESTIMATORS),
            "question_balanced_sample_weights": True,
            "stage186_hyperparameters_frozen_without_retuning": True,
        },
        "gain_ranker_contract": {
            "rankers": list(GAIN_RANKERS),
            "pairwise_pareto_logistic": {
                "training_unit": "within-question action pair",
                "feature_map": "runtime_feature(left) - runtime_feature(right)",
                "preference_rule": [
                    "strict_gain tier outranks safe_zero tier",
                    "safe_zero tier outranks unsafe tier",
                    (
                        "inside the same non-unsafe tier, retain a pair only when one "
                        "action componentwise Pareto-dominates the other on citation/F1"
                    ),
                    "omit incomparable citation/F1 trade-off pairs",
                ],
                "both_pair_orientations_emitted": True,
                "pair_sampling": False,
                "question_balanced_pair_weights": True,
                "estimator": "class-balanced logistic regression",
            },
            "linear_listnet_top_frontier": {
                "training_unit": "complete within-question action list",
                "target_distribution": (
                    "uniform over the citation/F1 Pareto frontier inside the highest "
                    "outcome tier available for that question"
                ),
                "baseline_guarantees_at_least_safe_zero_tier": True,
                "all_actions_retained": True,
                "list_sampling": False,
                "objective": "question-mean top-one ListNet cross-entropy plus L2",
                "optimizer": {
                    "name": "deterministic full-batch Adam",
                    "learning_rate": 0.05,
                    "l2_regularization": 0.001,
                    "maximum_iterations": 400,
                    "gradient_tolerance": 1e-7,
                    "patience": 20,
                    "random_initialization": False,
                },
            },
            "incomparable_tradeoff_preference_fabricated": False,
        },
        "relative_safety_frontier": {
            "citation_excess": "p(citation_loss) - minimum question p(citation_loss)",
            "f1_excess": "p(f1_loss) - minimum question p(f1_loss)",
            "joint_excess": "max(citation_excess, f1_excess)",
            "admissible_rule": (
                "joint_excess <= minimum question joint_excess + frozen frontier margin"
            ),
            "frontier_margins": list(SAFETY_FRONTIER_MARGINS),
            "mathematically_nonempty_for_every_nonempty_candidate_set": True,
            "selection_inside_frontier": [
                "maximize learned gain ranker score",
                "minimize joint safety excess",
                "canonical runtime action order",
            ],
            "runtime_gold_filter_used": False,
            "fallback_branch_used": False,
        },
        "candidate_grid": {
            "feature_representations": list(FEATURE_REPRESENTATIONS),
            "safety_estimators": list(SAFETY_ESTIMATORS),
            "gain_rankers": list(GAIN_RANKERS),
            "safety_frontier_margins": list(SAFETY_FRONTIER_MARGINS),
            "policy_config_count": policy_config_count,
            "safety_predictions_shared_across_rankers_and_margins": True,
            "gain_scores_shared_across_safety_estimators_and_margins": True,
        },
        "cross_validation": {
            "outer_fold_count": OUTER_FOLD_COUNT,
            "inner_fold_count": INNER_FOLD_COUNT,
            "inner_partition_count": inner_partition_count,
            "outer_refit_count": outer_refit_count,
            "model_fits_per_representation_per_partition": model_fits_per_representation,
            "model_fits_per_partition": model_fits_per_partition,
            "maximum_model_fit_count": (
                (inner_partition_count + outer_refit_count) * model_fits_per_partition
            ),
            "inner_selection_uses_only_inner_oof_predictions": True,
            "outer_fold_used_once_after_inner_selection": True,
            "no_inner_eligible_config_behavior": (
                "record the outer fold as no-eligible and do not evaluate a weaker configuration"
            ),
            "no_retry": True,
            "no_fallback": True,
        },
        "inner_selection": {
            "eligibility_constraints": [
                "aggregate citation delta >= 0",
                "aggregate mean F1 delta >= 0",
                "citation nonregression in at least 3 of 4 inner folds",
                "F1 nonregression in at least 3 of 4 inner folds",
                "changed-question count >= 10% of inner questions",
                "strict-success count >= 8% of inner questions",
                "strict-success precision >= 0.60",
            ],
            "lexicographic_objective": [
                "maximize strict-success count",
                "maximize strict-success precision",
                "minimize F1-regression action count",
                "minimize citation-loss action count",
                "maximize gold-citation delta",
                "maximize mean F1 delta",
                "maximize repaired Stage 182 F1 regressions",
                "deterministic candidate name",
            ],
            "weaker_ineligible_candidate_substitution": False,
        },
        "paired_bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "unit": "question",
            "metrics": ["gold citation delta", "mean answerable F1 delta"],
        },
        "advancement_gates": _advancement_gates(),
        "resource_contract": {
            "pair_rows_materialized_as_sparse_differences": True,
            "pair_rows_processed_one_partition_and_representation_at_a_time": True,
            "histogram_dense_matrix_released_before_pairwise_construction": True,
            "all_comparable_pairs_retained_without_sampling": True,
            "gpu_required": False,
            "insufficient_memory_behavior": (
                "do not start the formal run; request resource clearance instead of "
                "reducing the protocol"
            ),
            "process_monitoring": (
                "one PowerShell Wait-Process call for the formal PID until natural exit"
            ),
        },
        "authorization_boundary": {
            "stage188_train_only_experiment_may_run_if_protocol_guards_pass": True,
            "development_evaluation_authorized": False,
            "test_evaluation_authorized": False,
            "runtime_e2e_authorized": False,
            "full_train_policy_selection_authorized": False,
            "replacement_policy_selection_authorized": False,
            "default_runtime_activation_authorized": False,
            "stage178b_authorized": False,
        },
    }


def _advancement_gates() -> list[dict[str, Any]]:
    return [
        _threshold("outer_folds_with_inner_eligible_config", ">=", 5, "count"),
        _threshold(
            "gold_citation_delta",
            ">=",
            STAGE182_CITATION_DELTA,
            "count",
        ),
        _threshold(
            "mean_f1_delta",
            ">=",
            STAGE182_MEAN_F1_DELTA,
            "rate",
        ),
        _threshold("citation_bootstrap_ci95_lower", ">=", 0.0, "count"),
        _threshold("f1_bootstrap_ci95_lower", ">=", 0.0, "rate"),
        _threshold("citation_nonregressing_outer_folds", ">=", 4, "count"),
        _threshold("f1_nonregressing_outer_folds", ">=", 4, "count"),
        _threshold("strict_success_count", ">=", 37, "count"),
        _threshold("strict_success_precision", ">=", 0.65, "rate"),
        _threshold("citation_loss_action_count", "<=", 4, "count"),
        _threshold(
            "f1_regression_action_count",
            "<=",
            STAGE182_F1_REGRESSION_COUNT // 2,
            "count",
        ),
        _threshold("stage182_regression_repair_rate", ">=", 0.50, "rate"),
        _threshold("new_f1_regression_rate", "<=", 0.02, "rate"),
        _threshold("changed_question_count", ">=", 37, "count"),
    ]


def _evidence_summary(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    stage181 = reports["stage181_report"]["action_audit"]
    stage182 = reports["stage182_report"]["dual_target_nested_cv"]["aggregate"]
    stage183 = reports["stage183_report"]["f1_risk_attribution"]
    stage184 = reports["stage184_report"]["f1_representation_cv"]
    stage185 = reports["stage185_report"]["frozen_protocol"]
    stage186_cv = reports["stage186_report"]["joint_constraint_nested_cv"]
    stage186 = stage186_cv["aggregate"]
    failed_gates = [row["name"] for row in stage186_cv["advancement_gates"] if not row["passed"]]
    return {
        "stage181": {
            "question_count": stage181["question_count"],
            "nonbaseline_action_count": stage181["nonbaseline_action_count"],
            "strict_action_count": stage181["strict_expected_action_count"],
            "questions_with_strict_action": stage181["questions_with_strict_expected_action"],
            "oracle_citation_delta": stage181["oracle"]["gold_citation_delta"],
            "oracle_mean_f1_delta": stage181["oracle"]["mean_answerable_f1_delta"],
        },
        "stage182": {
            "selected_question_count": stage182["selected_question_count"],
            "strict_success_count": stage182["strict_expected_count"],
            "strict_success_precision": stage182["strict_expected_precision"],
            "citation_loss_action_count": stage182["citation_loss_action_count"],
            "f1_regression_action_count": stage182["f1_regression_action_count"],
            "gold_citation_delta": stage182["gold_citation_delta"],
            "mean_f1_delta": stage182["mean_f1_delta_all_questions"],
        },
        "stage183": {
            "safe_alternative_rate": stage183["safe_alternative_headroom"][
                "same_or_better_citation_safe_alternative_rate"
            ],
            "frozen_model_safe_top3_rate": stage183["safe_alternative_headroom"][
                "same_or_better_safe_alternative_in_model_top3_rate"
            ],
            "frozen_model_safe_top5_rate": stage183["safe_alternative_headroom"][
                "same_or_better_safe_alternative_in_model_top5_rate"
            ],
            "primary_bottleneck": stage183["diagnostic_findings"]["primary_bottleneck"],
        },
        "stage184": {
            "selected_representation": stage184["selection"]["selected_candidate"],
            "selected_auc": stage184["selection"]["selected_roc_auc"],
            "quality_gate_pass_count": stage184["selection"]["quality_gate_pass_count"],
        },
        "stage185": {
            "policy_config_count": stage185["candidate_grid"]["policy_config_count"],
            "maximum_model_fit_count": stage185["cross_validation"]["maximum_model_head_fit_count"],
            "strict_success_precision_gate": next(
                row["threshold"]
                for row in stage185["advancement_gates"]
                if row["name"] == "strict_success_precision"
            ),
        },
        "stage186": {
            "question_count": stage186["question_count"],
            "changed_question_count": stage186["changed_question_count"],
            "strict_success_count": stage186["strict_success_count"],
            "strict_success_precision": stage186["strict_success_precision"],
            "citation_gain_action_count": stage186["citation_gain_action_count"],
            "citation_loss_action_count": stage186["citation_loss_action_count"],
            "f1_regression_action_count": stage186["f1_regression_action_count"],
            "gold_citation_delta": stage186["gold_citation_delta"],
            "mean_f1_delta": stage186["mean_f1_delta"],
            "citation_delta_vs_stage182": stage186["citation_delta_vs_reference"],
            "mean_f1_delta_vs_stage182": stage186["mean_f1_delta_vs_reference"],
            "repaired_reference_regression_count": stage186["repaired_reference_regression_count"],
            "head_metrics": {
                target: {
                    "roc_auc": stage186_cv["head_metrics"][target]["roc_auc"],
                    "average_precision": stage186_cv["head_metrics"][target]["average_precision"],
                }
                for target in ("citation_loss", "f1_loss", "strict_gain")
            },
            "selected_ranking_rule_counts": _ranking_rule_counts(
                stage186_cv["selected_spec_counts"]
            ),
            "advancement_gates": stage186_cv["advancement_gates"],
            "advancement_gate_pass_count": stage186_cv["advancement_gate_pass_count"],
            "failed_gates": failed_gates,
        },
        "design_conclusion": (
            "Stage 186 proved that minimizing continuous safety risk before gain "
            "collapses to zero-delta actions. Stage 188 must learn within-question "
            "gain preferences directly, make gain primary inside a guaranteed-nonempty "
            "relative safety frontier, retain hard offline citation/F1 gates, and "
            "reject any result that fails to preserve the Stage 182 aggregate gains."
        ),
    }


def _ranking_rule_counts(selected_spec_counts: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, count in selected_spec_counts.items():
        for rule in (
            "max_safety_risk_lexicographic",
            "citation_first_lexicographic",
            "pareto_constraint_dominance",
        ):
            if f"__{rule}__" in name:
                counts[rule] = counts.get(rule, 0) + int(count)
                break
    return dict(sorted(counts.items()))


def _guard_checks(
    report: Mapping[str, Any],
    *,
    source_reports: Mapping[str, Mapping[str, Any]],
    source_files: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    protocol = report["frozen_protocol"]
    evidence = report["evidence_summary"]
    boundaries = report["execution_boundaries"]
    stage186_decision = source_reports["stage186_report"]["decision"]
    stage186_evidence = evidence["stage186"]
    return [
        _guard("user_confirmed", report["user_confirmation"]["confirmed"]),
        *[
            _guard(
                f"{name}_sha256_matches",
                source_files[name]["sha256"] == EXPECTED_SOURCE_SHA256[name],
            )
            for name in EXPECTED_SOURCE_SHA256
        ],
        *[
            _guard(
                f"{name}_status_matches",
                source_reports[name]["decision"]["status"] == EXPECTED_SOURCE_STATUS[name],
            )
            for name in EXPECTED_SOURCE_STATUS
        ],
        _guard("stage186_experiment_valid", stage186_decision["experiment_valid"] is True),
        _guard(
            "stage186_candidate_family_not_accepted",
            stage186_decision["candidate_family_accepted"] is False,
        ),
        _guard(
            "stage186_failed_only_strict_precision_gate",
            stage186_evidence["failed_gates"] == ["strict_success_precision_at_least_0_65"],
        ),
        _guard(
            "stage186_selected_zero_strict_successes",
            stage186_evidence["strict_success_count"] == 0,
        ),
        _guard(
            "stage186_all_selected_rules_were_max_safety",
            stage186_evidence["selected_ranking_rule_counts"]
            == {"max_safety_risk_lexicographic": 5},
        ),
        _guard(
            "policy_config_count_is_32",
            protocol["candidate_grid"]["policy_config_count"] == 32,
        ),
        _guard(
            "maximum_model_fit_count_is_300",
            protocol["cross_validation"]["maximum_model_fit_count"] == 300,
        ),
        _guard(
            "question_grouped_nested_cv",
            protocol["split_contract"]["all_actions_for_one_question_remain_in_one_fold"] is True,
        ),
        _guard(
            "inner_selection_uses_inner_oof_only",
            protocol["cross_validation"]["inner_selection_uses_only_inner_oof_predictions"] is True,
        ),
        _guard(
            "pairwise_training_has_no_sampling",
            protocol["gain_ranker_contract"]["pairwise_pareto_logistic"]["pair_sampling"] is False,
        ),
        _guard(
            "listwise_training_has_no_sampling",
            protocol["gain_ranker_contract"]["linear_listnet_top_frontier"]["list_sampling"]
            is False,
        ),
        _guard(
            "incomparable_tradeoffs_not_fabricated",
            protocol["gain_ranker_contract"]["incomparable_tradeoff_preference_fabricated"]
            is False,
        ),
        _guard(
            "relative_safety_frontier_is_nonempty",
            protocol["relative_safety_frontier"][
                "mathematically_nonempty_for_every_nonempty_candidate_set"
            ]
            is True,
        ),
        _guard(
            "gold_outcomes_not_runtime_features",
            protocol["outcome_tiers"]["gold_outcomes_available_to_runtime"] is False,
        ),
        _guard(
            "stage182_gains_are_advancement_floors",
            _gate_threshold(protocol, "gold_citation_delta") == STAGE182_CITATION_DELTA
            and _gate_threshold(protocol, "mean_f1_delta") == STAGE182_MEAN_F1_DELTA,
        ),
        _guard(
            "no_fallback",
            protocol["action_contract"]["fallback_enabled"] is False
            and protocol["cross_validation"]["no_fallback"] is True
            and protocol["relative_safety_frontier"]["fallback_branch_used"] is False
            and boundaries["fallback_action_count"] == 0,
        ),
        _guard("development_closed", boundaries["development_loaded"] is False),
        _guard("test_closed", boundaries["test_loaded"] is False),
        _guard("no_model_fit", boundaries["model_fit_count"] == 0),
        _guard("no_pair_rows_materialized", boundaries["pair_rows_materialized"] == 0),
        _guard(
            "no_listwise_questions_materialized",
            boundaries["listwise_questions_materialized"] == 0,
        ),
        _guard("no_policy_evaluation", boundaries["policy_evaluation_run"] is False),
        _guard(
            "no_replacement_policy_selected",
            boundaries["replacement_policy_selected"] is False,
        ),
        _guard("runtime_e2e_not_run", boundaries["runtime_e2e_run"] is False),
        _guard("default_runtime_unchanged", boundaries["runtime_registered_as_default"] is False),
        _guard("stage178b_not_run", boundaries["stage178b_run"] is False),
        _guard("no_retry", boundaries["retry_action_count"] == 0),
        _guard("public_report_safe", not _forbidden_keys_found(report)),
    ]


def _decision(guards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = all(row["passed"] for row in guards)
    return {
        "status": (
            "stage187_gain_sensitive_ranking_protocol_frozen"
            if valid
            else "stage187_gain_sensitive_ranking_protocol_invalid"
        ),
        "protocol_valid": valid,
        "stage188_train_only_experiment_authorized": valid,
        "development_opened": False,
        "test_opened": False,
        "runtime_e2e_authorized": False,
        "full_train_policy_selection_authorized": False,
        "replacement_policy_selected": False,
        "default_runtime_activation": False,
    }


def _chart(
    title: str,
    bars: Sequence[BarDatum],
    x_label: str,
    *,
    margin_left: int = 720,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label=x_label,
        width=1680,
        margin_left=margin_left,
        margin_right=260,
    )


def _gate_threshold(protocol: Mapping[str, Any], name: str) -> float:
    return next(row["threshold"] for row in protocol["advancement_gates"] if row["name"] == name)


def _threshold(name: str, operator: str, threshold: float, unit: str) -> dict[str, Any]:
    return {
        "name": name,
        "operator": operator,
        "threshold": threshold,
        "unit": unit,
    }


def _guard(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    payload = resolved.read_bytes()
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_size": len(payload),
    }


def _load_json_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _forbidden_keys_found(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        found = {str(key) for key in value if str(key) in FORBIDDEN_PUBLIC_KEYS}
        for nested in value.values():
            found.update(_forbidden_keys_found(nested))
        return found
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        found: set[str] = set()
        for nested in value:
            found.update(_forbidden_keys_found(nested))
        return found
    return set()
