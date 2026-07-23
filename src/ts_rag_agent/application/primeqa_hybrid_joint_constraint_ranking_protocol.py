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

STAGE = "Stage 185"
CREATED_AT = "2026-07-23"
PROTOCOL_ID = "primeqa_hybrid_joint_constraint_ranking_protocol_v1"
NEXT_STAGE = "Stage 186"
EXPECTED_SOURCE_SHA256 = {
    "stage181_report": "a9c557d7346eb2b4958cddd2505937eba828556c7671d7e936bf883d80cfe88b",
    "stage182_report": "c3dfbe7484a604b8491bed0531fc82b20bd092016fd7ddf303955b7c7c89044a",
    "stage183_report": "4dd611c9a759fd791288886c638bd9ec36b7564328eb53f2fef1742544540f1a",
    "stage184_report": "bdbb49bf31a0f889a431924ee1630c7593ec485fb1bf283def44048776a29eea",
}
EXPECTED_SOURCE_STATUS = {
    "stage181_report": "stage181_counterfactual_action_audit_complete",
    "stage182_report": "stage182_dual_target_nested_cv_insufficient",
    "stage183_report": "stage183_f1_risk_failure_attribution_complete",
    "stage184_report": "stage184_f1_representation_cv_insufficient",
}
FEATURE_REPRESENTATIONS = ("raw_runtime", "question_relative_runtime")
ESTIMATOR_FAMILIES = ("class_balanced_logistic", "histogram_gradient_boosting")
MODEL_TARGETS = ("citation_loss", "f1_loss", "strict_gain")
RANKING_RULES = (
    "max_safety_risk_lexicographic",
    "citation_first_lexicographic",
    "pareto_constraint_dominance",
)
SAFETY_DOMINANCE_MARGINS = (0.0, 0.02, 0.05)
STRICT_GAIN_MARGINS = (0.0, 0.05)
OUTER_FOLD_COUNT = 5
INNER_FOLD_COUNT = 4
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 185
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
class Stage185Visualization:
    """One public-safe Stage 185 protocol visualization."""

    name: str
    path: str


def freeze_joint_constraint_ranking_protocol(
    *,
    stage181_report_path: Path,
    stage182_report_path: Path,
    stage183_report_path: Path,
    stage184_report_path: Path,
    user_confirmed: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the Stage 185 train-only joint-constraint ranking protocol."""

    started_at = time.perf_counter()
    source_paths = {
        "stage181_report": stage181_report_path,
        "stage182_report": stage182_report_path,
        "stage183_report": stage183_report_path,
        "stage184_report": stage184_report_path,
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
            "Train-only protocol freeze for a new citation/F1 joint-constraint action "
            "ranking experiment. This stage reads only aggregate public-safe Stage "
            "181-184 reports, does not load split rows or documents, does not fit a "
            "model, does not evaluate a policy, keeps development and test closed, "
            "adds no fallback, and does not change runtime defaults."
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
    return {
        **report,
        "public_safe_contract": {
            "forbidden_public_keys": sorted(FORBIDDEN_PUBLIC_KEYS),
            "forbidden_keys_found": sorted(_forbidden_keys_found(report)),
            "private_action_rows_persisted": False,
            "private_predictions_persisted": False,
            "public_report_safe": not _forbidden_keys_found(report),
        },
    }


def write_stage185_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage185Visualization, ...]:
    """Write and XML-validate Stage 185 protocol charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = report["evidence_summary"]
    protocol = report["frozen_protocol"]
    charts = {
        "stage185_evidence_headroom.svg": _chart(
            "Stage 185 evidence for joint-constraint ranking",
            (
                BarDatum(
                    "questions with strict action",
                    evidence["stage181"]["questions_with_strict_action_rate"],
                    f"{evidence['stage181']['questions_with_strict_action_rate']:.3f}",
                ),
                BarDatum(
                    "old regressions with safe alternative",
                    evidence["stage183"]["safe_alternative_rate"],
                    f"{evidence['stage183']['safe_alternative_rate']:.3f}",
                ),
                BarDatum(
                    "selected representation safe top3",
                    evidence["stage184"]["selected_safe_top3_rate"],
                    f"{evidence['stage184']['selected_safe_top3_rate']:.3f}",
                ),
                BarDatum(
                    "selected representation safe top5",
                    evidence["stage184"]["selected_safe_top5_rate"],
                    f"{evidence['stage184']['selected_safe_top5_rate']:.3f}",
                ),
            ),
            "fraction of applicable train questions",
        ),
        "stage185_representation_auc.svg": _chart(
            "Stage 185 source F1-risk representation AUC",
            tuple(
                BarDatum(name, value, f"{value:.3f}")
                for name, value in evidence["stage184"]["representation_auc"].items()
            ),
            "five-fold OOF ROC AUC",
        ),
        "stage185_candidate_grid.svg": _chart(
            "Stage 185 candidate grid dimensions",
            (
                BarDatum(
                    "feature representations",
                    len(protocol["candidate_grid"]["feature_representations"]),
                    str(len(protocol["candidate_grid"]["feature_representations"])),
                ),
                BarDatum(
                    "estimator families",
                    len(protocol["candidate_grid"]["estimator_families"]),
                    str(len(protocol["candidate_grid"]["estimator_families"])),
                ),
                BarDatum(
                    "ranking rules",
                    len(protocol["candidate_grid"]["ranking_rules"]),
                    str(len(protocol["candidate_grid"]["ranking_rules"])),
                ),
                BarDatum(
                    "policy configurations",
                    protocol["candidate_grid"]["policy_config_count"],
                    str(protocol["candidate_grid"]["policy_config_count"]),
                ),
            ),
            "frozen count",
        ),
        "stage185_fit_budget.svg": _chart(
            "Stage 185 nested-CV fit budget",
            (
                BarDatum(
                    "inner partitions",
                    protocol["cross_validation"]["inner_partition_count"],
                    str(protocol["cross_validation"]["inner_partition_count"]),
                ),
                BarDatum(
                    "outer refits",
                    protocol["cross_validation"]["outer_refit_count"],
                    str(protocol["cross_validation"]["outer_refit_count"]),
                ),
                BarDatum(
                    "heads per partition",
                    protocol["cross_validation"]["head_fits_per_partition"],
                    str(protocol["cross_validation"]["head_fits_per_partition"]),
                ),
                BarDatum(
                    "maximum head fits",
                    protocol["cross_validation"]["maximum_model_head_fit_count"],
                    str(protocol["cross_validation"]["maximum_model_head_fit_count"]),
                ),
            ),
            "count",
        ),
        "stage185_advancement_gates.svg": _chart(
            "Stage 185 Stage 186 advancement thresholds",
            tuple(
                BarDatum(
                    gate["name"],
                    float(gate["threshold"]),
                    f"{gate['operator']} {gate['threshold']} {gate['unit']}",
                )
                for gate in protocol["advancement_gates"]
            ),
            "threshold value",
            margin_left=900,
        ),
        "stage185_decision_flags.svg": _chart(
            "Stage 185 protocol decision flags",
            tuple(
                BarDatum(name, float(value), "true" if value else "false")
                for name, value in report["decision"].items()
                if isinstance(value, bool)
            ),
            "1 means true",
            margin_left=760,
        ),
        "stage185_guard_checks.svg": _chart(
            "Stage 185 protocol guard checks",
            tuple(
                BarDatum(
                    row["name"],
                    float(row["passed"]),
                    "pass" if row["passed"] else "fail",
                )
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
        written.append(Stage185Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _frozen_protocol() -> dict[str, Any]:
    policy_config_count = (
        len(FEATURE_REPRESENTATIONS)
        * len(ESTIMATOR_FAMILIES)
        * len(RANKING_RULES)
        * len(SAFETY_DOMINANCE_MARGINS)
        * len(STRICT_GAIN_MARGINS)
    )
    head_fits_per_partition = (
        len(FEATURE_REPRESENTATIONS) * len(ESTIMATOR_FAMILIES) * len(MODEL_TARGETS)
    )
    inner_partition_count = OUTER_FOLD_COUNT * INNER_FOLD_COUNT
    outer_refit_count = OUTER_FOLD_COUNT
    return {
        "experiment_name": "train_only_joint_constraint_action_ranking_nested_cv",
        "split_contract": {
            "selection_split": "train",
            "frozen_question_grouped_outer_folds": OUTER_FOLD_COUNT,
            "inner_folds_per_outer_fold": INNER_FOLD_COUNT,
            "development_opened": False,
            "test_opened": False,
            "all_actions_for_one_question_remain_in_one_fold": True,
        },
        "reference_action_contract": {
            "reference": "Stage 182 emitted action for each train question",
            "candidate_set": (
                "all unique runtime-generatable Stage 181 counterfactual actions plus "
                "the Stage 182 reference action"
            ),
            "reference_is_an_ordinary_ranked_candidate": True,
            "reference_receives_no_tie_break_preference": True,
            "empty_candidate_set_possible": False,
            "fallback_enabled": False,
            "deterministic_tie_break": "canonical runtime action generation order",
        },
        "target_contract": {
            "citation_loss": "citation_delta < 0",
            "f1_loss": "f1_delta < -1e-12",
            "strict_gain": (
                "citation_delta >= 0 and f1_delta >= -1e-12 and at least one delta "
                "is strictly positive"
            ),
            "gold_targets_used_for_fit_and_offline_evaluation_only": True,
            "gold_targets_available_to_runtime": False,
        },
        "feature_contract": {
            "feature_representations": list(FEATURE_REPRESENTATIONS),
            "raw_source": "Stage 181 frozen runtime-visible action features",
            "relative_source": "Stage 184 label-free within-question transforms",
            "selected_stage184_representation_is_not_privileged": True,
            "forbidden_runtime_features": list(FORBIDDEN_RUNTIME_FEATURES),
            "second_retrieval_required": False,
            "second_model_call_required": False,
        },
        "candidate_grid": {
            "feature_representations": list(FEATURE_REPRESENTATIONS),
            "estimator_families": list(ESTIMATOR_FAMILIES),
            "shared_model_targets": list(MODEL_TARGETS),
            "ranking_rules": list(RANKING_RULES),
            "safety_dominance_margins": list(SAFETY_DOMINANCE_MARGINS),
            "strict_gain_margins": list(STRICT_GAIN_MARGINS),
            "policy_config_count": policy_config_count,
            "predictions_shared_across_ranking_rules_and_margins": True,
        },
        "ranking_semantics": {
            "max_safety_risk_lexicographic": [
                "minimize max(predicted citation-loss risk, predicted F1-loss risk)",
                "minimize sum of both predicted loss risks",
                "maximize predicted strict-gain probability",
            ],
            "citation_first_lexicographic": [
                "minimize predicted citation-loss risk",
                "minimize predicted F1-loss risk",
                "maximize predicted strict-gain probability",
            ],
            "pareto_constraint_dominance": [
                "prefer actions no worse than the reference by the frozen safety margin",
                "then require the frozen strict-gain probability margin",
                "rank remaining exact ties by canonical runtime action order",
            ],
            "runtime_gold_filter_used": False,
            "fallback_branch_used": False,
        },
        "cross_validation": {
            "outer_fold_count": OUTER_FOLD_COUNT,
            "inner_fold_count": INNER_FOLD_COUNT,
            "inner_partition_count": inner_partition_count,
            "outer_refit_count": outer_refit_count,
            "head_fits_per_partition": head_fits_per_partition,
            "maximum_model_head_fit_count": (
                (inner_partition_count + outer_refit_count) * head_fits_per_partition
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
            ],
            "lexicographic_objective": [
                "maximize repaired Stage 182 F1 regressions",
                "minimize newly induced F1 regressions",
                "maximize strict-success precision",
                "maximize gold-citation delta",
                "maximize mean F1 delta",
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
        "authorization_boundary": {
            "stage186_train_only_experiment_may_run_if_protocol_guards_pass": True,
            "development_evaluation_authorized": False,
            "test_evaluation_authorized": False,
            "runtime_e2e_authorized": False,
            "replacement_policy_selection_authorized": False,
            "default_runtime_activation_authorized": False,
            "stage178b_authorized": False,
        },
    }


def _advancement_gates() -> list[dict[str, Any]]:
    return [
        _threshold("outer_folds_with_inner_eligible_config", ">=", 5, "count"),
        _threshold("gold_citation_delta", ">=", 0, "count"),
        _threshold("mean_f1_delta", ">=", 0.0, "rate"),
        _threshold("citation_bootstrap_ci95_lower", ">=", 0.0, "count"),
        _threshold("f1_bootstrap_ci95_lower", ">=", 0.0, "rate"),
        _threshold("citation_nonregressing_outer_folds", ">=", 4, "count"),
        _threshold("f1_nonregressing_outer_folds", ">=", 4, "count"),
        _threshold("stage182_regression_repair_rate", ">=", 0.5, "rate"),
        _threshold("new_f1_regression_rate", "<=", 0.02, "rate"),
        _threshold("citation_loss_action_count", "<=", 4, "count"),
        _threshold("strict_success_precision", ">=", 0.65, "rate"),
        _threshold("changed_question_count", ">=", 37, "count"),
    ]


def _evidence_summary(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    stage181 = reports["stage181_report"]["action_audit"]
    stage182 = reports["stage182_report"]["dual_target_nested_cv"]["aggregate"]
    stage183 = reports["stage183_report"]["f1_risk_attribution"]
    stage184 = reports["stage184_report"]["f1_representation_cv"]
    representations = stage184["representations"]
    selection = stage184["selection"]
    selected = representations[selection["selected_candidate"]]
    return {
        "stage181": {
            "question_count": stage181["question_count"],
            "nonbaseline_action_count": stage181["nonbaseline_action_count"],
            "strict_action_count": stage181["strict_expected_action_count"],
            "questions_with_strict_action": stage181["questions_with_strict_expected_action"],
            "questions_with_strict_action_rate": round(
                stage181["questions_with_strict_expected_action"] / stage181["question_count"],
                6,
            ),
            "oracle_citation_delta": stage181["oracle"]["gold_citation_delta"],
            "oracle_mean_f1_delta": stage181["oracle"]["mean_answerable_f1_delta"],
        },
        "stage182": {
            "selected_question_count": stage182["selected_question_count"],
            "strict_success_precision": stage182["strict_expected_precision"],
            "citation_loss_action_count": stage182["citation_loss_action_count"],
            "f1_regression_action_count": stage182["f1_regression_action_count"],
            "gold_citation_delta": stage182["gold_citation_delta"],
            "mean_f1_delta": stage182["mean_f1_delta_all_questions"],
        },
        "stage183": {
            "selected_f1_regression_rate": stage183["selected_action_summary"][
                "f1_regression_rate"
            ],
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
            "best_raw_reference": selection["best_raw_reference"],
            "selected_representation": selection["selected_candidate"],
            "selected_representation_accepted": selection[
                "candidate_accepted_for_nested_policy_experiment"
            ],
            "selected_auc": selection["selected_roc_auc"],
            "auc_gain_vs_raw": selection["roc_auc_gain_vs_best_raw"],
            "selected_safe_top3_rate": selected["stage182_regression_headroom"][
                "safe_alternative_top3_rate"
            ],
            "selected_safe_top5_rate": selected["stage182_regression_headroom"][
                "safe_alternative_top5_rate"
            ],
            "representation_auc": {
                name: row["aggregate"]["roc_auc"] for name, row in representations.items()
            },
            "quality_gate_pass_count": selection["quality_gate_pass_count"],
        },
        "design_conclusion": (
            "Do not promote the Stage 184 AUC winner. Compare raw and relative "
            "features inside a new nested, citation/F1 joint-constraint ranking "
            "experiment with explicit non-triviality and regression-repair gates."
        ),
    }


def _guard_checks(
    report: Mapping[str, Any],
    *,
    source_reports: Mapping[str, Mapping[str, Any]],
    source_files: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    protocol = report["frozen_protocol"]
    boundaries = report["execution_boundaries"]
    stage184_decision = source_reports["stage184_report"]["decision"]
    guards = [
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
        _guard("stage184_experiment_valid", stage184_decision["experiment_valid"] is True),
        _guard(
            "stage184_candidate_not_promoted",
            stage184_decision["representation_candidate_accepted"] is False,
        ),
        _guard(
            "policy_config_count_is_72",
            protocol["candidate_grid"]["policy_config_count"] == 72,
        ),
        _guard(
            "maximum_model_head_fit_count_is_300",
            protocol["cross_validation"]["maximum_model_head_fit_count"] == 300,
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
            "gold_targets_not_runtime_features",
            protocol["target_contract"]["gold_targets_available_to_runtime"] is False,
        ),
        _guard(
            "no_fallback",
            protocol["reference_action_contract"]["fallback_enabled"] is False
            and protocol["cross_validation"]["no_fallback"] is True
            and boundaries["fallback_action_count"] == 0,
        ),
        _guard("development_closed", boundaries["development_loaded"] is False),
        _guard("test_closed", boundaries["test_loaded"] is False),
        _guard("no_model_fit", boundaries["model_fit_count"] == 0),
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
    return guards


def _decision(guards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = all(row["passed"] for row in guards)
    return {
        "status": (
            "stage185_joint_constraint_ranking_protocol_frozen"
            if valid
            else "stage185_joint_constraint_ranking_protocol_invalid"
        ),
        "protocol_valid": valid,
        "stage186_train_only_experiment_authorized": valid,
        "development_opened": False,
        "test_opened": False,
        "runtime_e2e_authorized": False,
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
