from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application import (
    primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_protocol as stage160,
)
from ts_rag_agent.application import primeqa_hybrid_gold_visible_refusal_diagnostics as stage164
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    _load_json_object,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector_training import (
    _fingerprint,
    _public_safe_contract,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 164 contract correction"
_CREATED_AT = "2026-07-19"
_ANALYSIS_ID = "primeqa_hybrid_gold_visible_refusal_contract_correction_v1"
_ORIGINAL_PUBLIC_SHA256 = "2a7dcef4fbc007f53d141cd246e7ad4bf327c3f5ad75899424f0a69c273ed3ae"
_ORIGINAL_PRIVATE_BYTE_SHA256 = "ddddc77f3e5bdfe1a756aa680dae2102e340a6b7c5452b90a53271ccc5f98507"
_ORIGINAL_PRIVATE_CANONICAL_SHA256 = (
    "20d42bae9b954e223dbc55798fa8de71fe5beec42dbed835f5f96b2e7aba3a63"
)
_STAGE160_HASHED_BYTE_SHA256 = "3f10cffe245a4405dfc56044f2a3c0d364fdd0f8723e6cc3ae401260199652db"
_STAGE160_HASHED_CANONICAL_SHA256 = (
    "1c8aa4260be5427e13322cb3304e518dd3609c2e38f839cda4f10ce01c911a0d"
)
_ORIGINAL_STATUS = "primeqa_hybrid_gold_visible_refusal_diagnostics_invalid"
_ORIGINAL_FAILED_GUARD = "gold_generation_ranks_bounded"
_CORRECTED_GUARD = "gold_generation_context_membership_exact"
_CORRECTED_STATUS = "primeqa_hybrid_gold_visible_refusal_diagnostics_completed"


@dataclass(frozen=True)
class PrimeQAHybridGoldVisibleRefusalCorrectionVisualization:
    name: str
    path: str


def run_stage164_contract_correction(
    *,
    original_public_report_path: Path,
    original_private_report_path: Path,
    stage160_hashed_report_path: Path,
    user_confirmed_correction: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Correct Stage164 rank semantics and interpretation without loading dev."""

    fingerprints = {
        "original_stage164_public": _fingerprint(original_public_report_path),
        "original_stage164_private": _fingerprint(original_private_report_path),
        "stage160_hashed": _fingerprint(stage160_hashed_report_path),
    }
    _authorize_fingerprints(fingerprints)
    original = _load_json_object(original_public_report_path)
    original_private = _load_json_object(original_private_report_path)
    stage160_hashed = _load_json_object(stage160_hashed_report_path)
    _authorize_reports(
        original=original,
        original_private=original_private,
        stage160_hashed=stage160_hashed,
    )

    profiles = tuple(
        stage164.GoldVisibleRefusalCaseProfile(**row) for row in original_private["rows"]
    )
    visible_stage160_rows = [
        row
        for row in stage160_hashed["rows"]
        if row["answerable"] and row["gold_generation_rank"] is not None
    ]
    corrected_process_guards = stage164._guard_checks(
        report=original,
        profiles=profiles,
        stage160_hashed=stage160_hashed,
    )
    corrected_assessment = stage164._primary_hypothesis_assessment(
        binary_associations=original["fixed_binary_associations"],
        fold_stability=original["fold_stability"],
    )
    corrected_context = {
        **original,
        "primary_hypothesis_assessment": corrected_assessment,
        "guard_checks": corrected_process_guards,
    }
    corrected_decision = stage164._decision(
        report=corrected_context,
        all_guards_passed=all(check["passed"] for check in corrected_process_guards),
    )
    metric_snapshot = _metric_snapshot(original)
    metric_sha256 = stage160.canonical_json_sha256(metric_snapshot)

    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Post-run correction of Stage164 generation-rank semantics and hypothesis "
            "interpretation. It reads only the immutable Stage164 public/private artifacts "
            "and Stage160 hashed diagnostics. No split, document corpus, retrieval, Agent, "
            "feature extraction, metric recomputation, fitting, tuning, or policy selection "
            "is performed."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_correction),
            "confirmation_note": confirmation_note,
            "selected_option": "A",
        },
        "source_authorization": {
            "fingerprints": fingerprints,
            "original_private_canonical_sha256": stage160.canonical_json_sha256(original_private),
            "stage160_hashed_canonical_sha256": stage160.canonical_json_sha256(stage160_hashed),
            "original_artifacts_remain_canonical": True,
        },
        "rank_semantics_correction": {
            "original_failed_guard": _ORIGINAL_FAILED_GUARD,
            "original_invalid_assumption": (
                "generation_result_rank_is_a_dense_position_from_1_through_10"
            ),
            "actual_rank_semantics": "candidate_pool_rank_preserved_on_generation_results",
            "gold_generation_rank_minimum": min(
                int(row["gold_generation_rank"]) for row in visible_stage160_rows
            ),
            "gold_generation_rank_maximum": max(
                int(row["gold_generation_rank"]) for row in visible_stage160_rows
            ),
            "rank_above_ten_count": sum(
                int(row["gold_generation_rank"]) > 10 for row in visible_stage160_rows
            ),
            "generation_context_count_minimum": min(
                int(row["generation_context_count"]) for row in visible_stage160_rows
            ),
            "generation_context_count_maximum": max(
                int(row["generation_context_count"]) for row in visible_stage160_rows
            ),
            "corrected_guard": _CORRECTED_GUARD,
            "corrected_guard_semantics": (
                "gold_membership_is_non_null_and_every_visible_case_has_context_count_10"
            ),
        },
        "hypothesis_interpretation_correction": {
            "original_aggregate_only_flag": original["primary_hypothesis_assessment"][
                "visibility_gap_observed"
            ],
            "corrected_assessment": corrected_assessment,
            "interpretation": (
                "answer_evidence_absence_has_a_positive_aggregate_association_but_the_"
                "direction_is_not_stable_across_grouped_folds"
            ),
            "causal_claim": False,
            "prompt_intervention_selected": False,
        },
        "stable_observed_patterns": {
            "question_gold_prompt_alignment_risk_auc": original["fixed_numeric_associations"][
                "question_token_recall_in_gold_prompt"
            ]["risk_aligned_auc"],
            "question_alignment_refused_median": original["fixed_numeric_associations"][
                "question_token_recall_in_gold_prompt"
            ]["refused"]["median"],
            "question_alignment_answered_median": original["fixed_numeric_associations"][
                "question_token_recall_in_gold_prompt"
            ]["answered"]["median"],
            "post_first_turn_refusal_rate_difference": original["fixed_binary_associations"][
                "turn_position_after_first"
            ]["refusal_rate_difference_risk_minus_reference"],
            "post_first_turn_risk_direction_fold_count": original["fold_stability"][
                "turn_position_after_first"
            ]["risk_direction_fold_count"],
            "post_first_turn_comparable_fold_count": original["fold_stability"][
                "turn_position_after_first"
            ]["comparable_fold_count"],
            "synthetic_history_is_not_natural_conversation": True,
            "causal_claim": False,
        },
        "process_correction": {
            "original_process_guard_pass_count": sum(
                check["passed"] for check in original["guard_checks"]
            ),
            "original_process_guard_count": len(original["guard_checks"]),
            "corrected_process_guard_pass_count": sum(
                check["passed"] for check in corrected_process_guards
            ),
            "corrected_process_guard_count": len(corrected_process_guards),
            "corrected_process_guards": corrected_process_guards,
        },
        "metric_integrity": {
            "metric_snapshot_sha256_before_correction": metric_sha256,
            "metric_snapshot_sha256_after_correction": stage160.canonical_json_sha256(
                _metric_snapshot(original)
            ),
            "metric_snapshot_changed": False,
            "development_rows_loaded_during_correction": 0,
            "documents_loaded_during_correction": 0,
            "feature_rows_recomputed_during_correction": 0,
            "agent_runs_during_correction": 0,
            "retrieval_runs_during_correction": 0,
        },
        "corrected_stage164_decision": corrected_decision,
        "closed_boundaries": {
            "train_loaded": False,
            "dev_loaded": False,
            "test_loaded": False,
            "agent_run": False,
            "retrieval_run": False,
            "policy_fit": False,
            "threshold_tuned": False,
            "runtime_registered_as_default": False,
            "fallback_strategies_enabled": False,
        },
    }
    report["guard_checks"] = _correction_guards(report=report, original=original)
    report["public_safe_contract"] = _public_safe_contract(report)
    all_guards_passed = all(check["passed"] for check in report["guard_checks"])
    report["decision"] = {
        "status": (
            "primeqa_hybrid_stage164_contract_correction_completed"
            if all_guards_passed
            else "primeqa_hybrid_stage164_contract_correction_invalid"
        ),
        "all_correction_guards_passed": all_guards_passed,
        "failed_correction_guards": [
            check["name"] for check in report["guard_checks"] if not check["passed"]
        ],
        "corrected_stage164_status": corrected_decision["status"],
        "fold_stable_visibility_gap_observed": corrected_assessment[
            "fold_stable_visibility_gap_observed"
        ],
        "diagnostic_only": True,
        "policy_selected": False,
        "dev_reloaded": False,
        "test_gate_opened": False,
        "runtime_registered_as_default": False,
        "fallback_strategies_enabled": False,
        "next_direction": corrected_decision["next_direction"],
    }
    if _fingerprint(original_public_report_path) != fingerprints["original_stage164_public"]:
        raise ValueError("original Stage164 public report changed during correction")
    if _fingerprint(original_private_report_path) != fingerprints["original_stage164_private"]:
        raise ValueError("original Stage164 private report changed during correction")
    report["source_authorization"]["original_artifacts_unchanged_after_correction"] = True
    return report


def write_stage164_contract_correction_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridGoldVisibleRefusalCorrectionVisualization]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rank = report["rank_semantics_correction"]
    hypothesis = report["hypothesis_interpretation_correction"]["corrected_assessment"]
    patterns = report["stable_observed_patterns"]
    specs = {
        "stage164_corrected_rank_semantics.svg": (
            "Stage164 generation context depth versus preserved source rank",
            [
                _bar("generation context depth", rank["generation_context_count_maximum"]),
                _bar("maximum preserved candidate rank", rank["gold_generation_rank_maximum"]),
                _bar("visible cases with rank above 10", rank["rank_above_ten_count"]),
            ],
        ),
        "stage164_corrected_pattern_stability.svg": (
            "Stage164 grouped-fold directional stability",
            [
                _bar(
                    "answer visibility risk direction",
                    hypothesis["exact_span_fold_direction_count"],
                ),
                _bar(
                    "answer visibility opposite direction",
                    hypothesis["exact_span_fold_opposite_direction_count"],
                ),
                _bar(
                    "post-first-turn risk direction",
                    patterns["post_first_turn_risk_direction_fold_count"],
                ),
            ],
        ),
    }
    artifacts = []
    for filename, (title, bars) in specs.items():
        path = output_dir / filename
        path.write_text(
            render_horizontal_bar_chart_svg(title=title, bars=bars, x_label="count"),
            encoding="utf-8",
        )
        artifacts.append(
            PrimeQAHybridGoldVisibleRefusalCorrectionVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _authorize_fingerprints(fingerprints: Mapping[str, Mapping[str, Any]]) -> None:
    expected = {
        "original_stage164_public": _ORIGINAL_PUBLIC_SHA256,
        "original_stage164_private": _ORIGINAL_PRIVATE_BYTE_SHA256,
        "stage160_hashed": _STAGE160_HASHED_BYTE_SHA256,
    }
    mismatches = {
        name: fingerprints[name]["sha256"]
        for name, expected_sha in expected.items()
        if fingerprints[name]["sha256"] != expected_sha
    }
    if mismatches:
        raise ValueError(f"Stage164 correction source fingerprint mismatch: {mismatches}")


def _authorize_reports(
    *,
    original: Mapping[str, Any],
    original_private: Mapping[str, Any],
    stage160_hashed: Mapping[str, Any],
) -> None:
    if original.get("decision", {}).get("status") != _ORIGINAL_STATUS:
        raise ValueError("Stage164 correction requires the original invalid report")
    if original.get("decision", {}).get("failed_process_guards") != [_ORIGINAL_FAILED_GUARD]:
        raise ValueError("Stage164 correction requires the single known rank guard failure")
    if stage160.canonical_json_sha256(original_private) != _ORIGINAL_PRIVATE_CANONICAL_SHA256:
        raise ValueError("Stage164 original private canonical content mismatch")
    if stage160.canonical_json_sha256(stage160_hashed) != _STAGE160_HASHED_CANONICAL_SHA256:
        raise ValueError("Stage160 hashed canonical content mismatch")
    if original_private.get("row_count") != 36 or len(original_private.get("rows", [])) != 36:
        raise ValueError("Stage164 correction requires exactly 36 hashed feature rows")


def _metric_snapshot(original: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cohort_summary": original["cohort_summary"],
        "answer_visibility_summary": original["answer_visibility_summary"],
        "fixed_binary_associations": original["fixed_binary_associations"],
        "fixed_numeric_associations": original["fixed_numeric_associations"],
        "question_route_summary": original["question_route_summary"],
        "fold_stability": original["fold_stability"],
        "exploratory_feature_ranking": original["exploratory_feature_ranking"],
    }


def _correction_guards(
    *,
    report: Mapping[str, Any],
    original: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source = report["source_authorization"]
    rank = report["rank_semantics_correction"]
    hypothesis = report["hypothesis_interpretation_correction"]["corrected_assessment"]
    process = report["process_correction"]
    integrity = report["metric_integrity"]
    decision = report["corrected_stage164_decision"]
    closed = report["closed_boundaries"]
    return [
        _check("user_confirmed_option_a", report["user_confirmation"]["confirmed"] is True),
        _check(
            "original_stage164_artifacts_exact",
            source["fingerprints"]["original_stage164_public"]["sha256"] == _ORIGINAL_PUBLIC_SHA256
            and source["fingerprints"]["original_stage164_private"]["sha256"]
            == _ORIGINAL_PRIVATE_BYTE_SHA256
            and source["original_private_canonical_sha256"] == _ORIGINAL_PRIVATE_CANONICAL_SHA256,
        ),
        _check(
            "stage160_hashed_source_exact",
            source["fingerprints"]["stage160_hashed"]["sha256"] == _STAGE160_HASHED_BYTE_SHA256
            and source["stage160_hashed_canonical_sha256"] == _STAGE160_HASHED_CANONICAL_SHA256,
        ),
        _check(
            "original_failure_is_single_known_guard",
            original["decision"]["failed_process_guards"] == [_ORIGINAL_FAILED_GUARD],
        ),
        _check(
            "generation_context_rank_semantics_corrected",
            rank["generation_context_count_minimum"] == 10
            and rank["generation_context_count_maximum"] == 10
            and rank["gold_generation_rank_maximum"] == 14
            and rank["rank_above_ten_count"] == 1,
        ),
        _check(
            "corrected_process_guards_all_pass",
            process["corrected_process_guard_count"] == 16
            and process["corrected_process_guard_pass_count"] == 16
            and any(
                check["name"] == _CORRECTED_GUARD and check["passed"]
                for check in process["corrected_process_guards"]
            ),
        ),
        _check(
            "aggregate_and_fold_interpretation_separated",
            hypothesis["aggregate_visibility_gap_observed"] is True
            and hypothesis["fold_stable_visibility_gap_observed"] is False
            and hypothesis["exact_span_fold_direction_count"] == 2
            and hypothesis["exact_span_fold_opposite_direction_count"] == 3,
        ),
        _check(
            "metric_snapshot_immutable",
            integrity["metric_snapshot_changed"] is False
            and integrity["metric_snapshot_sha256_before_correction"]
            == integrity["metric_snapshot_sha256_after_correction"],
        ),
        _check(
            "corrected_decision_is_diagnostic_only",
            decision["status"] == _CORRECTED_STATUS
            and decision["fold_stable_visibility_gap_observed"] is False
            and decision["policy_selected"] is False
            and decision["agent_rerun"] is False,
        ),
        _check(
            "no_data_feature_agent_or_retrieval_reexecution",
            integrity["development_rows_loaded_during_correction"] == 0
            and integrity["documents_loaded_during_correction"] == 0
            and integrity["feature_rows_recomputed_during_correction"] == 0
            and integrity["agent_runs_during_correction"] == 0
            and integrity["retrieval_runs_during_correction"] == 0,
        ),
        _check(
            "train_dev_test_runtime_and_fallback_closed",
            closed["train_loaded"] is False
            and closed["dev_loaded"] is False
            and closed["test_loaded"] is False
            and closed["agent_run"] is False
            and closed["retrieval_run"] is False
            and closed["policy_fit"] is False
            and closed["threshold_tuned"] is False
            and closed["runtime_registered_as_default"] is False
            and closed["fallback_strategies_enabled"] is False,
        ),
    ]


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _bar(label: str, value: int | float) -> BarDatum:
    return BarDatum(label=label, value=float(value), value_label=str(value))
