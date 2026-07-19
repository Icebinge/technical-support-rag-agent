from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.primeqa_hybrid_protected_context_selector_training import (
    _canonical_json_sha256,
    _fingerprint,
    _load_json_object,
    _public_safe_contract,
)
from ts_rag_agent.application.primeqa_hybrid_untouched_rrf_dev_validation import (
    _decision as _stage163_decision,
)
from ts_rag_agent.application.primeqa_hybrid_untouched_rrf_dev_validation import (
    _guard_checks as _stage163_guard_checks,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 163 contract correction"
_CREATED_AT = "2026-07-19"
_ANALYSIS_ID = "primeqa_hybrid_untouched_rrf_dev_contract_correction_v1"
_ORIGINAL_REPORT_SHA256 = "66f3b3185d4a7a3447fc9524a78729bc1e307f5feab04b6308268cdb06642e05"
_STAGE160_REPORT_SHA256 = "e17e5fe5bbc5fef4e25e41234e47b89daf19ea4ef18f3c7270601f0fee7d9377"
_STAGE160_PRIVATE_FILE_SHA256 = "3f10cffe245a4405dfc56044f2a3c0d364fdd0f8723e6cc3ae401260199652db"
_STAGE160_PRIVATE_CANONICAL_SHA256 = (
    "1c8aa4260be5427e13322cb3304e518dd3609c2e38f839cda4f10ce01c911a0d"
)
_ORIGINAL_INVALID_STATUS = "primeqa_hybrid_untouched_rrf_dev_validation_invalid"
_CORRECTED_POLICY_STATUS = "primeqa_hybrid_untouched_rrf_not_dev_safe"
_ORIGINAL_FAILED_PROCESS_GUARD = "candidate_pool_exact"
_CORRECTED_PROCESS_GUARD = "stage116_top200_candidate_pool_shape_exact"
_EXPECTED_POLICY_FAILURES = [
    "verified_f1_all_not_below_current",
    "every_fold_f1_not_below_current",
]


@dataclass(frozen=True)
class PrimeQAHybridUntouchedRRFDevCorrectionVisualization:
    """One public-safe Stage163 contract-correction chart."""

    name: str
    path: str


def run_stage163_contract_correction(
    *,
    original_report_path: Path,
    stage160_report_path: Path,
    stage160_private_report_path: Path,
    user_confirmed_correction: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Correct the process-contract audit without loading or reevaluating dev."""

    source_fingerprints = {
        "original_stage163_report": _fingerprint(original_report_path),
        "stage160_public_report": _fingerprint(stage160_report_path),
        "stage160_hashed_diagnostic_report": _fingerprint(stage160_private_report_path),
    }
    _authorize_source_fingerprints(source_fingerprints)
    original = _load_json_object(original_report_path)
    stage160 = _load_json_object(stage160_report_path)
    stage160_private = _load_json_object(stage160_private_report_path)
    _authorize_report_contracts(
        original=original,
        stage160=stage160,
        stage160_private=stage160_private,
    )

    runtime_contract = _summarize_runtime_contract(stage160_private["rows"])
    offline_contract = _summarize_stage163_contract(original)
    original_metric_snapshot = _metric_snapshot(original)
    metric_snapshot_sha256 = _canonical_json_sha256(original_metric_snapshot)
    corrected_process_guards = _stage163_guard_checks(original)
    corrected_stage163_context = {**original, "guard_checks": corrected_process_guards}
    corrected_stage163_decision = _stage163_decision(
        report=corrected_stage163_context,
        process_guards_passed=all(check["passed"] for check in corrected_process_guards),
    )

    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Post-run process-contract correction over the immutable Stage163 public report. "
            "The correction distinguishes the Stage160 runtime Top400 candidate pool from the "
            "Stage163 Stage116 offline Top200 pool. It does not load a split, rebuild retrieval, "
            "reevaluate a case, change a metric, tune a policy, or enable a fallback."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_correction),
            "confirmation_note": confirmation_note,
            "selected_option": "A",
        },
        "source_authorization": {
            "fingerprints": source_fingerprints,
            "stage160_hashed_diagnostic_canonical_sha256": _canonical_json_sha256(stage160_private),
            "stage160_public_declared_canonical_sha256": stage160[
                "private_diagnostic_artifact_contract"
            ]["canonical_content_sha256"],
            "original_stage163_report_remains_canonical": True,
        },
        "contract_evidence": {
            "stage160_runtime": runtime_contract,
            "stage163_offline": offline_contract,
            "comparison": {
                "candidate_pool_depths_are_same": False,
                "stage160_candidate_pool_depth": runtime_contract["candidate_pool_depth_minimum"],
                "stage163_candidate_pool_depth": offline_contract["candidate_pool_depth_minimum"],
                "gold_pool_hit_counts_are_cross_contract_comparable": False,
                "stage160_gold_pool_hit_count": stage160["aggregate_diagnostics"][
                    "answerable_refusal_flow"
                ]["gold_present_candidate_pool_count"],
                "stage163_gold_pool_hit_count": offline_contract["gold_pool_hit_count"],
                "mismatch_classification": (
                    "frozen_process_guard_compared_runtime_top400_with_offline_top200"
                ),
            },
        },
        "correction": {
            "original_guard_name": _ORIGINAL_FAILED_PROCESS_GUARD,
            "original_invalid_condition": (
                "required_stage163_top200_gold_pool_hit_count_to_equal_stage160_top400_count_70"
            ),
            "corrected_guard_name": _CORRECTED_PROCESS_GUARD,
            "corrected_condition": (
                "requires_121_pools_times_200_records_with_exact_top200_depth_only"
            ),
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
            "metric_snapshot_sha256_before_correction": metric_snapshot_sha256,
            "metric_snapshot_sha256_after_correction": _canonical_json_sha256(
                _metric_snapshot(original)
            ),
            "metric_snapshot_changed": False,
            "development_rows_loaded_during_correction": 0,
            "development_cases_reevaluated_during_correction": 0,
            "retrieval_channels_rebuilt_during_correction": False,
            "candidate_pools_rebuilt_during_correction": False,
        },
        "policy_result": {
            "strict_policy_guard_pass_count": sum(original["policy_guard_results"].values()),
            "strict_policy_guard_count": len(original["policy_guard_results"]),
            "failed_policy_guards": original["policy_adoption"]["failed_policy_guards"],
            "candidate_policy_adopted": False,
            "corrected_stage163_decision": corrected_stage163_decision,
        },
        "closed_boundaries": {
            "train_loaded": False,
            "dev_loaded": False,
            "test_loaded": False,
            "dev_reevaluated": False,
            "test_metrics_run": False,
            "policy_tuned": False,
            "runtime_registered_as_default": False,
            "fallback_strategies_enabled": False,
        },
    }
    report["guard_checks"] = _correction_guards(report=report, original=original)
    report["public_safe_contract"] = _public_safe_contract(report)
    all_guards_passed = all(check["passed"] for check in report["guard_checks"])
    report["decision"] = {
        "status": (
            "primeqa_hybrid_stage163_contract_correction_completed"
            if all_guards_passed
            else "primeqa_hybrid_stage163_contract_correction_invalid"
        ),
        "all_correction_guards_passed": all_guards_passed,
        "failed_correction_guards": [
            check["name"] for check in report["guard_checks"] if not check["passed"]
        ],
        "corrected_stage163_status": corrected_stage163_decision["status"],
        "candidate_policy_adopted": False,
        "dev_reevaluated": False,
        "test_gate_opened": False,
        "runtime_registered_as_default": False,
        "fallback_strategies_enabled": False,
        "next_direction": corrected_stage163_decision["next_direction"],
    }
    unchanged_fingerprint = _fingerprint(original_report_path)
    if unchanged_fingerprint != source_fingerprints["original_stage163_report"]:
        raise ValueError("original Stage163 report changed during contract correction")
    report["source_authorization"]["original_stage163_report_unchanged_after_correction"] = True
    return report


def write_stage163_contract_correction_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridUntouchedRRFDevCorrectionVisualization]:
    """Write two public-safe SVGs for the Stage163 contract correction."""

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = report["contract_evidence"]["stage160_runtime"]
    offline = report["contract_evidence"]["stage163_offline"]
    correction = report["correction"]
    policy = report["policy_result"]
    chart_specs = {
        "stage163_contract_depths.svg": (
            "Stage163 corrected candidate-pool contract depths",
            [
                _bar("Stage160 runtime candidate pool", runtime["candidate_pool_depth_minimum"]),
                _bar("Stage160 verification prefix", runtime["verification_context_depth_minimum"]),
                _bar("Stage163 offline candidate pool", offline["candidate_pool_depth_minimum"]),
                _bar("generation context", runtime["generation_context_depth_minimum"]),
            ],
        ),
        "stage163_corrected_gate_status.svg": (
            "Stage163 original, corrected, and policy gate pass counts",
            [
                _bar("original process", correction["original_process_guard_pass_count"]),
                _bar("corrected process", correction["corrected_process_guard_pass_count"]),
                _bar("strict policy", policy["strict_policy_guard_pass_count"]),
            ],
        ),
    }
    artifacts = []
    for filename, (title, bars) in chart_specs.items():
        path = output_dir / filename
        path.write_text(
            render_horizontal_bar_chart_svg(title=title, bars=bars, x_label="passed or depth"),
            encoding="utf-8",
        )
        artifacts.append(
            PrimeQAHybridUntouchedRRFDevCorrectionVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _authorize_source_fingerprints(fingerprints: Mapping[str, Mapping[str, Any]]) -> None:
    expected = {
        "original_stage163_report": _ORIGINAL_REPORT_SHA256,
        "stage160_public_report": _STAGE160_REPORT_SHA256,
        "stage160_hashed_diagnostic_report": _STAGE160_PRIVATE_FILE_SHA256,
    }
    mismatches = {
        name: fingerprints[name]["sha256"]
        for name, sha256 in expected.items()
        if fingerprints[name]["sha256"] != sha256
    }
    if mismatches:
        raise ValueError(f"Stage163 correction source fingerprint mismatch: {mismatches}")


def _authorize_report_contracts(
    *,
    original: Mapping[str, Any],
    stage160: Mapping[str, Any],
    stage160_private: Mapping[str, Any],
) -> None:
    if original.get("decision", {}).get("status") != _ORIGINAL_INVALID_STATUS:
        raise ValueError("Stage163 correction requires the original invalid decision")
    if original.get("decision", {}).get("failed_process_guards") != [
        _ORIGINAL_FAILED_PROCESS_GUARD
    ]:
        raise ValueError("Stage163 correction requires only the known pool-contract failure")
    if original.get("policy_adoption", {}).get("failed_policy_guards") != (
        _EXPECTED_POLICY_FAILURES
    ):
        raise ValueError("Stage163 correction requires the immutable policy failure set")
    declared = stage160.get("private_diagnostic_artifact_contract", {}).get(
        "canonical_content_sha256"
    )
    if declared != _STAGE160_PRIVATE_CANONICAL_SHA256:
        raise ValueError("Stage160 public report declares an unexpected diagnostic hash")
    if _canonical_json_sha256(stage160_private) != declared:
        raise ValueError("Stage160 hashed diagnostic canonical content mismatch")
    private_contract = stage160["private_diagnostic_artifact_contract"]
    if (
        private_contract.get("contains_raw_answer") is not False
        or private_contract.get("contains_raw_document_id") is not False
        or private_contract.get("contains_raw_document_text") is not False
        or private_contract.get("contains_raw_question") is not False
    ):
        raise ValueError("Stage160 diagnostic artifact is not authorized for aggregate audit")


def _summarize_runtime_contract(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Stage160 hashed diagnostic rows must not be empty")
    return {
        "row_count": len(rows),
        "candidate_pool_depth_minimum": min(int(row["candidate_pool_count"]) for row in rows),
        "candidate_pool_depth_maximum": max(int(row["candidate_pool_count"]) for row in rows),
        "verification_context_depth_minimum": min(
            int(row["verification_context_count"]) for row in rows
        ),
        "verification_context_depth_maximum": max(
            int(row["verification_context_count"]) for row in rows
        ),
        "generation_context_depth_minimum": min(
            int(row["generation_context_count"]) for row in rows
        ),
        "generation_context_depth_maximum": max(
            int(row["generation_context_count"]) for row in rows
        ),
        "contains_case_rows_in_public_correction": False,
    }


def _summarize_stage163_contract(original: Mapping[str, Any]) -> dict[str, Any]:
    pool = original["candidate_pool_summary"]
    return {
        "row_count": original["loaded_data_summary"]["dev_row_count"],
        "candidate_record_count": pool["candidate_record_count_in_memory"],
        "candidate_pool_depth_minimum": pool["minimum_pool_depth"],
        "candidate_pool_depth_maximum": pool["maximum_pool_depth"],
        "gold_pool_hit_count": pool["answerable_gold_pool_hit_count"],
        "candidate_pool_source": original["frozen_protocol"]["candidate_pool_source"],
    }


def _metric_snapshot(original: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dev_results": original["dev_results"],
        "policy_comparison": original["policy_comparison"],
        "policy_guard_results": original["policy_guard_results"],
        "policy_adoption": original["policy_adoption"],
    }


def _correction_guards(
    *,
    report: Mapping[str, Any],
    original: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source = report["source_authorization"]
    runtime = report["contract_evidence"]["stage160_runtime"]
    offline = report["contract_evidence"]["stage163_offline"]
    comparison = report["contract_evidence"]["comparison"]
    correction = report["correction"]
    integrity = report["metric_integrity"]
    policy = report["policy_result"]
    boundaries = report["closed_boundaries"]
    return [
        _check("user_confirmed_option_a", report["user_confirmation"]["confirmed"] is True),
        _check(
            "original_stage163_report_exact",
            source["fingerprints"]["original_stage163_report"]["sha256"] == _ORIGINAL_REPORT_SHA256,
        ),
        _check(
            "stage160_sources_exact",
            source["fingerprints"]["stage160_public_report"]["sha256"] == _STAGE160_REPORT_SHA256
            and source["fingerprints"]["stage160_hashed_diagnostic_report"]["sha256"]
            == _STAGE160_PRIVATE_FILE_SHA256
            and source["stage160_hashed_diagnostic_canonical_sha256"]
            == _STAGE160_PRIVATE_CANONICAL_SHA256,
        ),
        _check(
            "stage160_runtime_contract_exact",
            runtime["row_count"] == 121
            and runtime["candidate_pool_depth_minimum"] == 400
            and runtime["candidate_pool_depth_maximum"] == 400
            and runtime["verification_context_depth_minimum"] == 200
            and runtime["verification_context_depth_maximum"] == 200
            and runtime["generation_context_depth_minimum"] == 10
            and runtime["generation_context_depth_maximum"] == 10,
        ),
        _check(
            "stage163_offline_contract_exact",
            offline["row_count"] == 121
            and offline["candidate_record_count"] == 24200
            and offline["candidate_pool_depth_minimum"] == 200
            and offline["candidate_pool_depth_maximum"] == 200
            and offline["candidate_pool_source"] == "stage116_original_rrf_top200",
        ),
        _check(
            "cross_contract_count_rejected",
            comparison["candidate_pool_depths_are_same"] is False
            and comparison["gold_pool_hit_counts_are_cross_contract_comparable"] is False
            and comparison["stage160_candidate_pool_depth"] == 400
            and comparison["stage163_candidate_pool_depth"] == 200,
        ),
        _check(
            "original_failure_is_single_known_guard",
            original["decision"]["failed_process_guards"] == [_ORIGINAL_FAILED_PROCESS_GUARD],
        ),
        _check(
            "corrected_process_guards_all_pass",
            correction["corrected_process_guard_count"] == 17
            and correction["corrected_process_guard_pass_count"] == 17
            and any(
                check["name"] == _CORRECTED_PROCESS_GUARD and check["passed"]
                for check in correction["corrected_process_guards"]
            ),
        ),
        _check(
            "metrics_immutable",
            integrity["metric_snapshot_changed"] is False
            and integrity["metric_snapshot_sha256_before_correction"]
            == integrity["metric_snapshot_sha256_after_correction"],
        ),
        _check(
            "strict_policy_failure_preserved",
            policy["strict_policy_guard_pass_count"] == 6
            and policy["strict_policy_guard_count"] == 8
            and policy["failed_policy_guards"] == _EXPECTED_POLICY_FAILURES
            and policy["candidate_policy_adopted"] is False
            and policy["corrected_stage163_decision"]["status"] == _CORRECTED_POLICY_STATUS,
        ),
        _check(
            "no_split_or_retrieval_reexecution",
            integrity["development_rows_loaded_during_correction"] == 0
            and integrity["development_cases_reevaluated_during_correction"] == 0
            and integrity["retrieval_channels_rebuilt_during_correction"] is False
            and integrity["candidate_pools_rebuilt_during_correction"] is False,
        ),
        _check(
            "train_dev_test_and_runtime_closed",
            boundaries["train_loaded"] is False
            and boundaries["dev_loaded"] is False
            and boundaries["test_loaded"] is False
            and boundaries["dev_reevaluated"] is False
            and boundaries["test_metrics_run"] is False
            and boundaries["policy_tuned"] is False
            and boundaries["runtime_registered_as_default"] is False
            and boundaries["fallback_strategies_enabled"] is False,
        ),
    ]


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _bar(label: str, value: int | float) -> BarDatum:
    return BarDatum(label=label, value=float(value), value_label=str(value))
