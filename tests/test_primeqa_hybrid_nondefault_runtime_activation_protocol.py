import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_nondefault_runtime_activation_protocol import (
    RuntimeActivationEvidence,
    RuntimeActivationState,
    StrictNonDefaultRuntimeActivationPolicy,
    _forbidden_keys_found,
    freeze_primeqa_hybrid_nondefault_runtime_activation_protocol,
    write_primeqa_hybrid_nondefault_runtime_activation_protocol_visualizations,
)


def test_activation_policy_remains_disabled_without_explicit_request() -> None:
    evaluation = StrictNonDefaultRuntimeActivationPolicy().evaluate(
        _compliant_evidence(explicit_activation_requested=False)
    )

    assert evaluation.state is RuntimeActivationState.DISABLED
    assert evaluation.rejection_reasons == ("explicit_activation_not_requested",)
    assert evaluation.runtime_activated is False


def test_activation_policy_rejects_missing_p99_and_strict_train_p95_failure() -> None:
    evaluation = StrictNonDefaultRuntimeActivationPolicy().evaluate(
        RuntimeActivationEvidence(
            **{
                **_compliant_evidence().__dict__,
                "train_all_folds_pass": False,
                "train_p95_seconds": 0.450798,
                "train_p99_seconds": None,
                "dev_report_only_pass": False,
                "dev_p99_seconds": None,
            }
        )
    )

    assert evaluation.state is RuntimeActivationState.REJECTED
    assert "train_fold_strict_slo_not_passed" in evaluation.rejection_reasons
    assert "train_p95_exceeds_strict_slo" in evaluation.rejection_reasons
    assert "train_p99_missing" in evaluation.rejection_reasons
    assert "dev_p99_missing" in evaluation.rejection_reasons
    assert evaluation.runtime_activated is False


def test_activation_policy_marks_exact_thresholds_eligible_without_activating() -> None:
    evaluation = StrictNonDefaultRuntimeActivationPolicy().evaluate(_compliant_evidence())

    assert evaluation.state is RuntimeActivationState.ELIGIBLE
    assert evaluation.rejection_reasons == ()
    assert evaluation.runtime_activated is False


def test_activation_policy_rejects_concurrent_request_scope() -> None:
    evaluation = StrictNonDefaultRuntimeActivationPolicy().evaluate(
        _compliant_evidence(concurrent_request_support_requested=True)
    )

    assert evaluation.state is RuntimeActivationState.REJECTED
    assert evaluation.rejection_reasons == (
        "concurrent_runtime_not_authorized_by_single_request_protocol",
    )


def test_stage141_freeze_passes_while_current_source_remains_ineligible(tmp_path: Path) -> None:
    source_path = tmp_path / "stage140.json"
    source_path.write_text(json.dumps(_stage140_source()), encoding="utf-8")

    report = freeze_primeqa_hybrid_nondefault_runtime_activation_protocol(
        stage140_validation_path=source_path,
        user_confirmed_protocol=True,
        confirmation_note="User selected strict C.",
        selected_slo_profile_id="strict_c_warm_single_request_v1",
    )

    assert all(check["passed"] for check in report["guard_checks"])
    assert report["decision"]["runtime_activation_protocol_frozen"] is True
    assert report["decision"]["strict_slo_currently_satisfied"] is False
    assert report["decision"]["runtime_activation_allowed_now"] is False
    assert report["decision"]["runtime_entrypoint_registered"] is False
    assert (
        report["canonical_activation_evaluations"]["stage140_source_requested_now"]["state"]
        == "rejected"
    )
    assert report["public_safe_contract"]["forbidden_keys_found"] == []


def test_stage141_freeze_blocks_unconfirmed_or_different_slo_profile(tmp_path: Path) -> None:
    source_path = tmp_path / "stage140.json"
    source_path.write_text(json.dumps(_stage140_source()), encoding="utf-8")

    report = freeze_primeqa_hybrid_nondefault_runtime_activation_protocol(
        stage140_validation_path=source_path,
        user_confirmed_protocol=False,
        confirmation_note="",
        selected_slo_profile_id="lenient_profile",
    )

    assert report["decision"]["runtime_activation_protocol_frozen"] is False
    assert "user_confirmed_strict_c_protocol" in report["decision"]["failed_checks"]


def test_stage141_freeze_blocks_source_identity_or_recall_drift(tmp_path: Path) -> None:
    source = _stage140_source()
    source["split_reports"]["dev"]["exact_candidate_pool_identity_violation_count"] = 1
    source_path = tmp_path / "stage140.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    report = freeze_primeqa_hybrid_nondefault_runtime_activation_protocol(
        stage140_validation_path=source_path,
        user_confirmed_protocol=True,
        confirmation_note="User selected strict C.",
        selected_slo_profile_id="strict_c_warm_single_request_v1",
    )

    assert report["decision"]["runtime_activation_protocol_frozen"] is False
    assert "stage140_identity_and_recall_preserved" in report["decision"]["failed_checks"]


def test_stage141_public_safety_checks_exact_keys() -> None:
    assert _forbidden_keys_found({"unique_answer_doc_ids": 5}) == set()
    assert _forbidden_keys_found({"nested": {"sample_id": "private"}}) == {"sample_id"}


def test_stage141_writes_all_visualizations(tmp_path: Path) -> None:
    source_path = tmp_path / "stage140.json"
    source_path.write_text(json.dumps(_stage140_source()), encoding="utf-8")
    report = freeze_primeqa_hybrid_nondefault_runtime_activation_protocol(
        stage140_validation_path=source_path,
        user_confirmed_protocol=True,
        confirmation_note="User selected strict C.",
        selected_slo_profile_id="strict_c_warm_single_request_v1",
    )

    visualizations = write_primeqa_hybrid_nondefault_runtime_activation_protocol_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    assert {visualization.name for visualization in visualizations} == {
        "stage141_source_p95_vs_strict_slo.svg",
        "stage141_percentile_evidence_availability.svg",
        "stage141_activation_case_states.svg",
        "stage141_runtime_permission_flags.svg",
        "stage141_guard_check_status.svg",
    }
    assert all(Path(visualization.path).is_file() for visualization in visualizations)


def _compliant_evidence(**overrides: object) -> RuntimeActivationEvidence:
    values = {
        "explicit_activation_requested": True,
        "concurrent_request_support_requested": False,
        "source_performance_validated": True,
        "warm_resources_ready": True,
        "candidate_pool_identity_preserved": True,
        "retrieval_recall_preserved": True,
        "train_fold_count": 5,
        "train_all_folds_pass": True,
        "train_p95_seconds": 0.3,
        "train_p99_seconds": 1.0,
        "dev_report_only_pass": True,
        "dev_p95_seconds": 0.3,
        "dev_p99_seconds": 1.0,
        "test_split_locked": True,
    }
    values.update(overrides)
    return RuntimeActivationEvidence(**values)  # type: ignore[arg-type]


def _stage140_source() -> dict:
    guards = [{"name": f"guard_{index}", "passed": True} for index in range(21)]
    return {
        "stage": "Stage 140",
        "analysis_id": "primeqa_hybrid_online_candidate_pool_performance_validation_v1",
        "analysis_scope": {
            "test_split_loaded": False,
            "test_metrics_run": False,
        },
        "selected_candidate_pool_contract": {
            "config_id": "prefix_existing_dense_broad_append200_v1",
            "channel_top_k": 400,
            "prefix_depth": 200,
            "target_pool_depth": 400,
            "rrf_k": 60,
            "channel_count": 7,
            "independent_channel_count": 6,
            "derived_channel_count": 1,
            "indexes_owned_outside_request_path": True,
            "query_specific_candidate_pool_built_per_request": True,
        },
        "split_reports": {
            "train": _split_source(562, 0.222661, 0.450798, 5.051437, fold_count=5),
            "dev": _split_source(121, 0.185714, 0.293909, 2.751323, fold_count=0),
        },
        "guard_checks": guards,
        "decision": {
            "status": "primeqa_hybrid_online_candidate_pool_performance_validation_passed",
            "online_candidate_pool_implementation_validated": True,
            "candidate_pool_identity_preserved": True,
            "retrieval_recall_preserved": True,
            "runtime_activation_allowed_now": False,
            "runtime_defaultization_allowed_now": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
        "public_safe_contract": {"forbidden_keys_found": []},
    }


def _split_source(
    row_count: int,
    p50: float,
    p95: float,
    maximum: float,
    *,
    fold_count: int,
) -> dict:
    return {
        "row_count": row_count,
        "exact_candidate_pool_identity_violation_count": 0,
        "candidate_pool_size": {"min": 400.0, "max": 400.0},
        "latency_seconds": {"p50": p50, "p95": p95, "max": maximum},
        "recall": {"hit_counts": {"10": 1, "50": 1, "100": 1, "200": 1, "400": 1}},
        "fold_reports": {f"fold_{index}": {} for index in range(fold_count)},
    }
