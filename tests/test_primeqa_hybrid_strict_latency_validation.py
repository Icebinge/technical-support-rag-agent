from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_strict_latency_validation import (
    _aggregate_train_observations,
    _decision,
    _distribution,
    _forbidden_keys_found,
    _PassObservation,
    _strict_latency_pass,
    _train_gate_passed,
    write_primeqa_hybrid_strict_latency_visualizations,
)


def test_stage142_distribution_includes_linear_interpolated_p99() -> None:
    summary = _distribution([0.1, 0.2, 0.3, 1.0])

    assert summary == {
        "count": 4,
        "min": 0.1,
        "average": 0.4,
        "p50": 0.25,
        "p95": 0.895,
        "p99": 0.979,
        "max": 1.0,
    }


def test_stage142_strict_slo_requires_both_p95_and_p99() -> None:
    assert _strict_latency_pass({"p95": 0.3, "p99": 1.0}) is True
    assert _strict_latency_pass({"p95": 0.300001, "p99": 0.5}) is False
    assert _strict_latency_pass({"p95": 0.2, "p99": 1.000001}) is False
    assert _strict_latency_pass({"p95": 0.2}) is False


def test_stage142_aggregate_requires_every_pass_and_fold() -> None:
    observations = [
        _observation(repetition=1, latencies=(0.1, 0.2)),
        _observation(repetition=2, latencies=(0.2, 0.3)),
        _observation(repetition=3, latencies=(0.1, 0.2)),
    ]

    report = _aggregate_train_observations(observations)

    assert report["measurement_repetitions"] == 3
    assert report["measured_request_count"] == 6
    assert report["all_passes_strict_slo_passed"] is True
    assert report["all_pass_folds_strict_slo_passed"] is True
    assert report["combined_strict_slo_passed"] is True
    assert report["all_combined_folds_strict_slo_passed"] is True
    assert report["total_exact_candidate_pool_identity_violation_count"] == 0


def test_stage142_train_gate_blocks_any_identity_or_latency_failure() -> None:
    observations = [
        _observation(repetition=1, latencies=(0.1, 0.2)),
        _observation(repetition=2, latencies=(0.1, 0.2)),
        _observation(repetition=3, latencies=(0.1, 0.2)),
    ]
    report = _aggregate_train_observations(observations)
    warmup = {
        "candidate_pool_exact_identity_violation_count": 0,
    }

    assert _train_gate_passed(report, warmup) is True

    report["total_exact_candidate_pool_identity_violation_count"] = 1
    assert _train_gate_passed(report, warmup) is False


def test_stage142_decision_never_activates_runtime() -> None:
    decision = _decision(
        [{"name": "all", "passed": True}],
        train_gate_passed=True,
    )

    assert decision["strict_slo_validation_passed"] is True
    assert decision["strict_slo_evidence_state"] == "eligible"
    assert decision["can_implement_nondefault_runtime_wiring_now"] is True
    assert decision["runtime_activation_allowed_now"] is False
    assert decision["runtime_activated_now"] is False
    assert decision["runtime_defaultization_allowed_now"] is False
    assert decision["test_gate_opened"] is False


def test_stage142_decision_blocks_failed_guards() -> None:
    decision = _decision(
        [{"name": "strict_latency", "passed": False}],
        train_gate_passed=False,
    )

    assert decision["strict_slo_validation_passed"] is False
    assert decision["strict_slo_evidence_state"] == "rejected"
    assert decision["failed_checks"] == ["strict_latency"]
    assert decision["can_implement_nondefault_runtime_wiring_now"] is False


def test_stage142_public_safety_checks_exact_keys() -> None:
    assert _forbidden_keys_found({"unique_answer_doc_ids": 4}) == set()
    assert _forbidden_keys_found({"nested": {"sample_id": "private"}}) == {"sample_id"}


def test_stage142_writes_all_visualizations(tmp_path: Path) -> None:
    report = _visualization_report()

    visualizations = write_primeqa_hybrid_strict_latency_visualizations(
        report=report,
        output_dir=tmp_path,
    )

    assert {visualization.name for visualization in visualizations} == {
        "stage142_train_pass_latency_vs_slo.svg",
        "stage142_train_fold_worst_latency.svg",
        "stage142_stage140_latency_comparison.svg",
        "stage142_train_channel_p95_latency.svg",
        "stage142_dev_latency_vs_slo.svg",
        "stage142_decision_flags.svg",
        "stage142_guard_check_status.svg",
    }
    assert all(Path(visualization.path).is_file() for visualization in visualizations)


def _observation(*, repetition: int, latencies: tuple[float, ...]) -> _PassObservation:
    return _PassObservation(
        report={
            "repetition": repetition,
            "row_count": len(latencies),
            "strict_slo_passed": True,
            "all_fold_strict_slo_passed": True,
            "exact_candidate_pool_identity_violation_count": 0,
        },
        latencies=latencies,
        channel_latencies={"channel": latencies},
        fusion_latencies=(0.001,) * len(latencies),
        materialization_latencies=(0.001,) * len(latencies),
        fold_latencies={f"fold_{index}": latencies for index in range(1, 6)},
    )


def _visualization_report() -> dict:
    latency = {"p95": 0.1, "p99": 0.2}
    folds = {
        f"fold_{index}": {"latency_seconds": latency, "strict_slo_passed": True}
        for index in range(1, 6)
    }
    pass_reports = [
        {
            "repetition": index,
            "latency_seconds": latency,
            "fold_reports": folds,
        }
        for index in range(1, 4)
    ]
    return {
        "source_stage140": {
            "latency_by_split": {
                "train": {"p95": 0.45},
                "dev": {"p95": 0.29},
            }
        },
        "train_validation": {
            "pass_reports": pass_reports,
            "combined_latency_seconds": latency,
            "combined_channel_latency_seconds": {"channel": latency},
        },
        "dev_report_only_validation": {"latency_seconds": latency},
        "decision": {
            "strict_slo_validation_passed": True,
            "can_implement_nondefault_runtime_wiring_now": True,
            "runtime_settings_flag_implemented": False,
            "runtime_entrypoint_registered": False,
            "runtime_activation_allowed_now": False,
            "runtime_activated_now": False,
            "concurrent_runtime_activation_allowed": False,
            "runtime_defaultization_allowed_now": False,
            "test_gate_opened": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
        },
        "guard_checks": [{"name": "strict_latency", "passed": True}],
    }
