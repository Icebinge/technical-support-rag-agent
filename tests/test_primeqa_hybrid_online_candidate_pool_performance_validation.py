from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_online_candidate_pool_performance_validation import (
    _decision,
    _distribution,
    _forbidden_keys_found,
    _post_checks,
    write_primeqa_hybrid_online_candidate_pool_performance_visualizations,
)


def test_stage140_post_checks_require_identity_recall_and_locked_boundaries() -> None:
    split_reports = {
        "train": _split_report(row_count=2),
        "dev": _split_report(row_count=1),
    }
    source_recall = {
        split: {"hit_counts": report["recall"]["hit_counts"]}
        for split, report in split_reports.items()
    }
    stage139 = {"timing_seconds": {"build_candidate_pools": 30.0}}

    checks = _post_checks(
        split_reports=split_reports,
        source_recall=source_recall,
        stage139=stage139,
    )
    decision = _decision(checks)

    assert all(check["passed"] for check in checks)
    assert decision["online_candidate_pool_implementation_validated"] is True
    assert decision["runtime_activation_allowed_now"] is False
    assert decision["latency_slo_user_confirmed"] is False
    assert decision["test_gate_opened"] is False
    assert decision["fallback_strategies_enabled"] is False


def test_stage140_post_checks_block_candidate_identity_drift() -> None:
    split_reports = {
        "train": _split_report(row_count=2),
        "dev": {
            **_split_report(row_count=1),
            "exact_candidate_pool_identity_violation_count": 1,
        },
    }
    source_recall = {
        split: {"hit_counts": report["recall"]["hit_counts"]}
        for split, report in split_reports.items()
    }

    checks = _post_checks(
        split_reports=split_reports,
        source_recall=source_recall,
        stage139={"timing_seconds": {"build_candidate_pools": 30.0}},
    )
    decision = _decision(checks)

    assert "dev_candidate_pool_exact_identity" in decision["failed_checks"]
    assert decision["online_candidate_pool_implementation_validated"] is False
    assert decision["runtime_activation_allowed_now"] is False


def test_stage140_public_safety_checks_exact_keys_not_aggregate_names() -> None:
    safe = {"unique_answer_doc_ids": 12, "nested": [{"row_count": 2}]}
    unsafe = {"nested": [{"answer_doc_id": "private"}]}

    assert _forbidden_keys_found(safe) == set()
    assert _forbidden_keys_found(unsafe) == {"answer_doc_id"}


def test_stage140_distribution_uses_all_rows() -> None:
    summary = _distribution([0.1, 0.2, 0.3, 1.0])

    assert summary == {
        "count": 4,
        "min": 0.1,
        "average": 0.4,
        "p50": 0.25,
        "p95": 0.895,
        "max": 1.0,
    }


def test_stage140_writes_all_public_safe_visualizations(tmp_path: Path) -> None:
    report = {
        "source_stage139": {"candidate_pool_build_seconds": 30.0},
        "timing_seconds": {"run_online_retriever_and_compare_all_rows": 1.0},
        "split_reports": {
            "train": _split_report(row_count=2),
            "dev": _split_report(row_count=1),
        },
        "guard_checks": [
            {"name": "identity", "passed": True},
            {"name": "recall", "passed": True},
        ],
    }

    artifacts = write_primeqa_hybrid_online_candidate_pool_performance_visualizations(
        report=report,
        output_dir=tmp_path,
    )

    assert {artifact.name for artifact in artifacts} == {
        "stage140_candidate_pool_wall_time.svg",
        "stage140_online_latency_distribution.svg",
        "stage140_train_channel_p95_latency.svg",
        "stage140_recall_at_k.svg",
        "stage140_guard_check_status.svg",
    }
    assert all(Path(artifact.path).is_file() for artifact in artifacts)


def _split_report(*, row_count: int) -> dict:
    hit_counts = {"10": 1, "50": 1, "100": 1, "200": 1, "400": 1}
    hit_at_k = {key: 1.0 for key in hit_counts}
    return {
        "row_count": row_count,
        "exact_candidate_pool_identity_violation_count": 0,
        "candidate_pool_size": {"min": 400.0, "max": 400.0},
        "latency_seconds": {
            "average": 0.2,
            "p50": 0.1,
            "p95": 0.3,
            "max": 0.4,
        },
        "channel_latency_seconds": {
            "full_document_bm25": {"p95": 0.1},
        },
        "recall": {"hit_counts": hit_counts, "hit_at_k": hit_at_k},
    }
