from __future__ import annotations

from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_sidecar_agent_orchestrator_validation import (
    _check,
    _decision,
    _forbidden_keys_found,
    _summarize_agent_traces,
    _trace_agent_validation,
    _train_cv_validation,
    create_sidecar_agent_validation_harness,
    create_stage116_control_runner,
    write_primeqa_hybrid_sidecar_agent_orchestrator_validation_visualizations,
)
from ts_rag_agent.domain.dataset import PrimeQADocument
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
)


def test_harness_records_exact_stage116_contexts_and_answers() -> None:
    trace = _trace(answer_doc_id="append-special")

    assert trace.generation_context_identity_violation is False
    assert trace.verification_context_identity_violation is False
    assert trace.bundle_generation_context_identity_violation is False
    assert trace.original_answer_identity_violation is False
    assert trace.verified_answer_identity_violation is False
    assert trace.verification_reason_identity_violation is False


def test_harness_keeps_sidecar_out_of_answer_paths() -> None:
    trace = _trace(answer_doc_id="append-special")

    assert trace.sidecar_generation_leak_count == 0
    assert trace.sidecar_verification_leak_count == 0
    assert trace.sidecar_primary_overlap_count == 0
    assert trace.public_trace_serialization_violation is False
    assert trace.public_trace_forbidden_key_count == 0
    assert trace.public_trace_contract_violation_count == 0


def test_offline_trace_records_append_opportunity_and_sidecar_capture() -> None:
    summary = _summarize_agent_traces([_trace(answer_doc_id="append-special")])

    assert summary["append_pool_incremental_gold_hit_count"] == 1
    assert summary["sidecar_incremental_gold_hit_count"] == 1
    assert summary["sidecar_capture_rate_of_append_gold_opportunities"] == 1.0


def test_split_summary_reports_zero_control_deltas() -> None:
    summary = _summarize_agent_traces(
        [_trace(answer_doc_id="append-special"), _trace(answer_doc_id="prefix-001")]
    )

    assert summary["generation_context_identity_violation_count"] == 0
    assert summary["verified_answer_identity_violation_count"] == 0
    assert all(value == 0.0 for value in summary["verified_metric_deltas_vs_stage116"].values())
    assert (
        summary["agent_verified_gold_citation_count"]
        == summary["control_verified_gold_citation_count"]
    )


def test_train_cv_requires_every_fold_to_preserve_identity() -> None:
    summary = _summarize_agent_traces([_trace(answer_doc_id="append-special")])
    fold = {
        "fold_id": "fold_1",
        "row_count": 1,
        "generation_context_identity_violation_count": 0,
        "verification_context_identity_violation_count": 0,
        "original_answer_identity_violation_count": 0,
        "verified_answer_identity_violation_count": 0,
        "sidecar_answer_path_leak_count": 0,
        "public_trace_violation_count": 0,
        "sidecar_observation_availability_rate": 1.0,
        "row_query_overlap_signal_coverage": 1.0,
        "row_novel_query_coverage_signal_coverage": 1.0,
    }

    passed = _train_cv_validation(train_summary=summary, fold_reports=[fold])
    failed = _train_cv_validation(
        train_summary=summary,
        fold_reports=[{**fold, "verified_answer_identity_violation_count": 1}],
    )

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert "all_folds_verified_answers_identical" in failed["failed_checks"]


def test_decision_opens_only_optional_entrypoint_protocol() -> None:
    decision = _decision(
        [_check(name="integrity", passed=True, observed=0, expected=0)],
        split_reports={
            "train": {"sidecar_incremental_gold_hit_count": 0},
            "dev": {"sidecar_incremental_gold_hit_count": 0},
        },
    )

    assert decision["agent_orchestrator_integration_validated"] is True
    assert decision["sidecar_effectiveness_status"] == "safe_but_neutral"
    assert decision["can_freeze_optional_agent_entrypoint_protocol_now"] is True
    assert decision["can_run_final_test_metrics_now"] is False
    assert decision["runtime_defaultization_allowed_now"] is False
    assert decision["fallback_strategies_enabled"] is False


def test_decision_blocks_next_protocol_when_guard_fails() -> None:
    decision = _decision([_check(name="identity", passed=False, observed=1, expected=0)])

    assert decision["agent_orchestrator_integration_validated"] is False
    assert decision["can_freeze_optional_agent_entrypoint_protocol_now"] is False
    assert decision["failed_checks"] == ["identity"]


def test_public_safe_detector_rejects_private_keys() -> None:
    assert _forbidden_keys_found({"metric": {"count": 1}}) == set()
    assert _forbidden_keys_found({"runtime_content_handle": "private"}) == {
        "runtime_content_handle"
    }


def test_visualizations_write_seven_public_safe_svgs(tmp_path: Path) -> None:
    split_summary = _summarize_agent_traces([_trace(answer_doc_id="append-special")])
    report = {
        "split_agent_reports": {"train": split_summary, "dev": split_summary},
        "train_fold_reports": [
            {
                "fold_id": "fold_1",
                "generation_context_identity_violation_count": 0,
                "verification_context_identity_violation_count": 0,
                "original_answer_identity_violation_count": 0,
                "verified_answer_identity_violation_count": 0,
            }
        ],
        "guard_checks": [_check(name="identity", passed=True, observed=0, expected=0)],
        "decision": _decision(
            [_check(name="identity", passed=True, observed=0, expected=0)],
            split_reports={
                "train": {"sidecar_incremental_gold_hit_count": 0},
                "dev": {"sidecar_incremental_gold_hit_count": 0},
            },
        ),
    }

    artifacts = write_primeqa_hybrid_sidecar_agent_orchestrator_validation_visualizations(
        report,
        tmp_path,
    )

    assert len(artifacts) == 7
    assert all(Path(artifact.path).exists() for artifact in artifacts)


def _trace(*, answer_doc_id: str):
    documents = _documents()
    return _trace_agent_validation(
        sample=_sample(answer_doc_id=answer_doc_id),
        pool={
            "prefix_pool": [f"prefix-{index:03d}" for index in range(1, 201)],
            "stage128_pool": [
                *[f"prefix-{index:03d}" for index in range(1, 201)],
                "append-special",
                "append-novel",
                "append-ordinary",
            ],
        },
        documents_by_id=documents,
        control_runner=create_stage116_control_runner(),
        agent_harness=create_sidecar_agent_validation_harness(),
    )


def _sample(*, answer_doc_id: str) -> PrimeQAHybridSplitSample:
    return PrimeQAHybridSplitSample(
        split_name="primeqa_hybrid_stage68_v1",
        protocol_version="primeqa_hybrid_split_v1",
        assigned_split="train",
        split_subtype="grouped_random",
        source_split="train",
        sample_id=f"sample-{answer_doc_id}",
        question_id=f"question-{answer_doc_id}",
        question_title="adapter installation",
        question_text="How do I fix special adapter token failure?",
        answerable=True,
        answer="Use the adapter technote.",
        answer_doc_id=answer_doc_id,
        candidate_doc_ids=(),
        start_offset=None,
        end_offset=None,
    )


def _documents() -> dict[str, PrimeQADocument]:
    documents = {
        f"prefix-{index:03d}": PrimeQADocument(
            id=f"prefix-{index:03d}",
            title="adapter reference",
            text=f"ordinary adapter installation reference {index}",
        )
        for index in range(1, 201)
    }
    documents.update(
        {
            "append-special": PrimeQADocument(
                id="append-special",
                title="special adapter token failure",
                text="fix special adapter token failure using the technote",
            ),
            "append-novel": PrimeQADocument(
                id="append-novel",
                title="adapter installation failure",
                text="special token installation workaround",
            ),
            "append-ordinary": PrimeQADocument(
                id="append-ordinary",
                title="adapter guide",
                text="adapter setup reference",
            ),
        }
    )
    return documents
