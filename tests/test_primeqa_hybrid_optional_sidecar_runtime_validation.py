from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_runtime_validation import (
    _agent_summary_matches,
    _decision,
    _public_safe_contract,
    _runtime_trace_violation_count,
    _RuntimeValidationObservation,
    _source_checks,
    _summarize_split,
    _train_gate_passed,
    write_primeqa_hybrid_optional_sidecar_runtime_validation_visualizations,
)
from ts_rag_agent.domain.answer import AnswerCitation, GeneratedAnswer
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import PrimeQAHybridSplitSample


def test_stage143_source_checks_require_all_frozen_sources() -> None:
    checks = _source_checks(
        stage141=_stage141(),
        stage142=_stage142(),
        stage139=_stage139(),
        user_confirmed_validation=True,
        confirmation_note="User confirmed Stage143 runtime validation.",
    )

    assert len(checks) == 7
    assert all(check["passed"] for check in checks)


def test_stage143_runtime_trace_contract_matches_allowlist() -> None:
    runtime_trace = _runtime_trace()
    entrypoint_trace = {"terminal_state": "complete"}

    assert (
        _runtime_trace_violation_count(
            runtime_trace=runtime_trace,
            entrypoint_trace=entrypoint_trace,
        )
        == 0
    )

    runtime_trace["question_id"] = "private"
    assert (
        _runtime_trace_violation_count(
            runtime_trace=runtime_trace,
            entrypoint_trace=entrypoint_trace,
        )
        == 2
    )


def test_stage143_split_summary_preserves_zero_terminal_count() -> None:
    summary = _summarize_split([_observation()])

    assert summary["row_count"] == 1
    assert summary["terminal_state_counts"] == {"complete": 1, "refuse": 0}
    assert summary["candidate_pool_depth"]["min"] == 400.0
    assert summary["recall"]["hit_counts"]["400"] == 1
    assert summary["verified_gold_citation_count"] == 1
    assert summary["runtime_request_trace_violation_count"] == 0
    assert summary["entrypoint_trace_violation_count"] == 0


def test_stage143_train_gate_requires_five_clean_folds_and_source_parity() -> None:
    train = _summarize_split([_observation()])
    train["row_count"] = 562
    folds = {
        f"fold_{index}": {
            "runtime_request_trace_violation_count": 0,
            "entrypoint_trace_violation_count": 0,
            "exact_five_transition_trace_rate": 1.0,
            "candidate_pool_depth": {"min": 400, "max": 400},
            "strict_retrieval_slo_passed": True,
            "retry_action_count": 0,
            "fallback_action_count": 0,
        }
        for index in range(1, 6)
    }
    expected_agent = {
        "verified_average_token_f1": train["verified_metrics"]["average_token_f1"],
        "verified_gold_citation_count": 1,
        "terminal_state_counts": {"complete": 1, "refuse": 0},
        "exact_five_transition_trace_rate": 1.0,
    }

    assert _train_gate_passed(
        train_report=train,
        train_folds=folds,
        expected_recall=train["recall"]["hit_counts"],
        expected_agent=expected_agent,
    )

    folds["fold_5"]["strict_retrieval_slo_passed"] = False
    assert not _train_gate_passed(
        train_report=train,
        train_folds=folds,
        expected_recall=train["recall"]["hit_counts"],
        expected_agent=expected_agent,
    )


def test_stage143_agent_summary_requires_exact_stage139_values() -> None:
    actual = {
        "verified_metrics": {"average_token_f1": 0.1946},
        "verified_gold_citation_count": 151,
        "terminal_state_counts": {"complete": 560, "refuse": 2},
        "exact_five_transition_trace_rate": 1.0,
    }
    expected = {
        "verified_average_token_f1": 0.1946,
        "verified_gold_citation_count": 151,
        "terminal_state_counts": {"complete": 560, "refuse": 2},
        "exact_five_transition_trace_rate": 1.0,
    }

    assert _agent_summary_matches(actual, expected)
    actual["verified_gold_citation_count"] = 150
    assert not _agent_summary_matches(actual, expected)


def test_stage143_public_contract_rejects_private_keys() -> None:
    assert _public_safe_contract({"unique_answer_doc_ids": 4})["forbidden_keys_found"] == []
    assert _public_safe_contract({"nested": {"sample_id": "private"}})["forbidden_keys_found"] == [
        "sample_id"
    ]


def test_stage143_decision_and_visualizations(tmp_path: Path) -> None:
    checks = [{"name": "all", "passed": True}]
    decision = _decision(checks, train_gate_passed=True)
    report = _visualization_report(decision=decision, checks=checks)

    visualizations = write_primeqa_hybrid_optional_sidecar_runtime_validation_visualizations(
        report=report,
        output_dir=tmp_path,
    )

    assert decision["optional_runtime_activation_validated"] is True
    assert decision["runtime_registered_as_default"] is False
    assert decision["concurrent_runtime_activation_allowed"] is False
    assert decision["test_gate_opened"] is False
    assert len(visualizations) == 7
    for visualization in visualizations:
        ET.parse(visualization.path)


def _observation() -> _RuntimeValidationObservation:
    sample = PrimeQAHybridSplitSample(
        split_name="stage68",
        protocol_version="v1",
        assigned_split="train",
        split_subtype="grouped",
        source_split="train",
        sample_id="private-sample",
        question_id="private-question",
        question_title="adapter token",
        question_text="How do I repair the adapter token?",
        answerable=True,
        answer="Apply the documented adapter token repair.",
        answer_doc_id="doc-400",
        candidate_doc_ids=(),
        start_offset=None,
        end_offset=None,
    )
    answer = GeneratedAnswer(
        question_id=sample.sample_id,
        answer="Apply the documented adapter token repair.",
        citations=[
            AnswerCitation(
                document_id="doc-400",
                title="adapter token",
                retrieval_rank=400,
                evidence_score=1.0,
            )
        ],
        refused=False,
    )
    return _RuntimeValidationObservation(
        sample=sample,
        verified_answer=answer,
        candidate_doc_ids=tuple(f"doc-{index:03d}" for index in range(1, 401)),
        retrieval_latency_seconds=0.05,
        runtime_trace=_runtime_trace(),
        entrypoint_trace={"terminal_state": "complete"},
        runtime_trace_violation_count=0,
        entrypoint_trace_violation_count=0,
    )


def _runtime_trace() -> dict:
    return {
        "runtime_mode": "optional_sidecar_agent_single_request",
        "activation_requested": True,
        "activation_state": "eligible",
        "slo_profile_id": "strict_c_warm_single_request_v1",
        "warm_resources_ready": True,
        "candidate_pool_depth": 400,
        "retrieval_latency_ms": 50.0,
        "latency_budget_passed": True,
        "terminal_state": "complete",
    }


def _stage141() -> dict:
    return {
        "guard_checks": [{"passed": True}] * 19,
        "decision": {"status": "primeqa_hybrid_nondefault_runtime_activation_protocol_frozen"},
        "frozen_protocol": {
            "runtime_interface": {
                "future_environment_flag": "TS_RAG_ENABLE_OPTIONAL_SIDECAR_AGENT",
                "default_value": False,
                "explicit_true_required": True,
            }
        },
    }


def _stage142() -> dict:
    return {
        "guard_checks": [{"passed": True}] * 25,
        "decision": {
            "status": "primeqa_hybrid_strict_warm_latency_validation_passed",
            "can_implement_nondefault_runtime_wiring_now": True,
            "runtime_activated_now": False,
            "concurrent_runtime_activation_allowed": False,
            "runtime_defaultization_allowed_now": False,
        },
        "public_safe_contract": {
            "forbidden_keys_found": [],
            "test_split_loaded": False,
        },
    }


def _stage139() -> dict:
    return {
        "guard_checks": [{"passed": True}] * 45,
        "decision": {
            "status": (
                "primeqa_hybrid_optional_sidecar_agent_entrypoint_train_cv_dev_validation_passed"
            )
        },
        "public_safe_contract": {
            "forbidden_keys_found": [],
            "test_split_loaded": False,
        },
    }


def _visualization_report(*, decision: dict, checks: list[dict]) -> dict:
    latency = {"p95": 0.1, "p99": 0.2}
    split = {
        "retrieval_latency_seconds": latency,
        "recall": {"rates": {str(top_k): 0.9 for top_k in (10, 50, 100, 200, 400)}},
    }
    return {
        "startup_cases": {
            "disabled": {"activation_state": "disabled"},
            "rejected": {"activation_state": "rejected"},
            "eligible": {"activation_state": "eligible"},
        },
        "resource_summary": {
            "dense_model_count": 2,
            "dense_embedding_cache_count": 2,
            "lexical_index_count": 4,
            "derived_route_count": 1,
            "candidate_pool_retriever_instance_count": 1,
            "optional_entrypoint_instance_count": 1,
        },
        "train_runtime_validation": split,
        "dev_runtime_report_only_validation": split,
        "train_fold_reports": {
            f"fold_{index}": {"retrieval_latency_seconds": latency} for index in range(1, 6)
        },
        "decision": decision,
        "guard_checks": checks,
    }
