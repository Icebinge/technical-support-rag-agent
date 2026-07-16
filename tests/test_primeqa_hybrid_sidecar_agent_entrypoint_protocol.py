from __future__ import annotations

import json
from pathlib import Path

import pytest

from ts_rag_agent.application.primeqa_hybrid_sidecar_agent_entrypoint_protocol import (
    FrozenSidecarAgentTransitionPolicy,
    InvalidSidecarAgentTransitionError,
    SidecarAgentAction,
    SidecarAgentActionStateMachine,
    SidecarAgentState,
    _forbidden_keys_found,
    freeze_primeqa_hybrid_sidecar_agent_entrypoint_protocol,
    sidecar_agent_action_state_contract,
    write_primeqa_hybrid_sidecar_agent_entrypoint_protocol_visualizations,
)


def test_state_machine_executes_accepted_and_refused_paths() -> None:
    accepted = SidecarAgentActionStateMachine()
    accepted.run(
        (
            SidecarAgentAction.RETRIEVE,
            SidecarAgentAction.ANSWER,
            SidecarAgentAction.VERIFY,
            SidecarAgentAction.OBSERVE,
            SidecarAgentAction.COMPLETE,
        )
    )
    refused = SidecarAgentActionStateMachine()
    refused.run(
        (
            SidecarAgentAction.RETRIEVE,
            SidecarAgentAction.ANSWER,
            SidecarAgentAction.VERIFY,
            SidecarAgentAction.OBSERVE,
            SidecarAgentAction.REFUSE,
        )
    )

    assert accepted.state is SidecarAgentState.COMPLETE
    assert accepted.terminal is True
    assert [row["action"] for row in accepted.public_trace()] == [
        "retrieve",
        "answer",
        "verify",
        "observe",
        "complete",
    ]
    assert refused.state is SidecarAgentState.REFUSE
    assert refused.terminal is True
    assert refused.public_trace()[-1]["next_state"] == "refuse"


def test_state_machine_rejects_invalid_order_without_state_change() -> None:
    machine = SidecarAgentActionStateMachine()

    with pytest.raises(InvalidSidecarAgentTransitionError, match="not allowed"):
        machine.advance(SidecarAgentAction.ANSWER)

    assert machine.state is SidecarAgentState.READY
    assert machine.trace == ()


def test_terminal_state_rejects_any_additional_action() -> None:
    machine = SidecarAgentActionStateMachine()
    machine.run(
        (
            SidecarAgentAction.RETRIEVE,
            SidecarAgentAction.ANSWER,
            SidecarAgentAction.VERIFY,
            SidecarAgentAction.OBSERVE,
            SidecarAgentAction.REFUSE,
        )
    )

    with pytest.raises(InvalidSidecarAgentTransitionError, match="refuse"):
        machine.advance(SidecarAgentAction.RETRIEVE)

    assert machine.state is SidecarAgentState.REFUSE
    assert len(machine.trace) == 5


def test_transition_policy_is_polymorphically_injectable() -> None:
    machine = SidecarAgentActionStateMachine(FrozenSidecarAgentTransitionPolicy())

    transition = machine.advance(SidecarAgentAction.RETRIEVE)

    assert transition.previous_state is SidecarAgentState.READY
    assert transition.next_state is SidecarAgentState.RETRIEVE


def test_action_state_contract_has_no_retry_loop_or_fallback() -> None:
    contract = sidecar_agent_action_state_contract()

    assert contract["initial_state"] == "ready"
    assert contract["terminal_states"] == ["complete", "refuse"]
    assert len(contract["allowed_transitions"]) == 6
    assert contract["retry_actions_available"] is False
    assert contract["fallback_transitions_available"] is False
    assert contract["transition_loops_available"] is False
    assert not any(
        row["previous_state"] == row["next_state"] for row in contract["allowed_transitions"]
    )


def test_freeze_passes_all_guards_and_keeps_runtime_closed(tmp_path: Path) -> None:
    source = tmp_path / "stage137.json"
    source.write_text(json.dumps(_stage137_report()), encoding="utf-8")

    report = freeze_primeqa_hybrid_sidecar_agent_entrypoint_protocol(
        stage137_validation_path=source,
        user_confirmed_protocol=True,
        confirmation_note="confirmed Stage138 protocol freeze",
    )

    assert report["decision"]["status"] == (
        "primeqa_hybrid_optional_sidecar_agent_entrypoint_protocol_frozen"
    )
    assert report["decision"]["optional_agent_entrypoint_protocol_frozen"] is True
    assert report["decision"]["can_implement_optional_agent_entrypoint_now"] is True
    assert report["decision"]["runtime_entrypoint_implemented"] is False
    assert report["decision"]["runtime_action_order_validated"] is False
    assert report["decision"]["runtime_defaultization_allowed_now"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert len(report["guard_checks"]) == 31
    assert all(check["passed"] for check in report["guard_checks"])
    assert report["public_safe_contract"]["test_split_loaded"] is False
    assert report["public_safe_contract"]["forbidden_keys_found"] == []


def test_freeze_blocks_when_source_invariance_drifts(tmp_path: Path) -> None:
    source_report = _stage137_report()
    source_report["split_agent_reports"]["dev"]["verified_answer_identity_violation_count"] = 1
    source = tmp_path / "stage137-drift.json"
    source.write_text(json.dumps(source_report), encoding="utf-8")

    report = freeze_primeqa_hybrid_sidecar_agent_entrypoint_protocol(
        stage137_validation_path=source,
        user_confirmed_protocol=True,
        confirmation_note="confirmed drift check",
    )

    assert report["decision"]["optional_agent_entrypoint_protocol_frozen"] is False
    assert (
        "stage137_answer_path_identity_violations_are_zero" in report["decision"]["failed_checks"]
    )


def test_freeze_requires_explicit_confirmation(tmp_path: Path) -> None:
    source = tmp_path / "stage137.json"
    source.write_text(json.dumps(_stage137_report()), encoding="utf-8")

    report = freeze_primeqa_hybrid_sidecar_agent_entrypoint_protocol(
        stage137_validation_path=source,
        user_confirmed_protocol=False,
        confirmation_note="not confirmed",
    )

    assert report["decision"]["optional_agent_entrypoint_protocol_frozen"] is False
    assert "user_confirmed_stage138_protocol" in report["decision"]["failed_checks"]


def test_visualizations_are_written_as_svg(tmp_path: Path) -> None:
    source = tmp_path / "stage137.json"
    source.write_text(json.dumps(_stage137_report()), encoding="utf-8")
    report = freeze_primeqa_hybrid_sidecar_agent_entrypoint_protocol(
        stage137_validation_path=source,
        user_confirmed_protocol=True,
        confirmation_note="confirmed visualization test",
    )

    artifacts = write_primeqa_hybrid_sidecar_agent_entrypoint_protocol_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    assert len(artifacts) == 6
    assert all(
        Path(artifact.path).read_text(encoding="utf-8").startswith("<svg") for artifact in artifacts
    )


def test_forbidden_key_scanner_detects_private_fields() -> None:
    assert _forbidden_keys_found({"nested": [{"question_text": "private"}]}) == {"question_text"}


def _stage137_report() -> dict[str, object]:
    split_reports = {
        "train": _split_report(rows=562, answerable=370, opportunities=9, observations=1686),
        "dev": _split_report(rows=121, answerable=76, opportunities=1, observations=363),
    }
    return {
        "stage": "Stage 137",
        "analysis_id": "primeqa_hybrid_sidecar_agent_orchestrator_train_cv_dev_validation_v1",
        "guard_checks": [{"name": f"guard_{index}", "passed": True} for index in range(36)],
        "split_agent_reports": split_reports,
        "decision": {
            "status": "primeqa_hybrid_sidecar_agent_orchestrator_train_cv_dev_validation_passed",
            "agent_orchestrator_integration_validated": True,
            "can_freeze_optional_agent_entrypoint_protocol_now": True,
            "sidecar_effectiveness_status": "safe_but_neutral",
            "sidecar_citation_verification_effectiveness_demonstrated": False,
            "can_claim_answer_quality_improvement": False,
            "can_claim_retrieval_improvement": False,
            "sidecar_can_generate_answer_text": False,
            "sidecar_can_enter_answer_verification_context": False,
            "sidecar_can_replace_primary_context": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "runtime_defaultization_allowed_now": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
        "public_safe_contract": {
            "test_split_loaded": False,
            "final_test_metrics_run": False,
            "forbidden_keys_found": [],
        },
    }


def _split_report(
    *, rows: int, answerable: int, opportunities: int, observations: int
) -> dict[str, object]:
    return {
        "row_count": rows,
        "answerable_count": answerable,
        "generation_context_identity_violation_count": 0,
        "verification_context_identity_violation_count": 0,
        "bundle_generation_context_identity_violation_count": 0,
        "original_answer_identity_violation_count": 0,
        "verified_answer_identity_violation_count": 0,
        "verification_reason_identity_violation_count": 0,
        "sidecar_generation_leak_count": 0,
        "sidecar_verification_leak_count": 0,
        "sidecar_primary_overlap_count": 0,
        "public_trace_serialization_violation_count": 0,
        "public_trace_forbidden_key_count": 0,
        "public_trace_contract_violation_count": 0,
        "sidecar_observation_count": observations,
        "sidecar_observation_availability_rate": 1.0,
        "append_pool_incremental_gold_hit_count": opportunities,
        "sidecar_incremental_gold_hit_count": 0,
        "sidecar_capture_rate_of_append_gold_opportunities": 0.0,
    }
