from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ts_rag_agent.application.primeqa_hybrid_agent_tool_orchestration_protocol import (
    AgentToolOrchestrationProtocolEvidence,
    AgentToolOrchestrationProtocolState,
    AgentToolWorkflowAction,
    AgentToolWorkflowState,
    AgentToolWorkflowStateMachine,
    InvalidAgentToolWorkflowTransitionError,
    StrictAgentToolOrchestrationProtocolPolicy,
    agent_tool_registry_contract,
    agent_tool_workflow_state_contract,
    freeze_primeqa_hybrid_agent_tool_orchestration_protocol,
    write_primeqa_hybrid_agent_tool_orchestration_protocol_visualizations,
)

_ROOT = Path(__file__).resolve().parents[1]
_STAGE152 = _ROOT / "artifacts" / "primeqa_hybrid_agent_service_entrypoint_validation_stage152.json"
_STAGE139 = (
    _ROOT
    / "artifacts"
    / "primeqa_hybrid_optional_sidecar_agent_entrypoint_validation_stage139.json"
)
_PATH_PREFIX = [
    AgentToolWorkflowAction.VALIDATE_REQUEST,
    AgentToolWorkflowAction.RETRIEVE_CANDIDATE_POOL,
    AgentToolWorkflowAction.PREPARE_CONTEXT,
    AgentToolWorkflowAction.COMPOSE_GROUNDED_ANSWER,
    AgentToolWorkflowAction.VERIFY_GROUNDED_ANSWER,
    AgentToolWorkflowAction.OBSERVE_DIAGNOSTICS,
]


def test_state_machine_complete_path_is_exact_and_terminal() -> None:
    machine = AgentToolWorkflowStateMachine()

    trace = machine.run([*_PATH_PREFIX, AgentToolWorkflowAction.COMPLETE])

    assert machine.state is AgentToolWorkflowState.COMPLETE
    assert machine.terminal is True
    assert len(trace) == 7
    assert [row.sequence_number for row in trace] == list(range(1, 8))
    assert [row.action for row in trace] == [*_PATH_PREFIX, AgentToolWorkflowAction.COMPLETE]
    assert machine.public_trace()[-1] == {
        "sequence_number": 7,
        "previous_state": "observed",
        "action": "complete",
        "next_state": "complete",
    }


def test_state_machine_refuse_path_is_exact_and_terminal() -> None:
    machine = AgentToolWorkflowStateMachine()

    machine.run([*_PATH_PREFIX, AgentToolWorkflowAction.REFUSE])

    assert machine.state is AgentToolWorkflowState.REFUSE
    assert machine.terminal is True
    assert len(machine.trace) == 7


@pytest.mark.parametrize(
    "action",
    [
        AgentToolWorkflowAction.RETRIEVE_CANDIDATE_POOL,
        AgentToolWorkflowAction.COMPOSE_GROUNDED_ANSWER,
        AgentToolWorkflowAction.COMPLETE,
        AgentToolWorkflowAction.REFUSE,
    ],
)
def test_invalid_first_action_raises_without_mutation(action: AgentToolWorkflowAction) -> None:
    machine = AgentToolWorkflowStateMachine()

    with pytest.raises(InvalidAgentToolWorkflowTransitionError):
        machine.advance(action)

    assert machine.state is AgentToolWorkflowState.RECEIVED
    assert machine.trace == ()
    assert machine.terminal is False


def test_terminal_state_rejects_further_actions_without_mutation() -> None:
    machine = AgentToolWorkflowStateMachine()
    machine.run([*_PATH_PREFIX, AgentToolWorkflowAction.COMPLETE])
    trace_before = machine.trace

    with pytest.raises(InvalidAgentToolWorkflowTransitionError):
        machine.advance(AgentToolWorkflowAction.RETRIEVE_CANDIDATE_POOL)

    assert machine.state is AgentToolWorkflowState.COMPLETE
    assert machine.trace == trace_before


def test_tool_registry_has_three_sequential_fail_closed_tools() -> None:
    tools = agent_tool_registry_contract()

    assert [tool.tool_id for tool in tools] == [
        "retrieve_candidate_pool",
        "compose_grounded_answer",
        "verify_grounded_answer",
    ]
    assert [tool.successful_path_call_limit for tool in tools] == [1, 1, 1]
    assert all(tool.errors_propagate_unchanged for tool in tools)
    assert not any(tool.retry_allowed for tool in tools)
    assert not any(tool.fallback_allowed for tool in tools)
    assert not any(tool.parallel_call_allowed for tool in tools)


def test_workflow_contract_is_acyclic_and_honestly_classified() -> None:
    contract = agent_tool_workflow_state_contract()

    assert contract["classification"] == "deterministic_tool_workflow_not_autonomous_agent"
    assert contract["nodes"] == [
        "validate_request",
        "retrieve_candidate_pool",
        "prepare_context",
        "compose_grounded_answer",
        "verify_grounded_answer",
        "observe_diagnostics",
        "finalize_response",
    ]
    assert len(contract["states"]) == 9
    assert len(contract["allowed_transitions"]) == 8
    assert contract["successful_transition_count"] == 7
    assert contract["conditional_edge_count"] == 1
    assert contract["conditional_edge_source"] == "observed"
    assert contract["transition_loops_available"] is False
    assert contract["graph_compile_count_per_process"] == 1
    assert contract["request_state_shared_across_invocations"] is False
    assert len(contract["private_state_fields"]) == 13
    assert len(contract["public_trace_fields"]) == 20


def test_strict_policy_accepts_only_compliant_protocol() -> None:
    result = StrictAgentToolOrchestrationProtocolPolicy().evaluate(_compliant_evidence())

    assert result.state is AgentToolOrchestrationProtocolState.ELIGIBLE
    assert result.rejection_reasons == ()


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"source_stage152_validated": False}, "stage152_source_not_validated"),
        ({"workflow_classification_honest": False}, "workflow_classification_misleading"),
        ({"graph_acyclic": False}, "graph_contains_cycle"),
        ({"llm_selects_tools": True}, "llm_tool_selection_not_validated"),
        ({"query_rewrite_enabled": True}, "query_rewrite_not_validated"),
        ({"second_retrieval_enabled": True}, "second_retrieval_not_validated"),
        ({"tool_errors_propagate_unchanged": False}, "tool_errors_not_propagated"),
        ({"sidecar_can_generate_answer": True}, "sidecar_answer_generation_not_allowed"),
        ({"application_waiting_queue_enabled": True}, "application_queue_not_allowed"),
        ({"retry_action_count": 1}, "retry_actions_not_allowed"),
        ({"fallback_action_count": 1}, "fallback_actions_not_allowed"),
        ({"test_gate_opened": True}, "test_gate_not_allowed"),
        ({"workflow_implemented": True}, "workflow_implementation_out_of_scope"),
        ({"langgraph_dependency_installed": True}, "dependency_install_out_of_scope"),
    ],
)
def test_strict_policy_rejects_unauthorized_behavior(
    overrides: dict[str, object],
    reason: str,
) -> None:
    result = StrictAgentToolOrchestrationProtocolPolicy().evaluate(_compliant_evidence(**overrides))

    assert result.state is AgentToolOrchestrationProtocolState.REJECTED
    assert reason in result.rejection_reasons


def test_preflight_fails_only_confirmation_guard_and_does_not_mutate_sources() -> None:
    stage152_before = _STAGE152.read_bytes()
    stage139_before = _STAGE139.read_bytes()

    report = _freeze(confirmed=False, note="test preflight")

    failed = [check["name"] for check in report["guard_checks"] if not check["passed"]]
    assert failed == ["stage153_user_confirmed"]
    assert report["decision"]["protocol_frozen"] is False
    assert report["public_safe_contract"]["questions_loaded"] is False
    assert report["public_safe_contract"]["documents_loaded"] is False
    assert report["public_safe_contract"]["langgraph_dependency_installed"] is False
    assert _STAGE152.read_bytes() == stage152_before
    assert _STAGE139.read_bytes() == stage139_before


def test_formal_protocol_passes_all_guards_and_keeps_boundaries_closed() -> None:
    report = _freeze(confirmed=True, note="user confirmed Stage153 protocol")

    assert len(report["guard_checks"]) == 46
    assert all(check["passed"] for check in report["guard_checks"])
    assert report["decision"]["status"] == (
        "primeqa_hybrid_local_agent_tool_orchestration_protocol_frozen"
    )
    assert report["decision"]["workflow_implementation_allowed_next"] is True
    assert report["decision"]["workflow_implemented"] is False
    assert report["decision"]["langgraph_dependency_installed"] is False
    assert report["decision"]["test_gate_opened"] is False
    assert report["decision"]["runtime_registered_as_default"] is False
    assert report["frozen_protocol"]["execution_contract"]["retry_action_count"] == 0
    assert report["frozen_protocol"]["execution_contract"]["fallback_action_count"] == 0
    assert (
        report["canonical_policy_evaluations"]["exact_deterministic_tool_workflow"]["state"]
        == "eligible"
    )
    assert all(
        row["state"] == "rejected"
        for name, row in report["canonical_policy_evaluations"].items()
        if name != "exact_deterministic_tool_workflow"
    )


def test_tampered_stage152_source_fails_closed(tmp_path: Path) -> None:
    tampered = json.loads(_STAGE152.read_text(encoding="utf-8"))
    tampered["decision"]["status"] = "tampered"
    path = tmp_path / "stage152.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")

    report = freeze_primeqa_hybrid_agent_tool_orchestration_protocol(
        stage152_validation_path=path,
        stage139_validation_path=_STAGE139,
        user_confirmed_protocol=True,
        confirmation_note="test tamper rejection",
    )

    failed = [check["name"] for check in report["guard_checks"] if not check["passed"]]
    assert "stage152_identity_valid" in failed
    assert report["decision"]["protocol_frozen"] is False


def test_visualizations_are_ten_parseable_svg_files(tmp_path: Path) -> None:
    report = _freeze(confirmed=True, note="test visuals")

    visualizations = write_primeqa_hybrid_agent_tool_orchestration_protocol_visualizations(
        report=report,
        output_dir=tmp_path,
    )

    assert len(visualizations) == 10
    for visualization in visualizations:
        assert Path(visualization.path).is_file()
        assert ET.parse(visualization.path).getroot().tag.endswith("svg")


def test_public_protocol_contains_no_private_request_or_document_content() -> None:
    report = _freeze(confirmed=True, note="test public safety")
    serialized = json.dumps(report, sort_keys=True)

    assert "private-question-text" not in serialized
    assert "private-answer-text" not in serialized
    assert "private-document-id" not in serialized
    assert report["public_safe_contract"]["forbidden_keys_found"] == []


def _freeze(*, confirmed: bool, note: str) -> dict:
    return freeze_primeqa_hybrid_agent_tool_orchestration_protocol(
        stage152_validation_path=_STAGE152,
        stage139_validation_path=_STAGE139,
        user_confirmed_protocol=confirmed,
        confirmation_note=note,
    )


def _compliant_evidence(**overrides: object) -> AgentToolOrchestrationProtocolEvidence:
    values: dict[str, object] = {
        "source_stage152_validated": True,
        "source_stage139_validated": True,
        "real_local_service_lifecycle_validated": True,
        "prior_action_order_validated": True,
        "prior_answer_path_invariance_validated": True,
        "workflow_classification_honest": True,
        "tool_registry_exact": True,
        "typed_private_state_exact": True,
        "public_trace_allowlist_exact": True,
        "graph_nodes_exact": True,
        "graph_edges_exact": True,
        "graph_acyclic": True,
        "conditional_routing_terminal_only": True,
        "successful_transition_bound_exact": True,
        "successful_tool_call_bound_exact": True,
        "graph_compiled_once_per_process": True,
        "request_state_isolated": True,
        "tool_calls_sequential": True,
        "llm_selects_tools": False,
        "query_rewrite_enabled": False,
        "second_retrieval_enabled": False,
        "memory_or_checkpointer_enabled": False,
        "streaming_enabled": False,
        "human_interrupt_enabled": False,
        "tool_errors_propagate_unchanged": True,
        "tool_error_messages_public": False,
        "retrieval_owns_candidate_pool": True,
        "generation_uses_stage116_context": True,
        "verification_uses_stage116_prefix": True,
        "sidecar_can_generate_answer": False,
        "sidecar_can_verify_answer": False,
        "sidecar_can_replace_primary_context": False,
        "final_response_uses_verified_answer_only": True,
        "outer_concurrency_limit_preserved": True,
        "application_waiting_queue_enabled": False,
        "in_graph_timeout_enabled": False,
        "retry_action_count": 0,
        "fallback_action_count": 0,
        "runtime_registered_as_default": False,
        "remote_exposure_authorized": False,
        "test_gate_opened": False,
        "test_metrics_run": False,
        "workflow_implemented": False,
        "langgraph_dependency_installed": False,
    }
    values.update(overrides)
    return AgentToolOrchestrationProtocolEvidence(**values)  # type: ignore[arg-type]
