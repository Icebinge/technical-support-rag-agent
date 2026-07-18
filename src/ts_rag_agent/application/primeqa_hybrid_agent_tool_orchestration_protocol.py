from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Protocol

from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    _forbidden_keys_found,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 153"
_CREATED_AT = "2026-07-18"
AGENT_TOOL_ORCHESTRATION_PROTOCOL_ID = "primeqa_hybrid_local_agent_tool_orchestration_protocol_v1"
AGENT_TOOL_WORKFLOW_GRAPH_ID = "primeqa_hybrid_deterministic_tool_workflow_v1"
_PROTOCOL_ID = AGENT_TOOL_ORCHESTRATION_PROTOCOL_ID
_GRAPH_ID = AGENT_TOOL_WORKFLOW_GRAPH_ID
_SOURCE_STAGE152_ANALYSIS_ID = "primeqa_hybrid_local_agent_service_entrypoint_validation_v1"
_SOURCE_STAGE152_STATUS = "primeqa_hybrid_local_agent_service_entrypoint_implemented_and_validated"
_SOURCE_STAGE139_ANALYSIS_ID = (
    "primeqa_hybrid_optional_sidecar_agent_entrypoint_train_cv_dev_validation_v1"
)
_SOURCE_STAGE139_STATUS = (
    "primeqa_hybrid_optional_sidecar_agent_entrypoint_train_cv_dev_validation_passed"
)
_EXPECTED_STAGE152_GUARDS = 46
_EXPECTED_STAGE139_GUARDS = 45
_FINAL_STATUS = "primeqa_hybrid_local_agent_tool_orchestration_protocol_frozen"
_NEXT_DIRECTION = "implement_deterministic_agent_tool_workflow_and_langgraph_adapter"
_TOOL_IDS = (
    "retrieve_candidate_pool",
    "compose_grounded_answer",
    "verify_grounded_answer",
)
_NODE_IDS = (
    "validate_request",
    "retrieve_candidate_pool",
    "prepare_context",
    "compose_grounded_answer",
    "verify_grounded_answer",
    "observe_diagnostics",
    "finalize_response",
)
_PRIVATE_STATE_FIELDS = (
    "request_handle",
    "runtime_query",
    "candidate_pool_results",
    "generation_context_results",
    "verification_context_results",
    "sidecar_observation_bundle",
    "original_answer",
    "verification_result",
    "terminal_response",
    "current_state",
    "visited_states",
    "tool_call_counts",
    "failure_stage",
)
_PUBLIC_TRACE_FIELDS = (
    "protocol_id",
    "graph_id",
    "terminal_state",
    "transition_count",
    "tool_call_count",
    "retrieval_tool_call_count",
    "answer_tool_call_count",
    "verification_tool_call_count",
    "candidate_pool_depth",
    "generation_context_count",
    "verification_context_count",
    "sidecar_observation_count",
    "verified_refused",
    "verified_citation_count",
    "citation_context_valid",
    "diagnostics_observed",
    "failure_stage",
    "queue_action_count",
    "retry_action_count",
    "fallback_action_count",
)


class AgentToolWorkflowState(str, Enum):
    """Frozen request-local states for the Stage153 workflow."""

    RECEIVED = "received"
    VALIDATED = "validated"
    RETRIEVED = "retrieved"
    CONTEXT_PREPARED = "context_prepared"
    ANSWERED = "answered"
    VERIFIED = "verified"
    OBSERVED = "observed"
    COMPLETE = "complete"
    REFUSE = "refuse"


class AgentToolWorkflowAction(str, Enum):
    """Actions allowed to advance the Stage153 workflow."""

    VALIDATE_REQUEST = "validate_request"
    RETRIEVE_CANDIDATE_POOL = "retrieve_candidate_pool"
    PREPARE_CONTEXT = "prepare_context"
    COMPOSE_GROUNDED_ANSWER = "compose_grounded_answer"
    VERIFY_GROUNDED_ANSWER = "verify_grounded_answer"
    OBSERVE_DIAGNOSTICS = "observe_diagnostics"
    COMPLETE = "complete"
    REFUSE = "refuse"


class InvalidAgentToolWorkflowTransitionError(ValueError):
    """Raised before mutation when an action is not allowed."""


@dataclass(frozen=True)
class AgentToolWorkflowTransition:
    sequence_number: int
    previous_state: AgentToolWorkflowState
    action: AgentToolWorkflowAction
    next_state: AgentToolWorkflowState

    def to_public_dict(self) -> dict[str, Any]:
        payload = {
            "sequence_number": self.sequence_number,
            "previous_state": self.previous_state.value,
            "action": self.action.value,
            "next_state": self.next_state.value,
        }
        forbidden = sorted(_forbidden_keys_found(payload))
        if forbidden:
            raise ValueError(f"workflow transition contains forbidden keys: {forbidden}")
        return payload


class AgentToolWorkflowTransitionPolicy(Protocol):
    def allowed_transitions(
        self,
    ) -> tuple[
        tuple[AgentToolWorkflowState, AgentToolWorkflowAction, AgentToolWorkflowState], ...
    ]: ...

    def next_state(
        self,
        *,
        current_state: AgentToolWorkflowState,
        action: AgentToolWorkflowAction,
    ) -> AgentToolWorkflowState: ...


class FrozenAgentToolWorkflowTransitionPolicy:
    """Acyclic workflow with one conditional terminal branch and no recovery edge."""

    _TRANSITIONS = (
        (
            AgentToolWorkflowState.RECEIVED,
            AgentToolWorkflowAction.VALIDATE_REQUEST,
            AgentToolWorkflowState.VALIDATED,
        ),
        (
            AgentToolWorkflowState.VALIDATED,
            AgentToolWorkflowAction.RETRIEVE_CANDIDATE_POOL,
            AgentToolWorkflowState.RETRIEVED,
        ),
        (
            AgentToolWorkflowState.RETRIEVED,
            AgentToolWorkflowAction.PREPARE_CONTEXT,
            AgentToolWorkflowState.CONTEXT_PREPARED,
        ),
        (
            AgentToolWorkflowState.CONTEXT_PREPARED,
            AgentToolWorkflowAction.COMPOSE_GROUNDED_ANSWER,
            AgentToolWorkflowState.ANSWERED,
        ),
        (
            AgentToolWorkflowState.ANSWERED,
            AgentToolWorkflowAction.VERIFY_GROUNDED_ANSWER,
            AgentToolWorkflowState.VERIFIED,
        ),
        (
            AgentToolWorkflowState.VERIFIED,
            AgentToolWorkflowAction.OBSERVE_DIAGNOSTICS,
            AgentToolWorkflowState.OBSERVED,
        ),
        (
            AgentToolWorkflowState.OBSERVED,
            AgentToolWorkflowAction.COMPLETE,
            AgentToolWorkflowState.COMPLETE,
        ),
        (
            AgentToolWorkflowState.OBSERVED,
            AgentToolWorkflowAction.REFUSE,
            AgentToolWorkflowState.REFUSE,
        ),
    )

    def allowed_transitions(
        self,
    ) -> tuple[tuple[AgentToolWorkflowState, AgentToolWorkflowAction, AgentToolWorkflowState], ...]:
        return self._TRANSITIONS

    def next_state(
        self,
        *,
        current_state: AgentToolWorkflowState,
        action: AgentToolWorkflowAction,
    ) -> AgentToolWorkflowState:
        for source, allowed_action, target in self._TRANSITIONS:
            if source is current_state and allowed_action is action:
                return target
        raise InvalidAgentToolWorkflowTransitionError(
            f"action {action.value!r} is not allowed from state {current_state.value!r}"
        )


class AgentToolWorkflowStateMachine:
    """Execute the frozen transition contract without workflow dependencies."""

    def __init__(self, policy: AgentToolWorkflowTransitionPolicy | None = None) -> None:
        self._policy = policy or FrozenAgentToolWorkflowTransitionPolicy()
        self._state = AgentToolWorkflowState.RECEIVED
        self._trace: tuple[AgentToolWorkflowTransition, ...] = ()

    @property
    def state(self) -> AgentToolWorkflowState:
        return self._state

    @property
    def trace(self) -> tuple[AgentToolWorkflowTransition, ...]:
        return self._trace

    @property
    def terminal(self) -> bool:
        return self._state in {
            AgentToolWorkflowState.COMPLETE,
            AgentToolWorkflowState.REFUSE,
        }

    def advance(self, action: AgentToolWorkflowAction) -> AgentToolWorkflowTransition:
        next_state = self._policy.next_state(current_state=self._state, action=action)
        transition = AgentToolWorkflowTransition(
            sequence_number=len(self._trace) + 1,
            previous_state=self._state,
            action=action,
            next_state=next_state,
        )
        transition.to_public_dict()
        self._state = next_state
        self._trace = (*self._trace, transition)
        return transition

    def run(
        self,
        actions: Sequence[AgentToolWorkflowAction],
    ) -> tuple[AgentToolWorkflowTransition, ...]:
        for action in actions:
            self.advance(action)
        return self._trace

    def public_trace(self) -> list[dict[str, Any]]:
        return [row.to_public_dict() for row in self._trace]


@dataclass(frozen=True)
class AgentToolContract:
    tool_id: str
    role: str
    input_schema: str
    output_schema: str
    allowed_source_state: str
    success_target_state: str
    successful_path_call_limit: int = 1
    errors_propagate_unchanged: bool = True
    retry_allowed: bool = False
    fallback_allowed: bool = False
    parallel_call_allowed: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentToolOrchestrationProtocolState(str, Enum):
    REJECTED = "rejected"
    ELIGIBLE = "eligible"


@dataclass(frozen=True)
class AgentToolOrchestrationProtocolEvidence:
    source_stage152_validated: bool
    source_stage139_validated: bool
    real_local_service_lifecycle_validated: bool
    prior_action_order_validated: bool
    prior_answer_path_invariance_validated: bool
    workflow_classification_honest: bool
    tool_registry_exact: bool
    typed_private_state_exact: bool
    public_trace_allowlist_exact: bool
    graph_nodes_exact: bool
    graph_edges_exact: bool
    graph_acyclic: bool
    conditional_routing_terminal_only: bool
    successful_transition_bound_exact: bool
    successful_tool_call_bound_exact: bool
    graph_compiled_once_per_process: bool
    request_state_isolated: bool
    tool_calls_sequential: bool
    llm_selects_tools: bool
    query_rewrite_enabled: bool
    second_retrieval_enabled: bool
    memory_or_checkpointer_enabled: bool
    streaming_enabled: bool
    human_interrupt_enabled: bool
    tool_errors_propagate_unchanged: bool
    tool_error_messages_public: bool
    retrieval_owns_candidate_pool: bool
    generation_uses_stage116_context: bool
    verification_uses_stage116_prefix: bool
    sidecar_can_generate_answer: bool
    sidecar_can_verify_answer: bool
    sidecar_can_replace_primary_context: bool
    final_response_uses_verified_answer_only: bool
    outer_concurrency_limit_preserved: bool
    application_waiting_queue_enabled: bool
    in_graph_timeout_enabled: bool
    retry_action_count: int
    fallback_action_count: int
    runtime_registered_as_default: bool
    remote_exposure_authorized: bool
    test_gate_opened: bool
    test_metrics_run: bool
    workflow_implemented: bool
    langgraph_dependency_installed: bool


@dataclass(frozen=True)
class AgentToolOrchestrationProtocolEvaluation:
    state: AgentToolOrchestrationProtocolState
    rejection_reasons: tuple[str, ...]

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "rejection_reasons": list(self.rejection_reasons),
        }


class StrictAgentToolOrchestrationProtocolPolicy:
    """Reject autonomy, recovery, leakage, or answer-path drift not yet validated."""

    def evaluate(
        self,
        evidence: AgentToolOrchestrationProtocolEvidence,
    ) -> AgentToolOrchestrationProtocolEvaluation:
        reasons: list[str] = []
        required_true = {
            "stage152_source_not_validated": evidence.source_stage152_validated,
            "stage139_source_not_validated": evidence.source_stage139_validated,
            "real_service_lifecycle_missing": evidence.real_local_service_lifecycle_validated,
            "prior_action_order_not_validated": evidence.prior_action_order_validated,
            "answer_path_invariance_not_validated": (
                evidence.prior_answer_path_invariance_validated
            ),
            "workflow_classification_misleading": evidence.workflow_classification_honest,
            "tool_registry_not_exact": evidence.tool_registry_exact,
            "private_state_not_exact": evidence.typed_private_state_exact,
            "public_trace_allowlist_not_exact": evidence.public_trace_allowlist_exact,
            "graph_nodes_not_exact": evidence.graph_nodes_exact,
            "graph_edges_not_exact": evidence.graph_edges_exact,
            "graph_contains_cycle": evidence.graph_acyclic,
            "conditional_routing_not_terminal_only": (evidence.conditional_routing_terminal_only),
            "transition_bound_not_exact": evidence.successful_transition_bound_exact,
            "tool_call_bound_not_exact": evidence.successful_tool_call_bound_exact,
            "graph_not_compiled_once": evidence.graph_compiled_once_per_process,
            "request_state_not_isolated": evidence.request_state_isolated,
            "tool_calls_not_sequential": evidence.tool_calls_sequential,
            "tool_errors_not_propagated": evidence.tool_errors_propagate_unchanged,
            "retrieval_does_not_own_pool": evidence.retrieval_owns_candidate_pool,
            "generation_context_drift": evidence.generation_uses_stage116_context,
            "verification_context_drift": evidence.verification_uses_stage116_prefix,
            "final_response_not_verified_only": evidence.final_response_uses_verified_answer_only,
            "outer_concurrency_not_preserved": evidence.outer_concurrency_limit_preserved,
        }
        for reason, passed in required_true.items():
            if not passed:
                reasons.append(reason)

        forbidden_true = {
            "llm_tool_selection_not_validated": evidence.llm_selects_tools,
            "query_rewrite_not_validated": evidence.query_rewrite_enabled,
            "second_retrieval_not_validated": evidence.second_retrieval_enabled,
            "memory_or_checkpointer_not_validated": evidence.memory_or_checkpointer_enabled,
            "streaming_not_validated": evidence.streaming_enabled,
            "human_interrupt_not_validated": evidence.human_interrupt_enabled,
            "tool_error_message_leak": evidence.tool_error_messages_public,
            "sidecar_answer_generation_not_allowed": evidence.sidecar_can_generate_answer,
            "sidecar_verification_not_allowed": evidence.sidecar_can_verify_answer,
            "sidecar_primary_replacement_not_allowed": (
                evidence.sidecar_can_replace_primary_context
            ),
            "application_queue_not_allowed": evidence.application_waiting_queue_enabled,
            "in_graph_timeout_not_allowed": evidence.in_graph_timeout_enabled,
            "runtime_defaultization_not_allowed": evidence.runtime_registered_as_default,
            "remote_exposure_not_allowed": evidence.remote_exposure_authorized,
            "test_gate_not_allowed": evidence.test_gate_opened,
            "test_metrics_not_allowed": evidence.test_metrics_run,
            "workflow_implementation_out_of_scope": evidence.workflow_implemented,
            "dependency_install_out_of_scope": evidence.langgraph_dependency_installed,
        }
        for reason, enabled in forbidden_true.items():
            if enabled:
                reasons.append(reason)
        if evidence.retry_action_count != 0:
            reasons.append("retry_actions_not_allowed")
        if evidence.fallback_action_count != 0:
            reasons.append("fallback_actions_not_allowed")
        return AgentToolOrchestrationProtocolEvaluation(
            state=(
                AgentToolOrchestrationProtocolState.ELIGIBLE
                if not reasons
                else AgentToolOrchestrationProtocolState.REJECTED
            ),
            rejection_reasons=tuple(reasons),
        )


@dataclass(frozen=True)
class AgentToolOrchestrationProtocolVisualization:
    name: str
    path: str


def agent_tool_registry_contract() -> tuple[AgentToolContract, ...]:
    """Return the three tools authorized by the Stage153 protocol."""

    return (
        AgentToolContract(
            tool_id="retrieve_candidate_pool",
            role="build_exact_frozen_top400_candidate_pool",
            input_schema="PrimeQARuntimeQuery",
            output_schema="tuple[RetrievalResult, ...]",
            allowed_source_state=AgentToolWorkflowState.VALIDATED.value,
            success_target_state=AgentToolWorkflowState.RETRIEVED.value,
        ),
        AgentToolContract(
            tool_id="compose_grounded_answer",
            role="compose_from_stage116_primary_context_only",
            input_schema="PrimeQARuntimeQuery_plus_generation_context",
            output_schema="GeneratedAnswer",
            allowed_source_state=AgentToolWorkflowState.CONTEXT_PREPARED.value,
            success_target_state=AgentToolWorkflowState.ANSWERED.value,
        ),
        AgentToolContract(
            tool_id="verify_grounded_answer",
            role="verify_citations_against_stage116_prefix_only",
            input_schema="GeneratedAnswer_plus_verification_context",
            output_schema="AnswerVerificationResult",
            allowed_source_state=AgentToolWorkflowState.ANSWERED.value,
            success_target_state=AgentToolWorkflowState.VERIFIED.value,
        ),
    )


def agent_tool_workflow_state_contract() -> dict[str, Any]:
    """Return the executable graph topology and state boundary."""

    policy = FrozenAgentToolWorkflowTransitionPolicy()
    transitions = [
        {
            "previous_state": source.value,
            "action": action.value,
            "next_state": target.value,
        }
        for source, action, target in policy.allowed_transitions()
    ]
    accepted_actions = [
        AgentToolWorkflowAction.VALIDATE_REQUEST,
        AgentToolWorkflowAction.RETRIEVE_CANDIDATE_POOL,
        AgentToolWorkflowAction.PREPARE_CONTEXT,
        AgentToolWorkflowAction.COMPOSE_GROUNDED_ANSWER,
        AgentToolWorkflowAction.VERIFY_GROUNDED_ANSWER,
        AgentToolWorkflowAction.OBSERVE_DIAGNOSTICS,
        AgentToolWorkflowAction.COMPLETE,
    ]
    refused_actions = [*accepted_actions[:-1], AgentToolWorkflowAction.REFUSE]
    return {
        "graph_id": _GRAPH_ID,
        "classification": "deterministic_tool_workflow_not_autonomous_agent",
        "initial_state": AgentToolWorkflowState.RECEIVED.value,
        "states": [state.value for state in AgentToolWorkflowState],
        "actions": [action.value for action in AgentToolWorkflowAction],
        "nodes": list(_NODE_IDS),
        "allowed_transitions": transitions,
        "accepted_path": [action.value for action in accepted_actions],
        "refused_path": [action.value for action in refused_actions],
        "terminal_states": [
            AgentToolWorkflowState.COMPLETE.value,
            AgentToolWorkflowState.REFUSE.value,
        ],
        "successful_transition_count": 7,
        "conditional_edge_count": 1,
        "conditional_edge_source": AgentToolWorkflowState.OBSERVED.value,
        "transition_loops_available": False,
        "private_state_fields": list(_PRIVATE_STATE_FIELDS),
        "public_trace_fields": list(_PUBLIC_TRACE_FIELDS),
        "input_schema": "AgentToolWorkflowInput",
        "output_schema": "AgentFacadeResponse",
        "graph_compile_count_per_process": 1,
        "request_state_shared_across_invocations": False,
    }


def freeze_primeqa_hybrid_agent_tool_orchestration_protocol(
    *,
    stage152_validation_path: Path,
    stage139_validation_path: Path,
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze Stage153 from saved public aggregates and synthetic policy cases."""

    started_at = time.perf_counter()
    stage152 = _load_json_object(stage152_validation_path)
    stage139 = _load_json_object(stage139_validation_path)
    loaded_at = time.perf_counter()
    stage152_summary = _stage152_summary(stage152)
    stage139_summary = _stage139_summary(stage139)
    workflow = agent_tool_workflow_state_contract()
    tools = [tool.to_public_dict() for tool in agent_tool_registry_contract()]
    policy = StrictAgentToolOrchestrationProtocolPolicy()
    evaluations = _canonical_policy_evaluations(policy)
    traces = _canonical_workflow_traces()
    sources_unchanged = stage152 == _load_json_object(
        stage152_validation_path
    ) and stage139 == _load_json_object(stage139_validation_path)
    protocol = _frozen_protocol(workflow=workflow, tools=tools)
    preliminary: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_id": _PROTOCOL_ID,
        "protocol_scope": (
            "Public-safe freeze of a deterministic local Agent tool workflow around the "
            "Stage152 service and Stage139 validated answer path. This stage reads only "
            "saved aggregate JSON reports and executes synthetic transitions and policy "
            "cases. It does not load split questions, documents, models, indexes, or "
            "candidate pools; install LangGraph; implement the workflow; bind a port; "
            "change runtime defaults; open test; or add loops, queues, retries, or fallback."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_protocol),
            "confirmation_note": confirmation_note,
        },
        "source_files": {
            "stage152_service_entrypoint_validation": _fingerprint(stage152_validation_path),
            "stage139_optional_entrypoint_validation": _fingerprint(stage139_validation_path),
        },
        "source_unchanged_after_protocol_freeze": sources_unchanged,
        "stage152_summary": stage152_summary,
        "stage139_summary": stage139_summary,
        "official_framework_research": _official_framework_research(),
        "local_framework_observation": {
            "langgraph_installed": find_spec("langgraph") is not None,
            "langchain_installed": find_spec("langchain") is not None,
            "dependency_install_performed_in_stage153": False,
        },
        "frozen_protocol": protocol,
        "canonical_policy_evaluations": evaluations,
        "canonical_workflow_traces": traces,
    }
    checks = _guard_checks(preliminary)
    checked_at = time.perf_counter()
    passed = all(check["passed"] for check in checks)
    report = {
        **preliminary,
        "guard_checks": checks,
        "decision": {
            "status": _FINAL_STATUS
            if passed
            else "primeqa_hybrid_agent_tool_orchestration_protocol_rejected",
            "failed_checks": [check["name"] for check in checks if not check["passed"]],
            "protocol_frozen": passed,
            "workflow_classification": ("deterministic_tool_workflow_not_autonomous_agent"),
            "workflow_implementation_allowed_next": passed,
            "langgraph_adapter_proof_required_next": passed,
            "workflow_implemented": False,
            "langgraph_dependency_installed": False,
            "runtime_registered_as_default": False,
            "remote_exposure_authorized": False,
            "test_gate_opened": False,
            "test_metrics_run": False,
            "queue_actions_enabled": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
            "next_direction": _NEXT_DIRECTION if passed else "repair_failed_stage153_guards",
        },
        "timing_seconds": {
            "load_saved_aggregates": round(loaded_at - started_at, 6),
            "freeze_and_guard": round(checked_at - loaded_at, 6),
            "total": round(checked_at - started_at, 6),
        },
    }
    report["public_safe_contract"] = _public_safe_contract(report)
    return report


def write_primeqa_hybrid_agent_tool_orchestration_protocol_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[AgentToolOrchestrationProtocolVisualization]:
    """Write ten Stage153 public-safe SVG charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = report.get("frozen_protocol") or {}
    graph = protocol.get("graph_contract") or {}
    tools = protocol.get("tool_registry") or []
    evaluations = report.get("canonical_policy_evaluations") or {}
    closed = protocol.get("closed_boundaries") or {}
    transitions = graph.get("allowed_transitions") or []
    outdegree = Counter(row["previous_state"] for row in transitions)
    charts = {
        "stage153_source_guards.svg": _chart(
            "Stage153 source aggregate guards",
            [
                _bar("Stage152 passed guards", report["stage152_summary"]["passed_guards"]),
                _bar("Stage139 passed guards", report["stage139_summary"]["passed_guards"]),
            ],
        ),
        "stage153_graph_state_outdegree.svg": _chart(
            "Stage153 workflow state outdegree",
            [_bar(state, outdegree.get(state, 0)) for state in graph.get("states", [])],
        ),
        "stage153_terminal_paths.svg": _chart(
            "Stage153 terminal path action counts",
            [
                _bar("complete path", len(graph.get("accepted_path") or [])),
                _bar("refuse path", len(graph.get("refused_path") or [])),
                _bar("conditional edges", graph.get("conditional_edge_count", 0)),
            ],
        ),
        "stage153_tool_call_budgets.svg": _chart(
            "Stage153 successful path tool budgets",
            [_bar(str(tool["tool_id"]), tool["successful_path_call_limit"]) for tool in tools],
        ),
        "stage153_node_roles.svg": _chart(
            "Stage153 graph node categories",
            [
                _bar("tool nodes", len(tools)),
                _bar("deterministic nodes", len(graph.get("nodes") or []) - len(tools)),
                _bar("parallel tool nodes", 0),
                _bar("autonomous LLM router nodes", 0),
            ],
        ),
        "stage153_state_boundaries.svg": _chart(
            "Stage153 private and public state boundaries",
            [
                _bar("private state fields", len(graph.get("private_state_fields") or [])),
                _bar("public trace fields", len(graph.get("public_trace_fields") or [])),
                _bar("raw content fields public", 0),
            ],
        ),
        "stage153_framework_research.svg": _chart(
            "Stage153 framework selection state",
            [
                _truth_bar("StateGraph adapter selected", True),
                _truth_bar("LangGraph installed", False),
                _truth_bar("create_agent loop selected", False),
                _truth_bar("ToolNode error handling selected", False),
            ],
        ),
        "stage153_policy_cases.svg": _chart(
            "Stage153 canonical policy cases",
            [
                BarDatum(
                    label=str(name),
                    value=1.0 if row.get("state") == "eligible" else 0.0,
                    value_label=str(row.get("state")),
                )
                for name, row in evaluations.items()
            ],
        ),
        "stage153_closed_boundaries.svg": _chart(
            "Stage153 closed boundaries",
            [
                _truth_bar(str(name), value is False)
                for name, value in closed.items()
                if isinstance(value, bool)
            ],
            width=2200,
            margin_left=1180,
        ),
        "stage153_guard_check_status.svg": _chart(
            "Stage153 protocol guard checks",
            [
                BarDatum(
                    label=str(check["name"]),
                    value=1.0 if check["passed"] else 0.0,
                    value_label="passed" if check["passed"] else "failed",
                )
                for check in report.get("guard_checks", [])
            ],
            width=2800,
            margin_left=1560,
        ),
    }
    artifacts: list[AgentToolOrchestrationProtocolVisualization] = []
    for name, svg in charts.items():
        path = output_dir / name
        path.write_text(svg, encoding="utf-8")
        artifacts.append(AgentToolOrchestrationProtocolVisualization(name=name, path=str(path)))
    return artifacts


def _frozen_protocol(
    *,
    workflow: Mapping[str, Any],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "identity": {
            "protocol_id": _PROTOCOL_ID,
            "graph_id": _GRAPH_ID,
            "scope": "local_nondefault_deterministic_agent_tool_workflow",
            "classification": "deterministic_tool_workflow_not_autonomous_agent",
        },
        "framework_adapter_contract": {
            "protocol_framework_neutral": True,
            "selected_implementation_adapter": "langgraph.graph.StateGraph",
            "adapter_required_for_protocol_semantics": False,
            "state_schema_style": "TypedDict_with_private_internal_fields",
            "compile_required_before_invoke": True,
            "compile_count_per_process": 1,
            "new_graph_state_per_request": True,
            "langchain_create_agent_selected": False,
            "prebuilt_tool_node_selected": False,
            "checkpointer_selected": False,
            "persistent_store_selected": False,
            "streaming_selected": False,
            "human_in_the_loop_selected": False,
            "dependency_install_allowed_in_stage153": False,
            "stage154_must_record_exact_installed_versions": True,
        },
        "graph_contract": dict(workflow),
        "tool_registry": tools,
        "context_authority_contract": {
            "candidate_pool_depth": 400,
            "generation_context_depth": 10,
            "verification_context_max_rank": 200,
            "sidecar_observation_slots": 3,
            "sidecar_answer_generation_allowed": False,
            "sidecar_answer_verification_allowed": False,
            "sidecar_primary_context_replacement_allowed": False,
            "final_response_source": "verified_answer_only",
        },
        "execution_contract": {
            "successful_tool_call_count": 3,
            "tool_calls_parallel": False,
            "llm_selects_tools": False,
            "query_rewrite_enabled": False,
            "second_retrieval_enabled": False,
            "tool_errors_propagate_unchanged": True,
            "tool_error_messages_in_public_trace": False,
            "invalid_transition": "raise_without_state_change",
            "outer_nonblocking_concurrency_limit": 4,
            "application_waiting_queue": False,
            "in_graph_timeout_seconds": None,
            "retry_action_count": 0,
            "fallback_action_count": 0,
        },
        "observability_contract": {
            "public_trace_fields": list(_PUBLIC_TRACE_FIELDS),
            "public_trace_field_count": len(_PUBLIC_TRACE_FIELDS),
            "private_state_fields": list(_PRIVATE_STATE_FIELDS),
            "private_state_field_count": len(_PRIVATE_STATE_FIELDS),
            "raw_question_public": False,
            "raw_answer_public": False,
            "raw_document_public": False,
            "document_identifiers_public": False,
            "request_handle_public": False,
            "exception_message_public": False,
        },
        "closed_boundaries": {
            "workflow_implemented": False,
            "langgraph_dependency_installed": False,
            "autonomous_llm_tool_selection": False,
            "query_rewrite": False,
            "second_retrieval": False,
            "memory_or_checkpointer": False,
            "streaming": False,
            "human_interrupt": False,
            "runtime_registered_as_default": False,
            "remote_exposure_authorized": False,
            "test_gate_opened": False,
            "test_metrics_run": False,
            "queue_actions_enabled": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
        },
    }


def _official_framework_research() -> dict[str, Any]:
    return {
        "researched_at": _CREATED_AT,
        "sources": [
            {
                "title": "LangGraph overview",
                "url": "https://docs.langchain.com/oss/python/langgraph/overview",
                "fact_used": "LangGraph is low-level orchestration and does not require LangChain",
            },
            {
                "title": "LangGraph Graph API overview",
                "url": "https://docs.langchain.com/oss/python/langgraph/graph-api",
                "fact_used": "graphs use state, nodes, edges, and compile before invoke",
            },
            {
                "title": "LangGraph workflows and agents",
                "url": "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
                "fact_used": (
                    "workflows have predetermined paths while agents dynamically choose tools"
                ),
            },
            {
                "title": "LangChain tools",
                "url": "https://docs.langchain.com/oss/python/langchain/tools",
                "fact_used": "ToolNode has built-in tool execution and error-handling behavior",
            },
            {
                "title": "langgraph PyPI",
                "url": "https://pypi.org/project/langgraph/",
                "fact_used": "latest observed release was 1.2.9 on 2026-07-18",
            },
        ],
        "latest_observed_langgraph_version": "1.2.9",
        "local_version_observed": None,
        "version_claim_is_install_proof": False,
    }


def _canonical_policy_evaluations(
    policy: StrictAgentToolOrchestrationProtocolPolicy,
) -> dict[str, dict[str, Any]]:
    cases = {
        "exact_deterministic_tool_workflow": _compliant_evidence(),
        "autonomous_llm_tool_loop": _compliant_evidence(
            workflow_classification_honest=False,
            graph_acyclic=False,
            conditional_routing_terminal_only=False,
            successful_transition_bound_exact=False,
            successful_tool_call_bound_exact=False,
            llm_selects_tools=True,
        ),
        "query_rewrite_second_retrieval": _compliant_evidence(
            query_rewrite_enabled=True,
            second_retrieval_enabled=True,
        ),
        "hidden_tool_recovery": _compliant_evidence(
            tool_errors_propagate_unchanged=False,
            tool_error_messages_public=True,
            retry_action_count=1,
            fallback_action_count=1,
        ),
        "shared_persistent_request_state": _compliant_evidence(
            request_state_isolated=False,
            memory_or_checkpointer_enabled=True,
        ),
        "sidecar_answer_authority": _compliant_evidence(
            generation_uses_stage116_context=False,
            verification_uses_stage116_prefix=False,
            sidecar_can_generate_answer=True,
            sidecar_can_verify_answer=True,
            sidecar_can_replace_primary_context=True,
        ),
        "default_remote_test_open": _compliant_evidence(
            runtime_registered_as_default=True,
            remote_exposure_authorized=True,
            test_gate_opened=True,
            test_metrics_run=True,
            workflow_implemented=True,
            langgraph_dependency_installed=True,
        ),
    }
    return {name: policy.evaluate(evidence).to_public_dict() for name, evidence in cases.items()}


def _canonical_workflow_traces() -> dict[str, Any]:
    prefix = [
        AgentToolWorkflowAction.VALIDATE_REQUEST,
        AgentToolWorkflowAction.RETRIEVE_CANDIDATE_POOL,
        AgentToolWorkflowAction.PREPARE_CONTEXT,
        AgentToolWorkflowAction.COMPOSE_GROUNDED_ANSWER,
        AgentToolWorkflowAction.VERIFY_GROUNDED_ANSWER,
        AgentToolWorkflowAction.OBSERVE_DIAGNOSTICS,
    ]
    traces: dict[str, Any] = {}
    for name, terminal in (
        ("complete", AgentToolWorkflowAction.COMPLETE),
        ("refuse", AgentToolWorkflowAction.REFUSE),
    ):
        machine = AgentToolWorkflowStateMachine()
        machine.run([*prefix, terminal])
        traces[name] = {
            "terminal_state": machine.state.value,
            "terminal": machine.terminal,
            "transition_count": len(machine.trace),
            "trace": machine.public_trace(),
        }
    machine = AgentToolWorkflowStateMachine()
    try:
        machine.advance(AgentToolWorkflowAction.COMPOSE_GROUNDED_ANSWER)
        rejected = False
    except InvalidAgentToolWorkflowTransitionError:
        rejected = True
    traces["invalid_first_action"] = {
        "rejected": rejected,
        "state_unchanged": machine.state is AgentToolWorkflowState.RECEIVED,
        "trace_unchanged": machine.trace == (),
    }
    return traces


def _compliant_evidence(**overrides: Any) -> AgentToolOrchestrationProtocolEvidence:
    values: dict[str, Any] = {
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
    return AgentToolOrchestrationProtocolEvidence(**values)


def _stage152_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    checks = report.get("guard_checks") or []
    decision = report.get("decision") or {}
    real = report.get("real_resource_service_lifecycle") or {}
    public = report.get("public_safe_contract") or {}
    return {
        "identity_valid": (
            report.get("stage") == "Stage 152"
            and report.get("analysis_id") == _SOURCE_STAGE152_ANALYSIS_ID
            and decision.get("status") == _SOURCE_STAGE152_STATUS
        ),
        "guard_count": len(checks),
        "passed_guards": sum(check.get("passed") is True for check in checks),
        "all_guards_passed": (
            len(checks) == _EXPECTED_STAGE152_GUARDS
            and all(check.get("passed") is True for check in checks)
        ),
        "service_entrypoint_implemented": decision.get("service_entrypoint_implemented"),
        "real_lifecycle_validated": decision.get("real_resource_lifecycle_validated"),
        "real_exit_code": real.get("exit_code"),
        "test_gate_opened": decision.get("test_gate_opened"),
        "test_metrics_run": decision.get("test_metrics_run"),
        "runtime_registered_as_default": decision.get("runtime_registered_as_default"),
        "remote_exposure_authorized": decision.get("remote_exposure_authorized"),
        "queue_action_count": public.get("queue_action_count"),
        "retry_action_count": public.get("retry_action_count"),
        "fallback_action_count": public.get("fallback_action_count"),
    }


def _stage139_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    checks = report.get("guard_checks") or []
    decision = report.get("decision") or {}
    contract = report.get("entrypoint_contract") or {}
    permissions = contract.get("permissions") or {}
    public = report.get("public_safe_contract") or {}
    return {
        "identity_valid": (
            report.get("stage") == "Stage 139"
            and report.get("analysis_id") == _SOURCE_STAGE139_ANALYSIS_ID
            and decision.get("status") == _SOURCE_STAGE139_STATUS
        ),
        "guard_count": len(checks),
        "passed_guards": sum(check.get("passed") is True for check in checks),
        "all_guards_passed": (
            len(checks) == _EXPECTED_STAGE139_GUARDS
            and all(check.get("passed") is True for check in checks)
        ),
        "execution_order": contract.get("execution_order"),
        "runtime_action_order_validated": decision.get("runtime_action_order_validated"),
        "answer_path_invariance_validated": decision.get("answer_path_invariance_validated"),
        "sidecar_can_generate_answer": permissions.get("sidecar_answer_generation_allowed"),
        "sidecar_can_verify_answer": permissions.get("sidecar_verification_context_allowed"),
        "sidecar_can_replace_primary_context": permissions.get(
            "sidecar_primary_context_replacement_allowed"
        ),
        "retry_actions_allowed": permissions.get("retry_actions_allowed"),
        "fallback_strategies_allowed": permissions.get("fallback_strategies_allowed"),
        "test_split_loaded": public.get("test_split_loaded"),
        "test_metrics_run": public.get("final_test_metrics_run"),
    }


def _guard_checks(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    stage152 = report.get("stage152_summary") or {}
    stage139 = report.get("stage139_summary") or {}
    protocol = report.get("frozen_protocol") or {}
    identity = protocol.get("identity") or {}
    adapter = protocol.get("framework_adapter_contract") or {}
    graph = protocol.get("graph_contract") or {}
    tools = protocol.get("tool_registry") or []
    context = protocol.get("context_authority_contract") or {}
    execution = protocol.get("execution_contract") or {}
    observability = protocol.get("observability_contract") or {}
    closed = protocol.get("closed_boundaries") or {}
    evaluations = report.get("canonical_policy_evaluations") or {}
    traces = report.get("canonical_workflow_traces") or {}
    expected_states = {
        "exact_deterministic_tool_workflow": "eligible",
        "autonomous_llm_tool_loop": "rejected",
        "query_rewrite_second_retrieval": "rejected",
        "hidden_tool_recovery": "rejected",
        "shared_persistent_request_state": "rejected",
        "sidecar_answer_authority": "rejected",
        "default_remote_test_open": "rejected",
    }
    return [
        _check("stage153_user_confirmed", report["user_confirmation"]["confirmed"] is True),
        _check(
            "stage153_confirmation_note_present",
            bool(report["user_confirmation"]["confirmation_note"]),
        ),
        _check("stage152_identity_valid", stage152.get("identity_valid") is True),
        _check("stage152_all_46_guards_passed", stage152.get("all_guards_passed") is True),
        _check(
            "stage152_real_lifecycle_validated",
            stage152.get("real_lifecycle_validated") is True
            and stage152.get("real_exit_code") == 0,
        ),
        _check(
            "stage152_closed_boundaries_preserved",
            [
                stage152.get("test_gate_opened"),
                stage152.get("test_metrics_run"),
                stage152.get("runtime_registered_as_default"),
                stage152.get("remote_exposure_authorized"),
            ]
            == [False, False, False, False],
        ),
        _check(
            "stage152_no_recovery_actions",
            [
                stage152.get("queue_action_count"),
                stage152.get("retry_action_count"),
                stage152.get("fallback_action_count"),
            ]
            == [0, 0, 0],
        ),
        _check("stage139_identity_valid", stage139.get("identity_valid") is True),
        _check("stage139_all_45_guards_passed", stage139.get("all_guards_passed") is True),
        _check(
            "stage139_action_order_exact",
            stage139.get("execution_order")
            == ["retrieve", "answer", "verify", "observe", "complete_or_refuse"]
            and stage139.get("runtime_action_order_validated") is True,
        ),
        _check(
            "stage139_answer_path_invariant",
            stage139.get("answer_path_invariance_validated") is True,
        ),
        _check(
            "stage139_sidecar_authority_closed",
            [
                stage139.get("sidecar_can_generate_answer"),
                stage139.get("sidecar_can_verify_answer"),
                stage139.get("sidecar_can_replace_primary_context"),
            ]
            == [False, False, False],
        ),
        _check(
            "saved_sources_unchanged", report.get("source_unchanged_after_protocol_freeze") is True
        ),
        _check(
            "protocol_identity_exact",
            identity.get("protocol_id") == _PROTOCOL_ID and identity.get("graph_id") == _GRAPH_ID,
        ),
        _check(
            "workflow_classification_honest",
            identity.get("classification") == "deterministic_tool_workflow_not_autonomous_agent",
        ),
        _check("framework_protocol_neutral", adapter.get("protocol_framework_neutral") is True),
        _check(
            "stategraph_adapter_selected",
            adapter.get("selected_implementation_adapter") == "langgraph.graph.StateGraph",
        ),
        _check(
            "stategraph_compile_and_isolation_exact",
            [
                adapter.get("compile_required_before_invoke"),
                adapter.get("compile_count_per_process"),
                adapter.get("new_graph_state_per_request"),
            ]
            == [True, 1, True],
        ),
        _check(
            "no_prebuilt_agent_or_toolnode",
            [
                adapter.get("langchain_create_agent_selected"),
                adapter.get("prebuilt_tool_node_selected"),
            ]
            == [False, False],
        ),
        _check(
            "no_persistence_streaming_interrupt",
            [
                adapter.get("checkpointer_selected"),
                adapter.get("persistent_store_selected"),
                adapter.get("streaming_selected"),
                adapter.get("human_in_the_loop_selected"),
            ]
            == [False, False, False, False],
        ),
        _check(
            "dependency_install_closed",
            adapter.get("dependency_install_allowed_in_stage153") is False
            and report["local_framework_observation"]["langgraph_installed"] is False,
        ),
        _check("graph_nodes_exact", graph.get("nodes") == list(_NODE_IDS)),
        _check(
            "graph_states_exact",
            graph.get("states") == [state.value for state in AgentToolWorkflowState],
        ),
        _check(
            "graph_transitions_exact",
            len(graph.get("allowed_transitions") or []) == 8
            and graph.get("transition_loops_available") is False,
        ),
        _check(
            "terminal_paths_exact",
            len(graph.get("accepted_path") or []) == 7
            and len(graph.get("refused_path") or []) == 7
            and graph.get("successful_transition_count") == 7,
        ),
        _check(
            "conditional_edge_terminal_only",
            graph.get("conditional_edge_count") == 1
            and graph.get("conditional_edge_source") == "observed",
        ),
        _check(
            "private_state_exact", graph.get("private_state_fields") == list(_PRIVATE_STATE_FIELDS)
        ),
        _check(
            "public_trace_exact", graph.get("public_trace_fields") == list(_PUBLIC_TRACE_FIELDS)
        ),
        _check("tool_registry_exact", [tool.get("tool_id") for tool in tools] == list(_TOOL_IDS)),
        _check(
            "tool_call_limits_exact",
            len(tools) == 3 and all(tool.get("successful_path_call_limit") == 1 for tool in tools),
        ),
        _check(
            "tool_error_semantics_exact",
            all(
                tool.get("errors_propagate_unchanged") is True
                and tool.get("retry_allowed") is False
                and tool.get("fallback_allowed") is False
                for tool in tools
            ),
        ),
        _check(
            "tool_execution_sequential",
            all(tool.get("parallel_call_allowed") is False for tool in tools)
            and execution.get("tool_calls_parallel") is False,
        ),
        _check(
            "context_depths_preserved",
            [
                context.get("candidate_pool_depth"),
                context.get("generation_context_depth"),
                context.get("verification_context_max_rank"),
                context.get("sidecar_observation_slots"),
            ]
            == [400, 10, 200, 3],
        ),
        _check(
            "sidecar_answer_authority_closed",
            [
                context.get("sidecar_answer_generation_allowed"),
                context.get("sidecar_answer_verification_allowed"),
                context.get("sidecar_primary_context_replacement_allowed"),
            ]
            == [False, False, False],
        ),
        _check(
            "verified_answer_only", context.get("final_response_source") == "verified_answer_only"
        ),
        _check(
            "autonomous_and_retrieval_loops_closed",
            [
                execution.get("llm_selects_tools"),
                execution.get("query_rewrite_enabled"),
                execution.get("second_retrieval_enabled"),
            ]
            == [False, False, False],
        ),
        _check(
            "outer_concurrency_preserved",
            execution.get("outer_nonblocking_concurrency_limit") == 4
            and execution.get("application_waiting_queue") is False,
        ),
        _check(
            "no_timeout_retry_fallback",
            execution.get("in_graph_timeout_seconds") is None
            and [execution.get("retry_action_count"), execution.get("fallback_action_count")]
            == [0, 0],
        ),
        _check(
            "observability_counts_exact",
            observability.get("public_trace_field_count") == len(_PUBLIC_TRACE_FIELDS)
            and observability.get("private_state_field_count") == len(_PRIVATE_STATE_FIELDS),
        ),
        _check(
            "public_content_closed",
            [
                observability.get("raw_question_public"),
                observability.get("raw_answer_public"),
                observability.get("raw_document_public"),
                observability.get("document_identifiers_public"),
                observability.get("request_handle_public"),
                observability.get("exception_message_public"),
            ]
            == [False] * 6,
        ),
        _check(
            "canonical_policy_states_exact",
            all(
                (evaluations.get(name) or {}).get("state") == expected
                for name, expected in expected_states.items()
            ),
        ),
        _check(
            "complete_trace_exact",
            (traces.get("complete") or {}).get("terminal_state") == "complete"
            and (traces.get("complete") or {}).get("transition_count") == 7,
        ),
        _check(
            "refuse_trace_exact",
            (traces.get("refuse") or {}).get("terminal_state") == "refuse"
            and (traces.get("refuse") or {}).get("transition_count") == 7,
        ),
        _check(
            "invalid_transition_no_mutation",
            all(
                (traces.get("invalid_first_action") or {}).get(key) is True
                for key in ("rejected", "state_unchanged", "trace_unchanged")
            ),
        ),
        _check(
            "closed_boundaries_all_false",
            all(value is False for value in closed.values() if isinstance(value, bool)),
        ),
        _check(
            "official_research_recorded",
            len((report.get("official_framework_research") or {}).get("sources") or []) == 5,
        ),
    ]


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = report.get("decision") or {}
    return {
        "source_kind": "saved_public_stage152_and_stage139_aggregates_only",
        "train_split_loaded": False,
        "dev_split_loaded": False,
        "test_split_loaded": False,
        "test_metrics_run": False,
        "questions_loaded": False,
        "documents_loaded": False,
        "models_loaded": False,
        "indexes_loaded": False,
        "candidate_pools_built": False,
        "network_service_started": False,
        "network_port_bound": False,
        "workflow_implemented": decision.get("workflow_implemented"),
        "langgraph_dependency_installed": decision.get("langgraph_dependency_installed"),
        "runtime_registered_as_default": False,
        "remote_exposure_authorized": False,
        "queue_action_count": 0,
        "retry_action_count": 0,
        "fallback_action_count": 0,
        "forbidden_keys_found": [],
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _fingerprint(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _bar(label: str, value: Any) -> BarDatum:
    numeric = float(value or 0)
    return BarDatum(label=label, value=numeric, value_label=str(value or 0))


def _truth_bar(label: str, value: bool) -> BarDatum:
    return BarDatum(
        label=label,
        value=1.0 if value else 0.0,
        value_label="true" if value else "false",
    )


def _chart(
    title: str,
    bars: list[BarDatum],
    *,
    width: int = 1900,
    margin_left: int = 960,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label="observed value",
        width=width,
        margin_left=margin_left,
    )
