from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any

from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    _forbidden_keys_found,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 156"
_CREATED_AT = "2026-07-18"
_ANALYSIS_ID = "primeqa_hybrid_bounded_agent_tool_selection_state_protocol_v1"
_SOURCE_STAGE155_ANALYSIS_ID = "primeqa_hybrid_agent_runtime_activation_observability_validation_v1"
_SOURCE_STAGE155_STATUS = "primeqa_hybrid_agent_runtime_activation_observability_validated"
_EXPECTED_STAGE155_GUARDS = 57
_FINAL_STATUS = "primeqa_hybrid_bounded_agent_tool_selection_state_protocol_frozen"
_NEXT_DIRECTION = "implement_and_validate_local_structured_decision_router"

_MODEL_ACTIONS = (
    "compose_grounded_answer",
    "refuse_insufficient_evidence",
)
_SYSTEM_ACTIONS = (
    "validate_request",
    "retrieve_candidate_pool",
    "prepare_context",
    "verify_grounded_answer",
    "observe_diagnostics",
    "finalize_verified_response",
    "finalize_insufficient_evidence_refusal",
)
_CROSS_TURN_PRIVATE_FIELDS = (
    "opaque_thread_handle",
    "completed_turn_sequence",
    "user_turn_input",
    "verified_terminal_response",
    "terminal_state",
)
_TURN_LOCAL_DISCARD_FIELDS = (
    "candidate_pool_results",
    "generation_context_results",
    "verification_context_results",
    "unverified_generated_response",
    "verification_details",
    "diagnostic_bundle",
    "exception_details",
    "model_internal_reasoning",
)
_PUBLIC_TRACE_FIELDS = (
    "protocol_id",
    "decision_schema_id",
    "terminal_state",
    "completed_turn_count",
    "retained_state_bytes",
    "model_decision_count",
    "selected_action",
    "retrieval_call_count",
    "composition_call_count",
    "verification_call_count",
    "diagnostic_observation_count",
    "thread_state_opened",
    "thread_state_closed",
    "state_limit_rejected",
    "failure_stage",
    "retry_action_count",
    "fallback_action_count",
)


class DynamicToolSelectionState(str, Enum):
    ELIGIBLE = "eligible"
    REJECTED = "rejected"


class DynamicDecisionAction(str, Enum):
    COMPOSE_GROUNDED_ANSWER = "compose_grounded_answer"
    REFUSE_INSUFFICIENT_EVIDENCE = "refuse_insufficient_evidence"


@dataclass(frozen=True)
class DynamicToolSelectionEvidence:
    source_stage155_validated: bool = True
    structured_decision_schema_exact: bool = True
    decision_after_context_preparation: bool = True
    model_decision_count: int = 1
    selected_action: str = DynamicDecisionAction.COMPOSE_GROUNDED_ANSWER.value
    unauthorized_tool_ids: tuple[str, ...] = ()
    retrieval_system_required: bool = True
    verification_system_required_after_composition: bool = True
    diagnostics_system_owned_read_only: bool = True
    composed_response_verified_only: bool = True
    fixed_refusal_system_owned: bool = True
    model_owns_final_answer_authority: bool = False
    model_can_call_retrieval: bool = False
    query_rewrite_requested: bool = False
    second_retrieval_requested: bool = False
    decision_loop_available: bool = False
    parallel_tool_calls_requested: bool = False
    retry_action_count: int = 0
    fallback_action_count: int = 0
    runtime_registered_as_default: bool = False
    remote_exposure_authorized: bool = False
    test_gate_opened: bool = False
    test_metrics_run: bool = False
    protocol_implemented_in_runtime: bool = False
    model_loaded_or_called: bool = False


@dataclass(frozen=True)
class DynamicToolSelectionEvaluation:
    state: DynamicToolSelectionState
    rejection_reasons: tuple[str, ...]

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "rejection_reasons": list(self.rejection_reasons),
        }


class BoundedDynamicToolSelectionPolicy:
    """Authorize exactly one structured answer-or-refuse decision per turn."""

    def evaluate(
        self,
        evidence: DynamicToolSelectionEvidence,
    ) -> DynamicToolSelectionEvaluation:
        reasons: list[str] = []
        required_true = {
            "stage155_source_not_validated": evidence.source_stage155_validated,
            "decision_schema_not_exact": evidence.structured_decision_schema_exact,
            "decision_position_not_exact": evidence.decision_after_context_preparation,
            "retrieval_not_system_required": evidence.retrieval_system_required,
            "verification_not_system_required": (
                evidence.verification_system_required_after_composition
            ),
            "diagnostics_not_system_owned_read_only": (evidence.diagnostics_system_owned_read_only),
            "composed_response_not_verified_only": evidence.composed_response_verified_only,
            "fixed_refusal_not_system_owned": evidence.fixed_refusal_system_owned,
        }
        for reason, passed in required_true.items():
            if not passed:
                reasons.append(reason)
        if evidence.model_decision_count != 1:
            reasons.append("model_decision_count_not_one")
        if evidence.selected_action not in _MODEL_ACTIONS:
            reasons.append("selected_action_not_allowed")
        if evidence.unauthorized_tool_ids:
            reasons.append("unauthorized_tool_requested")
        forbidden_true = {
            "model_final_answer_authority_not_allowed": evidence.model_owns_final_answer_authority,
            "model_retrieval_authority_not_allowed": evidence.model_can_call_retrieval,
            "query_rewrite_not_allowed": evidence.query_rewrite_requested,
            "second_retrieval_not_allowed": evidence.second_retrieval_requested,
            "decision_loop_not_allowed": evidence.decision_loop_available,
            "parallel_tool_calls_not_allowed": evidence.parallel_tool_calls_requested,
            "runtime_defaultization_not_allowed": evidence.runtime_registered_as_default,
            "remote_exposure_not_allowed": evidence.remote_exposure_authorized,
            "test_gate_not_allowed": evidence.test_gate_opened,
            "test_metrics_not_allowed": evidence.test_metrics_run,
            "runtime_implementation_out_of_scope": evidence.protocol_implemented_in_runtime,
            "model_execution_out_of_scope": evidence.model_loaded_or_called,
        }
        for reason, enabled in forbidden_true.items():
            if enabled:
                reasons.append(reason)
        if evidence.retry_action_count != 0:
            reasons.append("retry_actions_not_allowed")
        if evidence.fallback_action_count != 0:
            reasons.append("fallback_actions_not_allowed")
        return DynamicToolSelectionEvaluation(
            state=(
                DynamicToolSelectionState.ELIGIBLE
                if not reasons
                else DynamicToolSelectionState.REJECTED
            ),
            rejection_reasons=tuple(reasons),
        )


class ThreadStatePolicyViolationError(ValueError):
    """Raised before mutation when volatile thread-state policy is violated."""


@dataclass(frozen=True)
class CompletedThreadTurn:
    sequence_number: int
    user_turn_input: str
    verified_terminal_response: str
    terminal_state: str

    @property
    def retained_bytes(self) -> int:
        return len(self.user_turn_input.encode("utf-8")) + len(
            self.verified_terminal_response.encode("utf-8")
        )


@dataclass(frozen=True)
class ThreadStateLimits:
    max_completed_turns: int
    max_retained_bytes: int
    allowed_terminal_states: tuple[str, ...] = ("complete", "refuse")

    def __post_init__(self) -> None:
        if self.max_completed_turns <= 0:
            raise ValueError("max_completed_turns must be positive")
        if self.max_retained_bytes <= 0:
            raise ValueError("max_retained_bytes must be positive")
        if not self.allowed_terminal_states:
            raise ValueError("allowed_terminal_states must not be empty")
        if len(set(self.allowed_terminal_states)) != len(self.allowed_terminal_states):
            raise ValueError("allowed_terminal_states must be unique")
        if any(not state.strip() for state in self.allowed_terminal_states):
            raise ValueError("allowed_terminal_states must be non-empty strings")


@dataclass(frozen=True)
class ThreadStateSummary:
    completed_turn_count: int
    retained_state_bytes: int
    opened: bool

    def to_public_dict(self) -> dict[str, int | bool]:
        return asdict(self)


class VolatileThreadStateLedger:
    """Process-local state with exact-thread access and reject-before-mutation limits."""

    def __init__(self, *, limits: ThreadStateLimits) -> None:
        self._limits = limits
        self._turns: dict[str, tuple[CompletedThreadTurn, ...]] = {}
        self._lock = RLock()

    def open_thread(self, opaque_thread_handle: str) -> None:
        self._validate_handle(opaque_thread_handle)
        with self._lock:
            if opaque_thread_handle in self._turns:
                raise ThreadStatePolicyViolationError("thread is already open")
            self._turns[opaque_thread_handle] = ()

    def append_completed_turn(
        self,
        opaque_thread_handle: str,
        turn: CompletedThreadTurn,
    ) -> ThreadStateSummary:
        self._validate_handle(opaque_thread_handle)
        if turn.terminal_state not in self._limits.allowed_terminal_states:
            raise ThreadStatePolicyViolationError("only terminal turns may be retained")
        if not turn.user_turn_input or not turn.verified_terminal_response:
            raise ThreadStatePolicyViolationError("retained terminal turn fields must be non-empty")
        with self._lock:
            current = self._require_open(opaque_thread_handle)
            expected_sequence = len(current) + 1
            if turn.sequence_number != expected_sequence:
                raise ThreadStatePolicyViolationError("completed turn sequence is not contiguous")
            proposed_count = len(current) + 1
            proposed_bytes = sum(item.retained_bytes for item in current) + turn.retained_bytes
            if proposed_count > self._limits.max_completed_turns:
                raise ThreadStatePolicyViolationError("completed turn limit exceeded")
            if proposed_bytes > self._limits.max_retained_bytes:
                raise ThreadStatePolicyViolationError("retained byte limit exceeded")
            self._turns[opaque_thread_handle] = (*current, turn)
            return ThreadStateSummary(proposed_count, proposed_bytes, True)

    def private_history(self, opaque_thread_handle: str) -> tuple[CompletedThreadTurn, ...]:
        self._validate_handle(opaque_thread_handle)
        with self._lock:
            return self._require_open(opaque_thread_handle)

    def public_summary(self, opaque_thread_handle: str) -> ThreadStateSummary:
        history = self.private_history(opaque_thread_handle)
        return ThreadStateSummary(
            completed_turn_count=len(history),
            retained_state_bytes=sum(turn.retained_bytes for turn in history),
            opened=True,
        )

    def close_thread(self, opaque_thread_handle: str) -> ThreadStateSummary:
        self._validate_handle(opaque_thread_handle)
        with self._lock:
            current = self._require_open(opaque_thread_handle)
            summary = ThreadStateSummary(
                completed_turn_count=len(current),
                retained_state_bytes=sum(turn.retained_bytes for turn in current),
                opened=False,
            )
            del self._turns[opaque_thread_handle]
            return summary

    def _require_open(self, opaque_thread_handle: str) -> tuple[CompletedThreadTurn, ...]:
        if opaque_thread_handle not in self._turns:
            raise ThreadStatePolicyViolationError("thread is not open")
        return self._turns[opaque_thread_handle]

    @staticmethod
    def _validate_handle(opaque_thread_handle: str) -> None:
        if not isinstance(opaque_thread_handle, str) or not opaque_thread_handle.strip():
            raise ThreadStatePolicyViolationError("opaque thread handle must be non-empty")


@dataclass(frozen=True)
class BoundedAgentStateProtocolVisualization:
    name: str
    path: str


def bounded_decision_contract() -> dict[str, Any]:
    return {
        "decision_schema_id": "bounded_answer_or_refuse_decision_v1",
        "decision_position": "after_prepare_context_before_answer_composition",
        "model_selectable_actions": list(_MODEL_ACTIONS),
        "system_required_actions": list(_SYSTEM_ACTIONS),
        "system_actions_by_model_action": {
            "compose_grounded_answer": [
                "verify_grounded_answer",
                "observe_diagnostics",
                "finalize_verified_response",
            ],
            "refuse_insufficient_evidence": ["finalize_insufficient_evidence_refusal"],
        },
        "model_decision_count_per_turn": 1,
        "retrieval_call_count_per_turn": 1,
        "composition_call_count_by_action": {
            "compose_grounded_answer": 1,
            "refuse_insufficient_evidence": 0,
        },
        "verification_call_count_by_action": {
            "compose_grounded_answer": 1,
            "refuse_insufficient_evidence": 0,
        },
        "diagnostic_observation_count_by_action": {
            "compose_grounded_answer": 1,
            "refuse_insufficient_evidence": 0,
        },
        "diagnostics_model_selectable": False,
        "diagnostics_mutation_allowed": False,
        "terminal_authority_by_action": {
            "compose_grounded_answer": "system_verifier",
            "refuse_insufficient_evidence": "fixed_system_refusal_constructor",
        },
        "transition_loops_available": False,
        "query_rewrite_enabled": False,
        "second_retrieval_enabled": False,
        "parallel_tool_calls_enabled": False,
        "retry_action_count": 0,
        "fallback_action_count": 0,
    }


def volatile_thread_state_contract() -> dict[str, Any]:
    return {
        "storage_scope": "process_local_volatile_memory_only",
        "implementation_port": "VolatileThreadStateLedger",
        "checkpointer_selected": False,
        "persistent_store_selected": False,
        "state_survives_process_restart": False,
        "thread_handle_required": True,
        "thread_handle_must_be_opaque": True,
        "cross_thread_read_allowed": False,
        "explicit_close_clears_state": True,
        "completed_terminal_turns_only": True,
        "cross_turn_private_fields": list(_CROSS_TURN_PRIVATE_FIELDS),
        "turn_local_fields_discarded": list(_TURN_LOCAL_DISCARD_FIELDS),
        "public_trace_fields": list(_PUBLIC_TRACE_FIELDS),
        "history_limit_required": True,
        "byte_limit_required": True,
        "runtime_limit_values_frozen_in_stage156": False,
        "runtime_limit_values_require_explicit_configuration": True,
        "overflow_behavior": "reject_before_mutation",
        "silent_truncation_allowed": False,
        "silent_eviction_allowed": False,
        "implicit_thread_creation_allowed": False,
        "state_reconstruction_fallback_allowed": False,
    }


def freeze_primeqa_hybrid_bounded_agent_state_protocol(
    *,
    stage155_validation_path: Path,
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze Stage156 from Stage155 aggregates and synthetic policy cases only."""

    started_at = time.perf_counter()
    stage155 = _load_json_object(stage155_validation_path)
    loaded_at = time.perf_counter()
    stage155_summary = _stage155_summary(stage155)
    policy_cases = _canonical_dynamic_policy_cases()
    state_cases = _canonical_thread_state_cases()
    source_unchanged = stage155 == _load_json_object(stage155_validation_path)
    preliminary: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "protocol_scope": (
            "Public-safe protocol freeze for one structured answer-or-refuse model decision "
            "per turn and strict process-local multi-turn state isolation. This stage reads "
            "only the saved Stage155 aggregate report and executes synthetic policy/state "
            "cases. It does not load questions, documents, models, indexes, or candidate "
            "pools; call a model; change runtime code or defaults; bind a port; open test; "
            "or add retries, fallback, query rewrite, repeated retrieval, autonomous loops, "
            "checkpoints, or persistent stores. Synthetic private strings used by state "
            "tests are not copied into this public report."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_protocol),
            "confirmation_note": confirmation_note,
        },
        "source_files": {
            "stage155_runtime_observability_validation": _fingerprint(stage155_validation_path)
        },
        "source_unchanged_after_protocol_freeze": source_unchanged,
        "stage155_summary": stage155_summary,
        "official_framework_research": _official_framework_research(),
        "bounded_decision_contract": bounded_decision_contract(),
        "volatile_thread_state_contract": volatile_thread_state_contract(),
        "canonical_dynamic_policy_cases": policy_cases,
        "canonical_thread_state_cases": state_cases,
    }
    checks = _guard_checks(preliminary)
    checked_at = time.perf_counter()
    passed = all(check["passed"] for check in checks)
    report = {
        **preliminary,
        "guard_checks": checks,
        "decision": {
            "status": _FINAL_STATUS if passed else "stage156_protocol_rejected",
            "failed_checks": [check["name"] for check in checks if not check["passed"]],
            "protocol_frozen": passed,
            "runtime_implementation_allowed_next": passed,
            "exact_runtime_state_limits_require_confirmation_next": passed,
            "model_provider_selected": False,
            "model_loaded_or_called": False,
            "runtime_changed": False,
            "runtime_registered_as_default": False,
            "remote_exposure_authorized": False,
            "test_gate_opened": False,
            "test_metrics_run": False,
            "queue_actions_enabled": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
            "next_direction": _NEXT_DIRECTION if passed else "repair_failed_stage156_guards",
        },
        "timing_seconds": {
            "load_saved_aggregate": round(loaded_at - started_at, 6),
            "freeze_and_guard": round(checked_at - loaded_at, 6),
            "total": round(checked_at - started_at, 6),
        },
    }
    report["public_safe_contract"] = _public_safe_contract(report)
    return report


def write_primeqa_hybrid_bounded_agent_state_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[BoundedAgentStateProtocolVisualization]:
    output_dir.mkdir(parents=True, exist_ok=True)
    decision = report.get("bounded_decision_contract") or {}
    state = report.get("volatile_thread_state_contract") or {}
    policy_cases = report.get("canonical_dynamic_policy_cases") or {}
    state_cases = report.get("canonical_thread_state_cases") or {}
    charts = {
        "stage156_decision_authority.svg": _chart(
            "Stage156 decision authority",
            [
                _bar(
                    "model-selectable actions",
                    len(decision.get("model_selectable_actions") or []),
                ),
                _bar(
                    "system-required actions",
                    len(decision.get("system_required_actions") or []),
                ),
                _bar("model decisions per turn", decision.get("model_decision_count_per_turn", 0)),
                _bar("available loops", int(bool(decision.get("transition_loops_available")))),
            ],
        ),
        "stage156_action_call_budget.svg": _chart(
            "Stage156 per-turn call budgets",
            [
                _bar("retrieval", decision.get("retrieval_call_count_per_turn", 0)),
                _bar("compose on answer", 1),
                _bar("verify on answer", 1),
                _bar(
                    "diagnostics on answer",
                    decision.get("diagnostic_observation_count_by_action", {}).get(
                        "compose_grounded_answer", 0
                    ),
                ),
                _bar("diagnostics on early refusal", 0),
            ],
        ),
        "stage156_dynamic_policy_cases.svg": _case_chart(
            "Stage156 dynamic policy cases", policy_cases
        ),
        "stage156_thread_state_cases.svg": _case_chart(
            "Stage156 volatile thread-state cases", state_cases
        ),
        "stage156_state_boundaries.svg": _chart(
            "Stage156 state boundaries",
            [
                _bar(
                    "cross-turn private fields",
                    len(state.get("cross_turn_private_fields") or []),
                ),
                _bar(
                    "turn-local discarded fields",
                    len(state.get("turn_local_fields_discarded") or []),
                ),
                _bar("public trace fields", len(state.get("public_trace_fields") or [])),
                _bar("raw content fields public", 0),
            ],
        ),
        "stage156_persistence_boundary.svg": _chart(
            "Stage156 persistence boundary",
            [
                _truth_bar("volatile process memory", True),
                _truth_bar("checkpointer", state.get("checkpointer_selected") is True),
                _truth_bar("persistent store", state.get("persistent_store_selected") is True),
                _truth_bar("survives restart", state.get("state_survives_process_restart") is True),
            ],
        ),
        "stage156_overflow_semantics.svg": _chart(
            "Stage156 state overflow semantics",
            [
                _truth_bar(
                    "reject before mutation",
                    state.get("overflow_behavior") == "reject_before_mutation",
                ),
                _truth_bar("silent truncation", state.get("silent_truncation_allowed") is True),
                _truth_bar("silent eviction", state.get("silent_eviction_allowed") is True),
                _truth_bar(
                    "reconstruction fallback",
                    state.get("state_reconstruction_fallback_allowed") is True,
                ),
            ],
        ),
        "stage156_official_design_mapping.svg": _chart(
            "Stage156 official design mapping",
            [
                _truth_bar("structured routing selected", True),
                _truth_bar("conditional edge selected", True),
                _truth_bar("ToolNode loop selected", False),
                _truth_bar("checkpointer selected", False),
            ],
        ),
        "stage156_closed_boundaries.svg": _chart(
            "Stage156 closed boundaries",
            [
                _truth_bar("query rewrite", decision.get("query_rewrite_enabled") is True),
                _truth_bar("second retrieval", decision.get("second_retrieval_enabled") is True),
                _truth_bar(
                    "parallel tool calls",
                    decision.get("parallel_tool_calls_enabled") is True,
                ),
                _truth_bar(
                    "runtime changed",
                    report.get("decision", {}).get("runtime_changed") is True,
                ),
                _truth_bar(
                    "test opened",
                    report.get("decision", {}).get("test_gate_opened") is True,
                ),
            ],
        ),
        "stage156_guard_status.svg": _chart(
            "Stage156 protocol guard checks",
            [
                BarDatum(
                    label=str(check["name"]),
                    value=1.0 if check["passed"] else 0.0,
                    value_label="passed" if check["passed"] else "failed",
                )
                for check in report.get("guard_checks", [])
            ],
            width=3000,
            margin_left=1700,
        ),
    }
    artifacts: list[BoundedAgentStateProtocolVisualization] = []
    for name, svg in charts.items():
        path = output_dir / name
        path.write_text(svg, encoding="utf-8")
        artifacts.append(BoundedAgentStateProtocolVisualization(name=name, path=str(path)))
    return artifacts


def _canonical_dynamic_policy_cases() -> dict[str, dict[str, Any]]:
    policy = BoundedDynamicToolSelectionPolicy()
    cases = {
        "bounded_compose": DynamicToolSelectionEvidence(),
        "bounded_refuse": DynamicToolSelectionEvidence(
            selected_action=DynamicDecisionAction.REFUSE_INSUFFICIENT_EVIDENCE.value
        ),
        "unauthorized_tool": DynamicToolSelectionEvidence(
            selected_action="retrieve_candidate_pool",
            unauthorized_tool_ids=("retrieve_candidate_pool",),
        ),
        "repeated_retrieval": DynamicToolSelectionEvidence(
            model_can_call_retrieval=True,
            second_retrieval_requested=True,
        ),
        "decision_loop": DynamicToolSelectionEvidence(
            model_decision_count=2,
            decision_loop_available=True,
        ),
        "model_final_authority": DynamicToolSelectionEvidence(
            verification_system_required_after_composition=False,
            composed_response_verified_only=False,
            fixed_refusal_system_owned=False,
            model_owns_final_answer_authority=True,
        ),
        "hidden_recovery": DynamicToolSelectionEvidence(
            retry_action_count=1,
            fallback_action_count=1,
        ),
        "default_remote_test_open": DynamicToolSelectionEvidence(
            runtime_registered_as_default=True,
            remote_exposure_authorized=True,
            test_gate_opened=True,
            test_metrics_run=True,
            protocol_implemented_in_runtime=True,
            model_loaded_or_called=True,
        ),
    }
    return {name: policy.evaluate(case).to_public_dict() for name, case in cases.items()}


def _canonical_thread_state_cases() -> dict[str, dict[str, Any]]:
    outcomes: dict[str, dict[str, Any]] = {}
    ledger = VolatileThreadStateLedger(
        limits=ThreadStateLimits(max_completed_turns=2, max_retained_bytes=512)
    )
    ledger.open_thread("synthetic-thread-a")
    ledger.open_thread("synthetic-thread-b")
    first = CompletedThreadTurn(1, "synthetic input a", "synthetic response a", "complete")
    summary = ledger.append_completed_turn("synthetic-thread-a", first)
    outcomes["complete_turn_retained"] = {
        "state": "eligible",
        "completed_turn_count": summary.completed_turn_count,
        "retained_state_bytes": summary.retained_state_bytes,
    }
    outcomes["cross_thread_isolated"] = {
        "state": "eligible" if ledger.private_history("synthetic-thread-b") == () else "rejected",
        "other_thread_completed_turn_count": len(ledger.private_history("synthetic-thread-b")),
    }
    before = ledger.private_history("synthetic-thread-a")
    try:
        ledger.append_completed_turn(
            "synthetic-thread-a",
            CompletedThreadTurn(2, "x" * 600, "synthetic response", "complete"),
        )
        overflow_rejected = False
    except ThreadStatePolicyViolationError:
        overflow_rejected = True
    outcomes["byte_overflow_rejected_without_mutation"] = {
        "state": "eligible"
        if overflow_rejected and ledger.private_history("synthetic-thread-a") == before
        else "rejected",
        "rejected": overflow_rejected,
        "state_unchanged": ledger.private_history("synthetic-thread-a") == before,
    }
    try:
        ledger.append_completed_turn(
            "missing-thread",
            CompletedThreadTurn(1, "synthetic", "synthetic", "complete"),
        )
        missing_rejected = False
    except ThreadStatePolicyViolationError:
        missing_rejected = True
    outcomes["implicit_thread_creation_rejected"] = {
        "state": "eligible" if missing_rejected else "rejected",
        "rejected": missing_rejected,
    }
    closed = ledger.close_thread("synthetic-thread-a")
    try:
        ledger.private_history("synthetic-thread-a")
        cleared = False
    except ThreadStatePolicyViolationError:
        cleared = True
    outcomes["explicit_close_clears_state"] = {
        "state": "eligible" if not closed.opened and cleared else "rejected",
        "closed": not closed.opened,
        "subsequent_read_rejected": cleared,
    }
    return outcomes


def _stage155_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    checks = report.get("guard_checks") or []
    decision = report.get("decision") or {}
    public = report.get("public_safe_contract") or {}
    return {
        "identity_exact": (
            report.get("stage") == "Stage 155"
            and report.get("analysis_id") == _SOURCE_STAGE155_ANALYSIS_ID
            and decision.get("status") == _SOURCE_STAGE155_STATUS
        ),
        "guard_count": len(checks),
        "passed_guard_count": sum(check.get("passed") is True for check in checks),
        "all_guards_passed": (
            len(checks) == _EXPECTED_STAGE155_GUARDS
            and all(check.get("passed") is True for check in checks)
        ),
        "activation_protocol_frozen": decision.get("activation_protocol_frozen"),
        "node_observability_implemented": decision.get("node_observability_implemented"),
        "real_lifecycle_validated": decision.get("real_resource_service_lifecycle_validated"),
        "runtime_registered_as_default": decision.get("runtime_registered_as_default"),
        "remote_exposure_authorized": decision.get("remote_exposure_authorized"),
        "test_gate_opened": decision.get("test_gate_opened"),
        "test_metrics_run": decision.get("test_metrics_run"),
        "request_content_saved": public.get("request_content_saved"),
        "document_content_saved": public.get("document_content_saved"),
        "retry_action_count": public.get("retry_action_count"),
        "fallback_action_count": public.get("fallback_action_count"),
    }


def _official_framework_research() -> dict[str, Any]:
    return {
        "researched_at": _CREATED_AT,
        "sources": [
            {
                "title": "LangGraph workflows and agents",
                "url": "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
                "fact_used": (
                    "structured model output can route conditional graph edges; agent tool "
                    "usage is dynamic and may loop"
                ),
            },
            {
                "title": "LangGraph persistence",
                "url": "https://docs.langchain.com/oss/python/langgraph/persistence",
                "fact_used": (
                    "checkpointers save graph state as thread-scoped checkpoints configured "
                    "with thread identifiers"
                ),
            },
            {
                "title": "LangChain agents",
                "url": "https://docs.langchain.com/oss/python/langchain/agents",
                "fact_used": "agents run tools in a loop until a stopping condition is met",
            },
            {
                "title": "LangChain tools",
                "url": "https://docs.langchain.com/oss/python/langchain/tools",
                "fact_used": (
                    "tools can access runtime state, context, and stores, so tool authority "
                    "and state visibility require explicit boundaries"
                ),
            },
        ],
        "selected_pattern": "structured_single_decision_router_with_conditional_edges",
        "patterns_not_selected": [
            "unbounded_agent_tool_loop",
            "model_selected_retrieval_loop",
            "checkpointer_backed_conversation_state",
            "persistent_cross_thread_store",
        ],
    }


def _guard_checks(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = report.get("stage155_summary") or {}
    decision = report.get("bounded_decision_contract") or {}
    state = report.get("volatile_thread_state_contract") or {}
    policy_cases = report.get("canonical_dynamic_policy_cases") or {}
    state_cases = report.get("canonical_thread_state_cases") or {}
    expected_policy_states = {
        "bounded_compose": "eligible",
        "bounded_refuse": "eligible",
        "unauthorized_tool": "rejected",
        "repeated_retrieval": "rejected",
        "decision_loop": "rejected",
        "model_final_authority": "rejected",
        "hidden_recovery": "rejected",
        "default_remote_test_open": "rejected",
    }
    return [
        _check("stage156_user_confirmed", report["user_confirmation"]["confirmed"] is True),
        _check(
            "stage156_confirmation_note_present",
            bool(report["user_confirmation"]["confirmation_note"]),
        ),
        _check("stage155_identity_exact", source.get("identity_exact") is True),
        _check("stage155_all_57_guards_passed", source.get("all_guards_passed") is True),
        _check(
            "stage155_runtime_evidence_exact",
            [
                source.get("activation_protocol_frozen"),
                source.get("node_observability_implemented"),
                source.get("real_lifecycle_validated"),
            ]
            == [True, True, True],
        ),
        _check(
            "stage155_closed_boundaries_preserved",
            [
                source.get("runtime_registered_as_default"),
                source.get("remote_exposure_authorized"),
                source.get("test_gate_opened"),
                source.get("test_metrics_run"),
            ]
            == [False, False, False, False],
        ),
        _check(
            "stage155_public_content_absent",
            [source.get("request_content_saved"), source.get("document_content_saved")]
            == [False, False],
        ),
        _check(
            "stage155_no_recovery_actions",
            [source.get("retry_action_count"), source.get("fallback_action_count")] == [0, 0],
        ),
        _check(
            "saved_source_unchanged",
            report.get("source_unchanged_after_protocol_freeze") is True,
        ),
        _check(
            "decision_actions_exact",
            decision.get("model_selectable_actions") == list(_MODEL_ACTIONS),
        ),
        _check(
            "system_actions_exact",
            decision.get("system_required_actions") == list(_SYSTEM_ACTIONS),
        ),
        _check(
            "branch_system_actions_exact",
            decision.get("system_actions_by_model_action")
            == {
                "compose_grounded_answer": [
                    "verify_grounded_answer",
                    "observe_diagnostics",
                    "finalize_verified_response",
                ],
                "refuse_insufficient_evidence": ["finalize_insufficient_evidence_refusal"],
            },
        ),
        _check(
            "single_decision_exact",
            decision.get("model_decision_count_per_turn") == 1,
        ),
        _check(
            "retrieval_system_owned_once",
            decision.get("retrieval_call_count_per_turn") == 1
            and "retrieve_candidate_pool" not in decision.get("model_selectable_actions", []),
        ),
        _check(
            "compose_verify_branch_exact",
            decision.get("composition_call_count_by_action")
            == {"compose_grounded_answer": 1, "refuse_insufficient_evidence": 0}
            and decision.get("verification_call_count_by_action")
            == {"compose_grounded_answer": 1, "refuse_insufficient_evidence": 0},
        ),
        _check(
            "diagnostics_system_read_only",
            decision.get("diagnostics_model_selectable") is False
            and decision.get("diagnostics_mutation_allowed") is False
            and decision.get("diagnostic_observation_count_by_action")
            == {"compose_grounded_answer": 1, "refuse_insufficient_evidence": 0},
        ),
        _check(
            "terminal_authority_exact",
            decision.get("terminal_authority_by_action")
            == {
                "compose_grounded_answer": "system_verifier",
                "refuse_insufficient_evidence": "fixed_system_refusal_constructor",
            },
        ),
        _check(
            "dynamic_loops_closed",
            decision.get("transition_loops_available") is False
            and decision.get("query_rewrite_enabled") is False
            and decision.get("second_retrieval_enabled") is False,
        ),
        _check(
            "parallel_recovery_closed",
            decision.get("parallel_tool_calls_enabled") is False
            and decision.get("retry_action_count") == 0
            and decision.get("fallback_action_count") == 0,
        ),
        _check(
            "volatile_storage_only",
            state.get("storage_scope") == "process_local_volatile_memory_only"
            and state.get("checkpointer_selected") is False
            and state.get("persistent_store_selected") is False
            and state.get("state_survives_process_restart") is False,
        ),
        _check(
            "thread_isolation_exact",
            state.get("thread_handle_required") is True
            and state.get("thread_handle_must_be_opaque") is True
            and state.get("cross_thread_read_allowed") is False,
        ),
        _check(
            "explicit_close_clears",
            state.get("explicit_close_clears_state") is True,
        ),
        _check(
            "cross_turn_fields_exact",
            state.get("cross_turn_private_fields") == list(_CROSS_TURN_PRIVATE_FIELDS),
        ),
        _check(
            "turn_local_discard_exact",
            state.get("turn_local_fields_discarded") == list(_TURN_LOCAL_DISCARD_FIELDS),
        ),
        _check(
            "public_trace_fields_exact",
            state.get("public_trace_fields") == list(_PUBLIC_TRACE_FIELDS),
        ),
        _check(
            "limits_mandatory_but_deferred",
            state.get("history_limit_required") is True
            and state.get("byte_limit_required") is True
            and state.get("runtime_limit_values_frozen_in_stage156") is False
            and state.get("runtime_limit_values_require_explicit_configuration") is True,
        ),
        _check(
            "overflow_rejects_before_mutation",
            state.get("overflow_behavior") == "reject_before_mutation"
            and state.get("silent_truncation_allowed") is False
            and state.get("silent_eviction_allowed") is False,
        ),
        _check(
            "implicit_creation_and_reconstruction_closed",
            state.get("implicit_thread_creation_allowed") is False
            and state.get("state_reconstruction_fallback_allowed") is False,
        ),
        *[
            _check(
                f"policy_case_{name}",
                policy_cases.get(name, {}).get("state") == expected,
            )
            for name, expected in expected_policy_states.items()
        ],
        *[
            _check(
                f"thread_case_{name}",
                state_cases.get(name, {}).get("state") == "eligible",
            )
            for name in (
                "complete_turn_retained",
                "cross_thread_isolated",
                "byte_overflow_rejected_without_mutation",
                "implicit_thread_creation_rejected",
                "explicit_close_clears_state",
            )
        ],
        _check(
            "official_sources_exact",
            len(report.get("official_framework_research", {}).get("sources") or []) == 4,
        ),
        _check(
            "test_remained_closed",
            source.get("test_gate_opened") is False and source.get("test_metrics_run") is False,
        ),
    ]


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "test_split_loaded": False,
        "test_metrics_run": False,
        "train_or_dev_rows_loaded": False,
        "questions_loaded": False,
        "documents_loaded": False,
        "models_loaded_or_called": False,
        "indexes_loaded": False,
        "candidate_pools_loaded": False,
        "request_content_saved": False,
        "request_identifiers_saved": False,
        "document_content_saved": False,
        "document_identifiers_saved": False,
        "synthetic_private_content_saved": False,
        "runtime_changed": False,
        "runtime_registered_as_default": False,
        "remote_exposure_authorized": False,
        "queue_action_count": 0,
        "retry_action_count": 0,
        "fallback_action_count": 0,
    }
    payload["forbidden_keys_found"] = sorted(_forbidden_keys_found(report))
    return payload


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _fingerprint(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {"size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _bar(label: str, value: int | float) -> BarDatum:
    return BarDatum(label=label, value=float(value), value_label=str(value))


def _truth_bar(label: str, enabled: bool) -> BarDatum:
    return BarDatum(
        label=label,
        value=1.0 if enabled else 0.0,
        value_label="enabled" if enabled else "closed",
    )


def _case_chart(title: str, cases: Mapping[str, Any]) -> str:
    return _chart(
        title,
        [
            BarDatum(
                label=str(name),
                value=1.0 if row.get("state") == "eligible" else 0.0,
                value_label=str(row.get("state")),
            )
            for name, row in cases.items()
        ],
    )


def _chart(
    title: str,
    data: list[BarDatum],
    *,
    width: int = 1800,
    margin_left: int = 850,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=data,
        x_label="contract value",
        width=width,
        margin_left=margin_left,
    )
