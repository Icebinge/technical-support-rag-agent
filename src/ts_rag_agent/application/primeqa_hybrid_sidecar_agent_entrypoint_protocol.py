from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from ts_rag_agent.application.primeqa_hybrid_sidecar_agent_orchestrator import (
    sidecar_agent_orchestrator_contract,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 138"
_CREATED_AT = "2026-07-17"
_PROTOCOL_ID = "primeqa_hybrid_optional_sidecar_agent_entrypoint_protocol_v1"
_SOURCE_STAGE137_STATUS = "primeqa_hybrid_sidecar_agent_orchestrator_train_cv_dev_validation_passed"
_SOURCE_STAGE137_ANALYSIS_ID = (
    "primeqa_hybrid_sidecar_agent_orchestrator_train_cv_dev_validation_v1"
)
_SOURCE_ORCHESTRATOR_ID = "stage116_primary_plus_stage128_sidecar_agent_orchestrator_v1"
_NEXT_DIRECTION = (
    "implement_optional_sidecar_agent_entrypoint_and_train_dev_action_trace_validation"
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer",
        "answer_doc_id",
        "answer_text",
        "candidate_doc_ids",
        "cited_doc_ids",
        "document_body",
        "document_id",
        "document_text",
        "document_title",
        "gold_answer",
        "matched_token_strings",
        "question_id",
        "question_text",
        "question_title",
        "raw_answer_text",
        "raw_document_text",
        "raw_question_text",
        "retrieved_doc_ids",
        "runtime_content_handle",
        "source_doc_ids",
    }
)


class SidecarAgentState(str, Enum):
    """Frozen states for the optional sidecar-agent entrypoint."""

    READY = "ready"
    RETRIEVE = "retrieve"
    ANSWER = "answer"
    VERIFY = "verify"
    OBSERVE = "observe"
    COMPLETE = "complete"
    REFUSE = "refuse"


class SidecarAgentAction(str, Enum):
    """Actions that move the optional entrypoint between frozen states."""

    RETRIEVE = "retrieve"
    ANSWER = "answer"
    VERIFY = "verify"
    OBSERVE = "observe"
    COMPLETE = "complete"
    REFUSE = "refuse"


class InvalidSidecarAgentTransitionError(ValueError):
    """Raised when an action is not allowed from the current state."""


@dataclass(frozen=True)
class SidecarAgentTransition:
    """One public-safe action-state transition."""

    sequence_number: int
    previous_state: SidecarAgentState
    action: SidecarAgentAction
    next_state: SidecarAgentState

    def to_public_dict(self) -> dict[str, Any]:
        payload = {
            "sequence_number": self.sequence_number,
            "previous_state": self.previous_state.value,
            "action": self.action.value,
            "next_state": self.next_state.value,
        }
        forbidden = sorted(_forbidden_keys_found(payload))
        if forbidden:
            raise ValueError(f"Action trace contains forbidden keys: {forbidden}")
        return payload


class SidecarAgentTransitionPolicy(Protocol):
    """Polymorphic transition policy accepted by the state machine."""

    def allowed_transitions(
        self,
    ) -> tuple[tuple[SidecarAgentState, SidecarAgentAction, SidecarAgentState], ...]: ...

    def next_state(
        self,
        *,
        current_state: SidecarAgentState,
        action: SidecarAgentAction,
    ) -> SidecarAgentState: ...


class FrozenSidecarAgentTransitionPolicy:
    """Deterministic Stage138 policy with no loops, retries, or fallback path."""

    _TRANSITIONS = (
        (SidecarAgentState.READY, SidecarAgentAction.RETRIEVE, SidecarAgentState.RETRIEVE),
        (
            SidecarAgentState.RETRIEVE,
            SidecarAgentAction.ANSWER,
            SidecarAgentState.ANSWER,
        ),
        (SidecarAgentState.ANSWER, SidecarAgentAction.VERIFY, SidecarAgentState.VERIFY),
        (
            SidecarAgentState.VERIFY,
            SidecarAgentAction.OBSERVE,
            SidecarAgentState.OBSERVE,
        ),
        (
            SidecarAgentState.OBSERVE,
            SidecarAgentAction.COMPLETE,
            SidecarAgentState.COMPLETE,
        ),
        (
            SidecarAgentState.OBSERVE,
            SidecarAgentAction.REFUSE,
            SidecarAgentState.REFUSE,
        ),
    )

    def allowed_transitions(
        self,
    ) -> tuple[tuple[SidecarAgentState, SidecarAgentAction, SidecarAgentState], ...]:
        return self._TRANSITIONS

    def next_state(
        self,
        *,
        current_state: SidecarAgentState,
        action: SidecarAgentAction,
    ) -> SidecarAgentState:
        for source, allowed_action, target in self._TRANSITIONS:
            if source is current_state and allowed_action is action:
                return target
        raise InvalidSidecarAgentTransitionError(
            f"Action {action.value!r} is not allowed from state {current_state.value!r}"
        )


class SidecarAgentActionStateMachine:
    """Execute only the frozen public-safe Stage138 transition contract."""

    def __init__(self, transition_policy: SidecarAgentTransitionPolicy | None = None) -> None:
        self._transition_policy = transition_policy or FrozenSidecarAgentTransitionPolicy()
        self._state = SidecarAgentState.READY
        self._trace: tuple[SidecarAgentTransition, ...] = ()

    @property
    def state(self) -> SidecarAgentState:
        return self._state

    @property
    def trace(self) -> tuple[SidecarAgentTransition, ...]:
        return self._trace

    @property
    def terminal(self) -> bool:
        return self._state in {SidecarAgentState.COMPLETE, SidecarAgentState.REFUSE}

    def advance(self, action: SidecarAgentAction) -> SidecarAgentTransition:
        next_state = self._transition_policy.next_state(
            current_state=self._state,
            action=action,
        )
        transition = SidecarAgentTransition(
            sequence_number=len(self._trace) + 1,
            previous_state=self._state,
            action=action,
            next_state=next_state,
        )
        transition.to_public_dict()
        self._state = next_state
        self._trace = (*self._trace, transition)
        return transition

    def run(self, actions: Sequence[SidecarAgentAction]) -> tuple[SidecarAgentTransition, ...]:
        for action in actions:
            self.advance(action)
        return self._trace

    def public_trace(self) -> list[dict[str, Any]]:
        return [transition.to_public_dict() for transition in self._trace]


@dataclass(frozen=True)
class PrimeQAHybridSidecarAgentEntrypointProtocolVisualization:
    """One generated Stage138 protocol chart."""

    name: str
    path: str


def sidecar_agent_action_state_contract() -> dict[str, Any]:
    """Return the executable Stage138 action-state contract."""

    policy = FrozenSidecarAgentTransitionPolicy()
    transitions = [
        {
            "previous_state": source.value,
            "action": action.value,
            "next_state": target.value,
        }
        for source, action, target in policy.allowed_transitions()
    ]
    return {
        "initial_state": SidecarAgentState.READY.value,
        "states": [state.value for state in SidecarAgentState],
        "actions": [action.value for action in SidecarAgentAction],
        "allowed_transitions": transitions,
        "terminal_states": [
            SidecarAgentState.COMPLETE.value,
            SidecarAgentState.REFUSE.value,
        ],
        "accepted_path": [
            SidecarAgentAction.RETRIEVE.value,
            SidecarAgentAction.ANSWER.value,
            SidecarAgentAction.VERIFY.value,
            SidecarAgentAction.OBSERVE.value,
            SidecarAgentAction.COMPLETE.value,
        ],
        "refused_path": [
            SidecarAgentAction.RETRIEVE.value,
            SidecarAgentAction.ANSWER.value,
            SidecarAgentAction.VERIFY.value,
            SidecarAgentAction.OBSERVE.value,
            SidecarAgentAction.REFUSE.value,
        ],
        "invalid_transition_behavior": "raise_without_state_change",
        "retry_actions_available": False,
        "fallback_transitions_available": False,
        "transition_loops_available": False,
    }


def freeze_primeqa_hybrid_sidecar_agent_entrypoint_protocol(
    *,
    stage137_validation_path: Path,
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Freeze the optional Stage138 entrypoint and action-state protocol."""

    started_at = time.perf_counter()
    stage137_validation = _load_json_object(stage137_validation_path)
    loaded_at = time.perf_counter()

    stage137_summary = _stage137_summary(stage137_validation)
    orchestrator_contract = sidecar_agent_orchestrator_contract()
    action_state_contract = sidecar_agent_action_state_contract()
    canonical_traces = _canonical_action_traces()
    frozen_protocol = _frozen_protocol(
        stage137_summary=stage137_summary,
        orchestrator_contract=orchestrator_contract,
        action_state_contract=action_state_contract,
        canonical_traces=canonical_traces,
    )
    preliminary = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "protocol_id": _PROTOCOL_ID,
        "protocol_scope": (
            "Public-safe Stage138 freeze of an optional sidecar-agent entrypoint and "
            "executable action-state contract around the Stage137-validated orchestrator. "
            "This stage reads only the saved public-safe Stage137 aggregate report and "
            "executes synthetic state transitions without questions, documents, candidate "
            "rows, model outputs, split files, gold labels, or test data. It does not wire "
            "the entrypoint into runtime, rerun retrieval or answer evaluation, claim "
            "sidecar effectiveness, open final-test gates, change defaults, or add retries "
            "or fallback strategies."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_protocol),
            "confirmation_note": confirmation_note,
        },
        "source_files": {
            "stage137_validation": _fingerprint(stage137_validation_path),
        },
        "stage137_summary": stage137_summary,
        "frozen_protocol": frozen_protocol,
    }
    guard_checks = _guard_checks(
        report=preliminary,
        stage137_summary=stage137_summary,
        orchestrator_contract=orchestrator_contract,
        action_state_contract=action_state_contract,
        canonical_traces=canonical_traces,
        user_confirmed_protocol=user_confirmed_protocol,
        confirmation_note=confirmation_note,
    )
    checked_at = time.perf_counter()
    report = {
        **preliminary,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks),
        "timing_seconds": {
            "load_source": round(loaded_at - started_at, 3),
            "freeze_and_guard": round(checked_at - loaded_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_sidecar_agent_entrypoint_protocol_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridSidecarAgentEntrypointProtocolVisualization]:
    """Write public-safe Stage138 SVG charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage138_stage137_identity_isolation_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage138 Stage137 source identity and isolation counts",
            bars=_source_identity_isolation_bars(report),
            x_label="violation count",
            width=1900,
            margin_left=1040,
        ),
        "stage138_sidecar_opportunity_capture.svg": render_horizontal_bar_chart_svg(
            title="Stage138 inherited sidecar opportunity boundary",
            bars=_opportunity_capture_bars(report),
            x_label="answerable row count",
            width=1660,
            margin_left=860,
        ),
        "stage138_state_outdegree.svg": render_horizontal_bar_chart_svg(
            title="Stage138 action-state outdegree",
            bars=_state_outdegree_bars(report),
            x_label="allowed outgoing transitions",
            width=1540,
            margin_left=760,
        ),
        "stage138_entrypoint_permission_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage138 optional entrypoint permission flags",
            bars=_entrypoint_permission_bars(report),
            x_label="1 means enabled",
            width=1880,
            margin_left=1040,
        ),
        "stage138_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage138 decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1920,
            margin_left=1060,
        ),
        "stage138_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage138 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=2260,
            margin_left=1300,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridSidecarAgentEntrypointProtocolVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage137_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = report.get("decision") or {}
    guard_checks = report.get("guard_checks") or []
    split_reports = report.get("split_agent_reports") or {}
    public_safe = report.get("public_safe_contract") or {}
    return {
        "stage": report.get("stage"),
        "analysis_id": report.get("analysis_id"),
        "status": decision.get("status"),
        "guard_check_count": len(guard_checks),
        "guard_check_passed_count": sum(bool(check.get("passed")) for check in guard_checks),
        "agent_orchestrator_integration_validated": decision.get(
            "agent_orchestrator_integration_validated"
        ),
        "can_freeze_optional_agent_entrypoint_protocol_now": decision.get(
            "can_freeze_optional_agent_entrypoint_protocol_now"
        ),
        "sidecar_effectiveness_status": decision.get("sidecar_effectiveness_status"),
        "sidecar_citation_verification_effectiveness_demonstrated": decision.get(
            "sidecar_citation_verification_effectiveness_demonstrated"
        ),
        "can_claim_answer_quality_improvement": decision.get(
            "can_claim_answer_quality_improvement"
        ),
        "can_claim_retrieval_improvement": decision.get("can_claim_retrieval_improvement"),
        "sidecar_can_generate_answer_text": decision.get("sidecar_can_generate_answer_text"),
        "sidecar_can_enter_answer_verification_context": decision.get(
            "sidecar_can_enter_answer_verification_context"
        ),
        "sidecar_can_replace_primary_context": decision.get("sidecar_can_replace_primary_context"),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "runtime_defaultization_allowed_now": decision.get("runtime_defaultization_allowed_now"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "train": _split_summary(split_reports.get("train") or {}),
        "dev": _split_summary(split_reports.get("dev") or {}),
        "public_safe_forbidden_keys_found": public_safe.get("forbidden_keys_found") or [],
        "test_split_loaded": public_safe.get("test_split_loaded"),
        "final_test_metrics_run": public_safe.get("final_test_metrics_run"),
    }


def _split_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "row_count",
        "answerable_count",
        "generation_context_identity_violation_count",
        "verification_context_identity_violation_count",
        "bundle_generation_context_identity_violation_count",
        "original_answer_identity_violation_count",
        "verified_answer_identity_violation_count",
        "verification_reason_identity_violation_count",
        "sidecar_generation_leak_count",
        "sidecar_verification_leak_count",
        "sidecar_primary_overlap_count",
        "public_trace_serialization_violation_count",
        "public_trace_forbidden_key_count",
        "public_trace_contract_violation_count",
        "sidecar_observation_count",
        "sidecar_observation_availability_rate",
        "append_pool_incremental_gold_hit_count",
        "sidecar_incremental_gold_hit_count",
        "sidecar_capture_rate_of_append_gold_opportunities",
    )
    return {key: summary.get(key) for key in keys}


def _frozen_protocol(
    *,
    stage137_summary: Mapping[str, Any],
    orchestrator_contract: Mapping[str, Any],
    action_state_contract: Mapping[str, Any],
    canonical_traces: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_id": _PROTOCOL_ID,
        "implementation_status": "action_state_machine_executable_entrypoint_not_runtime_wired",
        "entrypoint_policy": {
            "optional_entrypoint": True,
            "explicit_activation_required": True,
            "registered_as_runtime_default": False,
            "runtime_defaultization_allowed": False,
            "retry_actions_allowed": False,
            "fallback_strategies_allowed": False,
            "test_access_allowed": False,
        },
        "orchestrator_reuse_contract": {
            "orchestrator_id": orchestrator_contract.get("orchestrator_id"),
            "source_validation_status": stage137_summary.get("status"),
            "source_integration_validated": stage137_summary.get(
                "agent_orchestrator_integration_validated"
            ),
            "answer_generation_channel": (orchestrator_contract.get("channel_routing") or {}).get(
                "answer_generation"
            ),
            "answer_verification_channel": (orchestrator_contract.get("channel_routing") or {}).get(
                "answer_verification"
            ),
            "sidecar_observation_channel": (orchestrator_contract.get("channel_routing") or {}).get(
                "sidecar_observation"
            ),
            "sidecar_used_for_answer_generation": False,
            "sidecar_used_for_answer_verification": False,
            "sidecar_replaces_primary_context": False,
        },
        "action_state_contract": dict(action_state_contract),
        "canonical_public_traces": dict(canonical_traces),
        "execution_semantics": [
            {
                "action": SidecarAgentAction.RETRIEVE.value,
                "owner": "future_optional_entrypoint_retrieval_port",
                "constraint": "produce_frozen_stage128_candidate_pool_without_test_access",
            },
            {
                "action": SidecarAgentAction.ANSWER.value,
                "owner": _SOURCE_ORCHESTRATOR_ID,
                "constraint": "use_only_stage116_primary_answer_context",
            },
            {
                "action": SidecarAgentAction.VERIFY.value,
                "owner": _SOURCE_ORCHESTRATOR_ID,
                "constraint": "use_only_stage116_prefix_verification_context",
            },
            {
                "action": SidecarAgentAction.OBSERVE.value,
                "owner": _SOURCE_ORCHESTRATOR_ID,
                "constraint": "publish_sidecar_diagnostics_only_after_verification",
            },
            {
                "action": SidecarAgentAction.COMPLETE.value,
                "owner": "future_optional_entrypoint",
                "constraint": "return_only_the_verified_non_refused_result",
            },
            {
                "action": SidecarAgentAction.REFUSE.value,
                "owner": "future_optional_entrypoint",
                "constraint": "terminate_with_the_verified_refusal_without_retry_or_fallback",
            },
        ],
        "implementation_boundary": {
            "stage138_executes_state_transitions": True,
            "stage138_executes_runtime_orchestrator": False,
            "stage138_claims_runtime_action_order_validation": False,
            "stage139_must_implement_optional_adapter": True,
            "stage139_must_revalidate_answer_path_invariance": True,
        },
        "effectiveness_boundary": {
            "status": "safe_but_neutral",
            "citation_verification_effectiveness_claim_allowed": False,
            "answer_quality_improvement_claim_allowed": False,
            "retrieval_improvement_claim_allowed": False,
        },
    }


def _canonical_action_traces() -> dict[str, Any]:
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
    return {
        "accepted": {
            "terminal_state": accepted.state.value,
            "terminal": accepted.terminal,
            "transitions": accepted.public_trace(),
        },
        "refused": {
            "terminal_state": refused.state.value,
            "terminal": refused.terminal,
            "transitions": refused.public_trace(),
        },
    }


def _guard_checks(
    *,
    report: Mapping[str, Any],
    stage137_summary: Mapping[str, Any],
    orchestrator_contract: Mapping[str, Any],
    action_state_contract: Mapping[str, Any],
    canonical_traces: Mapping[str, Any],
    user_confirmed_protocol: bool,
    confirmation_note: str,
) -> list[dict[str, Any]]:
    identity_keys = (
        "generation_context_identity_violation_count",
        "verification_context_identity_violation_count",
        "bundle_generation_context_identity_violation_count",
        "original_answer_identity_violation_count",
        "verified_answer_identity_violation_count",
        "verification_reason_identity_violation_count",
    )
    isolation_keys = (
        "sidecar_generation_leak_count",
        "sidecar_verification_leak_count",
        "sidecar_primary_overlap_count",
    )
    trace_keys = (
        "public_trace_serialization_violation_count",
        "public_trace_forbidden_key_count",
        "public_trace_contract_violation_count",
    )
    consumer_policy = orchestrator_contract.get("consumer_policy") or {}
    transitions = action_state_contract.get("allowed_transitions") or []
    expected_transitions = [
        {"previous_state": "ready", "action": "retrieve", "next_state": "retrieve"},
        {"previous_state": "retrieve", "action": "answer", "next_state": "answer"},
        {"previous_state": "answer", "action": "verify", "next_state": "verify"},
        {"previous_state": "verify", "action": "observe", "next_state": "observe"},
        {"previous_state": "observe", "action": "complete", "next_state": "complete"},
        {"previous_state": "observe", "action": "refuse", "next_state": "refuse"},
    ]
    return [
        _check(
            name="user_confirmed_stage138_protocol",
            passed=user_confirmed_protocol and bool(confirmation_note.strip()),
            observed={"confirmed": user_confirmed_protocol, "note": confirmation_note},
            expected="confirmed with non-empty note",
        ),
        _check(
            name="stage137_source_identity_matches",
            passed=stage137_summary.get("stage") == "Stage 137"
            and stage137_summary.get("analysis_id") == _SOURCE_STAGE137_ANALYSIS_ID,
            observed={
                "stage": stage137_summary.get("stage"),
                "analysis_id": stage137_summary.get("analysis_id"),
            },
            expected={"stage": "Stage 137", "analysis_id": _SOURCE_STAGE137_ANALYSIS_ID},
        ),
        _check(
            name="stage137_validation_passed",
            passed=stage137_summary.get("status") == _SOURCE_STAGE137_STATUS,
            observed=stage137_summary.get("status"),
            expected=_SOURCE_STAGE137_STATUS,
        ),
        _check(
            name="stage137_all_guards_passed",
            passed=stage137_summary.get("guard_check_count") == 36
            and stage137_summary.get("guard_check_passed_count") == 36,
            observed={
                "passed": stage137_summary.get("guard_check_passed_count"),
                "total": stage137_summary.get("guard_check_count"),
            },
            expected={"passed": 36, "total": 36},
        ),
        _check(
            name="stage137_agent_integration_validated",
            passed=stage137_summary.get("agent_orchestrator_integration_validated") is True,
            observed=stage137_summary.get("agent_orchestrator_integration_validated"),
            expected=True,
        ),
        _check(
            name="stage137_authorizes_optional_entrypoint_protocol_freeze",
            passed=stage137_summary.get("can_freeze_optional_agent_entrypoint_protocol_now")
            is True,
            observed=stage137_summary.get("can_freeze_optional_agent_entrypoint_protocol_now"),
            expected=True,
        ),
        _check(
            name="stage137_source_splits_are_nonempty",
            passed=all(
                _numeric_field_positive(stage137_summary.get(split) or {}, "row_count")
                for split in ("train", "dev")
            ),
            observed={
                split: (stage137_summary.get(split) or {}).get("row_count")
                for split in ("train", "dev")
            },
            expected="positive train and dev row counts",
        ),
        _check(
            name="stage137_answer_path_identity_violations_are_zero",
            passed=_all_split_values_zero(stage137_summary, identity_keys),
            observed=_split_values(stage137_summary, identity_keys),
            expected=0,
        ),
        _check(
            name="stage137_sidecar_isolation_violations_are_zero",
            passed=_all_split_values_zero(stage137_summary, isolation_keys),
            observed=_split_values(stage137_summary, isolation_keys),
            expected=0,
        ),
        _check(
            name="stage137_public_trace_violations_are_zero",
            passed=_all_split_values_zero(stage137_summary, trace_keys),
            observed=_split_values(stage137_summary, trace_keys),
            expected=0,
        ),
        _check(
            name="stage137_sidecar_boundary_is_safe_but_neutral",
            passed=stage137_summary.get("sidecar_effectiveness_status") == "safe_but_neutral"
            and stage137_summary.get("sidecar_citation_verification_effectiveness_demonstrated")
            is False,
            observed={
                "status": stage137_summary.get("sidecar_effectiveness_status"),
                "effectiveness": stage137_summary.get(
                    "sidecar_citation_verification_effectiveness_demonstrated"
                ),
            },
            expected="safe_but_neutral without effectiveness claim",
        ),
        _check(
            name="stage137_opportunities_remain_uncaptured",
            passed=(stage137_summary.get("train") or {}).get(
                "append_pool_incremental_gold_hit_count"
            )
            == 9
            and (stage137_summary.get("dev") or {}).get("append_pool_incremental_gold_hit_count")
            == 1
            and all(
                _numeric_field_equals(
                    stage137_summary.get(split) or {}, "sidecar_incremental_gold_hit_count", 0
                )
                for split in ("train", "dev")
            ),
            observed={
                split: {
                    "opportunities": (stage137_summary.get(split) or {}).get(
                        "append_pool_incremental_gold_hit_count"
                    ),
                    "captures": (stage137_summary.get(split) or {}).get(
                        "sidecar_incremental_gold_hit_count"
                    ),
                }
                for split in ("train", "dev")
            },
            expected={"train": "9/0", "dev": "1/0"},
        ),
        _check(
            name="stage137_improvement_claims_remain_blocked",
            passed=stage137_summary.get("can_claim_answer_quality_improvement") is False
            and stage137_summary.get("can_claim_retrieval_improvement") is False,
            observed={
                "answer_quality": stage137_summary.get("can_claim_answer_quality_improvement"),
                "retrieval": stage137_summary.get("can_claim_retrieval_improvement"),
            },
            expected=False,
        ),
        _check(
            name="stage137_test_gate_remains_locked",
            passed=stage137_summary.get("can_open_final_test_gate_now") is False
            and stage137_summary.get("can_run_final_test_metrics_now") is False
            and stage137_summary.get("can_use_test_for_tuning") is False
            and stage137_summary.get("test_split_loaded") is False
            and stage137_summary.get("final_test_metrics_run") is False,
            observed={
                "open_gate": stage137_summary.get("can_open_final_test_gate_now"),
                "run_metrics": stage137_summary.get("can_run_final_test_metrics_now"),
                "tune": stage137_summary.get("can_use_test_for_tuning"),
                "loaded": stage137_summary.get("test_split_loaded"),
            },
            expected="test locked and unloaded",
        ),
        _check(
            name="stage137_runtime_defaults_unchanged",
            passed=stage137_summary.get("runtime_defaultization_allowed_now") is False
            and stage137_summary.get("default_runtime_policy") == "unchanged",
            observed={
                "allowed": stage137_summary.get("runtime_defaultization_allowed_now"),
                "policy": stage137_summary.get("default_runtime_policy"),
            },
            expected="unchanged",
        ),
        _check(
            name="stage137_fallback_strategies_disabled",
            passed=stage137_summary.get("fallback_strategies_enabled") is False,
            observed=stage137_summary.get("fallback_strategies_enabled"),
            expected=False,
        ),
        _check(
            name="stage137_public_source_has_no_forbidden_keys",
            passed=stage137_summary.get("public_safe_forbidden_keys_found") == [],
            observed=stage137_summary.get("public_safe_forbidden_keys_found"),
            expected=[],
        ),
        _check(
            name="validated_orchestrator_identity_is_reused",
            passed=orchestrator_contract.get("orchestrator_id") == _SOURCE_ORCHESTRATOR_ID,
            observed=orchestrator_contract.get("orchestrator_id"),
            expected=_SOURCE_ORCHESTRATOR_ID,
        ),
        _check(
            name="orchestrator_sidecar_answer_permissions_remain_blocked",
            passed=all(
                consumer_policy.get(key) is False
                for key in (
                    "answer_generation_allowed",
                    "answer_verification_context_allowed",
                    "primary_context_replacement_allowed",
                )
            ),
            observed=consumer_policy,
            expected="all sidecar answer-path permissions false",
        ),
        _check(
            name="entrypoint_is_optional_and_not_default",
            passed=(report.get("frozen_protocol") or {})
            .get("entrypoint_policy", {})
            .get("optional_entrypoint")
            is True
            and (report.get("frozen_protocol") or {})
            .get("entrypoint_policy", {})
            .get("registered_as_runtime_default")
            is False,
            observed=(report.get("frozen_protocol") or {}).get("entrypoint_policy"),
            expected="optional true and runtime default false",
        ),
        _check(
            name="stage138_is_protocol_only_not_runtime_wired",
            passed=(report.get("frozen_protocol") or {}).get("implementation_status")
            == "action_state_machine_executable_entrypoint_not_runtime_wired",
            observed=(report.get("frozen_protocol") or {}).get("implementation_status"),
            expected="action_state_machine_executable_entrypoint_not_runtime_wired",
        ),
        _check(
            name="action_state_catalog_is_exact",
            passed=action_state_contract.get("states")
            == [state.value for state in SidecarAgentState]
            and action_state_contract.get("actions")
            == [action.value for action in SidecarAgentAction],
            observed={
                "states": action_state_contract.get("states"),
                "actions": action_state_contract.get("actions"),
            },
            expected="frozen Stage138 state and action catalogs",
        ),
        _check(
            name="allowed_transition_graph_is_exact",
            passed=transitions == expected_transitions,
            observed=transitions,
            expected=expected_transitions,
        ),
        _check(
            name="transition_graph_has_no_loops",
            passed=action_state_contract.get("transition_loops_available") is False
            and all(row.get("previous_state") != row.get("next_state") for row in transitions),
            observed=transitions,
            expected="acyclic transition graph",
        ),
        _check(
            name="complete_and_refuse_are_terminal",
            passed=action_state_contract.get("terminal_states") == ["complete", "refuse"]
            and not any(row.get("previous_state") in {"complete", "refuse"} for row in transitions),
            observed=action_state_contract.get("terminal_states"),
            expected=["complete", "refuse"],
        ),
        _check(
            name="observe_occurs_only_after_verify",
            passed=[
                row for row in transitions if row.get("action") == SidecarAgentAction.OBSERVE.value
            ]
            == [{"previous_state": "verify", "action": "observe", "next_state": "observe"}],
            observed=[row for row in transitions if row.get("action") == "observe"],
            expected="verify -> observe",
        ),
        _check(
            name="refuse_occurs_only_after_observe",
            passed=[row for row in transitions if row.get("action") == "refuse"]
            == [{"previous_state": "observe", "action": "refuse", "next_state": "refuse"}],
            observed=[row for row in transitions if row.get("action") == "refuse"],
            expected="observe -> refuse",
        ),
        _check(
            name="canonical_accepted_path_reaches_complete",
            passed=(canonical_traces.get("accepted") or {}).get("terminal") is True
            and (canonical_traces.get("accepted") or {}).get("terminal_state") == "complete"
            and len((canonical_traces.get("accepted") or {}).get("transitions") or []) == 5,
            observed=canonical_traces.get("accepted"),
            expected="five transitions ending complete",
        ),
        _check(
            name="canonical_refused_path_reaches_refuse",
            passed=(canonical_traces.get("refused") or {}).get("terminal") is True
            and (canonical_traces.get("refused") or {}).get("terminal_state") == "refuse"
            and len((canonical_traces.get("refused") or {}).get("transitions") or []) == 5,
            observed=canonical_traces.get("refused"),
            expected="five transitions ending refuse",
        ),
        _check(
            name="invalid_transitions_raise_without_fallback",
            passed=action_state_contract.get("invalid_transition_behavior")
            == "raise_without_state_change"
            and action_state_contract.get("retry_actions_available") is False
            and action_state_contract.get("fallback_transitions_available") is False,
            observed={
                "behavior": action_state_contract.get("invalid_transition_behavior"),
                "retry": action_state_contract.get("retry_actions_available"),
                "fallback": action_state_contract.get("fallback_transitions_available"),
            },
            expected="raise, no retry, no fallback",
        ),
        _check(
            name="stage138_public_protocol_has_no_forbidden_keys",
            passed=not _forbidden_keys_found(report),
            observed=sorted(_forbidden_keys_found(report)),
            expected=[],
        ),
    ]


def _decision(guard_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    base = {
        "protocol_id": _PROTOCOL_ID,
        "entrypoint_optional": True,
        "entrypoint_registered_as_runtime_default": False,
        "state_machine_executable": True,
        "runtime_entrypoint_implemented": False,
        "runtime_action_order_validated": False,
        "agent_orchestrator_integration_source_validated": True,
        "sidecar_effectiveness_status": "safe_but_neutral",
        "can_claim_citation_verification_effectiveness": False,
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
        "retry_actions_enabled": False,
        "default_runtime_policy": "unchanged",
    }
    if failed_checks:
        return {
            **base,
            "status": "primeqa_hybrid_optional_sidecar_agent_entrypoint_protocol_blocked",
            "failed_checks": failed_checks,
            "optional_agent_entrypoint_protocol_frozen": False,
            "can_implement_optional_agent_entrypoint_now": False,
            "recommended_next_direction": "review_stage138_entrypoint_protocol_failures",
        }
    return {
        **base,
        "status": "primeqa_hybrid_optional_sidecar_agent_entrypoint_protocol_frozen",
        "failed_checks": [],
        "optional_agent_entrypoint_protocol_frozen": True,
        "can_implement_optional_agent_entrypoint_now": True,
        "recommended_next_direction": _NEXT_DIRECTION,
    }


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "public_safe_summary_only": True,
        "synthetic_action_traces_written": True,
        "runtime_orchestrator_executed": False,
        "raw_question_text_written": False,
        "raw_answer_text_written": False,
        "raw_document_text_written": False,
        "raw_document_ids_written": False,
        "raw_runtime_content_handles_written": False,
        "raw_candidate_rows_written": False,
        "raw_sample_ids_written": False,
        "split_files_loaded": False,
        "corpus_documents_loaded": False,
        "test_split_loaded": False,
        "final_test_metrics_run": False,
        "forbidden_keys_found": sorted(_forbidden_keys_found(report)),
    }


def _source_identity_isolation_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    summary = report.get("stage137_summary") or {}
    keys = (
        "generation_context_identity_violation_count",
        "verification_context_identity_violation_count",
        "verified_answer_identity_violation_count",
        "verification_reason_identity_violation_count",
        "sidecar_generation_leak_count",
        "sidecar_verification_leak_count",
        "sidecar_primary_overlap_count",
        "public_trace_contract_violation_count",
    )
    return [
        BarDatum(
            label=f"{split} {key}",
            value=float((summary.get(split) or {}).get(key) or 0),
            value_label=str(int((summary.get(split) or {}).get(key) or 0)),
        )
        for split in ("train", "dev")
        for key in keys
    ]


def _opportunity_capture_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    summary = report.get("stage137_summary") or {}
    bars = []
    for split in ("train", "dev"):
        split_summary = summary.get(split) or {}
        for key, label in (
            ("append_pool_incremental_gold_hit_count", "append opportunities"),
            ("sidecar_incremental_gold_hit_count", "sidecar captures"),
        ):
            value = int(split_summary.get(key) or 0)
            bars.append(
                BarDatum(
                    label=f"{split} {label}",
                    value=float(value),
                    value_label=str(value),
                )
            )
    return bars


def _state_outdegree_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    contract = (report.get("frozen_protocol") or {}).get("action_state_contract") or {}
    transitions = contract.get("allowed_transitions") or []
    return [
        BarDatum(
            label=state,
            value=float(sum(row.get("previous_state") == state for row in transitions)),
            value_label=str(sum(row.get("previous_state") == state for row in transitions)),
        )
        for state in contract.get("states") or []
    ]


def _entrypoint_permission_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    policy = (report.get("frozen_protocol") or {}).get("entrypoint_policy") or {}
    return [
        BarDatum(
            label=key,
            value=1.0 if policy.get(key) else 0.0,
            value_label="enabled" if policy.get(key) else "disabled",
        )
        for key in (
            "optional_entrypoint",
            "explicit_activation_required",
            "registered_as_runtime_default",
            "runtime_defaultization_allowed",
            "retry_actions_allowed",
            "fallback_strategies_allowed",
            "test_access_allowed",
        )
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    return [
        BarDatum(
            label=key,
            value=1.0 if decision.get(key) else 0.0,
            value_label="true" if decision.get(key) else "false",
        )
        for key in (
            "optional_agent_entrypoint_protocol_frozen",
            "state_machine_executable",
            "runtime_entrypoint_implemented",
            "runtime_action_order_validated",
            "can_implement_optional_agent_entrypoint_now",
            "can_claim_answer_quality_improvement",
            "can_open_final_test_gate_now",
            "runtime_defaultization_allowed_now",
            "fallback_strategies_enabled",
        )
    ]


def _guard_check_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(check.get("name")),
            value=1.0 if check.get("passed") else 0.0,
            value_label="passed" if check.get("passed") else "failed",
        )
        for check in report.get("guard_checks") or []
    ]


def _all_split_values_zero(summary: Mapping[str, Any], keys: Sequence[str]) -> bool:
    return all(
        _numeric_field_equals(summary.get(split) or {}, key, 0)
        for split in ("train", "dev")
        for key in keys
    )


def _split_values(summary: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {
        split: {key: (summary.get(split) or {}).get(key) for key in keys}
        for split in ("train", "dev")
    }


def _numeric_field_equals(summary: Mapping[str, Any], key: str, expected: int) -> bool:
    value = summary.get(key)
    return type(value) in (int, float) and value == expected


def _numeric_field_positive(summary: Mapping[str, Any], key: str) -> bool:
    value = summary.get(key)
    return type(value) in (int, float) and value > 0


def _forbidden_keys_found(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in _FORBIDDEN_PUBLIC_KEYS:
                found.add(key_text)
            found.update(_forbidden_keys_found(child))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for child in value:
            found.update(_forbidden_keys_found(child))
    return found


def _check(*, name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
