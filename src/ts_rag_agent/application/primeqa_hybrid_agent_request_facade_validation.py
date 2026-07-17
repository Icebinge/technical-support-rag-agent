from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event, Thread
from typing import Any

from ts_rag_agent.application.primeqa_hybrid_agent_request_facade import (
    AgentFacadeRequest,
    AgentRequestCancellationSignal,
    AgentRequestFacadeCancelledError,
    AgentRequestFacadeCapacityExceededError,
    AgentRequestFacadeClosedError,
    AgentRequestFacadeDrainingError,
    AgentRequestFacadeInvalidRequestError,
    AgentRequestFacadeNotActiveError,
    AgentRequestFacadeState,
    agent_request_facade_contract,
    create_primeqa_hybrid_agent_request_facade,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_runtime_activation import (
    PrimeQAHybridConcurrentRuntimeBootstrapResult,
    PublicSafeConcurrentRuntimeStartupTrace,
)
from ts_rag_agent.application.primeqa_hybrid_concurrent_sidecar_agent_runtime import (
    ConcurrentArrivalPattern,
    PrimeQAHybridConcurrentCapacityExceededError,
    PublicSafeConcurrentRuntimeRequestTrace,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    _forbidden_keys_found,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.answer import AnswerCitation, GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQARuntimeQuery

_STAGE = "Stage 148"
_CREATED_AT = "2026-07-17"
_ANALYSIS_ID = "primeqa_hybrid_transport_neutral_agent_request_facade_validation_v1"
_SOURCE_PROTOCOL_ID = "primeqa_hybrid_agent_request_facade_protocol_v1"
_SOURCE_STATUS = "primeqa_hybrid_agent_request_facade_protocol_frozen"
_EXPECTED_SOURCE_GUARDS = 34
_FINAL_STATUS = "primeqa_hybrid_transport_neutral_agent_request_facade_validation_passed"
_NEXT_DIRECTION = "freeze_agent_network_transport_protocol"


@dataclass(frozen=True)
class PrimeQAHybridAgentRequestFacadeValidationVisualization:
    """One generated Stage148 aggregate/synthetic visualization."""

    name: str
    path: str


@dataclass(frozen=True)
class _SyntheticRuntimeRun:
    verified_answer: GeneratedAnswer
    public_safe_trace: PublicSafeConcurrentRuntimeRequestTrace


class _SyntheticRuntime:
    def __init__(self, *, refused: bool = False, error: RuntimeError | None = None) -> None:
        self.refused = refused
        self.error = error
        self.call_count = 0
        self.received_query_type = ""
        self.received_query_fields: tuple[str, ...] = ()
        self.forbidden_query_attributes: tuple[str, ...] = ()
        self.arrival_pattern = ""

    def run(
        self,
        question: PrimeQARuntimeQuery,
        *,
        arrival_pattern: ConcurrentArrivalPattern,
    ) -> _SyntheticRuntimeRun:
        self.call_count += 1
        self.received_query_type = type(question).__name__
        self.received_query_fields = tuple(sorted(type(question).model_fields))
        self.forbidden_query_attributes = tuple(
            name
            for name in (
                "answer",
                "answerable",
                "answer_doc_id",
                "doc_ids",
                "start_offset",
                "end_offset",
                "test_membership",
            )
            if hasattr(question, name)
        )
        self.arrival_pattern = arrival_pattern.value
        if self.error is not None:
            raise self.error
        return _synthetic_run(question.id, refused=self.refused)


class _CapacityRuntime(_SyntheticRuntime):
    def run(
        self,
        question: PrimeQARuntimeQuery,
        *,
        arrival_pattern: ConcurrentArrivalPattern,
    ) -> _SyntheticRuntimeRun:
        self.call_count += 1
        self.arrival_pattern = arrival_pattern.value
        raise PrimeQAHybridConcurrentCapacityExceededError(
            _runtime_trace(
                admission_state="rejected_capacity",
                terminal_state="capacity_rejected",
                candidate_pool_depth=0,
                in_flight=4,
            )
        )


class _BlockingRuntime(_SyntheticRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.entered = Event()
        self.release = Event()

    def run(
        self,
        question: PrimeQARuntimeQuery,
        *,
        arrival_pattern: ConcurrentArrivalPattern,
    ) -> _SyntheticRuntimeRun:
        self.call_count += 1
        self.arrival_pattern = arrival_pattern.value
        self.entered.set()
        self.release.wait()
        return _synthetic_run(question.id)


def run_primeqa_hybrid_agent_request_facade_validation(
    *,
    stage147_protocol_path: Path,
    user_confirmed_validation: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Validate the Stage148 facade using only Stage147 aggregate and synthetic runtimes."""

    started_at = time.perf_counter()
    source = _load_json_object(stage147_protocol_path)
    loaded_at = time.perf_counter()
    source_summary = _stage147_summary(source)
    source_checks = _source_gate_checks(
        source_summary=source_summary,
        user_confirmed_validation=user_confirmed_validation,
        confirmation_note=confirmation_note,
    )
    source_gate_passed = all(check["passed"] for check in source_checks)
    preliminary = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "validation_scope": (
            "Validate the transport-neutral Agent request facade against the frozen "
            "Stage147 protocol using synthetic runtimes only. This stage does not load "
            "train, dev, test, questions, documents, models, indexes, or candidate pools; "
            "start a network service; change defaults; or add queues, retries, fallback, "
            "hard cancellation, forced shutdown, or an implicit shutdown timeout."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_validation),
            "confirmation_note": confirmation_note,
        },
        "source_files": {"stage147_protocol": _fingerprint(stage147_protocol_path)},
        "stage147_summary": source_summary,
        "source_gate_checks": source_checks,
        "source_gate_passed": source_gate_passed,
        "implementation_contract": agent_request_facade_contract(),
    }
    if not source_gate_passed:
        checked_at = time.perf_counter()
        report = {
            **preliminary,
            "synthetic_validation_executed": False,
            "synthetic_validation": {},
            "guard_checks": source_checks,
            "decision": _decision(source_checks, synthetic_validation_executed=False),
            "timing_seconds": {
                "load_public_stage147_aggregate": round(loaded_at - started_at, 6),
                "validate_synthetic_facade": 0.0,
                "total": round(checked_at - started_at, 6),
            },
        }
        return {**report, "public_safe_contract": _public_safe_contract(report)}

    synthetic = _run_synthetic_validation()
    source_unchanged_after_validation = _load_json_object(stage147_protocol_path) == source
    validated_at = time.perf_counter()
    guards = _guard_checks(
        report=preliminary,
        source_checks=source_checks,
        source_summary=source_summary,
        contract=preliminary["implementation_contract"],
        synthetic=synthetic,
        source_unchanged_after_validation=source_unchanged_after_validation,
    )
    report = {
        **preliminary,
        "synthetic_validation_executed": True,
        "source_unchanged_after_validation": source_unchanged_after_validation,
        "synthetic_validation": synthetic,
        "guard_checks": guards,
        "decision": _decision(guards, synthetic_validation_executed=True),
        "timing_seconds": {
            "load_public_stage147_aggregate": round(loaded_at - started_at, 6),
            "validate_synthetic_facade": round(validated_at - loaded_at, 6),
            "total": round(validated_at - started_at, 6),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_agent_request_facade_validation_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridAgentRequestFacadeValidationVisualization]:
    """Write ten Stage148 aggregate/synthetic SVG charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic = report.get("synthetic_validation") or {}
    charts = {
        "stage148_source_gate.svg": _chart(
            "Stage148 Stage147 source gate",
            _check_bars(report.get("source_gate_checks") or []),
            width=2140,
            margin_left=1180,
        ),
        "stage148_runtime_query_boundary.svg": _chart(
            "Stage148 label-free runtime query boundary",
            _flag_bars(
                synthetic.get("runtime_query_boundary") or {},
                ("type_exact", "field_set_exact", "forbidden_attributes_absent"),
            ),
        ),
        "stage148_private_response_mapping.svg": _chart(
            "Stage148 private response mapping",
            _flag_bars(
                synthetic.get("private_response_mapping") or {},
                (
                    "complete_text_mapped",
                    "complete_refusal_mapped",
                    "citation_fields_mapped",
                    "refusal_mapped",
                    "refusal_has_zero_citations",
                ),
            ),
        ),
        "stage148_public_telemetry_fields.svg": _chart(
            "Stage148 public telemetry field counts",
            _telemetry_field_bars(report),
            x_label="field count",
        ),
        "stage148_error_outcomes.svg": _chart(
            "Stage148 facade error outcomes",
            _error_outcome_bars(synthetic),
        ),
        "stage148_dispatch_counts.svg": _chart(
            "Stage148 synthetic runtime dispatch counts",
            _dispatch_bars(synthetic),
            x_label="runtime calls",
        ),
        "stage148_lifecycle.svg": _chart(
            "Stage148 natural draining lifecycle",
            _flag_bars(
                synthetic.get("lifecycle_validation") or {},
                (
                    "entered_draining",
                    "new_call_rejected_while_draining",
                    "in_flight_completed_naturally",
                    "closed_after_in_flight_zero",
                    "closed_call_rejected",
                    "shutdown_idempotent",
                ),
            ),
        ),
        "stage148_closed_boundaries.svg": _chart(
            "Stage148 closed serving boundaries",
            _closed_boundary_bars(report),
        ),
        "stage148_decision_flags.svg": _chart(
            "Stage148 facade validation decision",
            _flag_bars(
                report.get("decision") or {},
                (
                    "transport_neutral_facade_implemented",
                    "facade_synthetic_validation_passed",
                    "network_transport_protocol_allowed_next",
                    "network_service_implemented",
                    "runtime_registered_as_default",
                    "test_gate_opened",
                    "queue_actions_enabled",
                    "retry_actions_enabled",
                    "fallback_strategies_enabled",
                ),
            ),
        ),
        "stage148_guard_check_status.svg": _chart(
            "Stage148 validation guard checks",
            _check_bars(report.get("guard_checks") or []),
            width=2460,
            margin_left=1400,
        ),
    }
    artifacts: list[PrimeQAHybridAgentRequestFacadeValidationVisualization] = []
    for name, content in charts.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridAgentRequestFacadeValidationVisualization(name=name, path=str(path))
        )
    return artifacts


def _run_synthetic_validation() -> dict[str, Any]:
    complete_runtime = _SyntheticRuntime()
    complete_facade = _active_facade(complete_runtime)
    complete_run = complete_facade.invoke(
        AgentFacadeRequest(
            request_handle="synthetic-complete",
            title="Adapter setup",
            text="How is the adapter configured?",
        )
    )
    refusal_runtime = _SyntheticRuntime(refused=True)
    refusal_run = _active_facade(refusal_runtime).invoke(
        AgentFacadeRequest(
            request_handle="synthetic-refusal",
            text="Question without sufficient evidence",
        )
    )

    rejection_runtime = _SyntheticRuntime()
    rejection_facade = _active_facade(rejection_runtime)
    invalid_error: AgentRequestFacadeInvalidRequestError | None = None
    try:
        rejection_facade.invoke(AgentFacadeRequest(request_handle="", text=" "))
    except AgentRequestFacadeInvalidRequestError as error:
        invalid_error = error
    cancellation = AgentRequestCancellationSignal()
    cancellation.cancel()
    cancellation_code = ""
    try:
        rejection_facade.invoke(
            AgentFacadeRequest(
                request_handle="synthetic-cancelled",
                text="Cancelled before dispatch",
                cancellation_signal=cancellation,
            )
        )
    except AgentRequestFacadeCancelledError as error:
        cancellation_code = error.code

    capacity_runtime = _CapacityRuntime()
    capacity_facade = _active_facade(capacity_runtime)
    capacity_error: AgentRequestFacadeCapacityExceededError | None = None
    try:
        capacity_facade.invoke(
            AgentFacadeRequest(request_handle="synthetic-capacity", text="Capacity probe")
        )
    except AgentRequestFacadeCapacityExceededError as error:
        capacity_error = error

    downstream_original = RuntimeError("synthetic downstream failure")
    downstream_facade = _active_facade(_SyntheticRuntime(error=downstream_original))
    downstream_caught: RuntimeError | None = None
    try:
        downstream_facade.invoke(
            AgentFacadeRequest(request_handle="synthetic-failure", text="Failure probe")
        )
    except RuntimeError as error:
        downstream_caught = error

    lifecycle = _lifecycle_validation()
    inactive_error: AgentRequestFacadeNotActiveError | None = None
    try:
        create_primeqa_hybrid_agent_request_facade(
            bootstrap_result=_bootstrap_result(runtime=None, active=False)
        )
    except AgentRequestFacadeNotActiveError as error:
        inactive_error = error

    runtime_query = {
        "received_type": complete_runtime.received_query_type,
        "received_fields": list(complete_runtime.received_query_fields),
        "forbidden_attributes_present": list(complete_runtime.forbidden_query_attributes),
        "arrival_pattern": complete_runtime.arrival_pattern,
        "type_exact": complete_runtime.received_query_type == "PrimeQARuntimeQuery",
        "field_set_exact": complete_runtime.received_query_fields == ("id", "text", "title"),
        "forbidden_attributes_absent": not complete_runtime.forbidden_query_attributes,
    }
    private_mapping = {
        "complete_text_mapped": complete_run.response.text == "Configure the adapter. [doc-1]",
        "complete_refusal_mapped": complete_run.response.refused is False,
        "citation_fields_mapped": (
            len(complete_run.response.citations) == 1
            and complete_run.response.citations[0].document_reference == "doc-1"
            and complete_run.response.citations[0].title == "Adapter configuration"
            and complete_run.response.citations[0].rank == 1
            and complete_run.response.citations[0].evidence_score == 9.5
        ),
        "refusal_mapped": refusal_run.response.refused is True,
        "refusal_has_zero_citations": refusal_run.response.citations == (),
        "private_payload_written_to_report": False,
    }
    invalid_trace = invalid_error.public_safe_event if invalid_error else None
    capacity_trace = capacity_error.public_safe_event if capacity_error else None
    capacity_runtime_trace = capacity_error.public_safe_runtime_trace if capacity_error else None
    downstream_trace = downstream_facade.last_public_event
    inactive_trace = inactive_error.public_safe_event if inactive_error else None
    return {
        "runtime_query_boundary": runtime_query,
        "private_response_mapping": private_mapping,
        "public_telemetry_validation": {
            "facade_event_field_count": len(complete_run.public_safe_event.to_public_dict()),
            "runtime_trace_field_count": len(
                complete_run.public_safe_runtime_trace.to_public_dict()
            ),
            "complete_event": complete_run.public_safe_event.to_public_dict(),
            "complete_runtime_trace": complete_run.public_safe_runtime_trace.to_public_dict(),
            "forbidden_keys_found": sorted(_forbidden_keys_found(complete_run.to_public_dict())),
        },
        "invalid_request_validation": {
            "typed_error_observed": invalid_error is not None,
            "error_code": invalid_error.code if invalid_error else "",
            "rejection_reason_count": len(invalid_error.reasons) if invalid_error else 0,
            "downstream_dispatched": (
                invalid_trace.downstream_dispatched if invalid_trace else None
            ),
            "runtime_call_count": rejection_runtime.call_count,
        },
        "predispatch_cancellation_validation": {
            "error_code": cancellation_code,
            "runtime_call_count": rejection_runtime.call_count,
            "cancelled_counter": rejection_facade.counters().cancelled_before_dispatch_count,
        },
        "capacity_mapping_validation": {
            "typed_facade_error_observed": capacity_error is not None,
            "facade_error_code": capacity_error.code if capacity_error else "",
            "source_error_preserved_as_cause": (
                isinstance(capacity_error.__cause__, PrimeQAHybridConcurrentCapacityExceededError)
                if capacity_error
                else False
            ),
            "downstream_dispatched": (
                capacity_trace.downstream_dispatched if capacity_trace else None
            ),
            "runtime_admission_state": (
                capacity_runtime_trace.admission_state if capacity_runtime_trace else ""
            ),
            "runtime_call_count": capacity_runtime.call_count,
            "queue_action_count": capacity_facade.counters().queue_action_count,
            "retry_action_count": capacity_facade.counters().retry_action_count,
            "fallback_action_count": capacity_facade.counters().fallback_action_count,
        },
        "downstream_error_validation": {
            "same_error_object_propagated": downstream_caught is downstream_original,
            "event_outcome_code": downstream_trace.outcome_code if downstream_trace else "",
            "downstream_dispatched": (
                downstream_trace.downstream_dispatched if downstream_trace else None
            ),
            "downstream_error_count": downstream_facade.counters().downstream_error_count,
            "error_converted_to_response": False,
        },
        "inactive_factory_validation": {
            "typed_error_observed": inactive_error is not None,
            "error_code": inactive_error.code if inactive_error else "",
            "facade_state": inactive_trace.facade_state if inactive_trace else "",
        },
        "lifecycle_validation": lifecycle,
        "complete_facade_counters": asdict(complete_facade.counters()),
    }


def _lifecycle_validation() -> dict[str, Any]:
    runtime = _BlockingRuntime()
    facade = _active_facade(runtime)
    completed = []
    invocation_errors = []

    def invoke() -> None:
        try:
            completed.append(
                facade.invoke(
                    AgentFacadeRequest(
                        request_handle="synthetic-in-flight",
                        text="Block until released",
                    )
                )
            )
        except Exception as error:
            invocation_errors.append(type(error).__name__)

    invoke_thread = Thread(target=invoke)
    invoke_thread.start()
    runtime.entered.wait()
    shutdown_thread = Thread(target=facade.shutdown)
    shutdown_thread.start()
    facade.wait_until_state(AgentRequestFacadeState.DRAINING)
    entered_draining = facade.state is AgentRequestFacadeState.DRAINING
    draining_error: AgentRequestFacadeDrainingError | None = None
    try:
        facade.invoke(
            AgentFacadeRequest(request_handle="synthetic-new", text="Reject while draining")
        )
    except AgentRequestFacadeDrainingError as error:
        draining_error = error
    in_flight_before_release = facade.counters().current_in_flight
    runtime.release.set()
    invoke_thread.join()
    shutdown_thread.join()
    closed_after_in_flight = (
        facade.state is AgentRequestFacadeState.CLOSED and facade.counters().current_in_flight == 0
    )
    closed_error: AgentRequestFacadeClosedError | None = None
    try:
        facade.invoke(AgentFacadeRequest(request_handle="synthetic-closed", text="Reject"))
    except AgentRequestFacadeClosedError as error:
        closed_error = error
    facade.shutdown()
    return {
        "entered_draining": entered_draining,
        "new_call_rejected_while_draining": draining_error is not None,
        "draining_downstream_dispatched": (
            draining_error.public_safe_event.downstream_dispatched if draining_error else None
        ),
        "in_flight_before_release": in_flight_before_release,
        "in_flight_completed_naturally": len(completed) == 1 and not invocation_errors,
        "completion_event_state": (
            completed[0].public_safe_event.facade_state if completed else ""
        ),
        "closed_after_in_flight_zero": closed_after_in_flight,
        "closed_call_rejected": closed_error is not None,
        "shutdown_idempotent": facade.state is AgentRequestFacadeState.CLOSED,
        "implicit_timeout_used": False,
        "force_cancel_used": False,
    }


def _source_gate_checks(
    *,
    source_summary: Mapping[str, Any],
    user_confirmed_validation: bool,
    confirmation_note: str,
) -> list[dict[str, Any]]:
    return [
        _check(
            "stage148_user_confirmed",
            user_confirmed_validation,
            user_confirmed_validation,
            True,
        ),
        _check(
            "stage148_confirmation_note_present",
            bool(confirmation_note.strip()),
            bool(confirmation_note.strip()),
            True,
        ),
        _check(
            "stage147_source_identity_valid",
            source_summary.get("source_identity_valid") is True,
            source_summary.get("source_identity_valid"),
            True,
        ),
        _check(
            "stage147_all_34_guards_passed",
            source_summary.get("all_source_guards_passed") is True,
            source_summary.get("source_passed_guard_count"),
            _EXPECTED_SOURCE_GUARDS,
        ),
        _check(
            "stage147_facade_implementation_authorized",
            source_summary.get("facade_implementation_allowed_next") is True,
            source_summary.get("facade_implementation_allowed_next"),
            True,
        ),
        _check(
            "stage147_facade_and_network_not_preimplemented",
            source_summary.get("facade_implemented_now") is False
            and source_summary.get("network_service_implemented") is False,
            [
                source_summary.get("facade_implemented_now"),
                source_summary.get("network_service_implemented"),
            ],
            [False, False],
        ),
        _check(
            "stage147_closed_boundaries_preserved",
            source_summary.get("closed_boundaries_preserved") is True,
            source_summary.get("closed_boundaries_preserved"),
            True,
        ),
        _check(
            "stage147_public_safety_passed",
            source_summary.get("forbidden_keys_found") == [],
            source_summary.get("forbidden_keys_found"),
            [],
        ),
    ]


def _guard_checks(
    *,
    report: Mapping[str, Any],
    source_checks: Sequence[Mapping[str, Any]],
    source_summary: Mapping[str, Any],
    contract: Mapping[str, Any],
    synthetic: Mapping[str, Any],
    source_unchanged_after_validation: bool,
) -> list[dict[str, Any]]:
    query = synthetic.get("runtime_query_boundary") or {}
    response = synthetic.get("private_response_mapping") or {}
    telemetry = synthetic.get("public_telemetry_validation") or {}
    invalid = synthetic.get("invalid_request_validation") or {}
    cancelled = synthetic.get("predispatch_cancellation_validation") or {}
    capacity = synthetic.get("capacity_mapping_validation") or {}
    downstream = synthetic.get("downstream_error_validation") or {}
    inactive = synthetic.get("inactive_factory_validation") or {}
    lifecycle = synthetic.get("lifecycle_validation") or {}
    counters = synthetic.get("complete_facade_counters") or {}
    checks = list(source_checks)
    checks.extend(
        [
            _check(
                "implementation_contract_matches_stage147_identity",
                contract.get("facade_id")
                == "primeqa_hybrid_transport_neutral_agent_request_facade_v1",
                contract.get("facade_id"),
                "primeqa_hybrid_transport_neutral_agent_request_facade_v1",
            ),
            _check(
                "runtime_query_type_is_label_free",
                contract.get("runtime_query_type") == "PrimeQARuntimeQuery"
                and contract.get("runtime_query_fields") == ["id", "title", "text"]
                and contract.get("runtime_query_contains_gold_labels") is False,
                [
                    contract.get("runtime_query_type"),
                    contract.get("runtime_query_fields"),
                    contract.get("runtime_query_contains_gold_labels"),
                ],
                ["PrimeQARuntimeQuery", ["id", "title", "text"], False],
            ),
            _check(
                "runtime_received_exact_label_free_query",
                query.get("type_exact") is True
                and query.get("field_set_exact") is True
                and query.get("forbidden_attributes_absent") is True,
                [
                    query.get("type_exact"),
                    query.get("field_set_exact"),
                    query.get("forbidden_attributes_absent"),
                ],
                [True, True, True],
            ),
            _check(
                "application_arrival_pattern_exact",
                query.get("arrival_pattern") == "application_request",
                query.get("arrival_pattern"),
                "application_request",
            ),
            _check(
                "complete_private_response_mapped",
                response.get("complete_text_mapped") is True
                and response.get("complete_refusal_mapped") is True
                and response.get("citation_fields_mapped") is True,
                [
                    response.get("complete_text_mapped"),
                    response.get("complete_refusal_mapped"),
                    response.get("citation_fields_mapped"),
                ],
                [True, True, True],
            ),
            _check(
                "refusal_private_response_mapped",
                response.get("refusal_mapped") is True
                and response.get("refusal_has_zero_citations") is True,
                [response.get("refusal_mapped"), response.get("refusal_has_zero_citations")],
                [True, True],
            ),
            _check(
                "private_payload_not_written",
                response.get("private_payload_written_to_report") is False,
                response.get("private_payload_written_to_report"),
                False,
            ),
            _check(
                "public_facade_event_has_exact_six_fields",
                telemetry.get("facade_event_field_count") == 6,
                telemetry.get("facade_event_field_count"),
                6,
            ),
            _check(
                "public_runtime_trace_has_exact_fourteen_fields",
                telemetry.get("runtime_trace_field_count") == 14,
                telemetry.get("runtime_trace_field_count"),
                14,
            ),
            _check(
                "public_success_telemetry_has_no_forbidden_keys",
                telemetry.get("forbidden_keys_found") == [],
                telemetry.get("forbidden_keys_found"),
                [],
            ),
            _check(
                "invalid_request_rejected_before_runtime",
                invalid.get("typed_error_observed") is True
                and invalid.get("error_code") == "invalid_request"
                and invalid.get("downstream_dispatched") is False,
                [
                    invalid.get("typed_error_observed"),
                    invalid.get("error_code"),
                    invalid.get("downstream_dispatched"),
                ],
                [True, "invalid_request", False],
            ),
            _check(
                "predispatch_cancellation_never_calls_runtime",
                cancelled.get("error_code") == "cancelled_before_dispatch"
                and cancelled.get("runtime_call_count") == 0
                and cancelled.get("cancelled_counter") == 1,
                [
                    cancelled.get("error_code"),
                    cancelled.get("runtime_call_count"),
                    cancelled.get("cancelled_counter"),
                ],
                ["cancelled_before_dispatch", 0, 1],
            ),
            _check(
                "capacity_error_mapping_exact",
                capacity.get("typed_facade_error_observed") is True
                and capacity.get("facade_error_code") == "capacity_exceeded"
                and capacity.get("source_error_preserved_as_cause") is True
                and capacity.get("runtime_admission_state") == "rejected_capacity",
                [
                    capacity.get("typed_facade_error_observed"),
                    capacity.get("facade_error_code"),
                    capacity.get("source_error_preserved_as_cause"),
                    capacity.get("runtime_admission_state"),
                ],
                [True, "capacity_exceeded", True, "rejected_capacity"],
            ),
            _check(
                "capacity_rejected_before_downstream",
                capacity.get("downstream_dispatched") is False,
                capacity.get("downstream_dispatched"),
                False,
            ),
            _check(
                "capacity_queue_retry_fallback_zero",
                [
                    capacity.get("queue_action_count"),
                    capacity.get("retry_action_count"),
                    capacity.get("fallback_action_count"),
                ]
                == [0, 0, 0],
                [
                    capacity.get("queue_action_count"),
                    capacity.get("retry_action_count"),
                    capacity.get("fallback_action_count"),
                ],
                [0, 0, 0],
            ),
            _check(
                "downstream_error_object_propagated_unchanged",
                downstream.get("same_error_object_propagated") is True
                and downstream.get("error_converted_to_response") is False,
                [
                    downstream.get("same_error_object_propagated"),
                    downstream.get("error_converted_to_response"),
                ],
                [True, False],
            ),
            _check(
                "downstream_error_event_is_request_local",
                downstream.get("event_outcome_code") == "downstream_error"
                and downstream.get("downstream_dispatched") is True
                and downstream.get("downstream_error_count") == 1,
                [
                    downstream.get("event_outcome_code"),
                    downstream.get("downstream_dispatched"),
                    downstream.get("downstream_error_count"),
                ],
                ["downstream_error", True, 1],
            ),
            _check(
                "inactive_bootstrap_rejected",
                inactive.get("typed_error_observed") is True
                and inactive.get("error_code") == "facade_not_active"
                and inactive.get("facade_state") == "not_active",
                [
                    inactive.get("typed_error_observed"),
                    inactive.get("error_code"),
                    inactive.get("facade_state"),
                ],
                [True, "facade_not_active", "not_active"],
            ),
            _check(
                "lifecycle_entered_draining_and_rejected_new_call",
                lifecycle.get("entered_draining") is True
                and lifecycle.get("new_call_rejected_while_draining") is True
                and lifecycle.get("draining_downstream_dispatched") is False,
                [
                    lifecycle.get("entered_draining"),
                    lifecycle.get("new_call_rejected_while_draining"),
                    lifecycle.get("draining_downstream_dispatched"),
                ],
                [True, True, False],
            ),
            _check(
                "shutdown_waited_for_one_in_flight_call",
                lifecycle.get("in_flight_before_release") == 1
                and lifecycle.get("in_flight_completed_naturally") is True
                and lifecycle.get("completion_event_state") == "draining",
                [
                    lifecycle.get("in_flight_before_release"),
                    lifecycle.get("in_flight_completed_naturally"),
                    lifecycle.get("completion_event_state"),
                ],
                [1, True, "draining"],
            ),
            _check(
                "shutdown_closed_after_in_flight_zero",
                lifecycle.get("closed_after_in_flight_zero") is True
                and lifecycle.get("closed_call_rejected") is True
                and lifecycle.get("shutdown_idempotent") is True,
                [
                    lifecycle.get("closed_after_in_flight_zero"),
                    lifecycle.get("closed_call_rejected"),
                    lifecycle.get("shutdown_idempotent"),
                ],
                [True, True, True],
            ),
            _check(
                "shutdown_used_no_timeout_or_force_cancel",
                lifecycle.get("implicit_timeout_used") is False
                and lifecycle.get("force_cancel_used") is False,
                [
                    lifecycle.get("implicit_timeout_used"),
                    lifecycle.get("force_cancel_used"),
                ],
                [False, False],
            ),
            _check(
                "complete_counter_invariants_hold",
                counters.get("invocation_attempt_count") == 1
                and counters.get("accepted_call_count") == 1
                and counters.get("runtime_dispatch_count") == 1
                and counters.get("completed_response_count") == 1
                and counters.get("current_in_flight") == 0,
                [
                    counters.get("invocation_attempt_count"),
                    counters.get("accepted_call_count"),
                    counters.get("runtime_dispatch_count"),
                    counters.get("completed_response_count"),
                    counters.get("current_in_flight"),
                ],
                [1, 1, 1, 1, 0],
            ),
            _check(
                "facade_does_not_own_runtime_resources",
                contract.get("facade_owns_runtime_resources") is False,
                contract.get("facade_owns_runtime_resources"),
                False,
            ),
            _check(
                "network_default_test_remain_closed",
                contract.get("network_service_implemented") is False
                and contract.get("registered_as_runtime_default") is False
                and contract.get("test_access_allowed") is False,
                [
                    contract.get("network_service_implemented"),
                    contract.get("registered_as_runtime_default"),
                    contract.get("test_access_allowed"),
                ],
                [False, False, False],
            ),
            _check(
                "queue_retry_fallback_remain_closed",
                contract.get("queue_actions_allowed") is False
                and contract.get("retry_actions_allowed") is False
                and contract.get("fallback_strategies_allowed") is False,
                [
                    contract.get("queue_actions_allowed"),
                    contract.get("retry_actions_allowed"),
                    contract.get("fallback_strategies_allowed"),
                ],
                [False, False, False],
            ),
            _check(
                "source_stage147_file_unchanged_after_validation",
                source_unchanged_after_validation,
                source_unchanged_after_validation,
                True,
            ),
            _check(
                "preliminary_report_contains_no_forbidden_public_keys",
                _forbidden_keys_found(report) == set(),
                sorted(_forbidden_keys_found(report)),
                [],
            ),
            _check(
                "synthetic_summary_contains_no_forbidden_public_keys",
                _forbidden_keys_found(synthetic) == set(),
                sorted(_forbidden_keys_found(synthetic)),
                [],
            ),
        ]
    )
    return checks


def _stage147_summary(source: Mapping[str, Any]) -> dict[str, Any]:
    decision = source.get("decision") or {}
    public = source.get("public_safe_contract") or {}
    checks = source.get("guard_checks") or []
    passed_count = sum(check.get("passed") is True for check in checks)
    closed = (
        decision.get("facade_implemented_now") is False
        and decision.get("network_service_implemented") is False
        and decision.get("runtime_registered_as_default") is False
        and decision.get("test_gate_opened") is False
        and decision.get("test_metrics_run") is False
        and decision.get("queue_actions_enabled") is False
        and decision.get("retry_actions_enabled") is False
        and decision.get("fallback_strategies_enabled") is False
        and public.get("test_split_loaded") is False
        and public.get("forbidden_keys_found") == []
    )
    return {
        "source_identity_valid": (
            source.get("stage") == "Stage 147"
            and source.get("protocol_id") == _SOURCE_PROTOCOL_ID
            and decision.get("status") == _SOURCE_STATUS
        ),
        "source_guard_count": len(checks),
        "source_passed_guard_count": passed_count,
        "all_source_guards_passed": (
            len(checks) == _EXPECTED_SOURCE_GUARDS and passed_count == _EXPECTED_SOURCE_GUARDS
        ),
        "facade_implementation_allowed_next": decision.get("facade_implementation_allowed_next"),
        "facade_implemented_now": decision.get("facade_implemented_now"),
        "network_service_implemented": decision.get("network_service_implemented"),
        "closed_boundaries_preserved": closed,
        "test_split_loaded": public.get("test_split_loaded"),
        "test_metrics_run": public.get("test_metrics_run"),
        "forbidden_keys_found": list(public.get("forbidden_keys_found") or []),
    }


def _decision(
    guards: Sequence[Mapping[str, Any]],
    *,
    synthetic_validation_executed: bool,
) -> dict[str, Any]:
    failed = [str(check.get("name")) for check in guards if check.get("passed") is not True]
    passed = synthetic_validation_executed and not failed
    return {
        "status": _FINAL_STATUS if passed else "primeqa_hybrid_agent_request_facade_rejected",
        "failed_checks": failed,
        "transport_neutral_facade_implemented": passed,
        "facade_synthetic_validation_passed": passed,
        "label_free_runtime_query_validated": passed,
        "private_response_mapping_validated": passed,
        "public_telemetry_allowlists_validated": passed,
        "capacity_error_mapping_validated": passed,
        "lifecycle_and_natural_shutdown_validated": passed,
        "network_transport_protocol_allowed_next": passed,
        "network_service_implemented": False,
        "runtime_registered_as_default": False,
        "runtime_defaultization_allowed_now": False,
        "test_gate_opened": False,
        "test_metrics_run": False,
        "queue_actions_enabled": False,
        "retry_actions_enabled": False,
        "fallback_strategies_enabled": False,
        "hard_cancellation_enabled": False,
        "implicit_shutdown_timeout_enabled": False,
        "next_direction": _NEXT_DIRECTION if passed else "repair_failed_stage148_guards",
    }


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "aggregate_and_synthetic_only": True,
        "private_payloads_written": False,
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
        "forbidden_keys_found": sorted(_forbidden_keys_found(report)),
    }


def _active_facade(runtime):
    return create_primeqa_hybrid_agent_request_facade(
        bootstrap_result=_bootstrap_result(runtime=runtime, active=True)
    )


def _bootstrap_result(*, runtime, active: bool) -> PrimeQAHybridConcurrentRuntimeBootstrapResult:
    return PrimeQAHybridConcurrentRuntimeBootstrapResult(
        runtime=runtime,
        startup_trace=PublicSafeConcurrentRuntimeStartupTrace(
            runtime_mode="optional_sidecar_agent_concurrent_four_request",
            settings_field="enable_concurrent_sidecar_agent",
            environment_flag="TS_RAG_ENABLE_CONCURRENT_SIDECAR_AGENT",
            activation_requested=active,
            activation_state="eligible" if active else "disabled",
            source_validation_state="eligible" if active else "not_evaluated_disabled",
            slo_profile_id="strict_practical_b_concurrency4_v1",
            max_in_flight=4,
            warm_resources_ready=active,
            resources_initialized=active,
            runtime_activated=active,
            resource_factory_build_count=1 if active else 0,
            warmup_request_count=1 if active else 0,
            warmup_arrival_pattern="warmup_single_request" if active else "",
            warmup_candidate_pool_depth=400 if active else 0,
            warmup_retrieval_latency_ms=1.0 if active else 0.0,
            warmup_end_to_end_latency_ms=2.0 if active else 0.0,
            rejection_reasons=() if active else ("explicit_concurrent_activation_not_requested",),
        ),
        resource_summary=None,
        source_evaluation=None,
    )


def _synthetic_run(request_handle: str, *, refused: bool = False) -> _SyntheticRuntimeRun:
    return _SyntheticRuntimeRun(
        verified_answer=GeneratedAnswer(
            question_id=request_handle,
            answer=(
                "I do not have enough retrieved evidence to answer this question."
                if refused
                else "Configure the adapter. [doc-1]"
            ),
            citations=(
                []
                if refused
                else [
                    AnswerCitation(
                        document_id="doc-1",
                        title="Adapter configuration",
                        retrieval_rank=1,
                        evidence_score=9.5,
                    )
                ]
            ),
            refused=refused,
        ),
        public_safe_trace=_runtime_trace(
            admission_state="admitted",
            terminal_state="refuse" if refused else "complete",
            candidate_pool_depth=400,
            in_flight=1,
        ),
    )


def _runtime_trace(
    *,
    admission_state: str,
    terminal_state: str,
    candidate_pool_depth: int,
    in_flight: int,
) -> PublicSafeConcurrentRuntimeRequestTrace:
    return PublicSafeConcurrentRuntimeRequestTrace(
        runtime_mode="optional_sidecar_agent_concurrent_four_request",
        activation_requested=True,
        activation_state="eligible",
        slo_profile_id="strict_practical_b_concurrency4_v1",
        warm_resources_ready=True,
        concurrency_limit=4,
        in_flight_at_admission=in_flight,
        admission_state=admission_state,
        arrival_pattern=ConcurrentArrivalPattern.APPLICATION.value,
        candidate_pool_depth=candidate_pool_depth,
        retrieval_latency_ms=1.0 if candidate_pool_depth else 0.0,
        end_to_end_latency_ms=2.0,
        latency_budget_passed=True,
        terminal_state=terminal_state,
    )


def _telemetry_field_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    telemetry = (report.get("synthetic_validation") or {}).get("public_telemetry_validation") or {}
    return [
        _bar("facade event allowlist", telemetry.get("facade_event_field_count") or 0),
        _bar("runtime trace allowlist", telemetry.get("runtime_trace_field_count") or 0),
    ]


def _error_outcome_bars(synthetic: Mapping[str, Any]) -> list[BarDatum]:
    sections = (
        "invalid_request_validation",
        "predispatch_cancellation_validation",
        "capacity_mapping_validation",
        "downstream_error_validation",
        "inactive_factory_validation",
    )
    return [
        BarDatum(
            label=section,
            value=1.0,
            value_label=str(
                (synthetic.get(section) or {}).get("error_code")
                or (synthetic.get(section) or {}).get("event_outcome_code")
                or "observed"
            ),
        )
        for section in sections
    ]


def _dispatch_bars(synthetic: Mapping[str, Any]) -> list[BarDatum]:
    rows = (
        (
            "invalid + cancelled shared runtime",
            (synthetic.get("predispatch_cancellation_validation") or {}).get("runtime_call_count"),
        ),
        (
            "capacity runtime",
            (synthetic.get("capacity_mapping_validation") or {}).get("runtime_call_count"),
        ),
        (
            "complete runtime",
            (synthetic.get("complete_facade_counters") or {}).get("runtime_dispatch_count"),
        ),
    )
    return [_bar(label, value or 0) for label, value in rows]


def _closed_boundary_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    keys = (
        "network_service_implemented",
        "runtime_registered_as_default",
        "test_gate_opened",
        "queue_actions_enabled",
        "retry_actions_enabled",
        "fallback_strategies_enabled",
        "hard_cancellation_enabled",
        "implicit_shutdown_timeout_enabled",
    )
    return [
        BarDatum(
            label=key,
            value=0.0 if decision.get(key) is False else 1.0,
            value_label="closed" if decision.get(key) is False else "open",
        )
        for key in keys
    ]


def _flag_bars(source: Mapping[str, Any], keys: Sequence[str]) -> list[BarDatum]:
    return [
        BarDatum(
            label=key,
            value=1.0 if source.get(key) is True else 0.0,
            value_label="true" if source.get(key) is True else "false",
        )
        for key in keys
    ]


def _check_bars(checks: Sequence[Mapping[str, Any]]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(check.get("name")),
            value=1.0 if check.get("passed") is True else 0.0,
            value_label="passed" if check.get("passed") is True else "failed",
        )
        for check in checks
    ]


def _bar(label: str, value: Any) -> BarDatum:
    numeric = float(value)
    return BarDatum(label=label, value=numeric, value_label=str(value))


def _chart(
    title: str,
    bars: Sequence[BarDatum],
    *,
    x_label: str = "1 means true",
    width: int = 1760,
    margin_left: int = 920,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label=x_label,
        width=width,
        margin_left=margin_left,
    )


def _check(name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
