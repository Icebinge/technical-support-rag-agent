from __future__ import annotations

import hashlib
import json
import statistics
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_service_entrypoint import (
    CanonicalBoundedDynamicAgentServicePaths,
    ExactLocalQwenBackendLoader,
    PrimeQAHybridBoundedDynamicAgentServiceEntrypoint,
)
from ts_rag_agent.application.primeqa_hybrid_structured_decision_router import (
    GeneratedRouterText,
    Qwen3VLTransformersTextGenerationBackend,
    StructuredRouterPromptPolicy,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.config import ProjectSettings

from . import primeqa_hybrid_train_history_isolation_validation as stage165
from .primeqa_hybrid_train_history_isolation_protocol import (
    Stage165Arm,
    Stage165ArmObservation,
    build_stage165_grouped_fold_assignment,
    build_stage165_paired_workload_plan,
    load_stage165_train_diagnostic_samples,
)

_STAGE = "Stage 165 memory probe"
_CREATED_AT = "2026-07-19"
_ANALYSIS_ID = "primeqa_hybrid_stage165_thread37_cuda_memory_probe_v1"
_PROBE_THREAD_ORDINAL = 37
_EXPECTED_PROBE_TURNS = 8
_EXPECTED_IDENTITIES = (
    "3dcac36e8a5e558ec15e2c4ff7fb025df75286bae438ce5b3748291cb79cb5d9",
    "3dfd47bee94d0c77c8b74e3906485007bef9aad31cdeac851b588309422a2840",
    "3e300e5c7289c032eb104ac77f3a6162b303caa7f9e22f5c0b8098bdc37db05d",
    "3f5c1a54feb8e33db0aeff8c677bbc11d1b2a00fb3c0167c6649f0568ae4891f",
)


@dataclass(frozen=True)
class Stage165MemoryProbeContext:
    phase: str
    private_identity_sha256: str | None
    synthetic_turn_position: int | None
    arm: Stage165Arm | None
    arm_order_position: int | None


@dataclass(frozen=True)
class Stage165MemoryProbeEvent:
    generation_attempt: int
    phase: str
    private_identity_sha256: str | None
    synthetic_turn_position: int | None
    arm: Stage165Arm | None
    arm_order_position: int | None
    prompt_sha256: str
    prompt_character_count: int
    input_token_count_preflight: int
    cuda_allocated_before: int
    cuda_reserved_before: int
    cuda_peak_allocated: int | None
    cuda_peak_reserved: int | None
    cuda_allocated_after: int | None
    cuda_reserved_after: int | None
    cuda_telemetry_failure_types: tuple[str, ...]
    generation_completed: bool
    output_token_count: int | None
    generation_latency_ms: float | None
    failure_kind: str | None
    failure_type: str | None

    def to_private_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage165MemoryProbeRun:
    public_report: dict[str, Any]
    private_report: dict[str, Any]


@dataclass(frozen=True)
class Stage165MemoryProbeVisualization:
    name: str
    path: str


class Stage165ObservedGenerationBackend:
    """Record prompt and CUDA memory facts without recovering a failed call."""

    def __init__(
        self,
        *,
        delegate: Qwen3VLTransformersTextGenerationBackend,
        event_jsonl_path: Path,
    ) -> None:
        self._delegate = delegate
        self._processor = delegate._processor
        self._torch = delegate._torch
        self._context = Stage165MemoryProbeContext(
            phase="warmup",
            private_identity_sha256=None,
            synthetic_turn_position=None,
            arm=None,
            arm_order_position=None,
        )
        self._events: list[Stage165MemoryProbeEvent] = []
        self._event_jsonl_path = event_jsonl_path
        self._event_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._event_jsonl_path.write_text("", encoding="utf-8")

    @property
    def generation_call_count(self) -> int:
        return self._delegate.generation_call_count

    @property
    def events(self) -> tuple[Stage165MemoryProbeEvent, ...]:
        return tuple(self._events)

    def set_probe_context(self, context: Stage165MemoryProbeContext) -> None:
        self._context = context

    def reset_warmup_context(self) -> None:
        self._context = Stage165MemoryProbeContext(
            phase="warmup",
            private_identity_sha256=None,
            synthetic_turn_position=None,
            arm=None,
            arm_order_position=None,
        )

    def generate(
        self,
        *,
        prompt: str,
        max_input_tokens: int,
        max_new_tokens: int,
    ) -> GeneratedRouterText:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        encoded = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_token_count = int(encoded["input_ids"].shape[-1])
        del encoded
        self._torch.cuda.synchronize()
        self._torch.cuda.reset_peak_memory_stats()
        allocated_before = int(self._torch.cuda.memory_allocated())
        reserved_before = int(self._torch.cuda.memory_reserved())
        completed = False
        output: GeneratedRouterText | None = None
        failure: Exception | None = None
        try:
            output = self._delegate.generate(
                prompt=prompt,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
            )
            completed = True
            return output
        except Exception as error:
            failure = error
            raise
        finally:
            after_metrics, telemetry_failures = _read_post_generation_cuda_metrics(self._torch)
            event = Stage165MemoryProbeEvent(
                generation_attempt=len(self._events) + 1,
                phase=self._context.phase,
                private_identity_sha256=self._context.private_identity_sha256,
                synthetic_turn_position=self._context.synthetic_turn_position,
                arm=self._context.arm,
                arm_order_position=self._context.arm_order_position,
                prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                prompt_character_count=len(prompt),
                input_token_count_preflight=input_token_count,
                cuda_allocated_before=allocated_before,
                cuda_reserved_before=reserved_before,
                cuda_peak_allocated=after_metrics["peak_allocated"],
                cuda_peak_reserved=after_metrics["peak_reserved"],
                cuda_allocated_after=after_metrics["allocated_after"],
                cuda_reserved_after=after_metrics["reserved_after"],
                cuda_telemetry_failure_types=telemetry_failures,
                generation_completed=completed,
                output_token_count=(output.output_token_count if output is not None else None),
                generation_latency_ms=(
                    output.generation_latency_ms if output is not None else None
                ),
                failure_kind=_failure_kind(failure),
                failure_type=type(failure).__name__ if failure is not None else None,
            )
            self._events.append(event)
            with self._event_jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_private_dict(), ensure_ascii=True))
                handle.write("\n")
                handle.flush()


class Stage165MemoryProbeBackendLoader:
    def __init__(self, *, event_jsonl_path: Path) -> None:
        self._event_jsonl_path = event_jsonl_path
        self.backend: Stage165ObservedGenerationBackend | None = None

    def load(self, snapshot_path: Path) -> Stage165ObservedGenerationBackend:
        delegate = ExactLocalQwenBackendLoader().load(snapshot_path)
        self.backend = Stage165ObservedGenerationBackend(
            delegate=delegate,
            event_jsonl_path=self._event_jsonl_path,
        )
        return self.backend


def run_stage165_memory_probe(
    *,
    settings: ProjectSettings,
    stage164_correction_path: Path,
    train_split_path: Path,
    private_event_jsonl_path: Path,
    user_confirmed_thread37_probe: bool,
) -> Stage165MemoryProbeRun:
    """Run at most eight paired turns on the exact first-formal failure boundary."""

    import torch

    if not user_confirmed_thread37_probe:
        raise ValueError("Stage165 memory probe requires explicit thread37 confirmation")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage165 memory probe requires CUDA")
    started_at = time.perf_counter()
    project_root = Path(__file__).resolve().parents[3]
    paths = CanonicalBoundedDynamicAgentServicePaths.from_settings(settings)
    source_authorization = stage165._authorize_sources(
        project_root=project_root,
        stage164_correction_path=stage164_correction_path,
        train_split_path=train_split_path,
        paths=paths,
    )
    diagnostic_set = load_stage165_train_diagnostic_samples(train_split_path)
    workload = build_stage165_paired_workload_plan(diagnostic_set)
    folds = build_stage165_grouped_fold_assignment(diagnostic_set.samples)
    thread = workload.threads[_PROBE_THREAD_ORDINAL - 1]
    actual_identities = tuple(sample.private_identity_sha256 for sample in thread.samples)
    if thread.ordinal != _PROBE_THREAD_ORDINAL or actual_identities != _EXPECTED_IDENTITIES:
        raise ValueError("Stage165 memory probe boundary identity is not exact")
    source_before = _current_source_fingerprints(project_root)

    loader = Stage165MemoryProbeBackendLoader(event_jsonl_path=private_event_jsonl_path)
    torch.cuda.reset_peak_memory_stats()
    prepared = PrimeQAHybridBoundedDynamicAgentServiceEntrypoint(
        settings=settings,
        paths=paths,
        backend_loader=loader,
    ).prepare()
    backend = loader.backend
    if backend is None:
        raise RuntimeError("Stage165 memory probe did not expose the observed backend")
    prepared_at = time.perf_counter()

    session = stage165.Stage165BoundedAgentSession(
        runtime=prepared.runtime,
        prompt_policy=StructuredRouterPromptPolicy(),
    )
    observations: list[Stage165ArmObservation] = []
    failure: dict[str, Any] | None = None
    synthetic_handle = "stage165-memory-probe-synthetic-037"
    session.open_thread(synthetic_handle)
    try:
        for turn_position, sample in enumerate(thread.samples, start=1):
            for arm_order_position, arm in enumerate(workload.arm_order(sample), start=1):
                backend.set_probe_context(
                    Stage165MemoryProbeContext(
                        phase="thread37_probe",
                        private_identity_sha256=sample.private_identity_sha256,
                        synthetic_turn_position=turn_position,
                        arm=arm,
                        arm_order_position=arm_order_position,
                    )
                )
                isolated_handle = f"stage165-memory-probe-isolated-037-{turn_position:02d}"
                try:
                    if arm == "isolated":
                        session.open_thread(isolated_handle)
                        try:
                            observation = session.measure_turn(
                                handle=isolated_handle,
                                sample=sample,
                                fold_id=folds.fold_by_private_identity[
                                    sample.private_identity_sha256
                                ],
                                synthetic_thread_ordinal=thread.ordinal,
                                synthetic_turn_position=turn_position,
                                arm=arm,
                                arm_order_position=arm_order_position,
                            )
                        finally:
                            session.close_thread(isolated_handle)
                    else:
                        observation = session.measure_turn(
                            handle=synthetic_handle,
                            sample=sample,
                            fold_id=folds.fold_by_private_identity[sample.private_identity_sha256],
                            synthetic_thread_ordinal=thread.ordinal,
                            synthetic_turn_position=turn_position,
                            arm=arm,
                            arm_order_position=arm_order_position,
                        )
                    observations.append(observation)
                except Exception as error:
                    failure = {
                        "turn_position": turn_position,
                        "arm": arm,
                        "arm_order_position": arm_order_position,
                        "failure_kind": _failure_kind(error),
                        "failure_type": type(error).__name__,
                    }
                    break
            if failure is not None:
                break
    finally:
        backend.reset_warmup_context()
        session.close_thread(synthetic_handle)
    executed_at = time.perf_counter()

    events = backend.events
    train_events = tuple(event for event in events if event.phase == "thread37_probe")
    private_report = {
        "artifact_id": "primeqa_hybrid_stage165_thread37_memory_probe_private_v1",
        "contains_raw_question": False,
        "contains_raw_answer": False,
        "contains_raw_document_id": False,
        "contains_raw_document_text": False,
        "contains_raw_model_output": False,
        "event_count": len(events),
        "train_probe_event_count": len(train_events),
        "events": [event.to_private_dict() for event in events],
    }
    private_safety = stage165._public_safe_contract(private_report)
    source_after = _current_source_fingerprints(project_root)
    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "One user-confirmed train-only CUDA memory probe on synthetic thread 37, "
            "the first incomplete boundary from Stage165 formal attempt 1. The probe "
            "runs at most eight paired turns, records each generation attempt before "
            "stopping on failure, and performs no retry or fallback."
        ),
        "user_confirmation": {
            "selected_option": "A",
            "thread37_probe_confirmed": user_confirmed_thread37_probe,
        },
        "source_authorization": source_authorization,
        "probe_contract": {
            "train_only": True,
            "synthetic_thread_ordinal": _PROBE_THREAD_ORDINAL,
            "sample_count": len(thread.samples),
            "maximum_agent_turns": _EXPECTED_PROBE_TURNS,
            "arm_schedule_sha256": workload.arm_schedule_sha256,
            "failure_stops_probe": True,
            "retry": False,
            "fallback": False,
            "cuda_empty_cache_called": False,
            "development_loaded": False,
            "test_loaded": False,
        },
        "execution": {
            "warmup_generation_count": sum(event.phase == "warmup" for event in events),
            "attempted_train_turn_count": len(train_events),
            "completed_train_turn_count": len(observations),
            "failed_train_turn_count": sum(
                not event.generation_completed for event in train_events
            ),
            "failure": failure,
            "session_open_count": session.open_count,
            "session_close_count": session.close_count,
            "session_opened_after_probe": session.opened_thread_count,
            "model_successful_generation_count": backend.generation_call_count,
        },
        "memory_diagnostics": _memory_summary(train_events),
        "private_event_artifact_contract": {
            "jsonl_path": str(private_event_jsonl_path),
            "canonical_content_sha256": _canonical_json_sha256(private_report),
            "event_count": len(events),
            "train_probe_event_count": len(train_events),
            "contains_raw_question": False,
            "contains_raw_answer": False,
            "contains_raw_document_id": False,
            "contains_raw_document_text": False,
            "contains_raw_model_output": False,
            "forbidden_keys_found": private_safety["forbidden_keys_found"],
            "git_policy": "ignored_local_artifact",
        },
        "closed_boundaries": {
            "development_loaded": False,
            "test_loaded": False,
            "model_fit": False,
            "threshold_tuned": False,
            "policy_selected": False,
            "runtime_registered_as_default": False,
            "retry_actions": 0,
            "fallback_actions": 0,
            "query_rewrite": False,
            "second_retrieval": False,
        },
        "timing_seconds": {
            "prepare": round(prepared_at - started_at, 6),
            "probe_execution": round(executed_at - prepared_at, 6),
            "total": round(executed_at - started_at, 6),
        },
        "current_source_fingerprints_before": source_before,
        "current_source_fingerprints_after": source_after,
    }
    report["guard_checks"] = _guard_checks(report)
    all_guards = all(check["passed"] for check in report["guard_checks"])
    probe_reproduced_oom = any(event.failure_kind == "cuda_out_of_memory" for event in train_events)
    report["decision"] = {
        "status": (
            "primeqa_hybrid_stage165_thread37_probe_reproduced_cuda_oom"
            if probe_reproduced_oom
            else "primeqa_hybrid_stage165_thread37_probe_completed_without_cuda_oom"
            if failure is None
            else "primeqa_hybrid_stage165_thread37_probe_failed_other"
        ),
        "all_process_guards_passed": all_guards,
        "failed_process_guards": [
            check["name"] for check in report["guard_checks"] if not check["passed"]
        ],
        "probe_reproduced_cuda_oom": probe_reproduced_oom,
        "probe_completed_all_eight_turns": failure is None and len(observations) == 8,
        "full_formal_rerun_authorized": False,
        "development_gate_opened": False,
        "test_gate_opened": False,
        "policy_selected": False,
        "next_direction": (
            "analyze_exact_prompt_peak_before_full_rerun"
            if probe_reproduced_oom
            else "design_deterministic_sharded_full_train_protocol"
            if failure is None
            else "analyze_non_cuda_probe_failure"
        ),
    }
    return Stage165MemoryProbeRun(public_report=report, private_report=private_report)


def write_stage165_memory_probe_visualizations(
    *,
    private_report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage165MemoryProbeVisualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    events = [event for event in private_report["events"] if event.get("phase") == "thread37_probe"]
    labels = [f"turn {event['synthetic_turn_position']} {event['arm']}" for event in events]
    charts = {
        "stage165_memory_probe_input_tokens.svg": _chart(
            "Stage165 thread37 probe input tokens",
            [
                _bar(label, event["input_token_count_preflight"])
                for label, event in zip(labels, events, strict=True)
            ],
            "input tokens",
        ),
        "stage165_memory_probe_peak_allocated_mib.svg": _chart(
            "Stage165 thread37 probe peak CUDA allocated",
            [
                _memory_bar(
                    label,
                    event.get("cuda_peak_allocated"),
                )
                for label, event in zip(labels, events, strict=True)
            ],
            "MiB",
        ),
        "stage165_memory_probe_peak_reserved_mib.svg": _chart(
            "Stage165 thread37 probe peak CUDA reserved",
            [
                _memory_bar(
                    label,
                    event.get("cuda_peak_reserved"),
                )
                for label, event in zip(labels, events, strict=True)
            ],
            "MiB",
        ),
    }
    written = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        written.append(Stage165MemoryProbeVisualization(name=filename, path=str(path)))
    return tuple(written)


def _memory_summary(events: Sequence[Stage165MemoryProbeEvent]) -> dict[str, Any]:
    completed = [event for event in events if event.generation_completed]
    failed = [event for event in events if not event.generation_completed]
    return {
        "attempt_count": len(events),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "input_token_count": _distribution([event.input_token_count_preflight for event in events]),
        "prompt_character_count": _distribution([event.prompt_character_count for event in events]),
        "cuda_allocated_before_bytes": _distribution(
            [event.cuda_allocated_before for event in events]
        ),
        "cuda_reserved_before_bytes": _distribution(
            [event.cuda_reserved_before for event in events]
        ),
        "cuda_peak_allocated_bytes": _distribution([event.cuda_peak_allocated for event in events]),
        "cuda_peak_reserved_bytes": _distribution([event.cuda_peak_reserved for event in events]),
        "cuda_telemetry_failure_count": sum(
            bool(event.cuda_telemetry_failure_types) for event in events
        ),
        "failed_attempt": (
            {
                "turn_position": failed[0].synthetic_turn_position,
                "arm": failed[0].arm,
                "input_token_count": failed[0].input_token_count_preflight,
                "prompt_character_count": failed[0].prompt_character_count,
                "cuda_allocated_before": failed[0].cuda_allocated_before,
                "cuda_reserved_before": failed[0].cuda_reserved_before,
                "cuda_peak_allocated": failed[0].cuda_peak_allocated,
                "cuda_peak_reserved": failed[0].cuda_peak_reserved,
                "failure_kind": failed[0].failure_kind,
            }
            if failed
            else None
        ),
    }


def _guard_checks(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    confirmation = report["user_confirmation"]
    contract = report["probe_contract"]
    execution = report["execution"]
    private = report["private_event_artifact_contract"]
    boundaries = report["closed_boundaries"]
    return [
        _check(
            "user_confirmed_option_a_thread37_probe",
            confirmation.get("selected_option") == "A"
            and confirmation.get("thread37_probe_confirmed") is True,
        ),
        _check(
            "upstream_stage165_sources_authorized",
            report["source_authorization"].get("authorized") is True,
        ),
        _check(
            "exact_failure_boundary_thread_selected",
            contract.get("synthetic_thread_ordinal") == _PROBE_THREAD_ORDINAL
            and contract.get("sample_count") == 4
            and contract.get("maximum_agent_turns") == _EXPECTED_PROBE_TURNS,
        ),
        _check(
            "probe_stops_at_failure_or_eight_turns",
            int(execution.get("attempted_train_turn_count", 0)) <= _EXPECTED_PROBE_TURNS
            and int(execution.get("completed_train_turn_count", 0))
            <= int(execution.get("attempted_train_turn_count", 0))
            and (
                execution.get("failure") is not None
                or execution.get("completed_train_turn_count") == _EXPECTED_PROBE_TURNS
            ),
        ),
        _check(
            "incremental_private_events_exact",
            private.get("train_probe_event_count") == execution.get("attempted_train_turn_count")
            and private.get("event_count")
            == int(execution.get("attempted_train_turn_count", 0)) + 1,
        ),
        _check(
            "thread_lifecycle_closed",
            execution.get("session_open_count") == execution.get("session_close_count")
            and execution.get("session_opened_after_probe") == 0,
        ),
        _check(
            "current_sources_unchanged_during_probe",
            report.get("current_source_fingerprints_before")
            == report.get("current_source_fingerprints_after"),
        ),
        _check(
            "private_events_content_free",
            all(
                private.get(key) is False
                for key in (
                    "contains_raw_question",
                    "contains_raw_answer",
                    "contains_raw_document_id",
                    "contains_raw_document_text",
                    "contains_raw_model_output",
                )
            )
            and private.get("forbidden_keys_found") == [],
        ),
        _check(
            "train_only_dev_test_closed",
            contract.get("train_only") is True
            and contract.get("development_loaded") is False
            and contract.get("test_loaded") is False
            and boundaries.get("development_loaded") is False
            and boundaries.get("test_loaded") is False,
        ),
        _check(
            "no_fit_tuning_policy_or_runtime_change",
            boundaries.get("model_fit") is False
            and boundaries.get("threshold_tuned") is False
            and boundaries.get("policy_selected") is False
            and boundaries.get("runtime_registered_as_default") is False,
        ),
        _check(
            "no_retry_fallback_cache_clear_rewrite_or_second_retrieval",
            contract.get("retry") is False
            and contract.get("fallback") is False
            and contract.get("cuda_empty_cache_called") is False
            and boundaries.get("retry_actions") == 0
            and boundaries.get("fallback_actions") == 0
            and boundaries.get("query_rewrite") is False
            and boundaries.get("second_retrieval") is False,
        ),
    ]


def _current_source_fingerprints(project_root: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "stage165_protocol": (
            project_root
            / "src"
            / "ts_rag_agent"
            / "application"
            / "primeqa_hybrid_train_history_isolation_protocol.py"
        ),
        "stage165_validation": (
            project_root
            / "src"
            / "ts_rag_agent"
            / "application"
            / "primeqa_hybrid_train_history_isolation_validation.py"
        ),
        "memory_probe": Path(__file__).resolve(),
        "memory_probe_cli": (
            project_root / "scripts" / "probe_primeqa_hybrid_train_history_isolation_memory.py"
        ),
    }
    return {name: _fingerprint(path) for name, path in paths.items()}


def _failure_kind(error: Exception | None) -> str | None:
    if error is None:
        return None
    message = str(error).lower()
    if "cuda" in message and "out of memory" in message:
        return "cuda_out_of_memory"
    if isinstance(error, MemoryError):
        return "host_memory_error"
    return "other_error"


def _distribution(
    values: Sequence[int | float | None],
) -> dict[str, int | float]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return {
            "count": 0,
            "minimum": 0.0,
            "median": 0.0,
            "maximum": 0.0,
            "average": 0.0,
        }
    ordered = sorted(present)
    return {
        "count": len(ordered),
        "minimum": round(ordered[0], 3),
        "median": round(float(statistics.median(ordered)), 3),
        "maximum": round(ordered[-1], 3),
        "average": round(statistics.fmean(ordered), 3),
    }


def _fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _bar(label: str, value: int | float) -> BarDatum:
    return BarDatum(label=label, value=float(value), value_label=str(value))


def _chart(title: str, bars: Sequence[BarDatum], x_label: str) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label=x_label,
        width=1180,
        margin_left=390,
    )


def _read_post_generation_cuda_metrics(
    torch_module: Any,
) -> tuple[dict[str, int | None], tuple[str, ...]]:
    metrics: dict[str, int | None] = {}
    failures: list[str] = []
    calls = {
        "peak_allocated": torch_module.cuda.max_memory_allocated,
        "peak_reserved": torch_module.cuda.max_memory_reserved,
        "allocated_after": torch_module.cuda.memory_allocated,
        "reserved_after": torch_module.cuda.memory_reserved,
    }
    for name, call in calls.items():
        try:
            metrics[name] = int(call())
        except Exception as error:
            metrics[name] = None
            failures.append(f"{name}:{type(error).__name__}")
    return metrics, tuple(failures)


def _memory_bar(label: str, value: Any) -> BarDatum:
    if value is None:
        return BarDatum(label=label, value=0.0, value_label="unavailable")
    mib = round(float(value) / (1024**2), 3)
    return BarDatum(label=label, value=mib, value_label=str(mib))
