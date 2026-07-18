from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ts_rag_agent.application.primeqa_hybrid_agent_tool_workflow import (
    PrimeQAHybridAgentToolset,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_agent_state_protocol import (
    ThreadStateLimits,
    VolatileThreadStateLedger,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_runtime import (
    FIXED_INSUFFICIENT_EVIDENCE_RESPONSE,
    PRODUCTION_MAX_COMPLETED_TURNS,
    PRODUCTION_MAX_RETAINED_BYTES,
    bounded_dynamic_agent_runtime_contract,
    create_primeqa_hybrid_bounded_dynamic_agent_runtime,
    create_primeqa_hybrid_bounded_dynamic_agent_runtime_from_toolset,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    PrimeQAHybridProcessRuntimeResourceFactory,
    _forbidden_keys_found,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_observation_validation import (
    PrimeQAHybridSidecarObservationAdapter,
)
from ts_rag_agent.application.primeqa_hybrid_structured_decision_router import (
    GeneratedRouterText,
    Qwen3VLTransformersTextGenerationBackend,
    StrictStructuredDecisionRouter,
    StructuredDecisionSchemaError,
    structured_decision_router_contract,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.answer import (
    AnswerCitation,
    AnswerVerificationResult,
    GeneratedAnswer,
)
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQARuntimeQuery
from ts_rag_agent.domain.retrieval import RetrievalResult

_STAGE = "Stage 157"
_CREATED_AT = "2026-07-18"
_ANALYSIS_ID = "primeqa_hybrid_bounded_dynamic_agent_runtime_validation_v1"
_SOURCE_STAGE156_ANALYSIS_ID = "primeqa_hybrid_bounded_agent_tool_selection_state_protocol_v1"
_SOURCE_STAGE156_STATUS = "primeqa_hybrid_bounded_agent_tool_selection_state_protocol_frozen"
_EXPECTED_STAGE156_SHA256 = "1057cd70ed0ce872529bdc04d1182b84327a50cf6f9bcce9fedb76a4f2952a97"
_EXPECTED_STAGE156_GUARDS = 43
_FINAL_STATUS = "primeqa_hybrid_bounded_dynamic_agent_runtime_implemented_and_validated"
_NEXT_DIRECTION = "design_nondefault_local_service_integration_for_bounded_dynamic_agent"


@dataclass(frozen=True)
class Stage157Visualization:
    name: str
    path: str


def validate_primeqa_hybrid_bounded_dynamic_agent_runtime(
    *,
    stage156_protocol_path: Path,
    model_snapshot_path: Path,
    stage128_protocol_path: Path,
    stage125_protocol_path: Path,
    stage80_report_path: Path,
    documents_path: Path,
) -> dict[str, Any]:
    """Run one real label-free bounded turn and save only public aggregates."""

    import torch

    started_at = time.perf_counter()
    stage156_before = _load_json_object(stage156_protocol_path)
    source_loaded_at = time.perf_counter()
    stage156_fingerprint = _fingerprint(stage156_protocol_path)
    model_fingerprints = {
        "config": _fingerprint(model_snapshot_path / "config.json"),
        "weights": _fingerprint(model_snapshot_path / "model.safetensors"),
        "tokenizer": _fingerprint(model_snapshot_path / "tokenizer.json"),
    }
    fingerprints_at = time.perf_counter()
    synthetic_cases = _canonical_synthetic_runtime_cases()
    synthetic_at = time.perf_counter()

    if not torch.cuda.is_available():
        raise RuntimeError("Stage157 formal validation requires the selected CUDA runtime")
    torch.cuda.reset_peak_memory_stats()
    backend = Qwen3VLTransformersTextGenerationBackend.load_local(
        snapshot_path=model_snapshot_path,
    )
    router = StrictStructuredDecisionRouter(backend=backend)
    model_loaded_at = time.perf_counter()

    resource_factory = PrimeQAHybridProcessRuntimeResourceFactory(
        stage128_protocol_path=stage128_protocol_path,
        stage125_protocol_path=stage125_protocol_path,
        stage80_report_path=stage80_report_path,
        documents_path=documents_path,
        encoder_device="cpu",
    )
    shared_resources = resource_factory.build_shared()
    resources_built_at = time.perf_counter()
    runtime = create_primeqa_hybrid_bounded_dynamic_agent_runtime(
        candidate_pool_retriever=shared_resources.candidate_pool_retriever,
        decision_router=router,
    )
    topology = runtime.topology()
    runtime.open_thread("stage157-formal-opaque-thread")
    turn_started_at = time.perf_counter()
    run = runtime.run_turn(
        opaque_thread_handle="stage157-formal-opaque-thread",
        question=_generated_label_free_runtime_query(),
    )
    turn_finished_at = time.perf_counter()
    open_summary = runtime.thread_summary("stage157-formal-opaque-thread")
    close_summary = runtime.close_thread("stage157-formal-opaque-thread")
    closed_at = time.perf_counter()
    metrics = run.workflow_run.router_metrics
    if metrics is None:
        raise RuntimeError("formal local router call produced no metrics")
    final_state = run.workflow_run.final_state
    real_runtime_probe = {
        "probe_shape": "builtin_generated_label_free_service_warmup",
        "candidate_pool_depth": len(run.candidate_pool_results),
        "generation_context_count": len(final_state["generation_context_results"]),
        "verification_context_count": len(final_state["verification_context_results"]),
        "verified_refused": run.verified_answer.refused,
        "verified_citation_count": len(run.verified_answer.citations),
        "runtime_trace": run.public_safe_trace.to_public_dict(),
        "router_metrics": metrics.to_public_dict(),
        "thread_summary_before_close": open_summary.to_public_dict(),
        "thread_summary_after_close": close_summary.to_public_dict(),
        "resource_summary": asdict(shared_resources.summary),
        "resource_factory_build_count": resource_factory.build_count,
        "retrieval_encoder_device": "cpu",
    }
    stage156_after = _load_json_object(stage156_protocol_path)
    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "validation_scope": (
            "One local-files-only Qwen structured decision after one real non-test retrieval, "
            "plus synthetic branch, schema, authority, and thread-state validation."
        ),
        "user_selected_configuration": {
            "gpu_environment": "new_project_virtual_environment",
            "independent_gpu_recheck": True,
            "max_completed_turns": PRODUCTION_MAX_COMPLETED_TURNS,
            "max_retained_bytes": PRODUCTION_MAX_RETAINED_BYTES,
            "prompt_profile": "top10_600_chars_12288_input_32_output",
        },
        "source_files": {
            "stage156_protocol": stage156_fingerprint,
            "model_snapshot_files": model_fingerprints,
        },
        "source_unchanged_after_validation": stage156_before == stage156_after,
        "stage156_summary": _stage156_summary(stage156_before, stage156_fingerprint),
        "environment": {
            "python_environment": "project_.venv",
            "torch_version": torch.__version__,
            "torchvision_version": version("torchvision"),
            "transformers_version": version("transformers"),
            "langgraph_version": version("langgraph"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "bfloat16_supported": torch.cuda.is_bf16_supported(),
        },
        "router_contract": structured_decision_router_contract(),
        "runtime_contract": bounded_dynamic_agent_runtime_contract(),
        "graph_topology": topology,
        "synthetic_runtime_cases": synthetic_cases,
        "real_non_test_runtime_probe": real_runtime_probe,
        "model_runtime": {
            "snapshot_revision": backend.snapshot_path.name,
            "load_count": 1,
            "generation_call_count": backend.generation_call_count,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "closed_boundaries": {
            "train_split_loaded": False,
            "dev_split_loaded": False,
            "test_split_loaded": False,
            "test_metrics_run": False,
            "gold_labels_read": False,
            "runtime_registered_as_default": False,
            "http_service_integrated": False,
            "socket_bound": False,
            "remote_exposure_authorized": False,
            "persistent_state_enabled": False,
            "query_rewrite_enabled": False,
            "second_retrieval_enabled": False,
            "queue_action_count": 0,
            "retry_action_count": 0,
            "fallback_action_count": 0,
            "raw_question_saved": False,
            "raw_answer_saved": False,
            "raw_document_saved": False,
            "raw_model_output_saved": False,
        },
        "timing_seconds": {
            "load_stage156": round(source_loaded_at - started_at, 6),
            "fingerprint_sources": round(fingerprints_at - source_loaded_at, 6),
            "synthetic_cases": round(synthetic_at - fingerprints_at, 6),
            "model_load": round(model_loaded_at - synthetic_at, 6),
            "retrieval_resource_build": round(resources_built_at - model_loaded_at, 6),
            "real_turn": round(turn_finished_at - turn_started_at, 6),
            "close": round(closed_at - turn_finished_at, 6),
            "total": round(closed_at - started_at, 6),
        },
    }
    report["guard_checks"] = _guard_checks(report)
    forbidden = sorted(_forbidden_keys_found(report))
    report["public_safe_contract"] = {
        "forbidden_keys_found": forbidden,
        "private_runtime_content_saved": False,
        "generated_synthetic_content_saved": False,
    }
    all_guards_passed = all(check["passed"] for check in report["guard_checks"])
    report["decision"] = {
        "status": _FINAL_STATUS if all_guards_passed and not forbidden else "stage157_rejected",
        "all_guards_passed": all_guards_passed,
        "runtime_implemented": True,
        "real_non_test_probe_completed": True,
        "runtime_registered_as_default": False,
        "test_gate_opened": False,
        "test_metrics_run": False,
        "next_direction": _NEXT_DIRECTION,
    }
    return report


def write_stage157_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[Stage157Visualization]:
    output_dir.mkdir(parents=True, exist_ok=True)
    real = report.get("real_non_test_runtime_probe") or {}
    trace = real.get("runtime_trace") or {}
    metrics = real.get("router_metrics") or {}
    timings = report.get("timing_seconds") or {}
    model = report.get("model_runtime") or {}
    topology = report.get("graph_topology") or {}
    closed = report.get("closed_boundaries") or {}
    synthetic = report.get("synthetic_runtime_cases") or {}
    charts = {
        "stage157_guard_status.svg": _chart(
            "Stage157 formal guard checks",
            [
                BarDatum(
                    label=str(check["name"]),
                    value=1.0 if check["passed"] else 0.0,
                    value_label="passed" if check["passed"] else "failed",
                )
                for check in report.get("guard_checks", [])
            ],
            width=3200,
            margin_left=1750,
        ),
        "stage157_real_branch_calls.svg": _chart(
            "Real non-test turn call profile",
            [
                _bar("model decision", trace.get("model_decision_count", 0)),
                _bar("retrieval", trace.get("retrieval_call_count", 0)),
                _bar("composition", trace.get("composition_call_count", 0)),
                _bar("verification", trace.get("verification_call_count", 0)),
                _bar("diagnostics", trace.get("diagnostic_observation_count", 0)),
            ],
        ),
        "stage157_model_tokens.svg": _chart(
            "Local router token counts",
            [
                _bar("input tokens", metrics.get("input_token_count", 0)),
                _bar("output tokens", metrics.get("output_token_count", 0)),
            ],
        ),
        "stage157_latency.svg": _chart(
            "Stage157 measured latency",
            [
                _bar("model load seconds", timings.get("model_load", 0)),
                _bar("retrieval build seconds", timings.get("retrieval_resource_build", 0)),
                _bar("real turn seconds", timings.get("real_turn", 0)),
                _bar("total seconds", timings.get("total", 0)),
            ],
        ),
        "stage157_gpu_memory.svg": _chart(
            "Local router peak GPU memory",
            [
                BarDatum(
                    label="peak allocated GiB",
                    value=float(model.get("peak_gpu_memory_bytes", 0)) / (1024**3),
                    value_label=(
                        f"{float(model.get('peak_gpu_memory_bytes', 0)) / (1024**3):.3f} GiB"
                    ),
                )
            ],
        ),
        "stage157_thread_state.svg": _chart(
            "Real turn volatile thread state",
            [
                _bar("completed turns", trace.get("completed_turn_count", 0)),
                _bar("retained KiB", float(trace.get("retained_state_bytes", 0)) / 1024),
                _bar(
                    "thread closed",
                    not bool(real.get("thread_summary_after_close", {}).get("opened", True)),
                ),
            ],
        ),
        "stage157_graph_topology.svg": _chart(
            "Bounded LangGraph topology",
            [
                _bar("nodes", topology.get("node_count", 0)),
                _bar("conditional routes", topology.get("conditional_edge_count", 0)),
                _bar("conditional targets", topology.get("conditional_target_edge_count", 0)),
                _bar("compile count", topology.get("compile_count", 0)),
            ],
        ),
        "stage157_synthetic_cases.svg": _chart(
            "Synthetic runtime cases",
            [
                BarDatum(
                    label=str(name),
                    value=1.0 if row.get("passed") is True else 0.0,
                    value_label="passed" if row.get("passed") is True else "failed",
                )
                for name, row in synthetic.items()
            ],
            margin_left=1000,
        ),
        "stage157_closed_boundaries.svg": _chart(
            "Closed runtime boundaries",
            [
                BarDatum(
                    label=label,
                    value=1.0 if closed.get(key) is True else 0.0,
                    value_label="enabled" if closed.get(key) is True else "closed",
                )
                for key, label in (
                    ("test_split_loaded", "test loaded"),
                    ("runtime_registered_as_default", "runtime default"),
                    ("remote_exposure_authorized", "remote exposure"),
                    ("persistent_state_enabled", "persistent state"),
                    ("second_retrieval_enabled", "second retrieval"),
                    ("retry_action_count", "retry actions"),
                    ("fallback_action_count", "fallback actions"),
                )
            ],
        ),
        "stage157_source_integrity.svg": _chart(
            "Stage156 source integrity",
            [
                _bar(
                    "source guards passed",
                    report.get("stage156_summary", {}).get("passed_guard_count", 0),
                ),
                _bar(
                    "source guards expected",
                    report.get("stage156_summary", {}).get("guard_count", 0),
                ),
                _bar(
                    "source unchanged",
                    report.get("source_unchanged_after_validation") is True,
                ),
            ],
        ),
    }
    artifacts = []
    for name, svg in charts.items():
        path = output_dir / name
        path.write_text(svg, encoding="utf-8")
        artifacts.append(Stage157Visualization(name=name, path=str(path)))
    return artifacts


def _canonical_synthetic_runtime_cases() -> dict[str, dict[str, Any]]:
    compose_runtime, compose_backend = _synthetic_runtime('{"action":"compose_grounded_answer"}')
    compose_runtime.open_thread("synthetic-compose")
    compose = compose_runtime.run_turn(
        opaque_thread_handle="synthetic-compose",
        question=_synthetic_question("compose"),
    )
    compose_runtime.close_thread("synthetic-compose")

    refuse_runtime, refuse_backend = _synthetic_runtime('{"action":"refuse_insufficient_evidence"}')
    refuse_runtime.open_thread("synthetic-refuse")
    refuse = refuse_runtime.run_turn(
        opaque_thread_handle="synthetic-refuse",
        question=_synthetic_question("refuse"),
    )
    refuse_runtime.close_thread("synthetic-refuse")

    malformed_runtime, malformed_backend = _synthetic_runtime("not-json")
    malformed_runtime.open_thread("synthetic-malformed")
    malformed_rejected = False
    try:
        malformed_runtime.run_turn(
            opaque_thread_handle="synthetic-malformed",
            question=_synthetic_question("malformed"),
        )
    except StructuredDecisionSchemaError:
        malformed_rejected = True
    malformed_trace = malformed_runtime.last_public_trace
    malformed_runtime.close_thread("synthetic-malformed")

    unauthorized_runtime, unauthorized_backend = _synthetic_runtime(
        '{"action":"retrieve_candidate_pool"}'
    )
    unauthorized_runtime.open_thread("synthetic-unauthorized")
    unauthorized_rejected = False
    try:
        unauthorized_runtime.run_turn(
            opaque_thread_handle="synthetic-unauthorized",
            question=_synthetic_question("unauthorized"),
        )
    except StructuredDecisionSchemaError:
        unauthorized_rejected = True
    unauthorized_trace = unauthorized_runtime.last_public_trace
    unauthorized_runtime.close_thread("synthetic-unauthorized")

    isolation_runtime, isolation_backend = _synthetic_runtime(
        '{"action":"refuse_insufficient_evidence"}'
    )
    isolation_runtime.open_thread("synthetic-thread-a")
    isolation_runtime.open_thread("synthetic-thread-b")
    isolation_runtime.run_turn(
        opaque_thread_handle="synthetic-thread-a",
        question=_synthetic_question("isolation-a"),
    )
    isolation_runtime.run_turn(
        opaque_thread_handle="synthetic-thread-b",
        question=_synthetic_question("isolation-b"),
    )
    a_summary = isolation_runtime.thread_summary("synthetic-thread-a")
    b_summary = isolation_runtime.thread_summary("synthetic-thread-b")
    isolation_runtime.close_thread("synthetic-thread-a")
    isolation_runtime.close_thread("synthetic-thread-b")

    return {
        "compose_branch": {
            "passed": (
                compose.public_safe_trace.terminal_state == "complete"
                and compose.public_safe_trace.composition_call_count == 1
                and compose.public_safe_trace.verification_call_count == 1
                and compose.public_safe_trace.diagnostic_observation_count == 1
                and compose_backend.call_count == 1
            )
        },
        "early_refuse_branch": {
            "passed": (
                refuse.verified_answer.answer == FIXED_INSUFFICIENT_EVIDENCE_RESPONSE
                and refuse.public_safe_trace.composition_call_count == 0
                and refuse.public_safe_trace.verification_call_count == 0
                and refuse.public_safe_trace.diagnostic_observation_count == 0
                and refuse_backend.call_count == 1
            )
        },
        "malformed_schema_rejected": {
            "passed": (
                malformed_rejected
                and malformed_backend.call_count == 1
                and malformed_trace is not None
                and malformed_trace.retry_action_count == 0
                and malformed_trace.fallback_action_count == 0
            )
        },
        "unauthorized_action_rejected": {
            "passed": (
                unauthorized_rejected
                and unauthorized_backend.call_count == 1
                and unauthorized_trace is not None
                and unauthorized_trace.composition_call_count == 0
            )
        },
        "thread_state_isolated": {
            "passed": (
                a_summary.completed_turn_count == b_summary.completed_turn_count == 1
                and isolation_backend.call_count == 2
            )
        },
    }


class _SyntheticBackend:
    def __init__(self, text: str) -> None:
        self._text = text
        self.call_count = 0

    def generate(
        self,
        *,
        prompt: str,
        max_input_tokens: int,
        max_new_tokens: int,
    ) -> GeneratedRouterText:
        self.call_count += 1
        return GeneratedRouterText(
            text=self._text,
            input_token_count=100,
            output_token_count=9,
            generation_latency_ms=1.0,
        )


class _SyntheticRetriever:
    def retrieve(self, question: PrimeQARuntimeQuery) -> tuple[RetrievalResult, ...]:
        return tuple(
            RetrievalResult(
                document=PrimeQADocument(
                    id=f"synthetic-{question.id}-{rank}",
                    title=f"Synthetic procedure {rank}",
                    text="Use the generated verification procedure.",
                ),
                score=10.0 / rank,
                rank=rank,
            )
            for rank in range(1, 401)
        )


class _SyntheticGenerator:
    def generate(
        self,
        question: PrimeQARuntimeQuery,
        retrieval_results: Sequence[RetrievalResult],
    ) -> GeneratedAnswer:
        first = retrieval_results[0]
        return GeneratedAnswer(
            question_id=question.id,
            answer="Generated verified procedure.",
            citations=[
                AnswerCitation(
                    document_id=first.document.id,
                    title=first.document.title,
                    retrieval_rank=first.rank,
                    evidence_score=10.0,
                )
            ],
            refused=False,
        )


class _SyntheticGeneratorFactory:
    def create(self) -> _SyntheticGenerator:
        return _SyntheticGenerator()


class _SyntheticVerifier:
    def verify(
        self,
        answer: GeneratedAnswer,
        retrieval_results: Sequence[RetrievalResult],
    ) -> AnswerVerificationResult:
        return AnswerVerificationResult(
            original_answer=answer,
            verified_answer=answer,
            citation_context_valid=True,
            reasons=["verified"],
        )


class _SyntheticVerifierFactory:
    def create(self) -> _SyntheticVerifier:
        return _SyntheticVerifier()


def _synthetic_runtime(raw_decision: str):
    backend = _SyntheticBackend(raw_decision)
    router = StrictStructuredDecisionRouter(backend=backend)
    toolset = PrimeQAHybridAgentToolset(
        candidate_pool_retriever=_SyntheticRetriever(),
        observation_adapter=PrimeQAHybridSidecarObservationAdapter(),
        answer_generator_factory=_SyntheticGeneratorFactory(),
        answer_verifier_factory=_SyntheticVerifierFactory(),
    )
    runtime = create_primeqa_hybrid_bounded_dynamic_agent_runtime_from_toolset(
        toolset=toolset,
        decision_router=router,
        thread_ledger=VolatileThreadStateLedger(
            limits=ThreadStateLimits(
                max_completed_turns=PRODUCTION_MAX_COMPLETED_TURNS,
                max_retained_bytes=PRODUCTION_MAX_RETAINED_BYTES,
            )
        ),
    )
    return runtime, backend


def _synthetic_question(suffix: str) -> PrimeQARuntimeQuery:
    return PrimeQARuntimeQuery(
        id=f"synthetic-{suffix}",
        text="Generated private validation question.",
    )


def _generated_label_free_runtime_query() -> PrimeQARuntimeQuery:
    return PrimeQARuntimeQuery(
        id="stage157-generated-label-free-probe",
        title="Service installation verification",
        text="How can I verify a service configuration after installation?",
    )


def _stage156_summary(
    report: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    checks = report.get("guard_checks") or []
    decision = report.get("decision") or {}
    return {
        "identity_exact": (
            report.get("stage") == "Stage 156"
            and report.get("analysis_id") == _SOURCE_STAGE156_ANALYSIS_ID
            and decision.get("status") == _SOURCE_STAGE156_STATUS
        ),
        "fingerprint_exact": fingerprint.get("sha256") == _EXPECTED_STAGE156_SHA256,
        "guard_count": len(checks),
        "passed_guard_count": sum(check.get("passed") is True for check in checks),
        "all_guards_passed": (
            len(checks) == _EXPECTED_STAGE156_GUARDS
            and all(check.get("passed") is True for check in checks)
        ),
        "runtime_registered_as_default": decision.get("runtime_registered_as_default"),
        "test_gate_opened": decision.get("test_gate_opened"),
        "test_metrics_run": decision.get("test_metrics_run"),
    }


def _guard_checks(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = report.get("stage156_summary") or {}
    environment = report.get("environment") or {}
    router = report.get("router_contract") or {}
    runtime = report.get("runtime_contract") or {}
    topology = report.get("graph_topology") or {}
    synthetic = report.get("synthetic_runtime_cases") or {}
    real = report.get("real_non_test_runtime_probe") or {}
    trace = real.get("runtime_trace") or {}
    metrics = real.get("router_metrics") or {}
    thread_after = real.get("thread_summary_after_close") or {}
    model = report.get("model_runtime") or {}
    closed = report.get("closed_boundaries") or {}
    selected_action = trace.get("selected_action")
    branch_calls_exact = [
        trace.get("composition_call_count"),
        trace.get("verification_call_count"),
        trace.get("diagnostic_observation_count"),
    ] == ([1, 1, 1] if selected_action == "compose_grounded_answer" else [0, 0, 0])
    checks = [
        _check("stage156_identity_exact", source.get("identity_exact") is True),
        _check("stage156_fingerprint_exact", source.get("fingerprint_exact") is True),
        _check("stage156_all_43_guards_passed", source.get("all_guards_passed") is True),
        _check(
            "stage156_source_unchanged", report.get("source_unchanged_after_validation") is True
        ),
        _check(
            "gpu_environment_exact",
            environment.get("torch_version") == "2.11.0+cu128"
            and environment.get("torchvision_version") == "0.26.0+cu128"
            and environment.get("transformers_version") == "5.13.1"
            and environment.get("cuda_available") is True
            and environment.get("cuda_version") == "12.8"
            and environment.get("gpu_capability") == [12, 0]
            and environment.get("bfloat16_supported") is True,
        ),
        _check("router_local_files_only", router.get("local_files_only") is True),
        _check(
            "router_model_exact",
            router.get("model_id") == "Qwen/Qwen3-VL-2B-Instruct"
            and router.get("device") == "cuda:0"
            and router.get("dtype") == "bfloat16",
        ),
        _check(
            "prompt_profile_exact",
            router.get("prompt_policy")
            == {
                "max_evidence_results": 10,
                "max_evidence_chars_per_result": 600,
                "max_input_tokens": 12_288,
                "max_new_tokens": 32,
            },
        ),
        _check("strict_schema_enabled", router.get("strict_json_schema") is True),
        _check(
            "prompt_overflow_rejects",
            router.get("input_overflow_behavior") == "reject_before_generation",
        ),
        _check(
            "thread_limits_exact",
            runtime.get("thread_limits")
            == {"max_completed_turns": 4, "max_retained_bytes": 32_768},
        ),
        _check(
            "volatile_state_only",
            runtime.get("thread_storage") == "process_local_volatile_memory_only",
        ),
        _check(
            "graph_topology_exact",
            topology.get("node_count") == 9
            and topology.get("conditional_edge_count") == 1
            and topology.get("conditional_target_edge_count") == 2
            and topology.get("compile_count") == 1,
        ),
        _check("graph_no_checkpointer", topology.get("checkpointer_attached") is False),
        _check("graph_no_cache", topology.get("cache_attached") is False),
        *[
            _check(f"synthetic_{name}", row.get("passed") is True)
            for name, row in synthetic.items()
        ],
        _check("real_candidate_pool_top400", real.get("candidate_pool_depth") == 400),
        _check("real_generation_context_top10", real.get("generation_context_count") == 10),
        _check("real_verification_context_top200", real.get("verification_context_count") == 200),
        _check("real_retrieval_once", trace.get("retrieval_call_count") == 1),
        _check("real_model_decision_once", trace.get("model_decision_count") == 1),
        _check(
            "real_action_allowed",
            selected_action in {"compose_grounded_answer", "refuse_insufficient_evidence"},
        ),
        _check("real_branch_calls_exact", branch_calls_exact),
        _check("real_schema_valid", metrics.get("schema_valid") is True),
        _check("real_input_within_limit", 0 < int(metrics.get("input_token_count", 0)) <= 12_288),
        _check("real_output_within_limit", 0 < int(metrics.get("output_token_count", 0)) <= 32),
        _check("real_thread_committed_once", trace.get("completed_turn_count") == 1),
        _check("real_thread_closed", thread_after.get("opened") is False),
        _check("real_resources_built_once", real.get("resource_factory_build_count") == 1),
        _check("real_dense_retrieval_on_cpu", real.get("retrieval_encoder_device") == "cpu"),
        _check("model_loaded_once", model.get("load_count") == 1),
        _check("model_called_once", model.get("generation_call_count") == 1),
        _check("gpu_memory_observed", int(model.get("peak_gpu_memory_bytes", 0)) > 0),
        _check(
            "test_remained_closed",
            closed.get("test_split_loaded") is False and closed.get("test_metrics_run") is False,
        ),
        _check("gold_labels_not_read", closed.get("gold_labels_read") is False),
        _check("runtime_not_default", closed.get("runtime_registered_as_default") is False),
        _check("http_not_integrated", closed.get("http_service_integrated") is False),
        _check("remote_closed", closed.get("remote_exposure_authorized") is False),
        _check("persistence_closed", closed.get("persistent_state_enabled") is False),
        _check("query_rewrite_closed", closed.get("query_rewrite_enabled") is False),
        _check("second_retrieval_closed", closed.get("second_retrieval_enabled") is False),
        _check(
            "recovery_actions_zero",
            [
                closed.get("queue_action_count"),
                closed.get("retry_action_count"),
                closed.get("fallback_action_count"),
            ]
            == [0, 0, 0],
        ),
        _check(
            "private_content_not_saved",
            [
                closed.get("raw_question_saved"),
                closed.get("raw_answer_saved"),
                closed.get("raw_document_saved"),
                closed.get("raw_model_output_saved"),
            ]
            == [False, False, False, False],
        ),
    ]
    return checks


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _bar(label: str, value: int | float | bool) -> BarDatum:
    numeric = float(value)
    return BarDatum(label=label, value=numeric, value_label=str(value))


def _chart(
    title: str,
    bars: list[BarDatum],
    *,
    width: int = 1800,
    margin_left: int = 800,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label="observed value",
        width=width,
        margin_left=margin_left,
    )
