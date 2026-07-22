from __future__ import annotations

import ctypes
import hashlib
import json
import locale
import os
import statistics
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from ctypes import wintypes
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ts_rag_agent.application.primeqa_hybrid_bounded_iterative_agent_runtime import (
    bounded_iterative_agent_runtime_contract,
)
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    _build_dense_channels,
    _build_lexical_channels,
    _build_train_fold_assignments,
    _load_json_object,
)
from ts_rag_agent.application.primeqa_hybrid_iterative_decision_router import (
    ClarificationKind,
    IterativeDecisionAction,
    IterativeDecisionPhase,
    StrictIterativeStructuredDecisionRouter,
    iterative_decision_router_contract,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector import (
    ContextCandidateRecord,
    records_by_sample,
    select_current_query_overlap_top10,
    select_original_rrf_top10,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector_training import (
    Stage161TrainCandidateDatasetBuilder,
)
from ts_rag_agent.application.primeqa_hybrid_structured_decision_router import (
    Qwen3VLTransformersTextGenerationBackend,
    StructuredDecisionSchemaError,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQARuntimeQuery
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)

_STAGE = "Stage 169"
_CREATED_AT = "2026-07-22"
_ANALYSIS_ID = "primeqa_hybrid_real_gpu_iterative_router_calibration_v1"
_EXPECTED_TRAIN_ROWS = 562
_EXPECTED_CANDIDATE_ROWS = 112_400
_TRAIN_CASES_PER_STRATUM = 10
_SYNTHETIC_INITIAL_CASES = 14
_SYNTHETIC_FINAL_CASES = 4
_EXPECTED_MODEL_CALLS = 118
_TRAIN_STRATA = (
    "initial_gold_visible",
    "alternate_only_gold_visible",
    "union_gold_missing_candidate_hit",
    "candidate_pool_gold_missing",
    "unanswerable",
)
_QUALITY_THRESHOLDS = {
    "synthetic_phase_action_accuracy_min": 0.80,
    "synthetic_clarification_kind_accuracy_min": 5 / 6,
    "real_initial_visible_compose_rate_min": 0.70,
    "real_alternate_only_inspect_rate_min": 0.50,
    "real_alternate_only_final_compose_rate_min": 0.70,
    "real_alternate_only_path_success_rate_min": 0.40,
    "real_insufficient_final_compose_rate_max": 0.20,
    "schema_valid_rate_min": 1.0,
}
_SOURCE_HASHES = {
    "stage168": "27dd3266414e9e2e766588095b0792be035b7e3e1610bc9355167b0243fcf80a",
    "stage80": "2441bb1cb1e7888299d3f57962b18cd59df84e2086ac281105abcacfc144880f",
    "train": "cabd93e0b972c47384c4bf5cc2cd215a7fc519b2df4f81fba61db73c931aa155",
    "documents": "f93b5e2d8dcfb2c7d12676ef32ce22b7809692f14081aad98096099a5256722b",
    "model_config": "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b",
    "model_weights": "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0",
    "model_tokenizer": "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
}
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer",
        "answer_doc_id",
        "answer_text",
        "candidate_doc_ids",
        "document_id",
        "document_text",
        "gold_answer",
        "question_id",
        "question_text",
        "raw_model_output",
        "sample_id",
    }
)

ProgressSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class SyntheticCalibrationCase:
    case_id: str
    question: PrimeQARuntimeQuery
    initial_evidence: tuple[RetrievalResult, ...]
    alternate_evidence: tuple[RetrievalResult, ...]
    expected_initial_action: str
    expected_initial_clarification_kind: str | None = None
    expected_final_action: str | None = None


@dataclass(frozen=True)
class RouterCallObservation:
    action: str | None
    clarification_kind: str | None
    schema_valid: bool
    input_token_count: int
    output_token_count: int
    generation_latency_ms: float
    process_working_set_bytes: int
    process_private_usage_bytes: int
    system_available_memory_bytes: int
    gpu_peak_allocated_bytes: int
    gpu_peak_reserved_bytes: int


@dataclass(frozen=True)
class TrainCalibrationCase:
    stratum: str
    fold_id: str
    question: PrimeQARuntimeQuery
    initial_evidence: tuple[RetrievalResult, ...]
    alternate_evidence: tuple[RetrievalResult, ...]


@dataclass(frozen=True)
class Stage169Visualization:
    name: str
    path: str


@dataclass(frozen=True)
class ResourceSnapshot:
    phase: str
    process_working_set_bytes: int
    process_peak_working_set_bytes: int
    process_private_usage_bytes: int
    system_available_memory_bytes: int
    gpu_allocated_bytes: int
    gpu_reserved_bytes: int
    process_cpu_time_seconds: float


class Stage169ResourceTracker:
    """Capture event-driven Windows process, system memory, and CUDA counters."""

    def __init__(self, *, torch_module: Any) -> None:
        self._torch = torch_module
        self._snapshots: list[ResourceSnapshot] = []

    @property
    def snapshots(self) -> tuple[ResourceSnapshot, ...]:
        return tuple(self._snapshots)

    @property
    def torch_module(self) -> Any:
        return self._torch

    def capture(self, phase: str) -> ResourceSnapshot:
        process = _windows_process_memory()
        snapshot = ResourceSnapshot(
            phase=phase,
            process_working_set_bytes=process["working_set_bytes"],
            process_peak_working_set_bytes=process["peak_working_set_bytes"],
            process_private_usage_bytes=process["private_usage_bytes"],
            system_available_memory_bytes=_windows_available_memory_bytes(),
            gpu_allocated_bytes=(
                int(self._torch.cuda.memory_allocated()) if self._torch.cuda.is_available() else 0
            ),
            gpu_reserved_bytes=(
                int(self._torch.cuda.memory_reserved()) if self._torch.cuda.is_available() else 0
            ),
            process_cpu_time_seconds=round(time.process_time(), 6),
        )
        self._snapshots.append(snapshot)
        return snapshot


def build_synthetic_calibration_cases() -> tuple[SyntheticCalibrationCase, ...]:
    direct = (
        _synthetic_case(
            "compose_password_reset",
            "How do I reset the admin password for Acme Console?",
            "Open Settings, choose Security, and select Reset admin password.",
            IterativeDecisionAction.COMPOSE.value,
        ),
        _synthetic_case(
            "compose_service_restart",
            "What command restarts the Acme indexing service on Linux?",
            "Run systemctl restart acme-indexer.service as an administrator.",
            IterativeDecisionAction.COMPOSE.value,
        ),
        _synthetic_case(
            "refuse_nontechnical_recipe",
            "Write a cake recipe for twelve people.",
            "This evidence describes Acme server log rotation.",
            IterativeDecisionAction.REFUSE.value,
        ),
        _synthetic_case(
            "refuse_nontechnical_market",
            "Which stock should I buy tomorrow?",
            "This evidence describes supported Acme firmware releases.",
            IterativeDecisionAction.REFUSE.value,
        ),
    )
    clarifications = tuple(
        _clarification_case(case_id, question, evidence, kind)
        for case_id, question, evidence, kind in (
            (
                "clarify_product",
                "How do I install the driver?",
                "Driver installation differs by product and component.",
                ClarificationKind.PRODUCT_OR_COMPONENT,
            ),
            (
                "clarify_version",
                "Why is this setting unavailable in my release?",
                "Setting availability differs between software versions and builds.",
                ClarificationKind.VERSION_OR_BUILD,
            ),
            (
                "clarify_error",
                "The application failed. How do I fix it?",
                "Failure diagnosis requires the exact error code or relevant log excerpt.",
                ClarificationKind.ERROR_CODE_OR_LOG,
            ),
            (
                "clarify_environment",
                "Which command should I use to connect the client?",
                "Connection commands differ by operating system and platform.",
                ClarificationKind.ENVIRONMENT_OR_PLATFORM,
            ),
            (
                "clarify_outcome",
                "I need help with the Acme server.",
                "Support procedures depend on the user's requested outcome.",
                ClarificationKind.REQUESTED_OUTCOME,
            ),
            (
                "clarify_reproduction",
                "The issue returned after I changed some settings. Why?",
                "Diagnosis requires the exact sequence of steps that reproduces the issue.",
                ClarificationKind.REPRODUCTION_STEPS,
            ),
        )
    )
    inspect = (
        _inspect_case(
            "inspect_then_compose_password",
            "How do I rotate the Acme gateway certificate?",
            "The initial evidence only discusses printer setup.",
            "Use acmectl certificate rotate, then restart the gateway.",
            IterativeDecisionAction.COMPOSE.value,
        ),
        _inspect_case(
            "inspect_then_compose_cache",
            "How do I clear the Acme repository cache?",
            "The initial evidence only discusses user interface themes.",
            "Run acmectl repository cache clear and confirm the operation.",
            IterativeDecisionAction.COMPOSE.value,
        ),
        _inspect_case(
            "inspect_then_refuse_unknown_protocol",
            "How do I enable ZX-900 mode in Acme Gateway?",
            "The initial evidence does not mention ZX-900 mode.",
            "The alternate evidence only documents standard HTTPS mode.",
            IterativeDecisionAction.REFUSE.value,
        ),
        _inspect_case(
            "inspect_then_refuse_missing_feature",
            "How do I activate quantum backup in Acme Console?",
            "The initial evidence contains no quantum backup feature.",
            "The alternate evidence documents ordinary snapshot backups only.",
            IterativeDecisionAction.REFUSE.value,
        ),
    )
    cases = (*direct, *clarifications, *inspect)
    if len(cases) != _SYNTHETIC_INITIAL_CASES:
        raise RuntimeError("Stage169 synthetic case count drifted")
    return cases


def select_train_calibration_cases(
    *,
    samples: Sequence[PrimeQAHybridSplitSample],
    grouped_records: Mapping[str, Sequence[ContextCandidateRecord]],
    documents_by_id: Mapping[str, PrimeQADocument],
    per_stratum: int = _TRAIN_CASES_PER_STRATUM,
) -> tuple[TrainCalibrationCase, ...]:
    grouped_samples: dict[str, list[PrimeQAHybridSplitSample]] = {
        stratum: [] for stratum in _TRAIN_STRATA
    }
    for sample in samples:
        records = grouped_records[sample.sample_id]
        initial_ids = {
            record.document_id for record in select_current_query_overlap_top10(records).selected
        }
        alternate_ids = {
            record.document_id for record in select_original_rrf_top10(records).selected
        }
        candidate_ids = {record.document_id for record in records}
        if not sample.answerable:
            stratum = "unanswerable"
        elif sample.answer_doc_id in initial_ids:
            stratum = "initial_gold_visible"
        elif sample.answer_doc_id in alternate_ids:
            stratum = "alternate_only_gold_visible"
        elif sample.answer_doc_id in candidate_ids:
            stratum = "union_gold_missing_candidate_hit"
        else:
            stratum = "candidate_pool_gold_missing"
        grouped_samples[stratum].append(sample)

    selected: list[TrainCalibrationCase] = []
    for stratum in _TRAIN_STRATA:
        ordered = sorted(
            grouped_samples[stratum], key=lambda sample: _sha256_text(sample.sample_id)
        )
        if len(ordered) < per_stratum:
            raise ValueError(f"Stage169 stratum {stratum!r} has fewer than {per_stratum} rows")
        for sample in ordered[:per_stratum]:
            records = grouped_records[sample.sample_id]
            initial = select_current_query_overlap_top10(records).selected
            alternate = select_original_rrf_top10(records).selected
            selected.append(
                TrainCalibrationCase(
                    stratum=stratum,
                    fold_id=records[0].fold_id,
                    question=PrimeQARuntimeQuery(
                        id=_sha256_text(sample.sample_id),
                        title=sample.question_title,
                        text=sample.question_text,
                    ),
                    initial_evidence=_records_to_results(initial, documents_by_id),
                    alternate_evidence=_records_to_results(alternate, documents_by_id),
                )
            )
    return tuple(selected)


def run_stage169_real_gpu_calibration(
    *,
    stage168_report_path: Path,
    stage80_report_path: Path,
    train_split_path: Path,
    documents_path: Path,
    model_snapshot_path: Path,
    prior_failed_stdout_path: Path,
    prior_failed_stderr_path: Path,
    prior_failed_exit_path: Path,
    encoder_batch_size: int = 64,
    progress_sink: ProgressSink | None = None,
) -> dict[str, Any]:
    import torch

    started_at = time.perf_counter()
    tracker = Stage169ResourceTracker(torch_module=torch)
    tracker.capture("process_started")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    source_paths = {
        "stage168": stage168_report_path,
        "stage80": stage80_report_path,
        "train": train_split_path,
        "documents": documents_path,
        "model_config": model_snapshot_path / "config.json",
        "model_weights": model_snapshot_path / "model.safetensors",
        "model_tokenizer": model_snapshot_path / "tokenizer.json",
    }
    fingerprints = {name: _fingerprint(path) for name, path in source_paths.items()}
    _authorize_sources(fingerprints)
    stage168 = _load_json_object(stage168_report_path)
    if stage168.get("decision", {}).get("status") != (
        "advance_to_stage169_real_gpu_router_calibration"
    ):
        raise ValueError("Stage168 did not authorize Stage169")
    fingerprinted_at = time.perf_counter()
    tracker.capture("sources_fingerprinted")
    _emit(progress_sink, phase="sources_authorized")

    samples = load_primeqa_hybrid_split_samples(train_split_path)
    if len(samples) != _EXPECTED_TRAIN_ROWS or any(
        sample.assigned_split != "train" for sample in samples
    ):
        raise ValueError("Stage169 accepts only the exact 562-row train split")
    documents_by_id = load_primeqa_documents(documents_path)
    sections_by_document = load_primeqa_document_sections(documents_path)
    documents = list(documents_by_id.values())
    fold_assignments = _build_train_fold_assignments(samples, fold_count=5)
    stage80 = _load_json_object(stage80_report_path)
    dense_channels, dense_summary = _build_dense_channels(
        include_dense_channels=True,
        stage80_report=stage80,
        stage80_report_path=stage80_report_path,
        documents=documents,
        document_ids=tuple(document.id for document in documents),
        encoder_batch_size=encoder_batch_size,
        encoder_device="cpu",
        encoder_factory=None,
    )
    if dense_summary["status"] != "dense_channels_ready":
        raise RuntimeError("Stage169 requires both authorized local dense channels")
    lexical_channels = _build_lexical_channels(
        documents=documents,
        sections_by_document=sections_by_document,
        bm25_k1=1.5,
        bm25_b=0.75,
        component_depth=200,
    )
    channels = tuple([*lexical_channels, *dense_channels])
    records = Stage161TrainCandidateDatasetBuilder(
        documents_by_id=documents_by_id,
        sections_by_document=sections_by_document,
        channels=channels,
        fold_assignments=fold_assignments,
        progress_sink=progress_sink,
        progress_stage=_STAGE,
        progress_phase="train_candidate_replay",
    ).build(samples)
    if len(records) != _EXPECTED_CANDIDATE_ROWS:
        raise RuntimeError("Stage169 candidate replay row count drifted")
    train_cases = select_train_calibration_cases(
        samples=samples,
        grouped_records=records_by_sample(records),
        documents_by_id=documents_by_id,
    )
    evidence_ready_at = time.perf_counter()
    tracker.capture("train_evidence_ready")
    _emit(progress_sink, phase="train_evidence_ready", selected_cases=len(train_cases))

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage169 requires CUDA with bfloat16 support")
    torch.cuda.reset_peak_memory_stats()
    backend = Qwen3VLTransformersTextGenerationBackend.load_local(snapshot_path=model_snapshot_path)
    router = StrictIterativeStructuredDecisionRouter(backend=backend)
    model_loaded_at = time.perf_counter()
    tracker.capture("router_model_loaded")
    _emit(progress_sink, phase="router_model_loaded")

    synthetic_results, synthetic_calls = _run_synthetic_calibration(router, tracker, progress_sink)
    synthetic_finished_at = time.perf_counter()
    train_results, train_calls = _run_train_calibration(router, tracker, train_cases, progress_sink)
    calibration_finished_at = time.perf_counter()
    all_calls = (*synthetic_calls, *train_calls)
    tracker.capture("calibration_complete")
    quality_metrics = _quality_metrics(synthetic_results, train_results, all_calls)
    quality_gates = _quality_gates(quality_metrics)
    prior_failure = _prior_failed_run_summary(
        stdout_path=prior_failed_stdout_path,
        stderr_path=prior_failed_stderr_path,
        exit_path=prior_failed_exit_path,
    )
    process_guards = [
        _check("stage168_authorized_stage169", True),
        _check("exact_train_row_count", len(samples) == _EXPECTED_TRAIN_ROWS),
        _check("exact_candidate_row_count", len(records) == _EXPECTED_CANDIDATE_ROWS),
        _check("five_strata_selected", len(train_results) == len(_TRAIN_STRATA)),
        _check(
            "ten_cases_per_stratum",
            all(value["case_count"] == 10 for value in train_results.values()),
        ),
        _check("model_call_count_exact", backend.generation_call_count == _EXPECTED_MODEL_CALLS),
        _check("local_files_only", True),
        _check("gpu_cuda_available", torch.cuda.is_available()),
        _check("gpu_bfloat16_supported", torch.cuda.is_bf16_supported()),
        _check("retrieval_encoder_cpu", True),
        _check("development_not_loaded", True),
        _check("test_not_loaded", True),
        _check("answer_generation_not_run", True),
        _check("retry_count_zero", True),
        _check("default_runtime_unchanged", True),
        _check(
            "prior_oom_failure_recorded",
            prior_failure["exit_code"] == 1 and prior_failure["cuda_oom_confirmed"],
        ),
    ]
    all_process_guards_passed = all(guard["passed"] for guard in process_guards)
    all_quality_gates_passed = all(gate["passed"] for gate in quality_gates)
    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "validation_scope": (
            "Real local-files-only Qwen calibration over frozen synthetic action cases and "
            "deterministically stratified train-only retrieval evidence. No answer generation, "
            "development evaluation, test evaluation, retry, or default activation."
        ),
        "source_authorization": fingerprints,
        "prior_failed_formal_run": prior_failure,
        "environment": {
            "python_environment": "project_.venv",
            "torch_version": torch.__version__,
            "torchvision_version": version("torchvision"),
            "transformers_version": version("transformers"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "bfloat16_supported": torch.cuda.is_bf16_supported(),
        },
        "router_contract": iterative_decision_router_contract(),
        "runtime_contract": bounded_iterative_agent_runtime_contract(),
        "frozen_calibration_protocol": {
            "synthetic_initial_case_count": _SYNTHETIC_INITIAL_CASES,
            "synthetic_final_case_count": _SYNTHETIC_FINAL_CASES,
            "train_strata": list(_TRAIN_STRATA),
            "train_cases_per_stratum": _TRAIN_CASES_PER_STRATUM,
            "train_sampling": "ascending_sha256_of_frozen_sample_id_within_stratum",
            "real_phase_calls_per_case": 2,
            "expected_model_call_count": _EXPECTED_MODEL_CALLS,
            "quality_thresholds": _QUALITY_THRESHOLDS,
            "thresholds_frozen_before_model_execution": True,
        },
        "synthetic_calibration": synthetic_results,
        "train_proxy_calibration": train_results,
        "quality_metrics": quality_metrics,
        "quality_gates": quality_gates,
        "model_runtime": {
            "snapshot_revision": backend.snapshot_path.name,
            "load_count": 1,
            "generation_call_count": backend.generation_call_count,
            "peak_gpu_memory_bytes": max(call.gpu_peak_allocated_bytes for call in all_calls),
        },
        "resource_consumption": _resource_consumption_summary(
            tracker=tracker,
            calls=all_calls,
            wall_time_seconds=calibration_finished_at - started_at,
        ),
        "closed_boundaries": {
            "train_split_loaded": True,
            "development_split_loaded": False,
            "test_split_loaded": False,
            "answer_generation_run": False,
            "f1_or_citation_metrics_run": False,
            "raw_question_saved": False,
            "raw_answer_saved": False,
            "raw_document_saved": False,
            "raw_model_output_saved": False,
            "query_rewrite_enabled": False,
            "second_retrieval_enabled": False,
            "retry_action_count": 0,
            "runtime_registered_as_default": False,
            "http_service_integrated": False,
        },
        "limitations": [
            "Train gold-document visibility is only a routing proxy, not answer quality.",
            "Synthetic expected actions test instruction following, not production prevalence.",
            "Only ten deterministic rows per train stratum receive real model decisions.",
            "Clarification quality lacks human labels and is evaluated only on synthetic cases.",
        ],
        "timing_seconds": {
            "source_fingerprinting": round(fingerprinted_at - started_at, 6),
            "train_evidence_build": round(evidence_ready_at - fingerprinted_at, 6),
            "model_load": round(model_loaded_at - evidence_ready_at, 6),
            "synthetic_calibration": round(synthetic_finished_at - model_loaded_at, 6),
            "train_calibration": round(calibration_finished_at - synthetic_finished_at, 6),
            "total_before_visualization": round(calibration_finished_at - started_at, 6),
        },
        "process_guards": process_guards,
        "public_safe_contract": {
            "forbidden_keys": sorted(_FORBIDDEN_PUBLIC_KEYS),
            "forbidden_keys_found": [],
        },
        "decision": {
            "all_process_guards_passed": all_process_guards_passed,
            "all_quality_gates_passed": all_quality_gates_passed,
            "status": (
                "advance_to_stage170_train_only_iterative_runtime_e2e"
                if all_process_guards_passed and all_quality_gates_passed
                else "stage169_router_requires_redesign"
            ),
            "default_runtime_activation": False,
            "development_opened": False,
            "test_opened": False,
        },
    }
    forbidden = sorted(_forbidden_keys_found(report))
    report["public_safe_contract"]["forbidden_keys_found"] = forbidden
    report["process_guards"].append(
        _check("public_report_contains_no_forbidden_keys", not forbidden)
    )
    report["decision"]["all_process_guards_passed"] = all(
        guard["passed"] for guard in report["process_guards"]
    )
    if not report["decision"]["all_process_guards_passed"]:
        report["decision"]["status"] = "stage169_process_invalid"
    _emit(progress_sink, phase="calibration_complete", decision=report["decision"])
    return report


def write_stage169_visualizations(
    *, report: Mapping[str, Any], output_dir: Path
) -> tuple[Stage169Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = report["quality_metrics"]
    synthetic = report["synthetic_calibration"]
    train = report["train_proxy_calibration"]
    resources = report["resource_consumption"]
    inspect_action = IterativeDecisionAction.INSPECT.value
    charts = {
        "synthetic_quality.svg": _chart(
            "Stage 169 synthetic router quality",
            (
                _rate_bar("Phase action accuracy", synthetic["phase_action_accuracy"]),
                _rate_bar(
                    "Clarification kind accuracy",
                    synthetic["clarification_kind_accuracy"],
                ),
                _rate_bar("Exact path accuracy", synthetic["exact_path_accuracy"]),
                _rate_bar("Schema valid rate", metrics["schema_valid_rate"]),
            ),
        ),
        "train_proxy_quality.svg": _chart(
            "Stage 169 train-only routing proxy quality",
            (
                _rate_bar(
                    "Initial-visible compose",
                    metrics["real_initial_visible_compose_rate"],
                ),
                _rate_bar(
                    "Alternate-only inspect",
                    metrics["real_alternate_only_inspect_rate"],
                ),
                _rate_bar(
                    "Alternate-only final compose",
                    metrics["real_alternate_only_final_compose_rate"],
                ),
                _rate_bar(
                    "Alternate-only exact path",
                    metrics["real_alternate_only_path_success_rate"],
                ),
                _rate_bar(
                    "Insufficient final compose",
                    metrics["real_insufficient_final_compose_rate"],
                ),
            ),
        ),
        "train_action_distribution.svg": _chart(
            "Stage 169 initial inspect counts by train stratum",
            tuple(
                BarDatum(
                    label=stratum.replace("_", " "),
                    value=float(train[stratum]["initial_action_counts"].get(inspect_action, 0)),
                    value_label=(
                        f"{train[stratum]['initial_action_counts'].get(inspect_action, 0)} / 10"
                    ),
                )
                for stratum in _TRAIN_STRATA
            ),
            x_label="Real Qwen initial inspect selections",
        ),
        "router_latency.svg": _chart(
            "Stage 169 real GPU router generation latency",
            (
                BarDatum(
                    "p50", metrics["latency_ms"]["p50"], f"{metrics['latency_ms']['p50']:.1f} ms"
                ),
                BarDatum(
                    "p95", metrics["latency_ms"]["p95"], f"{metrics['latency_ms']['p95']:.1f} ms"
                ),
                BarDatum(
                    "maximum",
                    metrics["latency_ms"]["max"],
                    f"{metrics['latency_ms']['max']:.1f} ms",
                ),
            ),
            x_label="Generation latency in milliseconds",
        ),
        "resource_peaks.svg": _chart(
            "Stage 169 process and GPU resource peaks",
            (
                _gib_bar("Process working set", resources["process_peak_working_set_bytes"]),
                _gib_bar("Process private usage", resources["process_peak_private_usage_bytes"]),
                _gib_bar("GPU allocated", resources["gpu_peak_allocated_bytes"]),
                _gib_bar("GPU reserved", resources["gpu_peak_reserved_bytes"]),
                _gib_bar(
                    "Minimum system available",
                    resources["minimum_system_available_memory_bytes"],
                ),
            ),
            x_label="GiB",
        ),
    }
    visualizations = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        visualizations.append(
            Stage169Visualization(name=filename.removesuffix(".svg"), path=str(path))
        )
    return tuple(visualizations)


def _run_synthetic_calibration(
    router: StrictIterativeStructuredDecisionRouter,
    tracker: Stage169ResourceTracker,
    progress_sink: ProgressSink | None,
) -> tuple[dict[str, Any], tuple[RouterCallObservation, ...]]:
    rows = []
    calls: list[RouterCallObservation] = []
    for index, case in enumerate(build_synthetic_calibration_cases(), start=1):
        initial = _invoke_router(
            router,
            phase=IterativeDecisionPhase.INITIAL,
            question=case.question,
            initial=case.initial_evidence,
            alternate=(),
            tracker=tracker,
        )
        calls.append(initial)
        final = None
        if case.expected_final_action is not None:
            final = _invoke_router(
                router,
                phase=IterativeDecisionPhase.FINAL_AFTER_INSPECTION,
                question=case.question,
                initial=case.initial_evidence,
                alternate=case.alternate_evidence,
                tracker=tracker,
            )
            calls.append(final)
        rows.append(
            {
                "case_id": case.case_id,
                "expected_initial_action": case.expected_initial_action,
                "observed_initial_action": initial.action,
                "expected_clarification_kind": case.expected_initial_clarification_kind,
                "observed_clarification_kind": initial.clarification_kind,
                "expected_final_action": case.expected_final_action,
                "observed_final_action": final.action if final else None,
                "initial_schema_valid": initial.schema_valid,
                "final_schema_valid": final.schema_valid if final else None,
            }
        )
        _emit(progress_sink, phase="synthetic_router_calls", completed=index, total=14)
    phase_expectations = sum(1 + int(row["expected_final_action"] is not None) for row in rows)
    phase_matches = sum(
        int(row["expected_initial_action"] == row["observed_initial_action"])
        + int(
            row["expected_final_action"] is not None
            and row["expected_final_action"] == row["observed_final_action"]
        )
        for row in rows
    )
    clarification_rows = [row for row in rows if row["expected_clarification_kind"]]
    clarification_matches = sum(
        row["expected_clarification_kind"] == row["observed_clarification_kind"]
        for row in clarification_rows
    )
    exact_paths = sum(
        row["expected_initial_action"] == row["observed_initial_action"]
        and row["expected_clarification_kind"] == row["observed_clarification_kind"]
        and row["expected_final_action"] == row["observed_final_action"]
        for row in rows
    )
    return (
        {
            "case_count": len(rows),
            "phase_expectation_count": phase_expectations,
            "phase_action_match_count": phase_matches,
            "phase_action_accuracy": round(phase_matches / phase_expectations, 6),
            "clarification_case_count": len(clarification_rows),
            "clarification_kind_match_count": clarification_matches,
            "clarification_kind_accuracy": round(
                clarification_matches / len(clarification_rows), 6
            ),
            "exact_path_match_count": exact_paths,
            "exact_path_accuracy": round(exact_paths / len(rows), 6),
            "cases": rows,
        },
        tuple(calls),
    )


def _run_train_calibration(
    router: StrictIterativeStructuredDecisionRouter,
    tracker: Stage169ResourceTracker,
    cases: Sequence[TrainCalibrationCase],
    progress_sink: ProgressSink | None,
) -> tuple[dict[str, Any], tuple[RouterCallObservation, ...]]:
    observations: dict[str, list[tuple[RouterCallObservation, RouterCallObservation]]] = {
        stratum: [] for stratum in _TRAIN_STRATA
    }
    calls: list[RouterCallObservation] = []
    for index, case in enumerate(cases, start=1):
        initial = _invoke_router(
            router,
            phase=IterativeDecisionPhase.INITIAL,
            question=case.question,
            initial=case.initial_evidence,
            alternate=(),
            tracker=tracker,
        )
        final = _invoke_router(
            router,
            phase=IterativeDecisionPhase.FINAL_AFTER_INSPECTION,
            question=case.question,
            initial=case.initial_evidence,
            alternate=case.alternate_evidence,
            tracker=tracker,
        )
        calls.extend((initial, final))
        observations[case.stratum].append((initial, final))
        _emit(progress_sink, phase="train_router_calls", completed=index, total=len(cases))
    report = {}
    for stratum, pairs in observations.items():
        report[stratum] = {
            "case_count": len(pairs),
            "initial_action_counts": _action_counts(pair[0].action for pair in pairs),
            "final_action_counts": _action_counts(pair[1].action for pair in pairs),
            "initial_schema_valid_count": sum(pair[0].schema_valid for pair in pairs),
            "final_schema_valid_count": sum(pair[1].schema_valid for pair in pairs),
            "inspect_then_compose_count": sum(
                pair[0].action == IterativeDecisionAction.INSPECT.value
                and pair[1].action == IterativeDecisionAction.COMPOSE.value
                for pair in pairs
            ),
        }
    return report, tuple(calls)


def _quality_metrics(
    synthetic: Mapping[str, Any],
    train: Mapping[str, Any],
    calls: Sequence[RouterCallObservation],
) -> dict[str, Any]:
    initial_visible = train["initial_gold_visible"]
    alternate_only = train["alternate_only_gold_visible"]
    insufficient_strata = (
        "union_gold_missing_candidate_hit",
        "candidate_pool_gold_missing",
        "unanswerable",
    )
    insufficient_final_compose = sum(
        train[stratum]["final_action_counts"].get(IterativeDecisionAction.COMPOSE.value, 0)
        for stratum in insufficient_strata
    )
    insufficient_count = sum(train[stratum]["case_count"] for stratum in insufficient_strata)
    latencies = sorted(call.generation_latency_ms for call in calls)
    return {
        "synthetic_phase_action_accuracy": synthetic["phase_action_accuracy"],
        "synthetic_clarification_kind_accuracy": synthetic["clarification_kind_accuracy"],
        "real_initial_visible_compose_rate": _action_rate(
            initial_visible, "initial", IterativeDecisionAction.COMPOSE.value
        ),
        "real_alternate_only_inspect_rate": _action_rate(
            alternate_only, "initial", IterativeDecisionAction.INSPECT.value
        ),
        "real_alternate_only_final_compose_rate": _action_rate(
            alternate_only, "final", IterativeDecisionAction.COMPOSE.value
        ),
        "real_alternate_only_path_success_rate": round(
            alternate_only["inspect_then_compose_count"] / alternate_only["case_count"], 6
        ),
        "real_insufficient_final_compose_rate": round(
            insufficient_final_compose / insufficient_count, 6
        ),
        "schema_valid_count": sum(call.schema_valid for call in calls),
        "schema_valid_rate": round(sum(call.schema_valid for call in calls) / len(calls), 6),
        "model_call_count": len(calls),
        "input_tokens": _distribution(call.input_token_count for call in calls),
        "output_tokens": _distribution(call.output_token_count for call in calls),
        "latency_ms": {
            "min": round(latencies[0], 3),
            "p50": round(_nearest_rank(latencies, 0.50), 3),
            "p95": round(_nearest_rank(latencies, 0.95), 3),
            "max": round(latencies[-1], 3),
            "mean": round(statistics.fmean(latencies), 3),
        },
    }


def _quality_gates(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    pairs = (
        (
            "synthetic_phase_action_accuracy",
            metrics["synthetic_phase_action_accuracy"],
            _QUALITY_THRESHOLDS["synthetic_phase_action_accuracy_min"],
            "min",
        ),
        (
            "synthetic_clarification_kind_accuracy",
            metrics["synthetic_clarification_kind_accuracy"],
            _QUALITY_THRESHOLDS["synthetic_clarification_kind_accuracy_min"],
            "min",
        ),
        (
            "real_initial_visible_compose_rate",
            metrics["real_initial_visible_compose_rate"],
            _QUALITY_THRESHOLDS["real_initial_visible_compose_rate_min"],
            "min",
        ),
        (
            "real_alternate_only_inspect_rate",
            metrics["real_alternate_only_inspect_rate"],
            _QUALITY_THRESHOLDS["real_alternate_only_inspect_rate_min"],
            "min",
        ),
        (
            "real_alternate_only_final_compose_rate",
            metrics["real_alternate_only_final_compose_rate"],
            _QUALITY_THRESHOLDS["real_alternate_only_final_compose_rate_min"],
            "min",
        ),
        (
            "real_alternate_only_path_success_rate",
            metrics["real_alternate_only_path_success_rate"],
            _QUALITY_THRESHOLDS["real_alternate_only_path_success_rate_min"],
            "min",
        ),
        (
            "real_insufficient_final_compose_rate",
            metrics["real_insufficient_final_compose_rate"],
            _QUALITY_THRESHOLDS["real_insufficient_final_compose_rate_max"],
            "max",
        ),
        (
            "schema_valid_rate",
            metrics["schema_valid_rate"],
            _QUALITY_THRESHOLDS["schema_valid_rate_min"],
            "min",
        ),
    )
    return [
        {
            "name": name,
            "observed": observed,
            "threshold": threshold,
            "direction": direction,
            "passed": observed >= threshold if direction == "min" else observed <= threshold,
        }
        for name, observed, threshold, direction in pairs
    ]


def _invoke_router(
    router: StrictIterativeStructuredDecisionRouter,
    *,
    phase: IterativeDecisionPhase,
    question: PrimeQARuntimeQuery,
    initial: Sequence[RetrievalResult],
    alternate: Sequence[RetrievalResult],
    tracker: Stage169ResourceTracker,
) -> RouterCallObservation:
    torch = tracker.torch_module
    torch.cuda.reset_peak_memory_stats()
    action = None
    clarification_kind = None
    try:
        decision = router.decide(
            phase=phase,
            question=question,
            initial_evidence_results=initial,
            alternate_evidence_results=alternate,
            completed_turns=(),
        )
        action = decision.action
        clarification_kind = decision.clarification_kind
    except StructuredDecisionSchemaError:
        pass
    metrics = router.last_metrics
    if metrics is None:
        raise RuntimeError("Stage169 router call produced no metrics")
    resources = tracker.capture(f"router_call_{phase.value}")
    return RouterCallObservation(
        action=action,
        clarification_kind=clarification_kind,
        schema_valid=metrics.schema_valid,
        input_token_count=metrics.input_token_count,
        output_token_count=metrics.output_token_count,
        generation_latency_ms=metrics.generation_latency_ms,
        process_working_set_bytes=resources.process_working_set_bytes,
        process_private_usage_bytes=resources.process_private_usage_bytes,
        system_available_memory_bytes=resources.system_available_memory_bytes,
        gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        gpu_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
    )


def _synthetic_case(
    case_id: str, question: str, evidence: str, expected_action: str
) -> SyntheticCalibrationCase:
    return SyntheticCalibrationCase(
        case_id=case_id,
        question=PrimeQARuntimeQuery(id=case_id, text=question),
        initial_evidence=_synthetic_results(f"{case_id}-initial", evidence),
        alternate_evidence=(),
        expected_initial_action=expected_action,
    )


def _clarification_case(
    case_id: str,
    question: str,
    evidence: str,
    kind: ClarificationKind,
) -> SyntheticCalibrationCase:
    return SyntheticCalibrationCase(
        case_id=case_id,
        question=PrimeQARuntimeQuery(id=case_id, text=question),
        initial_evidence=_synthetic_results(f"{case_id}-initial", evidence),
        alternate_evidence=(),
        expected_initial_action=IterativeDecisionAction.CLARIFY.value,
        expected_initial_clarification_kind=kind.value,
    )


def _inspect_case(
    case_id: str,
    question: str,
    initial_evidence: str,
    alternate_evidence: str,
    expected_final_action: str,
) -> SyntheticCalibrationCase:
    return SyntheticCalibrationCase(
        case_id=case_id,
        question=PrimeQARuntimeQuery(id=case_id, text=question),
        initial_evidence=_synthetic_results(f"{case_id}-initial", initial_evidence),
        alternate_evidence=_synthetic_results(f"{case_id}-alternate", alternate_evidence),
        expected_initial_action=IterativeDecisionAction.INSPECT.value,
        expected_final_action=expected_final_action,
    )


def _synthetic_results(prefix: str, evidence: str) -> tuple[RetrievalResult, ...]:
    return tuple(
        RetrievalResult(
            document=PrimeQADocument(
                id=f"{prefix}-{rank}",
                title=f"Synthetic technical note {rank}",
                text=evidence,
            ),
            score=1.0 / rank,
            rank=rank,
        )
        for rank in range(1, 11)
    )


def _records_to_results(
    records: Sequence[ContextCandidateRecord],
    documents_by_id: Mapping[str, PrimeQADocument],
) -> tuple[RetrievalResult, ...]:
    return tuple(
        RetrievalResult(
            document=documents_by_id[record.document_id],
            score=float(record.features.get("stage116_rrf_score", 0.0)),
            rank=record.baseline_rank,
        )
        for record in records
    )


def _action_rate(summary: Mapping[str, Any], phase: str, action: str) -> float:
    return round(summary[f"{phase}_action_counts"].get(action, 0) / summary["case_count"], 6)


def _action_counts(actions: Any) -> dict[str, int]:
    counts = Counter(action if action is not None else "schema_invalid" for action in actions)
    return dict(sorted(counts.items()))


def _distribution(values: Sequence[int] | Any) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": float(ordered[0]),
        "p50": float(_nearest_rank(ordered, 0.50)),
        "p95": float(_nearest_rank(ordered, 0.95)),
        "max": float(ordered[-1]),
        "mean": round(statistics.fmean(ordered), 6),
    }


def _resource_consumption_summary(
    *,
    tracker: Stage169ResourceTracker,
    calls: Sequence[RouterCallObservation],
    wall_time_seconds: float,
) -> dict[str, Any]:
    snapshots = tracker.snapshots
    generation_seconds = sum(call.generation_latency_ms for call in calls) / 1000
    return {
        "sampling_mode": "event_driven_in_process_without_monitor_polling",
        "phase_snapshots": [
            {
                "phase": snapshot.phase,
                "process_working_set_bytes": snapshot.process_working_set_bytes,
                "process_private_usage_bytes": snapshot.process_private_usage_bytes,
                "system_available_memory_bytes": snapshot.system_available_memory_bytes,
                "gpu_allocated_bytes": snapshot.gpu_allocated_bytes,
                "gpu_reserved_bytes": snapshot.gpu_reserved_bytes,
                "process_cpu_time_seconds": snapshot.process_cpu_time_seconds,
            }
            for snapshot in snapshots
            if not snapshot.phase.startswith("router_call_")
        ],
        "process_peak_working_set_bytes": max(
            snapshot.process_peak_working_set_bytes for snapshot in snapshots
        ),
        "process_peak_private_usage_bytes": max(
            snapshot.process_private_usage_bytes for snapshot in snapshots
        ),
        "minimum_system_available_memory_bytes": min(
            snapshot.system_available_memory_bytes for snapshot in snapshots
        ),
        "gpu_peak_allocated_bytes": max(call.gpu_peak_allocated_bytes for call in calls),
        "gpu_peak_reserved_bytes": max(call.gpu_peak_reserved_bytes for call in calls),
        "per_call_gpu_peak_allocated_bytes": _distribution(
            call.gpu_peak_allocated_bytes for call in calls
        ),
        "per_call_gpu_peak_reserved_bytes": _distribution(
            call.gpu_peak_reserved_bytes for call in calls
        ),
        "per_call_process_working_set_bytes": _distribution(
            call.process_working_set_bytes for call in calls
        ),
        "per_call_process_private_usage_bytes": _distribution(
            call.process_private_usage_bytes for call in calls
        ),
        "minimum_per_call_system_available_memory_bytes": min(
            call.system_available_memory_bytes for call in calls
        ),
        "total_input_tokens": sum(call.input_token_count for call in calls),
        "total_output_tokens": sum(call.output_token_count for call in calls),
        "total_generation_seconds": round(generation_seconds, 6),
        "generation_calls_per_second": round(len(calls) / generation_seconds, 6),
        "wall_time_seconds": round(wall_time_seconds, 6),
        "process_cpu_time_seconds": round(
            snapshots[-1].process_cpu_time_seconds - snapshots[0].process_cpu_time_seconds,
            6,
        ),
    }


class _ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("page_fault_count", ctypes.c_ulong),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
        ("private_usage", ctypes.c_size_t),
    ]


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


def _windows_process_memory() -> dict[str, int]:
    if os.name != "nt":
        raise RuntimeError("Stage169 resource accounting requires Windows")
    counters = _ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessMemoryCountersEx),
        wintypes.DWORD,
    )
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    succeeded = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not succeeded:
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "working_set_bytes": int(counters.working_set_size),
        "peak_working_set_bytes": int(counters.peak_working_set_size),
        "private_usage_bytes": int(counters.private_usage),
    }


def _windows_available_memory_bytes() -> int:
    if os.name != "nt":
        raise RuntimeError("Stage169 resource accounting requires Windows")
    status = _MemoryStatusEx()
    status.length = ctypes.sizeof(status)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GlobalMemoryStatusEx.argtypes = (ctypes.POINTER(_MemoryStatusEx),)
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(status.available_physical)


def _prior_failed_run_summary(
    *, stdout_path: Path, stderr_path: Path, exit_path: Path
) -> dict[str, Any]:
    for path in (stdout_path, stderr_path, exit_path):
        if not path.is_file():
            raise FileNotFoundError(f"Stage169 prior failure audit input is missing: {path}")
    progress = []
    for line in stdout_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("stage") == _STAGE:
            progress.append(event)
    stderr = stderr_path.read_text(encoding=locale.getpreferredencoding(False))
    synthetic_completed = max(
        (
            int(event["completed"])
            for event in progress
            if event.get("phase") == "synthetic_router_calls"
        ),
        default=0,
    )
    train_completed = max(
        (
            int(event["completed"])
            for event in progress
            if event.get("phase") == "train_router_calls"
        ),
        default=0,
    )
    return {
        "nature": "first_formal_run_failed_before_report_creation",
        "exit_code": int(exit_path.read_text(encoding="utf-8").strip()),
        "cuda_oom_confirmed": "CUDA error: out of memory" in stderr,
        "synthetic_cases_completed": synthetic_completed,
        "train_cases_completed": train_completed,
        "report_created": False,
        "automatic_retry_performed": False,
        "stdout": _fingerprint(stdout_path),
        "stderr": _fingerprint(stderr_path),
        "exit_file": _fingerprint(exit_path),
    }


def _nearest_rank(values: Sequence[int | float], quantile: float) -> int | float:
    return values[max(0, min(len(values) - 1, int((len(values) * quantile) + 0.999999) - 1))]


def _fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _authorize_sources(fingerprints: Mapping[str, Mapping[str, Any]]) -> None:
    for name, expected in _SOURCE_HASHES.items():
        observed = fingerprints.get(name, {}).get("sha256")
        if observed != expected:
            raise ValueError(f"Stage169 source hash mismatch for {name}: {observed}")


def _forbidden_keys_found(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in _FORBIDDEN_PUBLIC_KEYS:
                found.add(str(key))
            found.update(_forbidden_keys_found(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_forbidden_keys_found(child))
    return found


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _emit(sink: ProgressSink | None, **event: Any) -> None:
    if sink is not None:
        sink({"stage": _STAGE, **event})


def _rate_bar(label: str, value: float) -> BarDatum:
    return BarDatum(label=label, value=float(value), value_label=f"{value:.1%}")


def _gib_bar(label: str, value: int) -> BarDatum:
    gib = value / (1024**3)
    return BarDatum(label=label, value=gib, value_label=f"{gib:.3f} GiB")


def _chart(
    title: str,
    bars: Sequence[BarDatum],
    *,
    x_label: str = "Observed rate",
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label=x_label,
        width=1000,
        margin_left=300,
        margin_right=170,
    )
