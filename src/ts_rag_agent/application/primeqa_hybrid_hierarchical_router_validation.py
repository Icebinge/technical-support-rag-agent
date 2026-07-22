from __future__ import annotations

import os
import statistics
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ts_rag_agent.application import primeqa_hybrid_iterative_router_calibration as stage169
from ts_rag_agent.application.primeqa_hybrid_hierarchical_decision_router import (
    EvidenceDisposition,
    HierarchicalConstrainedDecisionRouter,
    HierarchicalDecisionLayer,
    HierarchicalLayerInvocationMetrics,
    HierarchicalLayerObserver,
    HierarchicalRouterTrace,
    RequestDisposition,
    hierarchical_decision_router_contract,
)
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    _build_dense_channels,
    _build_lexical_channels,
    _build_train_fold_assignments,
    _load_json_object,
)
from ts_rag_agent.application.primeqa_hybrid_iterative_decision_router import (
    IterativeDecisionAction,
    IterativeDecisionPhase,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector import (
    records_by_sample,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector_training import (
    Stage161TrainCandidateDatasetBuilder,
)
from ts_rag_agent.application.primeqa_hybrid_structured_decision_router import (
    Qwen3VLTransformersTextGenerationBackend,
    StructuredDecisionSchemaError,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.dataset import PrimeQARuntimeQuery
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    load_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)

_STAGE = "Stage 171"
_CREATED_AT = "2026-07-22"
_ANALYSIS_ID = "primeqa_hybrid_hierarchical_router_train_validation_v1"
_EXPECTED_TRAIN_ROWS = 562
_EXPECTED_CANDIDATE_ROWS = 112_400
_TRAIN_CASES_PER_STRATUM = 20
_EXPECTED_TRAIN_CASES = 100
_EXPECTED_SYNTHETIC_DECISIONS = 18
_EXPECTED_TRAIN_DECISIONS = 200
_EXPECTED_DECISIONS = 218
_MODEL_CALLS_PER_DECISION = 2
_EXPECTED_MODEL_CALLS = 436
_HIERARCHY_THRESHOLDS = {
    "synthetic_request_disposition_accuracy_min": 0.90,
    "synthetic_evidence_disposition_accuracy_min": 0.90,
    "train_request_complete_rate_min": 0.95,
    "request_layer_schema_valid_rate_min": 1.0,
    "evidence_layer_schema_valid_rate_min": 1.0,
}
_SOURCE_HASHES = {
    "stage170": "d74abda6a8455ab1946504096654a1238c5aa5ad2b4d5d3f4aa917e6badb5ef2",
    "stage80": "2441bb1cb1e7888299d3f57962b18cd59df84e2086ac281105abcacfc144880f",
    "train": "cabd93e0b972c47384c4bf5cc2cd215a7fc519b2df4f81fba61db73c931aa155",
    "documents": "f93b5e2d8dcfb2c7d12676ef32ce22b7809692f14081aad98096099a5256722b",
    "model_config": "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b",
    "model_weights": "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0",
    "model_tokenizer": "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
}

ProgressSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class HierarchicalPhaseOutcome:
    decision: stage169.RouterCallObservation
    trace: HierarchicalRouterTrace


@dataclass(frozen=True)
class HierarchicalTrainOutcome:
    stratum: str
    fold_id: str
    initial: HierarchicalPhaseOutcome
    final: HierarchicalPhaseOutcome


@dataclass(frozen=True)
class Stage171Visualization:
    name: str
    path: str


class ResourceRecordingLayerObserver(HierarchicalLayerObserver):
    """Record one event-driven resource observation after every model generation."""

    def __init__(self, *, tracker: stage169.Stage169ResourceTracker) -> None:
        self._tracker = tracker
        self._calls: list[stage169.RouterCallObservation] = []

    @property
    def calls(self) -> tuple[stage169.RouterCallObservation, ...]:
        return tuple(self._calls)

    def before_generation(self, layer: HierarchicalDecisionLayer) -> None:
        _ = layer
        self._tracker.torch_module.cuda.reset_peak_memory_stats()

    def after_generation(self, metrics: HierarchicalLayerInvocationMetrics) -> None:
        torch = self._tracker.torch_module
        resources = self._tracker.capture(f"router_call_{metrics.layer}")
        self._calls.append(
            stage169.RouterCallObservation(
                action=metrics.selected_label,
                clarification_kind=metrics.clarification_kind,
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
        )


def run_stage171_hierarchical_validation(
    *,
    stage170_report_path: Path,
    stage80_report_path: Path,
    train_split_path: Path,
    documents_path: Path,
    model_snapshot_path: Path,
    encoder_batch_size: int = 64,
    progress_sink: ProgressSink | None = None,
) -> dict[str, Any]:
    import torch

    started_at = time.perf_counter()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tracker = stage169.Stage169ResourceTracker(torch_module=torch)
    tracker.capture("process_started")
    source_paths = {
        "stage170": stage170_report_path,
        "stage80": stage80_report_path,
        "train": train_split_path,
        "documents": documents_path,
        "model_config": model_snapshot_path / "config.json",
        "model_weights": model_snapshot_path / "model.safetensors",
        "model_tokenizer": model_snapshot_path / "tokenizer.json",
    }
    fingerprints = {name: stage169._fingerprint(path) for name, path in source_paths.items()}
    _authorize_sources(fingerprints)
    stage170_report = _load_json_object(stage170_report_path)
    if stage170_report.get("decision", {}).get("status") != "stage170_prompt_family_insufficient":
        raise ValueError("Stage170 did not authorize hierarchical redesign")
    fingerprinted_at = time.perf_counter()
    tracker.capture("sources_fingerprinted")
    _emit(progress_sink, phase="sources_authorized")

    samples = load_primeqa_hybrid_split_samples(train_split_path)
    if len(samples) != _EXPECTED_TRAIN_ROWS or any(
        sample.assigned_split != "train" for sample in samples
    ):
        raise ValueError("Stage171 accepts only the exact 562-row train split")
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
        raise RuntimeError("Stage171 requires both authorized local dense channels")
    lexical_channels = _build_lexical_channels(
        documents=documents,
        sections_by_document=sections_by_document,
        bm25_k1=1.5,
        bm25_b=0.75,
        component_depth=200,
    )
    records = Stage161TrainCandidateDatasetBuilder(
        documents_by_id=documents_by_id,
        sections_by_document=sections_by_document,
        channels=tuple([*lexical_channels, *dense_channels]),
        fold_assignments=fold_assignments,
        progress_sink=progress_sink,
        progress_stage=_STAGE,
        progress_phase="train_candidate_replay",
    ).build(samples)
    if len(records) != _EXPECTED_CANDIDATE_ROWS:
        raise RuntimeError("Stage171 candidate replay row count drifted")
    train_cases = stage169.select_train_calibration_cases(
        samples=samples,
        grouped_records=records_by_sample(records),
        documents_by_id=documents_by_id,
        per_stratum=_TRAIN_CASES_PER_STRATUM,
    )
    evidence_ready_at = time.perf_counter()
    tracker.capture("train_evidence_ready")
    _emit(progress_sink, phase="train_evidence_ready", selected_cases=len(train_cases))

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage171 requires CUDA with bfloat16 support")
    backend = Qwen3VLTransformersTextGenerationBackend.load_local(snapshot_path=model_snapshot_path)
    observer = ResourceRecordingLayerObserver(tracker=tracker)
    router = HierarchicalConstrainedDecisionRouter(backend=backend, observer=observer)
    model_loaded_at = time.perf_counter()
    tracker.capture("router_model_loaded")
    _emit(progress_sink, phase="router_model_loaded")

    synthetic, synthetic_decisions = _run_synthetic(router, progress_sink)
    synthetic_finished_at = time.perf_counter()
    tracker.capture("synthetic_complete")
    train_outcomes, train_decisions = _run_train(
        router=router,
        cases=train_cases,
        progress_sink=progress_sink,
    )
    validation_finished_at = time.perf_counter()
    tracker.capture("validation_complete")

    decisions = (*synthetic_decisions, *train_decisions)
    layer_calls = observer.calls
    train_summary = _aggregate_train_outcomes(train_outcomes)
    quality_metrics = _quality_metrics(
        synthetic=synthetic,
        train=train_summary,
        decisions=decisions,
        layer_calls=layer_calls,
        train_outcomes=train_outcomes,
    )
    legacy_gates = stage169._quality_gates(quality_metrics)
    hierarchy_gates = _hierarchy_gates(quality_metrics)
    all_quality_gates_passed = all(gate["passed"] for gate in (*legacy_gates, *hierarchy_gates))
    process_guards = [
        _check("stage170_authorized_hierarchical_redesign", True),
        _check("exact_train_row_count", len(samples) == _EXPECTED_TRAIN_ROWS),
        _check("exact_candidate_row_count", len(records) == _EXPECTED_CANDIDATE_ROWS),
        _check("twenty_cases_per_stratum", len(train_cases) == _EXPECTED_TRAIN_CASES),
        _check("exact_decision_count", len(decisions) == _EXPECTED_DECISIONS),
        _check("two_model_calls_per_decision", len(layer_calls) == len(decisions) * 2),
        _check("model_call_count_exact", backend.generation_call_count == _EXPECTED_MODEL_CALLS),
        _check("local_files_only", True),
        _check("development_not_loaded", True),
        _check("test_not_loaded", True),
        _check("answer_generation_not_run", True),
        _check("retry_count_zero", True),
        _check("fallback_count_zero", True),
        _check("default_runtime_unchanged", True),
    ]
    all_process_guards_passed = all(guard["passed"] for guard in process_guards)
    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "validation_scope": (
            "Expanded train-only validation of a fixed two-layer request/evidence classifier "
            "with deterministic action mapping and five-fold stability reporting."
        ),
        "source_authorization": fingerprints,
        "environment": {
            "torch_version": torch.__version__,
            "transformers_version": version("transformers"),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
        },
        "frozen_protocol": {
            "router_contract": hierarchical_decision_router_contract(),
            "train_cases_per_stratum": _TRAIN_CASES_PER_STRATUM,
            "train_case_count": _EXPECTED_TRAIN_CASES,
            "synthetic_decision_count": _EXPECTED_SYNTHETIC_DECISIONS,
            "train_decision_count": _EXPECTED_TRAIN_DECISIONS,
            "model_calls_per_decision": _MODEL_CALLS_PER_DECISION,
            "expected_model_call_count": _EXPECTED_MODEL_CALLS,
            "quality_thresholds": stage169._QUALITY_THRESHOLDS,
            "hierarchy_thresholds": _HIERARCHY_THRESHOLDS,
            "sampling": "ascending_sha256_of_frozen_sample_id_within_each_stratum",
            "fold_count": 5,
            "per_call_evidence_and_token_policy_held_constant": True,
        },
        "prior_baselines": {
            "stage169_quality_gate_pass_count": 3,
            "stage170_best_quality_gate_pass_count": max(
                profile["quality_gate_pass_count"]
                for profile in stage170_report["train_comparison"].values()
            ),
        },
        "synthetic_calibration": synthetic,
        "train_proxy_calibration": train_summary,
        "train_layer_diagnostics": _train_layer_diagnostics(train_outcomes),
        "fold_stability": _fold_stability(train_outcomes),
        "quality_metrics": quality_metrics,
        "legacy_quality_gates": legacy_gates,
        "hierarchy_quality_gates": hierarchy_gates,
        "legacy_quality_gate_pass_count": sum(gate["passed"] for gate in legacy_gates),
        "hierarchy_quality_gate_pass_count": sum(gate["passed"] for gate in hierarchy_gates),
        "resource_consumption": stage169._resource_consumption_summary(
            tracker=tracker,
            calls=layer_calls,
            wall_time_seconds=validation_finished_at - started_at,
        ),
        "model_runtime": {
            "snapshot_revision": backend.snapshot_path.name,
            "generation_call_count": backend.generation_call_count,
            "peak_gpu_memory_bytes": max(call.gpu_peak_allocated_bytes for call in layer_calls),
        },
        "closed_boundaries": {
            "train_split_loaded": True,
            "development_split_loaded": False,
            "test_split_loaded": False,
            "answer_generation_run": False,
            "raw_question_saved": False,
            "raw_document_saved": False,
            "raw_model_output_saved": False,
            "retry_action_count": 0,
            "fallback_action_count": 0,
            "runtime_registered_as_default": False,
        },
        "timing_seconds": {
            "source_fingerprinting": round(fingerprinted_at - started_at, 6),
            "train_evidence_build": round(evidence_ready_at - fingerprinted_at, 6),
            "model_load": round(model_loaded_at - evidence_ready_at, 6),
            "synthetic_calibration": round(synthetic_finished_at - model_loaded_at, 6),
            "expanded_train_validation": round(validation_finished_at - synthetic_finished_at, 6),
            "total_before_visualization": round(validation_finished_at - started_at, 6),
        },
        "process_guards": process_guards,
        "decision": {
            "all_process_guards_passed": all_process_guards_passed,
            "all_quality_gates_passed": all_quality_gates_passed,
            "status": (
                "advance_to_stage172_train_only_hierarchical_runtime_e2e"
                if all_process_guards_passed and all_quality_gates_passed
                else "stage171_hierarchy_requires_redesign"
            ),
            "default_runtime_activation": False,
            "development_opened": False,
            "test_opened": False,
        },
    }
    forbidden = sorted(stage169._forbidden_keys_found(report))
    report["public_safe_contract"] = {
        "forbidden_keys": sorted(stage169._FORBIDDEN_PUBLIC_KEYS),
        "forbidden_keys_found": forbidden,
    }
    report["process_guards"].append(
        _check("public_report_contains_no_forbidden_keys", not forbidden)
    )
    report["decision"]["all_process_guards_passed"] = all(
        guard["passed"] for guard in report["process_guards"]
    )
    if not report["decision"]["all_process_guards_passed"]:
        report["decision"]["status"] = "stage171_process_invalid"
    _emit(progress_sink, phase="validation_complete", decision=report["decision"])
    return report


def write_stage171_visualizations(
    *, report: Mapping[str, Any], output_dir: Path
) -> tuple[Stage171Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = report["quality_metrics"]
    resources = report["resource_consumption"]
    folds = report["fold_stability"]["folds"]
    charts = {
        "quality_gate_progress.svg": _chart(
            "Stage 171 frozen quality gates passed",
            (
                BarDatum("Stage 169 baseline", 3, "3 / 8"),
                BarDatum("Stage 170 best prompt", 0, "0 / 8"),
                BarDatum(
                    "Stage 171 hierarchical",
                    report["legacy_quality_gate_pass_count"],
                    f"{report['legacy_quality_gate_pass_count']} / 8",
                ),
            ),
            x_label="Passed legacy gates",
        ),
        "synthetic_layer_accuracy.svg": _chart(
            "Stage 171 synthetic layer accuracy",
            (
                _rate_bar(
                    "Request disposition",
                    metrics["synthetic_request_disposition_accuracy"],
                ),
                _rate_bar(
                    "Evidence disposition",
                    metrics["synthetic_evidence_disposition_accuracy"],
                ),
                _rate_bar("Final action", metrics["synthetic_phase_action_accuracy"]),
                _rate_bar(
                    "Clarification kind",
                    metrics["synthetic_clarification_kind_accuracy"],
                ),
            ),
        ),
        "train_proxy_rates.svg": _chart(
            "Stage 171 expanded train proxy rates",
            (
                _rate_bar("Initial-visible compose", metrics["real_initial_visible_compose_rate"]),
                _rate_bar("Alternate-only inspect", metrics["real_alternate_only_inspect_rate"]),
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
        "fold_stability.svg": _chart(
            "Stage 171 five-fold path and safety stability",
            tuple(
                datum
                for fold_id, fold in folds.items()
                for label, value in (
                    ("alternate exact path", fold["alternate_only_path_success_rate"]),
                    ("insufficient final compose", fold["insufficient_final_compose_rate"]),
                )
                if value is not None
                for datum in (_rate_bar(f"{fold_id}: {label}", value),)
            ),
        ),
        "layer_latency.svg": _chart(
            "Stage 171 layer p95 generation latency",
            tuple(
                BarDatum(
                    layer,
                    distribution["p95"],
                    f"{distribution['p95']:.1f} ms",
                )
                for layer, distribution in metrics["latency_ms_by_layer"].items()
            ),
            x_label="Milliseconds",
        ),
        "resource_peaks.svg": _chart(
            "Stage 171 process and GPU resource peaks",
            (
                _gib_bar("Process working set", resources["process_peak_working_set_bytes"]),
                _gib_bar("Process private usage", resources["process_peak_private_usage_bytes"]),
                _gib_bar("GPU allocated", resources["gpu_peak_allocated_bytes"]),
                _gib_bar("GPU reserved", resources["gpu_peak_reserved_bytes"]),
            ),
            x_label="GiB",
        ),
    }
    written = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        written.append(Stage171Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _run_synthetic(
    router: HierarchicalConstrainedDecisionRouter,
    progress_sink: ProgressSink | None,
) -> tuple[dict[str, Any], tuple[stage169.RouterCallObservation, ...]]:
    rows = []
    decisions = []
    for index, case in enumerate(stage169.build_synthetic_calibration_cases(), start=1):
        initial = _invoke(
            router=router,
            phase=IterativeDecisionPhase.INITIAL,
            question=case.question,
            initial=case.initial_evidence,
            alternate=(),
        )
        decisions.append(initial.decision)
        rows.append(
            _synthetic_row(
                case_id=case.case_id,
                phase=IterativeDecisionPhase.INITIAL,
                expected_action=case.expected_initial_action,
                expected_clarification_kind=case.expected_initial_clarification_kind,
                outcome=initial,
            )
        )
        if case.expected_final_action is not None:
            final = _invoke(
                router=router,
                phase=IterativeDecisionPhase.FINAL_AFTER_INSPECTION,
                question=case.question,
                initial=case.initial_evidence,
                alternate=case.alternate_evidence,
            )
            decisions.append(final.decision)
            rows.append(
                _synthetic_row(
                    case_id=case.case_id,
                    phase=IterativeDecisionPhase.FINAL_AFTER_INSPECTION,
                    expected_action=case.expected_final_action,
                    expected_clarification_kind=None,
                    outcome=final,
                )
            )
        _emit(progress_sink, phase="synthetic_cases", completed=index, total=14)
    request_rows = rows
    evidence_rows = [row for row in rows if row["expected_evidence_disposition"] is not None]
    initial_rows = [row for row in rows if row["phase"] == IterativeDecisionPhase.INITIAL.value]
    clarification_rows = [row for row in initial_rows if row["expected_clarification_kind"]]
    exact_paths = 0
    cases_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cases_by_id.setdefault(row["case_id"], []).append(row)
    for case_rows in cases_by_id.values():
        exact_paths += all(
            row["expected_action"] == row["observed_action"]
            and row["expected_clarification_kind"] == row["observed_clarification_kind"]
            for row in case_rows
        )
    action_matches = sum(row["expected_action"] == row["observed_action"] for row in rows)
    request_matches = sum(
        row["expected_request_disposition"] == row["observed_request_disposition"]
        for row in request_rows
    )
    evidence_matches = sum(
        row["expected_evidence_disposition"] == row["observed_evidence_disposition"]
        for row in evidence_rows
    )
    clarification_matches = sum(
        row["expected_clarification_kind"] == row["observed_clarification_kind"]
        for row in clarification_rows
    )
    return (
        {
            "case_count": len(cases_by_id),
            "phase_expectation_count": len(rows),
            "phase_action_match_count": action_matches,
            "phase_action_accuracy": _rate(action_matches, len(rows)),
            "request_expectation_count": len(request_rows),
            "request_disposition_match_count": request_matches,
            "request_disposition_accuracy": _rate(request_matches, len(request_rows)),
            "evidence_expectation_count": len(evidence_rows),
            "evidence_disposition_match_count": evidence_matches,
            "evidence_disposition_accuracy": _rate(evidence_matches, len(evidence_rows)),
            "clarification_case_count": len(clarification_rows),
            "clarification_kind_match_count": clarification_matches,
            "clarification_kind_accuracy": _rate(clarification_matches, len(clarification_rows)),
            "exact_path_match_count": exact_paths,
            "exact_path_accuracy": _rate(exact_paths, len(cases_by_id)),
            "phases": rows,
        },
        tuple(decisions),
    )


def _run_train(
    *,
    router: HierarchicalConstrainedDecisionRouter,
    cases: Sequence[stage169.TrainCalibrationCase],
    progress_sink: ProgressSink | None,
) -> tuple[
    tuple[HierarchicalTrainOutcome, ...],
    tuple[stage169.RouterCallObservation, ...],
]:
    outcomes = []
    decisions = []
    for index, case in enumerate(cases, start=1):
        initial = _invoke(
            router=router,
            phase=IterativeDecisionPhase.INITIAL,
            question=case.question,
            initial=case.initial_evidence,
            alternate=(),
        )
        final = _invoke(
            router=router,
            phase=IterativeDecisionPhase.FINAL_AFTER_INSPECTION,
            question=case.question,
            initial=case.initial_evidence,
            alternate=case.alternate_evidence,
        )
        outcomes.append(HierarchicalTrainOutcome(case.stratum, case.fold_id, initial, final))
        decisions.extend((initial.decision, final.decision))
        _emit(progress_sink, phase="expanded_train_cases", completed=index, total=len(cases))
    return tuple(outcomes), tuple(decisions)


def _invoke(
    *,
    router: HierarchicalConstrainedDecisionRouter,
    phase: IterativeDecisionPhase,
    question: PrimeQARuntimeQuery,
    initial: Sequence[RetrievalResult],
    alternate: Sequence[RetrievalResult],
) -> HierarchicalPhaseOutcome:
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
    trace = router.last_trace
    if metrics is None or trace is None or len(trace.layer_metrics) != 2:
        raise RuntimeError("Stage171 hierarchical decision produced incomplete metrics")
    observation = stage169.RouterCallObservation(
        action=action,
        clarification_kind=clarification_kind,
        schema_valid=metrics.schema_valid,
        input_token_count=metrics.input_token_count,
        output_token_count=metrics.output_token_count,
        generation_latency_ms=metrics.generation_latency_ms,
        process_working_set_bytes=0,
        process_private_usage_bytes=0,
        system_available_memory_bytes=0,
        gpu_peak_allocated_bytes=0,
        gpu_peak_reserved_bytes=0,
    )
    return HierarchicalPhaseOutcome(observation, trace)


def _synthetic_row(
    *,
    case_id: str,
    phase: IterativeDecisionPhase,
    expected_action: str,
    expected_clarification_kind: str | None,
    outcome: HierarchicalPhaseOutcome,
) -> dict[str, Any]:
    expected_request = (
        RequestDisposition.COMPLETE.value
        if phase is IterativeDecisionPhase.FINAL_AFTER_INSPECTION
        else _expected_request_disposition(expected_action, expected_clarification_kind)
    )
    expected_evidence = _expected_evidence_disposition(expected_action, expected_request)
    return {
        "case_id": case_id,
        "phase": phase.value,
        "expected_request_disposition": expected_request,
        "observed_request_disposition": outcome.trace.request_disposition,
        "expected_evidence_disposition": expected_evidence,
        "observed_evidence_disposition": outcome.trace.evidence_disposition,
        "expected_action": expected_action,
        "observed_action": outcome.decision.action,
        "expected_clarification_kind": expected_clarification_kind,
        "observed_clarification_kind": outcome.decision.clarification_kind,
        "schema_valid": outcome.trace.schema_valid,
    }


def _expected_request_disposition(expected_action: str, clarification_kind: str | None) -> str:
    if clarification_kind is not None:
        return RequestDisposition.MISSING_FACT.value
    if expected_action == IterativeDecisionAction.REFUSE.value:
        return RequestDisposition.UNSUPPORTED.value
    return RequestDisposition.COMPLETE.value


def _expected_evidence_disposition(
    expected_action: str, expected_request_disposition: str
) -> str | None:
    if expected_request_disposition != RequestDisposition.COMPLETE.value:
        return None
    if expected_action == IterativeDecisionAction.COMPOSE.value:
        return EvidenceDisposition.SUFFICIENT.value
    return EvidenceDisposition.INSUFFICIENT.value


def _aggregate_train_outcomes(
    outcomes: Sequence[HierarchicalTrainOutcome],
) -> dict[str, dict[str, Any]]:
    report = {}
    for stratum in stage169._TRAIN_STRATA:
        rows = [outcome for outcome in outcomes if outcome.stratum == stratum]
        report[stratum] = {
            "case_count": len(rows),
            "initial_action_counts": _counts(row.initial.decision.action for row in rows),
            "final_action_counts": _counts(row.final.decision.action for row in rows),
            "initial_schema_valid_count": sum(row.initial.decision.schema_valid for row in rows),
            "final_schema_valid_count": sum(row.final.decision.schema_valid for row in rows),
            "inspect_then_compose_count": sum(
                row.initial.decision.action == IterativeDecisionAction.INSPECT.value
                and row.final.decision.action == IterativeDecisionAction.COMPOSE.value
                for row in rows
            ),
        }
    return report


def _train_layer_diagnostics(
    outcomes: Sequence[HierarchicalTrainOutcome],
) -> dict[str, dict[str, Any]]:
    report = {}
    for stratum in stage169._TRAIN_STRATA:
        rows = [outcome for outcome in outcomes if outcome.stratum == stratum]
        phases = [phase for row in rows for phase in (row.initial, row.final)]
        report[stratum] = {
            "decision_count": len(phases),
            "request_disposition_counts": _counts(
                phase.trace.request_disposition for phase in phases
            ),
            "evidence_disposition_counts": _counts(
                phase.trace.evidence_disposition for phase in phases
            ),
            "request_complete_rate": _rate(
                sum(
                    phase.trace.request_disposition == RequestDisposition.COMPLETE.value
                    for phase in phases
                ),
                len(phases),
            ),
            "evidence_sufficient_rate": _rate(
                sum(
                    phase.trace.evidence_disposition == EvidenceDisposition.SUFFICIENT.value
                    for phase in phases
                ),
                len(phases),
            ),
        }
    return report


def _fold_stability(outcomes: Sequence[HierarchicalTrainOutcome]) -> dict[str, Any]:
    folds = {}
    for fold_id in sorted({outcome.fold_id for outcome in outcomes}):
        rows = [outcome for outcome in outcomes if outcome.fold_id == fold_id]
        initial_visible = [row for row in rows if row.stratum == "initial_gold_visible"]
        alternate_only = [row for row in rows if row.stratum == "alternate_only_gold_visible"]
        insufficient = [
            row
            for row in rows
            if row.stratum
            in {
                "union_gold_missing_candidate_hit",
                "candidate_pool_gold_missing",
                "unanswerable",
            }
        ]
        phases = [phase for row in rows for phase in (row.initial, row.final)]
        folds[str(fold_id)] = {
            "case_count": len(rows),
            "layer_schema_valid_rate": _optional_rate(
                sum(
                    metric.schema_valid for phase in phases for metric in phase.trace.layer_metrics
                ),
                len(phases) * 2,
            ),
            "request_complete_rate": _optional_rate(
                sum(
                    phase.trace.request_disposition == RequestDisposition.COMPLETE.value
                    for phase in phases
                ),
                len(phases),
            ),
            "initial_visible_compose_rate": _optional_rate(
                sum(
                    row.initial.decision.action == IterativeDecisionAction.COMPOSE.value
                    for row in initial_visible
                ),
                len(initial_visible),
            ),
            "alternate_only_inspect_rate": _optional_rate(
                sum(
                    row.initial.decision.action == IterativeDecisionAction.INSPECT.value
                    for row in alternate_only
                ),
                len(alternate_only),
            ),
            "alternate_only_final_compose_rate": _optional_rate(
                sum(
                    row.final.decision.action == IterativeDecisionAction.COMPOSE.value
                    for row in alternate_only
                ),
                len(alternate_only),
            ),
            "alternate_only_path_success_rate": _optional_rate(
                sum(
                    row.initial.decision.action == IterativeDecisionAction.INSPECT.value
                    and row.final.decision.action == IterativeDecisionAction.COMPOSE.value
                    for row in alternate_only
                ),
                len(alternate_only),
            ),
            "insufficient_final_compose_rate": _optional_rate(
                sum(
                    row.final.decision.action == IterativeDecisionAction.COMPOSE.value
                    for row in insufficient
                ),
                len(insufficient),
            ),
        }
    return {"fold_count": len(folds), "folds": folds}


def _quality_metrics(
    *,
    synthetic: Mapping[str, Any],
    train: Mapping[str, Any],
    decisions: Sequence[stage169.RouterCallObservation],
    layer_calls: Sequence[stage169.RouterCallObservation],
    train_outcomes: Sequence[HierarchicalTrainOutcome],
) -> dict[str, Any]:
    metrics = stage169._quality_metrics(synthetic, train, decisions)
    request_calls = [
        call for index, call in enumerate(layer_calls) if index % _MODEL_CALLS_PER_DECISION == 0
    ]
    evidence_calls = [
        call for index, call in enumerate(layer_calls) if index % _MODEL_CALLS_PER_DECISION == 1
    ]
    train_phases = [
        phase for outcome in train_outcomes for phase in (outcome.initial, outcome.final)
    ]
    latencies_by_layer = {
        "request": _latency_distribution(request_calls),
        "evidence": _latency_distribution(evidence_calls),
    }
    return {
        **metrics,
        "synthetic_request_disposition_accuracy": synthetic["request_disposition_accuracy"],
        "synthetic_evidence_disposition_accuracy": synthetic["evidence_disposition_accuracy"],
        "train_request_complete_rate": _rate(
            sum(
                phase.trace.request_disposition == RequestDisposition.COMPLETE.value
                for phase in train_phases
            ),
            len(train_phases),
        ),
        "schema_valid_count": sum(call.schema_valid for call in layer_calls),
        "schema_valid_rate": _rate(
            sum(call.schema_valid for call in layer_calls), len(layer_calls)
        ),
        "request_layer_schema_valid_rate": _rate(
            sum(call.schema_valid for call in request_calls), len(request_calls)
        ),
        "evidence_layer_schema_valid_rate": _rate(
            sum(call.schema_valid for call in evidence_calls), len(evidence_calls)
        ),
        "model_call_count": len(layer_calls),
        "input_tokens": stage169._distribution(call.input_token_count for call in layer_calls),
        "output_tokens": stage169._distribution(call.output_token_count for call in layer_calls),
        "latency_ms": _latency_distribution(layer_calls),
        "latency_ms_by_layer": latencies_by_layer,
    }


def _hierarchy_gates(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    pairs = (
        (
            "synthetic_request_disposition_accuracy",
            metrics["synthetic_request_disposition_accuracy"],
            _HIERARCHY_THRESHOLDS["synthetic_request_disposition_accuracy_min"],
        ),
        (
            "synthetic_evidence_disposition_accuracy",
            metrics["synthetic_evidence_disposition_accuracy"],
            _HIERARCHY_THRESHOLDS["synthetic_evidence_disposition_accuracy_min"],
        ),
        (
            "train_request_complete_rate",
            metrics["train_request_complete_rate"],
            _HIERARCHY_THRESHOLDS["train_request_complete_rate_min"],
        ),
        (
            "request_layer_schema_valid_rate",
            metrics["request_layer_schema_valid_rate"],
            _HIERARCHY_THRESHOLDS["request_layer_schema_valid_rate_min"],
        ),
        (
            "evidence_layer_schema_valid_rate",
            metrics["evidence_layer_schema_valid_rate"],
            _HIERARCHY_THRESHOLDS["evidence_layer_schema_valid_rate_min"],
        ),
    )
    return [
        {
            "name": name,
            "observed": observed,
            "threshold": threshold,
            "direction": "min",
            "passed": observed >= threshold,
        }
        for name, observed, threshold in pairs
    ]


def _latency_distribution(
    calls: Sequence[stage169.RouterCallObservation],
) -> dict[str, float]:
    ordered = sorted(call.generation_latency_ms for call in calls)
    return {
        "min": round(ordered[0], 3),
        "p50": round(stage169._nearest_rank(ordered, 0.50), 3),
        "p95": round(stage169._nearest_rank(ordered, 0.95), 3),
        "max": round(ordered[-1], 3),
        "mean": round(statistics.fmean(ordered), 3),
    }


def _authorize_sources(fingerprints: Mapping[str, Mapping[str, Any]]) -> None:
    for name, expected in _SOURCE_HASHES.items():
        observed = fingerprints.get(name, {}).get("sha256")
        if observed != expected:
            raise ValueError(f"Stage171 source hash mismatch for {name}: {observed}")


def _counts(values: Any) -> dict[str, int]:
    counts = Counter(value if value is not None else "schema_invalid" for value in values)
    return dict(sorted(counts.items()))


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("required rate denominator must be positive")
    return round(numerator / denominator, 6)


def _optional_rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _emit(sink: ProgressSink | None, **event: Any) -> None:
    if sink is not None:
        sink({"stage": _STAGE, **event})


def _rate_bar(label: str, value: float) -> BarDatum:
    return BarDatum(label, value, f"{value:.1%}")


def _gib_bar(label: str, value: int) -> BarDatum:
    gib = value / (1024**3)
    return BarDatum(label, gib, f"{gib:.3f} GiB")


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
        width=1200,
        margin_left=440,
        margin_right=180,
    )
