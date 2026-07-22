from __future__ import annotations

import gc
import hashlib
import json
import os
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ts_rag_agent.application import (
    primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_protocol as stage160,
)
from ts_rag_agent.application import primeqa_hybrid_grouped_ranking_cv as stage175
from ts_rag_agent.application import primeqa_hybrid_iterative_router_calibration as stage169
from ts_rag_agent.application import primeqa_hybrid_listwise_reranker_cv as stage177
from ts_rag_agent.application import primeqa_hybrid_semantic_evidence_cv as stage173
from ts_rag_agent.application import primeqa_hybrid_supervised_cross_encoder_cv as stage174
from ts_rag_agent.application.listwise_runtime_reranker import (
    ListwiseUnionPrimaryContextSelectionPolicy,
    LocalListwiseCheckpointScoreProvider,
    PrecomputedListwiseScoreProvider,
    checkpoint_public_summary,
    listwise_runtime_policy_contract,
    write_listwise_checkpoint_manifest,
)
from ts_rag_agent.application.primeqa_hybrid_agent_runtime_observability import (
    AgentWorkflowObservationSink,
    PublicSafeAgentWorkflowObservationEvent,
)
from ts_rag_agent.application.primeqa_hybrid_agent_tool_workflow import (
    FrozenAnswerGeneratorFactory,
    FrozenAnswerVerifierFactory,
    PrimeQAHybridAgentToolset,
    create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset,
)
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    _build_dense_channels,
    _build_lexical_channels,
    _build_train_fold_assignments,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    PrimeQAHybridProcessRuntimeResourceFactory,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector import (
    ContextCandidateRecord,
    records_by_sample,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector_training import (
    Stage161TrainCandidateDatasetBuilder,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_observation_validation import (
    PrimeQAHybridSidecarObservationAdapter,
)
from ts_rag_agent.application.rag_answering import evaluate_answers
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.answer import GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQAQuestion, PrimeQARuntimeQuery
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)

_STAGE = "Stage 178A"
_CREATED_AT = "2026-07-22"
_ANALYSIS_ID = "primeqa_hybrid_listwise_tool_agent_grouped_oof_e2e_v1"
_EXPECTED_TRAIN_ROWS = 562
_EXPECTED_ANSWERABLE_ROWS = 370
_EXPECTED_CANDIDATE_ROWS = 112_400
_EXPECTED_PAIR_ROWS = 9_714
_EXPECTED_OOF_FITS = 5
_EXPECTED_FULL_FITS = 1
_BOOTSTRAP_REPLICATES = 2_000
_BOOTSTRAP_SEED = 178_000
_MINIMUM_STABLE_FOLDS = 4
_CPU_LATENCY_PROBE_COUNT = 50
_CPU_RERANK_P95_LIMIT_SECONDS = 1.0
_SOURCE_HASHES = {
    "stage178_alignment": "e2398024edf128ad0628900d25eb1ccc9c83c437fb474921fe136e2603e47272",
    "stage177": "6e028ed9e90fe153fda39f3073861c5d0b8eb019675635edb21a6825a472be50",
    "stage128": "012ca36c0559f3533ea2e89160fcb3cee7fb12daa89fb68c69dcf27d9d2ce63e",
    "stage125": "dfaee9eb5688a2a91e3f3d3695def5e32ad87494cdfdd31a00a0434df53ccd65",
    "stage80": "2441bb1cb1e7888299d3f57962b18cd59df84e2086ac281105abcacfc144880f",
    "train": "cabd93e0b972c47384c4bf5cc2cd215a7fc519b2df4f81fba61db73c931aa155",
    "documents": "f93b5e2d8dcfb2c7d12676ef32ce22b7809692f14081aad98096099a5256722b",
    **{key: value for key, value in stage173._SOURCE_HASHES.items() if key.startswith("model_")},
}
_FORBIDDEN_PUBLIC_KEYS = stage177._FORBIDDEN_PUBLIC_KEYS | {
    "answer_doc_id",
    "candidate_doc_ids",
    "document_id",
    "gold_answer",
    "question_text",
    "sample_id",
}

ProgressSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class Stage178AVisualization:
    name: str
    path: str


@dataclass(frozen=True)
class AgentE2ETrace:
    sample: PrimeQAHybridSplitSample
    fold_id: str
    question: PrimeQAQuestion
    candidate_pool_results: tuple[RetrievalResult, ...]
    generation_context_results: tuple[RetrievalResult, ...]
    verified_answer: GeneratedAnswer
    latency_seconds: float
    tool_call_count: int
    retry_action_count: int
    fallback_action_count: int

    @property
    def answer_f1(self) -> float:
        if not self.sample.answerable:
            return 0.0
        return stage160.score_answer(
            self.verified_answer.answer,
            self.sample.answer,
            refused=self.verified_answer.refused,
        )

    @property
    def context_gold_hit(self) -> bool:
        return bool(
            self.sample.answerable
            and self.sample.answer_doc_id is not None
            and any(
                result.document.id == self.sample.answer_doc_id
                for result in self.generation_context_results
            )
        )

    @property
    def gold_cited(self) -> bool:
        return bool(
            self.sample.answerable
            and self.sample.answer_doc_id is not None
            and any(
                citation.document_id == self.sample.answer_doc_id
                for citation in self.verified_answer.citations
            )
        )


class ValidatingMemoryObservationSink(AgentWorkflowObservationSink):
    def __init__(self) -> None:
        self.event_count = 0

    def emit(self, event: PublicSafeAgentWorkflowObservationEvent) -> None:
        event.to_public_dict()
        self.event_count += 1


def run_stage178a_listwise_agent_e2e(
    *,
    stage178_alignment_path: Path,
    stage177_report_path: Path,
    stage128_protocol_path: Path,
    stage125_protocol_path: Path,
    stage80_report_path: Path,
    train_split_path: Path,
    documents_path: Path,
    model_snapshot_path: Path,
    checkpoint_path: Path,
    encoder_batch_size: int = 64,
    progress_sink: ProgressSink | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started_at = time.perf_counter()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    source_paths = {
        "stage178_alignment": stage178_alignment_path,
        "stage177": stage177_report_path,
        "stage128": stage128_protocol_path,
        "stage125": stage125_protocol_path,
        "stage80": stage80_report_path,
        "train": train_split_path,
        "documents": documents_path,
        **{
            source_name: model_snapshot_path / filename
            for source_name, filename in stage173._MODEL_SOURCE_FILES.items()
        },
    }
    fingerprints = {
        name: stage173._resolved_fingerprint(path) for name, path in source_paths.items()
    }
    _authorize_sources(fingerprints)
    alignment_report = json.loads(stage178_alignment_path.read_text(encoding="utf-8"))
    alignment_decision = alignment_report.get("decision") or {}
    if not all(
        alignment_decision.get(key) is True
        for key in (
            "full_prefix_contract_exact",
            "selection_surface_exact",
            "live_union_fully_covered_by_stage177_pairs",
        )
    ):
        raise ValueError("Stage178 candidate alignment audit did not authorize Agent E2E")
    stage177_report = json.loads(stage177_report_path.read_text(encoding="utf-8"))
    if stage177_report.get("decision", {}).get("status") != (
        "advance_to_stage178_listwise_reranker_agent_e2e"
    ):
        raise ValueError("Stage177 did not authorize Stage178 Agent E2E")
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="sources_authorized")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Stage178A formal training requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    tracker = stage169.Stage169ResourceTracker(torch_module=torch)
    tracker.capture("analysis_start")
    samples = load_primeqa_hybrid_split_samples(train_split_path)
    if len(samples) != _EXPECTED_TRAIN_ROWS or any(
        sample.assigned_split != "train" for sample in samples
    ):
        raise ValueError("Stage178A accepts only the exact train split")
    if sum(sample.answerable for sample in samples) != _EXPECTED_ANSWERABLE_ROWS:
        raise ValueError("Stage178A answerable row count drifted")
    documents_by_id = load_primeqa_documents(documents_path)
    sections_by_document = load_primeqa_document_sections(documents_path)
    documents = list(documents_by_id.values())
    fold_assignments = _build_train_fold_assignments(samples, fold_count=5)
    dense_channels, dense_summary = _build_dense_channels(
        include_dense_channels=True,
        stage80_report=json.loads(stage80_report_path.read_text(encoding="utf-8")),
        stage80_report_path=stage80_report_path,
        documents=documents,
        document_ids=tuple(document.id for document in documents),
        encoder_batch_size=encoder_batch_size,
        encoder_device="cpu",
        encoder_factory=None,
    )
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
        raise RuntimeError("Stage178A candidate replay row count drifted")
    grouped_records = records_by_sample(records)
    replay_ready_at = time.perf_counter()
    tracker.capture("candidate_replay_ready")

    frozen_scorer = stage173.LocalCrossEncoderSemanticScorer(
        snapshot_path=model_snapshot_path,
        tracker=tracker,
    )
    base_cases, pairs, frozen_scores, frozen_summary = stage173.build_semantic_evidence_cases(
        samples=samples,
        grouped_records=grouped_records,
        documents_by_id=documents_by_id,
        scorer=frozen_scorer,
        text_policy=stage173.QueryAwareCrossEncoderTextPolicy(),
        progress_sink=progress_sink,
    )
    del frozen_scorer
    gc.collect()
    torch.cuda.empty_cache()
    pair_rows = stage174.build_pair_fold_rows(
        pairs=pairs,
        base_cases=base_cases,
        frozen_scores=frozen_scores,
    )
    if len(pair_rows) != _EXPECTED_PAIR_ROWS:
        raise RuntimeError("Stage178A complete pair count drifted")
    pair_data_ready_at = time.perf_counter()
    tracker.capture("pair_data_ready")

    trainer = stage175.LocalGroupedRankingTrainer(
        snapshot_path=model_snapshot_path,
        tracker=tracker,
        torch_module=torch,
    )
    oof = stage177.run_grouped_oof_listwise_reranking(
        samples=samples,
        grouped_records=grouped_records,
        pair_rows=pair_rows,
        trainer=trainer,
        progress_sink=progress_sink,
    )
    oof_ready_at = time.perf_counter()
    full_predictions, full_summary = trainer.fit_predict(
        family="listwise_none",
        training_rows=pair_rows,
        evaluation_rows=(),
        fit_id="full_train_runtime_checkpoint",
        training_fold_count=5,
        evaluation_fold_count=0,
        progress_sink=progress_sink,
        checkpoint_path=checkpoint_path,
    )
    if full_predictions:
        raise RuntimeError("Stage178A full-train checkpoint fit emitted evaluation scores")
    checkpoint_manifest = write_listwise_checkpoint_manifest(
        checkpoint_path=checkpoint_path,
        stage177_report_sha256=_SOURCE_HASHES["stage177"],
        train_source_sha256=_SOURCE_HASHES["train"],
        training_row_count=len(samples),
        training_pair_count=full_summary.training_pair_count,
        optimizer_step_count=full_summary.optimizer_step_count,
        first_epoch_mean_loss=full_summary.first_epoch_mean_loss,
        final_epoch_mean_loss=full_summary.final_epoch_mean_loss,
    )
    checkpoint_summary = checkpoint_public_summary(checkpoint_path)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    checkpoint_ready_at = time.perf_counter()
    tracker.capture("checkpoint_ready")

    resource_factory = PrimeQAHybridProcessRuntimeResourceFactory(
        stage128_protocol_path=stage128_protocol_path,
        stage125_protocol_path=stage125_protocol_path,
        stage80_report_path=stage80_report_path,
        documents_path=documents_path,
        encoder_batch_size=encoder_batch_size,
        encoder_device="cpu",
    )
    shared_resources = resource_factory.build_shared()
    resources_ready_at = time.perf_counter()
    tracker.capture("runtime_resources_ready")
    score_provider = PrecomputedListwiseScoreProvider(oof["pair_logits"])
    baseline_sink = ValidatingMemoryObservationSink()
    candidate_sink = ValidatingMemoryObservationSink()
    baseline_workflow = _workflow(
        candidate_pool_retriever=shared_resources.candidate_pool_retriever,
        observation_adapter=PrimeQAHybridSidecarObservationAdapter(),
        observation_sink=baseline_sink,
    )
    candidate_workflow = _workflow(
        candidate_pool_retriever=shared_resources.candidate_pool_retriever,
        observation_adapter=PrimeQAHybridSidecarObservationAdapter(
            primary_context_selection_policy=ListwiseUnionPrimaryContextSelectionPolicy(
                score_provider=score_provider
            )
        ),
        observation_sink=candidate_sink,
    )
    traces = _run_paired_tool_agent(
        samples=samples,
        grouped_records=grouped_records,
        baseline_workflow=baseline_workflow,
        candidate_workflow=candidate_workflow,
        progress_sink=progress_sink,
    )
    agent_ready_at = time.perf_counter()
    tracker.capture("paired_tool_agent_complete")
    evaluation = evaluate_agent_e2e(traces)

    cpu_provider = LocalListwiseCheckpointScoreProvider(
        checkpoint_path=checkpoint_path,
        device="cpu",
    )
    latency_probe = _run_cpu_latency_probe(
        traces=traces["baseline"],
        score_provider=cpu_provider,
    )
    runtime_smoke_sink = ValidatingMemoryObservationSink()
    runtime_smoke_workflow = _workflow(
        candidate_pool_retriever=shared_resources.candidate_pool_retriever,
        observation_adapter=PrimeQAHybridSidecarObservationAdapter(
            primary_context_selection_policy=ListwiseUnionPrimaryContextSelectionPolicy(
                score_provider=cpu_provider
            )
        ),
        observation_sink=runtime_smoke_sink,
    )
    smoke_started_at = time.perf_counter()
    smoke_run = runtime_smoke_workflow.run(
        PrimeQARuntimeQuery(
            id="stage178a-label-free-runtime-smoke",
            title="Service installation verification",
            text="How can I verify a service configuration after installation?",
        )
    )
    runtime_smoke = {
        "completed": True,
        "latency_seconds": round(time.perf_counter() - smoke_started_at, 6),
        "candidate_pool_depth": len(smoke_run.candidate_pool_results),
        "generation_context_depth": len(smoke_run.generation_context_results),
        "verified_refused": smoke_run.verified_answer.refused,
        "tool_call_count": smoke_run.public_safe_trace.tool_call_count,
        "retry_action_count": smoke_run.public_safe_trace.retry_action_count,
        "fallback_action_count": smoke_run.public_safe_trace.fallback_action_count,
        "observation_event_count": runtime_smoke_sink.event_count,
    }
    evaluated_at = time.perf_counter()
    tracker.capture("cpu_checkpoint_runtime_validated")

    gates = _quality_gates(evaluation=evaluation, latency_probe=latency_probe)
    all_gates_passed = all(gate["passed"] for gate in gates)
    observed_retry_actions = sum(
        row.retry_action_count for rows in traces.values() for row in rows
    ) + int(runtime_smoke["retry_action_count"])
    observed_fallback_actions = sum(
        row.fallback_action_count for rows in traces.values() for row in rows
    ) + int(runtime_smoke["fallback_action_count"])
    private_report = {
        "format_id": "stage178a_oof_pair_logits_v1",
        "pair_count": len(oof["pair_logits"]),
        "scores": dict(sorted(oof["pair_logits"].items())),
    }
    private_sha256 = _canonical_sha256(private_report)
    snapshots = tracker.snapshots
    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train-only five-fold OOF comparison of the frozen query-overlap Agent "
            "context against the Stage177 listwise union reranker in the real LangGraph "
            "retrieval-compose-verify workflow. A sixth full-train fit writes an "
            "authenticated optional runtime checkpoint but is not used for quality metrics."
        ),
        "source_authorization": fingerprints,
        "frozen_protocol": {
            "user_selected_route": "C_stage178a_then_stage178b_if_all_gates_pass",
            "quality_score_source": "five_fold_grouped_oof_only",
            "oof_fit_count": _EXPECTED_OOF_FITS,
            "full_checkpoint_fit_count": _EXPECTED_FULL_FITS,
            "bootstrap_replicates": _BOOTSTRAP_REPLICATES,
            "minimum_stable_folds": _MINIMUM_STABLE_FOLDS,
            "cpu_latency_probe_count": _CPU_LATENCY_PROBE_COUNT,
            "cpu_rerank_p95_limit_seconds": _CPU_RERANK_P95_LIMIT_SECONDS,
            "runtime_policy": listwise_runtime_policy_contract(),
            "development_and_test_closed": True,
            "corrected_runtime_alignment_required": True,
        },
        "training": {
            "complete_pair_count": len(pair_rows),
            "oof_fit_count": len(oof["fit_summaries"]),
            "full_checkpoint_fit": asdict(full_summary),
            "checkpoint": checkpoint_summary,
            "checkpoint_manifest_file_count_before_manifest": len(checkpoint_manifest["files"]),
            "quality_metrics_use_full_checkpoint": False,
        },
        "agent_e2e": evaluation,
        "precomputed_oof_provider": asdict(score_provider.counters()),
        "cpu_checkpoint_latency_probe": latency_probe,
        "cpu_checkpoint_runtime_smoke": runtime_smoke,
        "workflow": {
            "baseline_topology": baseline_workflow.topology(),
            "candidate_topology": candidate_workflow.topology(),
            "baseline_counters": asdict(baseline_workflow.counters()),
            "candidate_counters": asdict(candidate_workflow.counters()),
            "baseline_observation_events": baseline_sink.event_count,
            "candidate_observation_events": candidate_sink.event_count,
        },
        "private_oof_artifact": {
            "format_id": private_report["format_id"],
            "pair_count": private_report["pair_count"],
            "canonical_sha256": private_sha256,
            "contains_raw_question": False,
            "contains_raw_document_text": False,
            "git_policy": "ignored_local_artifact",
        },
        "resource_consumption": {
            "wall_time_seconds": round(evaluated_at - started_at, 6),
            "process_peak_working_set_bytes": max(
                snapshot.process_peak_working_set_bytes for snapshot in snapshots
            ),
            "process_peak_private_usage_bytes": max(
                snapshot.process_private_usage_bytes for snapshot in snapshots
            ),
            "minimum_system_available_memory_bytes": min(
                snapshot.system_available_memory_bytes for snapshot in snapshots
            ),
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "gpu_device": torch.cuda.get_device_name(0),
            "resource_factory_build_count": resource_factory.build_count,
            "frozen_semantic_pair_count": frozen_summary.pair_count,
        },
        "timing_seconds": {
            "source_authorization": round(authorized_at - started_at, 6),
            "candidate_replay": round(replay_ready_at - authorized_at, 6),
            "pair_build_and_frozen_score": round(pair_data_ready_at - replay_ready_at, 6),
            "five_fold_oof_training": round(oof_ready_at - pair_data_ready_at, 6),
            "full_checkpoint_training": round(checkpoint_ready_at - oof_ready_at, 6),
            "runtime_resource_build": round(resources_ready_at - checkpoint_ready_at, 6),
            "paired_tool_agent_e2e": round(agent_ready_at - resources_ready_at, 6),
            "cpu_checkpoint_validation": round(evaluated_at - agent_ready_at, 6),
        },
        "quality_gates": gates,
        "execution_boundaries": {
            "train_loaded": True,
            "development_loaded": False,
            "test_loaded": False,
            "gold_projected_into_runtime": False,
            "agent_turns_run": sum(len(rows) for rows in traces.values()) + 1,
            "answer_quality_uses_oof_only": True,
            "sufficiency_gate_run": False,
            "fallback_action_count": observed_fallback_actions,
            "retry_action_count": observed_retry_actions,
            "runtime_registered_as_default": False,
        },
        "decision": {
            "candidate_selected": all_gates_passed,
            "status": (
                "advance_to_stage178b_sharded_qwen_agent_e2e"
                if all_gates_passed
                else "stage178a_listwise_tool_agent_e2e_insufficient"
            ),
            "stage178b_authorized": all_gates_passed,
            "development_opened": False,
            "test_opened": False,
            "default_runtime_activation": False,
        },
    }
    forbidden = sorted(stage177._forbidden_keys_found(report) | _forbidden_keys_found(report))
    report["public_safe_contract"] = {
        "forbidden_keys": sorted(_FORBIDDEN_PUBLIC_KEYS),
        "forbidden_keys_found": forbidden,
    }
    process_guards = _process_guards(
        report=report,
        records=records,
        pair_rows=pair_rows,
        oof=oof,
        traces=traces,
        dense_summary=dense_summary,
        forbidden=forbidden,
    )
    report["process_guards"] = process_guards
    if not all(guard["passed"] for guard in process_guards):
        report["decision"]["candidate_selected"] = False
        report["decision"]["stage178b_authorized"] = False
        report["decision"]["status"] = "stage178a_process_invalid"
    _emit(progress_sink, phase="analysis_complete", decision=report["decision"])
    return report, private_report


def _workflow(
    *,
    candidate_pool_retriever: Any,
    observation_adapter: PrimeQAHybridSidecarObservationAdapter,
    observation_sink: AgentWorkflowObservationSink,
) -> Any:
    toolset = PrimeQAHybridAgentToolset(
        candidate_pool_retriever=candidate_pool_retriever,
        observation_adapter=observation_adapter,
        answer_generator_factory=FrozenAnswerGeneratorFactory(),
        answer_verifier_factory=FrozenAnswerVerifierFactory(),
    )
    return create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset(
        toolset=toolset,
        observation_sink=observation_sink,
    )


def _run_paired_tool_agent(
    *,
    samples: Sequence[PrimeQAHybridSplitSample],
    grouped_records: Mapping[str, Sequence[ContextCandidateRecord]],
    baseline_workflow: Any,
    candidate_workflow: Any,
    progress_sink: ProgressSink | None,
) -> dict[str, tuple[AgentE2ETrace, ...]]:
    traces: dict[str, list[AgentE2ETrace]] = {"baseline": [], "candidate": []}
    workflows = {"baseline": baseline_workflow, "candidate": candidate_workflow}
    for index, sample in enumerate(samples, start=1):
        order = (
            ("baseline", "candidate")
            if int(hashlib.sha256(sample.sample_id.encode()).hexdigest(), 16) % 2 == 0
            else ("candidate", "baseline")
        )
        question = sample.to_primeqa_question()
        runtime_query = PrimeQARuntimeQuery(
            id=sample.sample_id,
            title=sample.question_title,
            text=sample.question_text,
        )
        expected_ids = tuple(
            record.document_id
            for record in sorted(
                grouped_records[sample.sample_id],
                key=lambda record: record.baseline_rank,
            )
        )
        fold_id = tuple(grouped_records[sample.sample_id])[0].fold_id
        for arm in order:
            started_at = time.perf_counter()
            run = workflows[arm].run(runtime_query)
            latency = time.perf_counter() - started_at
            actual_ids = tuple(
                result.document.id for result in run.candidate_pool_results[: len(expected_ids)]
            )
            if actual_ids != expected_ids:
                raise RuntimeError("Stage178A live candidate prefix drifted from OOF replay")
            trace = run.public_safe_trace
            traces[arm].append(
                AgentE2ETrace(
                    sample=sample,
                    fold_id=fold_id,
                    question=question,
                    candidate_pool_results=tuple(run.candidate_pool_results),
                    generation_context_results=tuple(run.generation_context_results),
                    verified_answer=run.verified_answer,
                    latency_seconds=latency,
                    tool_call_count=trace.tool_call_count,
                    retry_action_count=trace.retry_action_count,
                    fallback_action_count=trace.fallback_action_count,
                )
            )
        if index % 25 == 0 or index == len(samples):
            _emit(
                progress_sink,
                phase="paired_tool_agent_progress",
                completed=index,
                total=len(samples),
            )
    return {name: tuple(rows) for name, rows in traces.items()}


def evaluate_agent_e2e(
    traces: Mapping[str, Sequence[AgentE2ETrace]],
) -> dict[str, Any]:
    baseline = tuple(traces["baseline"])
    candidate = tuple(traces["candidate"])
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Stage178A paired traces must be complete")
    _assert_pair_alignment(baseline, candidate)
    profiles = {
        "baseline": _profile_metrics(baseline),
        "candidate": _profile_metrics(candidate),
    }
    folds = {}
    for fold_id in sorted({row.fold_id for row in baseline}):
        fold_baseline = tuple(row for row in baseline if row.fold_id == fold_id)
        fold_candidate = tuple(row for row in candidate if row.fold_id == fold_id)
        folds[fold_id] = {
            "baseline": _profile_metrics(fold_baseline),
            "candidate": _profile_metrics(fold_candidate),
        }
    bootstrap = _paired_bootstrap(baseline=baseline, candidate=candidate)
    return {
        "paired_question_count": len(baseline),
        "profiles": profiles,
        "deltas": _profile_deltas(
            baseline=profiles["baseline"],
            candidate=profiles["candidate"],
        ),
        "folds": folds,
        "paired_bootstrap": bootstrap,
        "changed_verified_answer_count": sum(
            left.verified_answer != right.verified_answer
            for left, right in zip(baseline, candidate, strict=True)
        ),
    }


def _profile_metrics(rows: Sequence[AgentE2ETrace]) -> dict[str, Any]:
    questions = [row.question for row in rows]
    answers = [row.verified_answer for row in rows]
    metrics = asdict(evaluate_answers(questions, answers))
    answerable = [row for row in rows if row.sample.answerable]
    unanswerable = [row for row in rows if not row.sample.answerable]
    return {
        "verified_metrics": metrics,
        "context_gold_hit_count": sum(row.context_gold_hit for row in answerable),
        "context_gold_hit_rate": round(
            sum(row.context_gold_hit for row in answerable) / len(answerable), 6
        ),
        "gold_citation_count": sum(row.gold_cited for row in answerable),
        "answerable_refusal_count": sum(row.verified_answer.refused for row in answerable),
        "unanswerable_false_answer_count": sum(
            not row.verified_answer.refused for row in unanswerable
        ),
        "latency_seconds": _distribution([row.latency_seconds for row in rows]),
        "tool_call_count": sum(row.tool_call_count for row in rows),
        "retry_action_count": sum(row.retry_action_count for row in rows),
        "fallback_action_count": sum(row.fallback_action_count for row in rows),
    }


def _profile_deltas(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "verified_average_token_f1": round(
            candidate["verified_metrics"]["average_token_f1"]
            - baseline["verified_metrics"]["average_token_f1"],
            6,
        ),
        "gold_citation_count": candidate["gold_citation_count"] - baseline["gold_citation_count"],
        "context_gold_hit_count": candidate["context_gold_hit_count"]
        - baseline["context_gold_hit_count"],
        "answerable_refusal_count": candidate["answerable_refusal_count"]
        - baseline["answerable_refusal_count"],
        "unanswerable_false_answer_count": candidate["unanswerable_false_answer_count"]
        - baseline["unanswerable_false_answer_count"],
        "latency_p95_seconds": round(
            candidate["latency_seconds"]["p95"] - baseline["latency_seconds"]["p95"],
            6,
        ),
    }


def _paired_bootstrap(
    *,
    baseline: Sequence[AgentE2ETrace],
    candidate: Sequence[AgentE2ETrace],
) -> dict[str, Any]:
    answerable_indices = [index for index, row in enumerate(baseline) if row.sample.answerable]
    metric_pairs = {
        "answer_f1": (
            np.asarray([candidate[index].answer_f1 for index in answerable_indices]),
            np.asarray([baseline[index].answer_f1 for index in answerable_indices]),
        ),
        "gold_citation": (
            np.asarray([candidate[index].gold_cited for index in answerable_indices], dtype=float),
            np.asarray([baseline[index].gold_cited for index in answerable_indices], dtype=float),
        ),
        "context_gold_hit": (
            np.asarray(
                [candidate[index].context_gold_hit for index in answerable_indices], dtype=float
            ),
            np.asarray(
                [baseline[index].context_gold_hit for index in answerable_indices], dtype=float
            ),
        ),
    }
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    sample_indices = rng.integers(
        0,
        len(answerable_indices),
        size=(_BOOTSTRAP_REPLICATES, len(answerable_indices)),
    )
    summaries = {}
    for name, (candidate_values, baseline_values) in metric_pairs.items():
        paired = candidate_values - baseline_values
        sampled = paired[sample_indices].mean(axis=1)
        summaries[name] = {
            "observed_delta": round(float(paired.mean()), 6),
            "ci95_lower": round(float(np.quantile(sampled, 0.025)), 6),
            "ci95_upper": round(float(np.quantile(sampled, 0.975)), 6),
        }
    return {
        "replicates": _BOOTSTRAP_REPLICATES,
        "seed": _BOOTSTRAP_SEED,
        "metrics": summaries,
    }


def _quality_gates(
    *,
    evaluation: Mapping[str, Any],
    latency_probe: Mapping[str, Any],
) -> list[dict[str, Any]]:
    bootstrap = evaluation["paired_bootstrap"]["metrics"]
    folds = evaluation["folds"]
    f1_nonregression = sum(
        values["candidate"]["verified_metrics"]["average_token_f1"]
        >= values["baseline"]["verified_metrics"]["average_token_f1"]
        for values in folds.values()
    )
    citation_nonregression = sum(
        values["candidate"]["gold_citation_count"] >= values["baseline"]["gold_citation_count"]
        for values in folds.values()
    )
    deltas = evaluation["deltas"]
    return [
        _gate("f1_delta_positive", deltas["verified_average_token_f1"] > 0),
        _gate("f1_bootstrap_ci_lower_nonnegative", bootstrap["answer_f1"]["ci95_lower"] >= 0),
        _gate("gold_citation_count_nonregression", deltas["gold_citation_count"] >= 0),
        _gate(
            "gold_citation_bootstrap_ci_lower_nonnegative",
            bootstrap["gold_citation"]["ci95_lower"] >= 0,
        ),
        _gate("context_gold_hit_strict_gain", deltas["context_gold_hit_count"] > 0),
        _gate(
            "context_hit_bootstrap_ci_lower_positive",
            bootstrap["context_gold_hit"]["ci95_lower"] > 0,
        ),
        _gate("answerable_refusal_nonincrease", deltas["answerable_refusal_count"] <= 0),
        _gate(
            "unanswerable_false_answer_nonincrease",
            deltas["unanswerable_false_answer_count"] <= 0,
        ),
        _gate("f1_fold_nonregression_4_of_5", f1_nonregression >= _MINIMUM_STABLE_FOLDS),
        _gate(
            "citation_fold_nonregression_4_of_5",
            citation_nonregression >= _MINIMUM_STABLE_FOLDS,
        ),
        _gate(
            "cpu_checkpoint_rerank_p95_within_limit",
            latency_probe["latency_seconds"]["p95"] <= _CPU_RERANK_P95_LIMIT_SECONDS,
        ),
    ]


def _run_cpu_latency_probe(
    *,
    traces: Sequence[AgentE2ETrace],
    score_provider: LocalListwiseCheckpointScoreProvider,
) -> dict[str, Any]:
    ordered = sorted(
        traces,
        key=lambda row: hashlib.sha256(row.sample.sample_id.encode()).hexdigest(),
    )[:_CPU_LATENCY_PROBE_COUNT]
    policy = ListwiseUnionPrimaryContextSelectionPolicy(score_provider=score_provider)
    adapter = PrimeQAHybridSidecarObservationAdapter(primary_context_selection_policy=policy)
    latencies = []
    for row in ordered:
        query = PrimeQARuntimeQuery(
            id=row.sample.sample_id,
            title=row.sample.question_title,
            text=row.sample.question_text,
        )
        started_at = time.perf_counter()
        bundle = adapter.observe(
            question=query,
            candidate_pool_results=row.candidate_pool_results,
        )
        latencies.append(time.perf_counter() - started_at)
        if len(bundle.answer_context_results) != 10:
            raise RuntimeError("Stage178A CPU checkpoint probe context depth drifted")
    return {
        "query_count": len(ordered),
        "device": "cpu",
        "latency_seconds": _distribution(latencies),
        "score_provider_counters": asdict(score_provider.counters()),
        "quality_metrics_computed": False,
    }


def _process_guards(
    *,
    report: Mapping[str, Any],
    records: Sequence[ContextCandidateRecord],
    pair_rows: Sequence[stage174.PairFoldRow],
    oof: Mapping[str, Any],
    traces: Mapping[str, Sequence[AgentE2ETrace]],
    dense_summary: Mapping[str, Any],
    forbidden: Sequence[str],
) -> list[dict[str, Any]]:
    boundaries = report["execution_boundaries"]
    return [
        _gate("stage177_authorized", True),
        _gate("corrected_runtime_alignment_authorized", True),
        _gate("exact_train_rows", report["agent_e2e"]["paired_question_count"] == 562),
        _gate("exact_candidate_rows", len(records) == _EXPECTED_CANDIDATE_ROWS),
        _gate("exact_pair_rows", len(pair_rows) == _EXPECTED_PAIR_ROWS),
        _gate("exact_oof_fits", len(oof["fit_summaries"]) == _EXPECTED_OOF_FITS),
        _gate("complete_oof_scores", len(oof["pair_logits"]) == len(pair_rows)),
        _gate("one_full_checkpoint_fit", report["training"]["full_checkpoint_fit"] != {}),
        _gate("checkpoint_hashes_valid", report["training"]["checkpoint"]["all_file_hashes_valid"]),
        _gate("dense_channels_ready", dense_summary.get("status") == "dense_channels_ready"),
        _gate("paired_agent_arms_complete", all(len(rows) == 562 for rows in traces.values())),
        _gate("quality_uses_oof_only", boundaries["answer_quality_uses_oof_only"] is True),
        _gate("development_not_loaded", boundaries["development_loaded"] is False),
        _gate("test_not_loaded", boundaries["test_loaded"] is False),
        _gate(
            "gold_not_projected_into_runtime", boundaries["gold_projected_into_runtime"] is False
        ),
        _gate("sufficiency_gate_not_run", boundaries["sufficiency_gate_run"] is False),
        _gate("retry_count_zero", boundaries["retry_action_count"] == 0),
        _gate("fallback_count_zero", boundaries["fallback_action_count"] == 0),
        _gate("default_runtime_unchanged", boundaries["runtime_registered_as_default"] is False),
        _gate("public_report_safe", not forbidden),
    ]


def write_stage178a_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage178AVisualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = report["agent_e2e"]["profiles"]
    deltas = report["agent_e2e"]["deltas"]
    folds = report["agent_e2e"]["folds"]
    gates = report["quality_gates"]
    resources = report["resource_consumption"]
    charts = {
        "verified_f1.svg": _chart(
            "Stage 178A OOF tool Agent verified F1",
            tuple(
                BarDatum(
                    name,
                    row["verified_metrics"]["average_token_f1"],
                    f"{row['verified_metrics']['average_token_f1']:.4f}",
                )
                for name, row in profiles.items()
            ),
            "verified average token F1",
        ),
        "context_and_citation.svg": _chart(
            "Stage 178A context and citation outcomes",
            tuple(
                BarDatum(
                    f"{name} context gold hit",
                    row["context_gold_hit_count"],
                    str(row["context_gold_hit_count"]),
                )
                for name, row in profiles.items()
            )
            + tuple(
                BarDatum(
                    f"{name} gold citation",
                    row["gold_citation_count"],
                    str(row["gold_citation_count"]),
                )
                for name, row in profiles.items()
            ),
            "answerable train questions",
        ),
        "metric_deltas.svg": _chart(
            "Stage 178A candidate minus baseline deltas",
            tuple(BarDatum(name, float(value), str(value)) for name, value in deltas.items()),
            "delta",
        ),
        "fold_f1.svg": _chart(
            "Stage 178A held-out fold verified F1",
            tuple(
                BarDatum(
                    f"{fold_id} {arm}",
                    row["verified_metrics"]["average_token_f1"],
                    f"{row['verified_metrics']['average_token_f1']:.4f}",
                )
                for fold_id, arms in folds.items()
                for arm, row in arms.items()
            ),
            "verified average token F1",
        ),
        "latency.svg": _chart(
            "Stage 178A request and checkpoint latency",
            (
                BarDatum(
                    "baseline p95",
                    profiles["baseline"]["latency_seconds"]["p95"],
                    f"{profiles['baseline']['latency_seconds']['p95']:.3f}s",
                ),
                BarDatum(
                    "candidate lookup p95",
                    profiles["candidate"]["latency_seconds"]["p95"],
                    f"{profiles['candidate']['latency_seconds']['p95']:.3f}s",
                ),
                BarDatum(
                    "checkpoint CPU rerank p95",
                    report["cpu_checkpoint_latency_probe"]["latency_seconds"]["p95"],
                    f"{report['cpu_checkpoint_latency_probe']['latency_seconds']['p95']:.3f}s",
                ),
                BarDatum(
                    "checkpoint runtime smoke",
                    report["cpu_checkpoint_runtime_smoke"]["latency_seconds"],
                    f"{report['cpu_checkpoint_runtime_smoke']['latency_seconds']:.3f}s",
                ),
            ),
            "seconds",
        ),
        "quality_gates.svg": _chart(
            "Stage 178A strict advancement gates",
            tuple(
                BarDatum(gate["name"], float(gate["passed"]), str(gate["passed"])) for gate in gates
            ),
            "1 means passed",
        ),
        "timing.svg": _chart(
            "Stage 178A phase wall times",
            tuple(
                BarDatum(name.replace("_", " "), value, f"{value:.2f}s")
                for name, value in report["timing_seconds"].items()
            ),
            "seconds",
        ),
        "resources.svg": _chart(
            "Stage 178A resource peaks",
            (
                _gib_bar("process working set", resources["process_peak_working_set_bytes"]),
                _gib_bar("process private usage", resources["process_peak_private_usage_bytes"]),
                _gib_bar("GPU allocated", resources["gpu_peak_allocated_bytes"]),
                _gib_bar("GPU reserved", resources["gpu_peak_reserved_bytes"]),
                _gib_bar(
                    "minimum system available", resources["minimum_system_available_memory_bytes"]
                ),
            ),
            "GiB",
        ),
    }
    written = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        written.append(Stage178AVisualization(name=filename.removesuffix(".svg"), path=str(path)))
    return tuple(written)


def _assert_pair_alignment(
    baseline: Sequence[AgentE2ETrace],
    candidate: Sequence[AgentE2ETrace],
) -> None:
    baseline_ids = [row.sample.sample_id for row in baseline]
    candidate_ids = [row.sample.sample_id for row in candidate]
    if baseline_ids != candidate_ids:
        raise ValueError("Stage178A paired Agent rows are not aligned")


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "maximum": 0.0}
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "mean": round(float(array.mean()), 6),
        "median": round(float(np.median(array)), 6),
        "p95": round(float(np.quantile(array, 0.95)), 6),
        "maximum": round(float(array.max()), 6),
    }


def _authorize_sources(fingerprints: Mapping[str, Mapping[str, Any]]) -> None:
    mismatches = [
        name
        for name, expected in _SOURCE_HASHES.items()
        if fingerprints[name]["sha256"] != expected
    ]
    if mismatches:
        raise ValueError(f"Stage178A source authorization failed: {mismatches}")


def _forbidden_keys_found(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        found = {str(key) for key in value if str(key) in _FORBIDDEN_PUBLIC_KEYS}
        for child in value.values():
            found.update(_forbidden_keys_found(child))
        return found
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        found: set[str] = set()
        for child in value:
            found.update(_forbidden_keys_found(child))
        return found
    return set()


def _canonical_sha256(value: Any) -> str:
    material = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _emit(sink: ProgressSink | None, **event: Any) -> None:
    if sink is not None:
        sink({"stage": _STAGE, **event})


def _gib_bar(label: str, value: int) -> BarDatum:
    gib = value / (1024**3)
    return BarDatum(label, gib, f"{gib:.3f} GiB")


def _chart(title: str, bars: Sequence[BarDatum], x_label: str) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label=x_label,
        width=1320,
        margin_left=560,
        margin_right=220,
    )
