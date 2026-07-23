from __future__ import annotations

import hashlib
import json
import statistics
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application import primeqa_hybrid_iterative_router_calibration as stage169
from ts_rag_agent.application import primeqa_hybrid_listwise_agent_e2e as stage178
from ts_rag_agent.application import (
    primeqa_hybrid_listwise_agent_failure_attribution as stage179,
)
from ts_rag_agent.application import primeqa_hybrid_semantic_evidence_cv as stage173
from ts_rag_agent.application.answer_composition import AnswerCompositionDecision
from ts_rag_agent.application.answer_verification import AnswerVerifier
from ts_rag_agent.application.candidate_reranker_cv import CandidateRerankerExample
from ts_rag_agent.application.candidate_reranker_dataset import (
    build_candidate_runtime_features,
)
from ts_rag_agent.application.citation_aware_composition_policy import (
    CitationAwareCompositionPolicy,
    CitationAwareCompositionSpec,
    DualTargetCandidateModel,
    stage180_policy_specs,
)
from ts_rag_agent.application.evidence_selection import (
    SentenceEvidenceCandidate,
    classify_question_route,
    create_sentence_evidence_selector,
)
from ts_rag_agent.application.listwise_runtime_reranker import (
    ListwiseUnionPrimaryContextSelectionPolicy,
    PrecomputedListwiseScoreProvider,
)
from ts_rag_agent.application.primeqa_hybrid_agent_runtime_observability import (
    AgentWorkflowObservationSink,
    PublicSafeAgentWorkflowObservationEvent,
)
from ts_rag_agent.application.primeqa_hybrid_agent_tool_workflow import (
    FrozenAnswerVerifierFactory,
    PrimeQAHybridAgentToolset,
    create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset,
)
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    _build_train_fold_assignments,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    PrimeQAHybridProcessRuntimeResourceFactory,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_observation_validation import (
    PrimeQAHybridSidecarObservationAdapter,
)
from ts_rag_agent.application.rag_answering import ExtractiveAnswerGenerator
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.application.text_metrics import token_f1
from ts_rag_agent.domain.answer import AnswerCitation, GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQAQuestion, PrimeQARuntimeQuery
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)

_STAGE = "Stage 180"
_CREATED_AT = "2026-07-22"
_ANALYSIS_ID = "primeqa_hybrid_runtime_visible_citation_aware_composition_nested_oof_v1"
_EXPECTED_ROWS = 562
_EXPECTED_ANSWERABLE = 370
_EXPECTED_FOLDS = 5
_EXPECTED_POLICY_SPECS = 14
_EXPECTED_RULE_SPECS = 8
_EXPECTED_LEARNED_SPECS = 6
_EXPECTED_INNER_PARTITIONS = 20
_EXPECTED_MODEL_HEAD_FITS = 50
_MAX_SENTENCES = 3
_MIN_SENTENCE_SCORE = 2.0
_SELECTOR_NAME = "bm25_sentence"
_MAX_CANDIDATES_PER_DOCUMENT = 3
_BOOTSTRAP_REPLICATES = 2_000
_SOURCE_HASHES = {
    "stage179": "80a7b82016eb54a480748466fabff7990d147843742e49114277e08155b45d8f",
    "stage178_public": "e57e3f09bcc65657a3f8783e97e6767b690095e2cffd5d252d51e181eaf533c9",
    "stage178_private": "6fffa820773dea8892dc1d441aff1c3ef3df54ff368b82bf1c9a09b961f0857a",
    "stage178_alignment": "e2398024edf128ad0628900d25eb1ccc9c83c437fb474921fe136e2603e47272",
    "stage128": "012ca36c0559f3533ea2e89160fcb3cee7fb12daa89fb68c69dcf27d9d2ce63e",
    "stage125": "dfaee9eb5688a2a91e3f3d3695def5e32ad87494cdfdd31a00a0434df53ccd65",
    "stage80": "2441bb1cb1e7888299d3f57962b18cd59df84e2086ac281105abcacfc144880f",
    "train": "cabd93e0b972c47384c4bf5cc2cd215a7fc519b2df4f81fba61db73c931aa155",
    "documents": "f93b5e2d8dcfb2c7d12676ef32ce22b7809692f14081aad98096099a5256722b",
}
_FORBIDDEN_PUBLIC_KEYS = stage179._FORBIDDEN_PUBLIC_KEYS | {
    "candidate_sentence",
    "gold_labels",
    "question_id",
    "runtime_features",
}

ProgressSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class Stage180Case:
    sample: PrimeQAHybridSplitSample
    fold_id: str
    question: PrimeQAQuestion
    runtime_query: PrimeQARuntimeQuery
    candidate_pool_results: tuple[RetrievalResult, ...]
    generation_context_results: tuple[RetrievalResult, ...]
    candidates: tuple[SentenceEvidenceCandidate, ...]
    examples: tuple[CandidateRerankerExample, ...]
    baseline_verified: GeneratedAnswer


@dataclass(frozen=True)
class Stage180Outcome:
    fold_id: str
    answerable: bool
    baseline_f1: float
    candidate_f1: float
    baseline_gold_cited: bool
    candidate_gold_cited: bool
    baseline_refused: bool
    candidate_refused: bool
    changed: bool


@dataclass(frozen=True)
class Stage180Visualization:
    name: str
    path: str


class _ObservationSink(AgentWorkflowObservationSink):
    def __init__(self) -> None:
        self.event_count = 0

    def emit(self, event: PublicSafeAgentWorkflowObservationEvent) -> None:
        event.to_public_dict()
        self.event_count += 1


class _DispatchCompositionPolicy:
    def __init__(
        self,
        *,
        policy_by_fold: Mapping[str, CitationAwareCompositionPolicy],
        fold_by_question: Mapping[str, str],
    ) -> None:
        self._policy_by_fold = dict(policy_by_fold)
        self._fold_by_question = dict(fold_by_question)

    @property
    def name(self) -> str:
        return "stage180_nested_oof_dispatch"

    def select(
        self,
        question: PrimeQAQuestion,
        candidates: Sequence[SentenceEvidenceCandidate],
        max_sentences: int,
    ) -> AnswerCompositionDecision:
        fold_id = self._fold_by_question[question.id]
        return self._policy_by_fold[fold_id].select(question, candidates, max_sentences)


@dataclass(frozen=True)
class _AnswerGeneratorFactory:
    policy: _DispatchCompositionPolicy

    def create(self) -> ExtractiveAnswerGenerator:
        return ExtractiveAnswerGenerator(
            max_sentences=_MAX_SENTENCES,
            min_sentence_score=_MIN_SENTENCE_SCORE,
            evidence_selector=create_sentence_evidence_selector(
                _SELECTOR_NAME,
                max_candidates_per_document=_MAX_CANDIDATES_PER_DOCUMENT,
            ),
            composition_policy=self.policy,
        )


def run_stage180_citation_aware_composition_cv(
    *,
    stage179_report_path: Path,
    stage178_public_path: Path,
    stage178_private_path: Path,
    stage178_alignment_path: Path,
    stage128_protocol_path: Path,
    stage125_protocol_path: Path,
    stage80_report_path: Path,
    train_split_path: Path,
    documents_path: Path,
    encoder_batch_size: int = 64,
    progress_sink: ProgressSink | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    source_paths = {
        "stage179": stage179_report_path,
        "stage178_public": stage178_public_path,
        "stage178_private": stage178_private_path,
        "stage178_alignment": stage178_alignment_path,
        "stage128": stage128_protocol_path,
        "stage125": stage125_protocol_path,
        "stage80": stage80_report_path,
        "train": train_split_path,
        "documents": documents_path,
    }
    fingerprints = {
        name: stage173._resolved_fingerprint(path) for name, path in source_paths.items()
    }
    _authorize_sources(fingerprints)
    stage179_report = _load_json(stage179_report_path)
    stage178_public = _load_json(stage178_public_path)
    stage178_private = _load_json(stage178_private_path)
    alignment = _load_json(stage178_alignment_path)
    _authorize_reports(
        stage179_report=stage179_report,
        stage178_public=stage178_public,
        stage178_private=stage178_private,
        alignment=alignment,
    )
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="sources_authorized")

    samples = load_primeqa_hybrid_split_samples(train_split_path)
    if len(samples) != _EXPECTED_ROWS or any(row.assigned_split != "train" for row in samples):
        raise ValueError("Stage180 accepts only the exact train split")
    if sum(row.answerable for row in samples) != _EXPECTED_ANSWERABLE:
        raise ValueError("Stage180 answerable row count drifted")
    fold_assignments = _build_train_fold_assignments(samples, fold_count=_EXPECTED_FOLDS)
    specs = stage180_policy_specs()
    _validate_specs(specs)
    loaded_at = time.perf_counter()

    import torch

    tracker = stage169.Stage169ResourceTracker(torch_module=torch)
    tracker.capture("analysis_start")
    resource_factory = PrimeQAHybridProcessRuntimeResourceFactory(
        stage128_protocol_path=stage128_protocol_path,
        stage125_protocol_path=stage125_protocol_path,
        stage80_report_path=stage80_report_path,
        documents_path=documents_path,
        encoder_batch_size=encoder_batch_size,
        encoder_device="cpu",
    )
    resources = resource_factory.build_shared()
    tracker.capture("runtime_resources_ready")
    provider = PrecomputedListwiseScoreProvider(stage178_private["scores"])
    collection_sink = _ObservationSink()
    collection_workflow = stage178._workflow(
        candidate_pool_retriever=resources.candidate_pool_retriever,
        observation_adapter=PrimeQAHybridSidecarObservationAdapter(
            primary_context_selection_policy=ListwiseUnionPrimaryContextSelectionPolicy(
                score_provider=provider
            )
        ),
        observation_sink=collection_sink,
    )
    cases, collection_traces = _collect_cases(
        samples=samples,
        fold_assignments=fold_assignments,
        workflow=collection_workflow,
        progress_sink=progress_sink,
    )
    tracker.capture("composition_dataset_ready")
    collected_at = time.perf_counter()

    nested = _run_nested_selection(cases=cases, specs=specs, progress_sink=progress_sink)
    tracker.capture("nested_selection_ready")
    selected_at = time.perf_counter()

    final = _run_final_paired_agent(
        cases=cases,
        resources=resources,
        provider=provider,
        policy_by_fold=nested["policy_by_fold"],
        expected_by_question=nested["outer_expected_by_question"],
        progress_sink=progress_sink,
    )
    tracker.capture("paired_agent_complete")
    agent_e2e = stage178.evaluate_agent_e2e(final["traces"])
    _assert_collection_reproduction(collection_traces, final["traces"]["baseline"])
    baseline_reproduced = _profile_core(agent_e2e["profiles"]["baseline"]) == _profile_core(
        stage178_public["agent_e2e"]["profiles"]["candidate"]
    )
    composition_analysis = _composition_analysis(
        baseline=final["traces"]["baseline"],
        candidate=final["traces"]["candidate"],
    )
    quality_gates = _quality_gates(agent_e2e)
    finished_at = time.perf_counter()
    tracker.capture("analysis_complete")

    snapshots = tracker.snapshots
    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train-only nested five-fold OOF comparison of frozen rule and dual-target "
            "runtime-visible citation-aware answer-composition policies."
        ),
        "source_authorization": fingerprints,
        "frozen_protocol": _frozen_protocol(specs),
        "composition_dataset": _dataset_summary(cases),
        "nested_selection": {
            "outer_folds": nested["outer_reports"],
            "selected_policy_counts": dict(sorted(Counter(nested["selected_policy_ids"]).items())),
            "inner_partition_count": nested["inner_partition_count"],
            "model_head_fit_count": nested["model_head_fit_count"],
        },
        "agent_e2e": agent_e2e,
        "stage178_candidate_profile_reproduced": baseline_reproduced,
        "composition_analysis": composition_analysis,
        "quality_gates": quality_gates,
        "runtime": {
            "resource_factory_build_count": resource_factory.build_count,
            "precomputed_score_provider": asdict(provider.counters()),
            "collection_observation_event_count": collection_sink.event_count,
            "paired_baseline_observation_event_count": final["baseline_event_count"],
            "paired_candidate_observation_event_count": final["candidate_event_count"],
            "collection_workflow_counters": asdict(collection_workflow.counters()),
            "paired_baseline_workflow_counters": final["baseline_workflow_counters"],
            "paired_candidate_workflow_counters": final["candidate_workflow_counters"],
        },
        "resource_consumption": {
            "sampling_mode": "event_driven_in_process_without_monitor_polling",
            "phase_snapshots": [asdict(snapshot) for snapshot in snapshots],
            "process_peak_working_set_bytes": max(
                row.process_peak_working_set_bytes for row in snapshots
            ),
            "process_peak_private_usage_bytes": max(
                row.process_private_usage_bytes for row in snapshots
            ),
            "minimum_system_available_memory_bytes": min(
                row.system_available_memory_bytes for row in snapshots
            ),
            "gpu_peak_allocated_bytes": max(row.gpu_allocated_bytes for row in snapshots),
            "gpu_peak_reserved_bytes": max(row.gpu_reserved_bytes for row in snapshots),
            "process_cpu_time_seconds": round(
                snapshots[-1].process_cpu_time_seconds - snapshots[0].process_cpu_time_seconds,
                6,
            ),
        },
        "timing_seconds": {
            "source_authorization": round(authorized_at - started_at, 6),
            "load_train_and_protocol": round(loaded_at - authorized_at, 6),
            "resource_build_and_case_collection": round(collected_at - loaded_at, 6),
            "nested_selection": round(selected_at - collected_at, 6),
            "paired_agent_and_analysis": round(finished_at - selected_at, 6),
            "wall": round(finished_at - started_at, 6),
        },
        "execution_boundaries": {
            "train_loaded": True,
            "development_loaded": False,
            "test_loaded": False,
            "gold_labels_used_only_during_fit_and_offline_evaluation": True,
            "runtime_policy_uses_only_runtime_features": True,
            "model_head_fit_count": nested["model_head_fit_count"],
            "agent_turn_count": _EXPECTED_ROWS * 3,
            "retry_action_count": sum(
                row.retry_action_count for arm in final["traces"].values() for row in arm
            ),
            "fallback_action_count": sum(
                row.fallback_action_count for arm in final["traces"].values() for row in arm
            ),
            "runtime_registered_as_default": False,
            "stage178b_run": False,
        },
    }
    forbidden = sorted(_stage180_forbidden_keys_found(report))
    report["public_safe_contract"] = {
        "forbidden_keys": sorted(_FORBIDDEN_PUBLIC_KEYS),
        "forbidden_keys_found": forbidden,
    }
    report["process_guards"] = _process_guards(
        report=report,
        cases=cases,
        specs=specs,
        final=final,
        forbidden=forbidden,
    )
    process_passed = all(row["passed"] for row in report["process_guards"])
    quality_passed = all(row["passed"] for row in quality_gates)
    if not process_passed:
        status = "stage180_citation_aware_composition_invalid"
    elif quality_passed:
        status = "advance_to_stage181_frozen_composition_validation"
    else:
        status = "stage180_citation_aware_composition_insufficient"
    report["decision"] = {
        "status": status,
        "candidate_selected": process_passed and quality_passed,
        "recommended_next_direction": (
            "freeze_selected_oof_policy_for_validation"
            if process_passed and quality_passed
            else "stop_or_redesign_composition_family"
        ),
        "development_opened": False,
        "test_opened": False,
        "stage178b_authorized": False,
        "default_runtime_activation": False,
    }
    _emit(progress_sink, phase="analysis_complete", decision=report["decision"])
    return report


def _authorize_sources(fingerprints: Mapping[str, Mapping[str, Any]]) -> None:
    mismatches = [
        name
        for name, expected in _SOURCE_HASHES.items()
        if fingerprints.get(name, {}).get("sha256") != expected
    ]
    if mismatches:
        raise ValueError(f"Stage180 source authorization failed: {mismatches}")


def _authorize_reports(
    *,
    stage179_report: Mapping[str, Any],
    stage178_public: Mapping[str, Any],
    stage178_private: Mapping[str, Any],
    alignment: Mapping[str, Any],
) -> None:
    if stage179_report.get("decision", {}).get("status") != (
        "stage179_failure_attribution_completed"
    ):
        raise ValueError("Stage179 did not complete failure attribution")
    if stage179_report.get("decision", {}).get("recommended_next_direction") != (
        "design_runtime_visible_citation_aware_composition_oof"
    ):
        raise ValueError("Stage179 did not recommend citation-aware composition OOF")
    stage179._authorize_stage178(
        public=stage178_public,
        private=stage178_private,
        alignment=alignment,
    )


def _validate_specs(specs: Sequence[CitationAwareCompositionSpec]) -> None:
    if len(specs) != _EXPECTED_POLICY_SPECS:
        raise ValueError("Stage180 policy family size drifted")
    if len({spec.policy_id for spec in specs}) != len(specs):
        raise ValueError("Stage180 policy ids must be unique")
    rule_count = sum(spec.family != "dual_target" for spec in specs)
    learned_count = sum(spec.family == "dual_target" for spec in specs)
    if rule_count != _EXPECTED_RULE_SPECS or learned_count != _EXPECTED_LEARNED_SPECS:
        raise ValueError("Stage180 rule/learned policy counts drifted")


def _collect_cases(
    *,
    samples: Sequence[PrimeQAHybridSplitSample],
    fold_assignments: Mapping[str, str],
    workflow: Any,
    progress_sink: ProgressSink | None,
) -> tuple[tuple[Stage180Case, ...], tuple[stage178.AgentE2ETrace, ...]]:
    selector = create_sentence_evidence_selector(
        _SELECTOR_NAME,
        max_candidates_per_document=_MAX_CANDIDATES_PER_DOCUMENT,
    )
    verifier = AnswerVerifier(min_evidence_score=7.0, max_citation_rank=200)
    cases = []
    traces = []
    for index, sample in enumerate(samples, start=1):
        question = sample.to_primeqa_question()
        runtime_query = PrimeQARuntimeQuery(
            id=sample.sample_id,
            title=sample.question_title,
            text=sample.question_text,
        )
        started_at = time.perf_counter()
        run = workflow.run(runtime_query)
        latency = time.perf_counter() - started_at
        candidates = tuple(
            candidate
            for candidate in selector.rank_sentence_candidates(
                question,
                run.generation_context_results,
            )
            if candidate.score >= _MIN_SENTENCE_SCORE
        )
        baseline_direct = _compose_and_verify(
            question=question,
            selected=candidates[:_MAX_SENTENCES],
            verification_context=run.candidate_pool_results,
            verifier=verifier,
        )
        if baseline_direct != run.verified_answer:
            raise RuntimeError("Stage180 direct top3 baseline did not reproduce Agent output")
        examples = _candidate_examples(question=question, candidates=candidates)
        case = Stage180Case(
            sample=sample,
            fold_id=fold_assignments[sample.sample_id],
            question=question,
            runtime_query=runtime_query,
            candidate_pool_results=tuple(run.candidate_pool_results),
            generation_context_results=tuple(run.generation_context_results),
            candidates=candidates,
            examples=examples,
            baseline_verified=run.verified_answer,
        )
        cases.append(case)
        public_trace = run.public_safe_trace
        traces.append(
            stage178.AgentE2ETrace(
                sample=sample,
                fold_id=case.fold_id,
                question=question,
                candidate_pool_results=case.candidate_pool_results,
                generation_context_results=case.generation_context_results,
                verified_answer=run.verified_answer,
                latency_seconds=latency,
                tool_call_count=public_trace.tool_call_count,
                retry_action_count=public_trace.retry_action_count,
                fallback_action_count=public_trace.fallback_action_count,
            )
        )
        if index % 25 == 0 or index == len(samples):
            _emit(
                progress_sink,
                phase="composition_dataset_progress",
                completed=index,
                total=len(samples),
            )
    return tuple(cases), tuple(traces)


def _candidate_examples(
    *,
    question: PrimeQAQuestion,
    candidates: Sequence[SentenceEvidenceCandidate],
) -> tuple[CandidateRerankerExample, ...]:
    f1_values = [
        token_f1(candidate.sentence, question.answer) if question.answerable else 0.0
        for candidate in candidates
    ]
    best_index = f1_values.index(max(f1_values)) if f1_values else None
    route = classify_question_route(question)
    return tuple(
        CandidateRerankerExample(
            split="train",
            question_id=question.id,
            candidate_id=f"{question.id}::stage180_candidate_{rank:03d}",
            candidate_rank=rank,
            question_route=route,
            runtime_features=build_candidate_runtime_features(
                question=question,
                candidate=candidate,
                question_route=route,
                selector_name=_SELECTOR_NAME,
            ),
            candidate_token_f1=f1_value,
            is_best_candidate_for_question=(rank - 1 == best_index),
            is_gold_document=bool(
                question.answerable
                and question.answer_doc_id is not None
                and candidate.retrieval_result.document.id == question.answer_doc_id
            ),
        )
        for rank, (candidate, f1_value) in enumerate(
            zip(candidates, f1_values, strict=True),
            start=1,
        )
    )


def _run_nested_selection(
    *,
    cases: Sequence[Stage180Case],
    specs: Sequence[CitationAwareCompositionSpec],
    progress_sink: ProgressSink | None,
) -> dict[str, Any]:
    fold_ids = sorted({case.fold_id for case in cases})
    if len(fold_ids) != _EXPECTED_FOLDS:
        raise ValueError("Stage180 requires five frozen outer folds")
    rule_specs = tuple(spec for spec in specs if spec.family != "dual_target")
    learned_specs = tuple(spec for spec in specs if spec.family == "dual_target")
    policy_by_fold = {}
    expected_by_question = {}
    outer_reports = []
    selected_policy_ids = []
    inner_partition_count = 0
    model_head_fit_count = 0

    for outer_index, outer_fold in enumerate(fold_ids, start=1):
        inner_folds = tuple(fold for fold in fold_ids if fold != outer_fold)
        outcomes_by_policy: dict[str, list[Stage180Outcome]] = {
            spec.policy_id: [] for spec in specs
        }
        for validation_fold in inner_folds:
            training_cases = [
                case for case in cases if case.fold_id not in {outer_fold, validation_fold}
            ]
            validation_cases = [case for case in cases if case.fold_id == validation_fold]
            model = _fit_dual_target_model(training_cases)
            model_head_fit_count += 2
            inner_partition_count += 1
            for spec in rule_specs:
                outcomes_by_policy[spec.policy_id].extend(
                    _evaluate_cases(
                        cases=validation_cases,
                        policy=CitationAwareCompositionPolicy(spec=spec),
                    )["outcomes"]
                )
            for spec in learned_specs:
                outcomes_by_policy[spec.policy_id].extend(
                    _evaluate_cases(
                        cases=validation_cases,
                        policy=CitationAwareCompositionPolicy(spec=spec, model=model),
                    )["outcomes"]
                )

        inner_summaries = {
            spec.policy_id: _outcome_summary(outcomes_by_policy[spec.policy_id]) for spec in specs
        }
        selected_spec = _select_spec(specs=specs, summaries=inner_summaries)
        selected_policy_ids.append(selected_spec.policy_id)
        outer_training_cases = [case for case in cases if case.fold_id != outer_fold]
        outer_model = _fit_dual_target_model(outer_training_cases)
        model_head_fit_count += 2
        selected_policy = CitationAwareCompositionPolicy(
            spec=selected_spec,
            model=outer_model if selected_spec.family == "dual_target" else None,
        )
        policy_by_fold[outer_fold] = selected_policy
        outer_cases = [case for case in cases if case.fold_id == outer_fold]
        outer_evaluation = _evaluate_cases(cases=outer_cases, policy=selected_policy)
        expected_by_question.update(outer_evaluation["verified_by_question"])
        outer_reports.append(
            {
                "outer_fold": outer_fold,
                "inner_fold_count": len(inner_folds),
                "selected_policy_id": selected_spec.policy_id,
                "selected_policy_family": selected_spec.family,
                "selection_order": _selection_order(inner_summaries[selected_spec.policy_id]),
                "inner_oof_policy_summaries": dict(sorted(inner_summaries.items())),
                "outer_selected_policy_summary": _outcome_summary(outer_evaluation["outcomes"]),
            }
        )
        _emit(
            progress_sink,
            phase="nested_outer_fold_complete",
            completed=outer_index,
            total=len(fold_ids),
            selected_policy_id=selected_spec.policy_id,
        )

    return {
        "policy_by_fold": policy_by_fold,
        "outer_expected_by_question": expected_by_question,
        "outer_reports": outer_reports,
        "selected_policy_ids": selected_policy_ids,
        "inner_partition_count": inner_partition_count,
        "model_head_fit_count": model_head_fit_count,
    }


def _fit_dual_target_model(cases: Sequence[Stage180Case]) -> DualTargetCandidateModel:
    examples = [example for case in cases for example in case.examples]
    model = DualTargetCandidateModel()
    model.fit(examples)
    return model


def _evaluate_cases(
    *,
    cases: Sequence[Stage180Case],
    policy: CitationAwareCompositionPolicy,
) -> dict[str, Any]:
    verifier = AnswerVerifier(min_evidence_score=7.0, max_citation_rank=200)
    outcomes = []
    verified_by_question = {}
    for case in cases:
        decision = policy.select(case.question, case.candidates, _MAX_SENTENCES)
        candidate = _compose_and_verify(
            question=case.question,
            selected=decision.selected_candidates,
            verification_context=case.candidate_pool_results,
            verifier=verifier,
        )
        verified_by_question[case.question.id] = candidate
        outcomes.append(_outcome(case=case, candidate=candidate))
    return {
        "outcomes": tuple(outcomes),
        "verified_by_question": verified_by_question,
    }


def _compose_and_verify(
    *,
    question: PrimeQAQuestion,
    selected: Sequence[SentenceEvidenceCandidate],
    verification_context: Sequence[RetrievalResult],
    verifier: AnswerVerifier,
) -> GeneratedAnswer:
    if not selected:
        generated = GeneratedAnswer(
            question_id=question.id,
            answer="I do not have enough retrieved evidence to answer this question.",
            citations=[],
            refused=True,
        )
    else:
        generated = GeneratedAnswer(
            question_id=question.id,
            answer=" ".join(
                f"{candidate.sentence} [{candidate.retrieval_result.document.id}]"
                for candidate in selected
            ),
            citations=[
                AnswerCitation(
                    document_id=candidate.retrieval_result.document.id,
                    title=candidate.retrieval_result.document.title,
                    retrieval_rank=candidate.retrieval_result.rank,
                    evidence_score=round(candidate.score, 4),
                )
                for candidate in selected
            ],
            refused=False,
        )
    return verifier.verify(generated, verification_context).verified_answer


def _outcome(*, case: Stage180Case, candidate: GeneratedAnswer) -> Stage180Outcome:
    baseline = case.baseline_verified
    gold_id = case.sample.answer_doc_id
    baseline_cited = bool(
        case.sample.answerable
        and gold_id is not None
        and any(row.document_id == gold_id for row in baseline.citations)
    )
    candidate_cited = bool(
        case.sample.answerable
        and gold_id is not None
        and any(row.document_id == gold_id for row in candidate.citations)
    )
    return Stage180Outcome(
        fold_id=case.fold_id,
        answerable=case.sample.answerable,
        baseline_f1=(
            stage178.stage160.score_answer(
                baseline.answer,
                case.sample.answer,
                refused=baseline.refused,
            )
            if case.sample.answerable
            else 0.0
        ),
        candidate_f1=(
            stage178.stage160.score_answer(
                candidate.answer,
                case.sample.answer,
                refused=candidate.refused,
            )
            if case.sample.answerable
            else 0.0
        ),
        baseline_gold_cited=baseline_cited,
        candidate_gold_cited=candidate_cited,
        baseline_refused=baseline.refused,
        candidate_refused=candidate.refused,
        changed=baseline != candidate,
    )


def _outcome_summary(outcomes: Sequence[Stage180Outcome]) -> dict[str, Any]:
    answerable = [row for row in outcomes if row.answerable]
    unanswerable = [row for row in outcomes if not row.answerable]
    baseline_f1 = statistics.fmean(row.baseline_f1 for row in answerable)
    candidate_f1 = statistics.fmean(row.candidate_f1 for row in answerable)
    fold_deltas = {
        fold_id: _fold_outcome_delta([row for row in outcomes if row.fold_id == fold_id])
        for fold_id in sorted({row.fold_id for row in outcomes})
    }
    f1_deltas = [row.candidate_f1 - row.baseline_f1 for row in answerable]
    return {
        "question_count": len(outcomes),
        "answerable_question_count": len(answerable),
        "baseline_answerable_f1": round(baseline_f1, 6),
        "candidate_answerable_f1": round(candidate_f1, 6),
        "answerable_f1_delta": round(candidate_f1 - baseline_f1, 6),
        "baseline_gold_citation_count": sum(row.baseline_gold_cited for row in answerable),
        "candidate_gold_citation_count": sum(row.candidate_gold_cited for row in answerable),
        "gold_citation_delta": sum(
            int(row.candidate_gold_cited) - int(row.baseline_gold_cited) for row in answerable
        ),
        "f1_improved_count": sum(delta > 0 for delta in f1_deltas),
        "f1_tied_count": sum(delta == 0 for delta in f1_deltas),
        "f1_regressed_count": sum(delta < 0 for delta in f1_deltas),
        "changed_verified_count": sum(row.changed for row in outcomes),
        "answerable_refusal_delta": sum(row.candidate_refused for row in answerable)
        - sum(row.baseline_refused for row in answerable),
        "unanswerable_false_answer_delta": sum(not row.candidate_refused for row in unanswerable)
        - sum(not row.baseline_refused for row in unanswerable),
        "minimum_fold_gold_citation_delta": min(
            row["gold_citation_delta"] for row in fold_deltas.values()
        ),
        "minimum_fold_answerable_f1_delta": min(
            row["answerable_f1_delta"] for row in fold_deltas.values()
        ),
        "fold_deltas": fold_deltas,
    }


def _fold_outcome_delta(outcomes: Sequence[Stage180Outcome]) -> dict[str, Any]:
    answerable = [row for row in outcomes if row.answerable]
    return {
        "question_count": len(outcomes),
        "answerable_question_count": len(answerable),
        "gold_citation_delta": sum(
            int(row.candidate_gold_cited) - int(row.baseline_gold_cited) for row in answerable
        ),
        "answerable_f1_delta": round(
            statistics.fmean(row.candidate_f1 - row.baseline_f1 for row in answerable),
            6,
        ),
    }


def _select_spec(
    *,
    specs: Sequence[CitationAwareCompositionSpec],
    summaries: Mapping[str, Mapping[str, Any]],
) -> CitationAwareCompositionSpec:
    return min(
        specs,
        key=lambda spec: (
            -int(summaries[spec.policy_id]["minimum_fold_gold_citation_delta"]),
            -int(summaries[spec.policy_id]["gold_citation_delta"]),
            -float(summaries[spec.policy_id]["minimum_fold_answerable_f1_delta"]),
            -float(summaries[spec.policy_id]["answerable_f1_delta"]),
            int(summaries[spec.policy_id]["f1_regressed_count"]),
            int(summaries[spec.policy_id]["changed_verified_count"]),
            spec.policy_id,
        ),
    )


def _selection_order(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "minimum_fold_gold_citation_delta": summary["minimum_fold_gold_citation_delta"],
        "aggregate_gold_citation_delta": summary["gold_citation_delta"],
        "minimum_fold_answerable_f1_delta": summary["minimum_fold_answerable_f1_delta"],
        "aggregate_answerable_f1_delta": summary["answerable_f1_delta"],
        "f1_regressed_count": summary["f1_regressed_count"],
        "changed_verified_count": summary["changed_verified_count"],
    }


def _run_final_paired_agent(
    *,
    cases: Sequence[Stage180Case],
    resources: Any,
    provider: PrecomputedListwiseScoreProvider,
    policy_by_fold: Mapping[str, CitationAwareCompositionPolicy],
    expected_by_question: Mapping[str, GeneratedAnswer],
    progress_sink: ProgressSink | None,
) -> dict[str, Any]:
    fold_by_question = {case.question.id: case.fold_id for case in cases}
    baseline_sink = _ObservationSink()
    candidate_sink = _ObservationSink()

    def adapter() -> PrimeQAHybridSidecarObservationAdapter:
        return PrimeQAHybridSidecarObservationAdapter(
            primary_context_selection_policy=ListwiseUnionPrimaryContextSelectionPolicy(
                score_provider=provider
            )
        )

    baseline_workflow = stage178._workflow(
        candidate_pool_retriever=resources.candidate_pool_retriever,
        observation_adapter=adapter(),
        observation_sink=baseline_sink,
    )
    dispatch = _DispatchCompositionPolicy(
        policy_by_fold=policy_by_fold,
        fold_by_question=fold_by_question,
    )
    candidate_toolset = PrimeQAHybridAgentToolset(
        candidate_pool_retriever=resources.candidate_pool_retriever,
        observation_adapter=adapter(),
        answer_generator_factory=_AnswerGeneratorFactory(dispatch),
        answer_verifier_factory=FrozenAnswerVerifierFactory(),
    )
    candidate_workflow = create_primeqa_hybrid_langgraph_agent_tool_workflow_from_toolset(
        toolset=candidate_toolset,
        observation_sink=candidate_sink,
    )
    workflows = {"baseline": baseline_workflow, "candidate": candidate_workflow}
    traces: dict[str, list[stage178.AgentE2ETrace]] = {"baseline": [], "candidate": []}
    direct_match_count = 0
    context_match_count = 0
    for index, case in enumerate(cases, start=1):
        order = (
            ("baseline", "candidate")
            if int(hashlib.sha256(case.question.id.encode()).hexdigest(), 16) % 2 == 0
            else ("candidate", "baseline")
        )
        runs = {}
        for arm in order:
            started_at = time.perf_counter()
            run = workflows[arm].run(case.runtime_query)
            latency = time.perf_counter() - started_at
            runs[arm] = run
            public_trace = run.public_safe_trace
            traces[arm].append(
                stage178.AgentE2ETrace(
                    sample=case.sample,
                    fold_id=case.fold_id,
                    question=case.question,
                    candidate_pool_results=tuple(run.candidate_pool_results),
                    generation_context_results=tuple(run.generation_context_results),
                    verified_answer=run.verified_answer,
                    latency_seconds=latency,
                    tool_call_count=public_trace.tool_call_count,
                    retry_action_count=public_trace.retry_action_count,
                    fallback_action_count=public_trace.fallback_action_count,
                )
            )
        expected_pool = tuple(row.document.id for row in case.candidate_pool_results)
        expected_context = tuple(row.document.id for row in case.generation_context_results)
        for run in runs.values():
            if tuple(row.document.id for row in run.candidate_pool_results) != expected_pool:
                raise RuntimeError("Stage180 candidate pool drifted during paired Agent replay")
            if tuple(row.document.id for row in run.generation_context_results) != expected_context:
                raise RuntimeError("Stage180 generation context drifted during paired Agent replay")
        context_match_count += 1
        if runs["candidate"].verified_answer != expected_by_question[case.question.id]:
            raise RuntimeError("Stage180 OOF Agent output did not match offline outer evaluation")
        direct_match_count += 1
        if index % 25 == 0 or index == len(cases):
            _emit(
                progress_sink,
                phase="paired_agent_progress",
                completed=index,
                total=len(cases),
            )
    return {
        "traces": {name: tuple(rows) for name, rows in traces.items()},
        "direct_match_count": direct_match_count,
        "context_match_count": context_match_count,
        "baseline_event_count": baseline_sink.event_count,
        "candidate_event_count": candidate_sink.event_count,
        "baseline_workflow_counters": asdict(baseline_workflow.counters()),
        "candidate_workflow_counters": asdict(candidate_workflow.counters()),
    }


def _assert_collection_reproduction(
    collection: Sequence[stage178.AgentE2ETrace],
    final_baseline: Sequence[stage178.AgentE2ETrace],
) -> None:
    if len(collection) != len(final_baseline):
        raise RuntimeError("Stage180 baseline collection count drifted")
    for first, second in zip(collection, final_baseline, strict=True):
        if first.question.id != second.question.id:
            raise RuntimeError("Stage180 baseline collection alignment drifted")
        if first.verified_answer != second.verified_answer:
            raise RuntimeError("Stage180 baseline Agent answer did not reproduce")
        if tuple(row.document.id for row in first.generation_context_results) != tuple(
            row.document.id for row in second.generation_context_results
        ):
            raise RuntimeError("Stage180 baseline Agent context did not reproduce")


def _composition_analysis(
    *,
    baseline: Sequence[stage178.AgentE2ETrace],
    candidate: Sequence[stage178.AgentE2ETrace],
) -> dict[str, Any]:
    answerable_pairs = [
        (left, right)
        for left, right in zip(baseline, candidate, strict=True)
        if left.sample.answerable
    ]
    context_hits = [(left, right) for left, right in answerable_pairs if left.context_gold_hit]
    baseline_cited = sum(left.gold_cited for left, _right in context_hits)
    candidate_cited = sum(right.gold_cited for _left, right in context_hits)
    deltas = [right.answer_f1 - left.answer_f1 for left, right in answerable_pairs]
    return {
        "answerable_question_count": len(answerable_pairs),
        "unchanged_context_gold_hit_count": len(context_hits),
        "baseline_context_to_citation_count": baseline_cited,
        "candidate_context_to_citation_count": candidate_cited,
        "context_to_citation_count_delta": candidate_cited - baseline_cited,
        "baseline_context_to_citation_rate": _ratio(baseline_cited, len(context_hits)),
        "candidate_context_to_citation_rate": _ratio(candidate_cited, len(context_hits)),
        "context_to_citation_rate_delta": round(
            _ratio(candidate_cited, len(context_hits)) - _ratio(baseline_cited, len(context_hits)),
            6,
        ),
        "f1_improved_count": sum(delta > 0 for delta in deltas),
        "f1_tied_count": sum(delta == 0 for delta in deltas),
        "f1_regressed_count": sum(delta < 0 for delta in deltas),
    }


def _quality_gates(agent_e2e: Mapping[str, Any]) -> list[dict[str, Any]]:
    profiles = agent_e2e["profiles"]
    baseline = profiles["baseline"]
    candidate = profiles["candidate"]
    deltas = agent_e2e["deltas"]
    bootstrap = agent_e2e["paired_bootstrap"]["metrics"]
    folds = agent_e2e["folds"]
    citation_nonregression = sum(
        row["candidate"]["gold_citation_count"] >= row["baseline"]["gold_citation_count"]
        for row in folds.values()
    )
    f1_nonregression = sum(
        row["candidate"]["verified_metrics"]["average_token_f1"]
        >= row["baseline"]["verified_metrics"]["average_token_f1"]
        for row in folds.values()
    )
    return [
        _gate("gold_citation_strict_gain", deltas["gold_citation_count"] >= 1),
        _gate(
            "gold_citation_bootstrap_ci_lower_nonnegative",
            bootstrap["gold_citation"]["ci95_lower"] >= 0,
        ),
        _gate("answerable_f1_strict_gain", deltas["verified_average_token_f1"] > 0),
        _gate(
            "answerable_f1_bootstrap_ci_lower_nonnegative",
            bootstrap["answer_f1"]["ci95_lower"] >= 0,
        ),
        _gate("citation_fold_nonregression_4_of_5", citation_nonregression >= 4),
        _gate("f1_fold_nonregression_4_of_5", f1_nonregression >= 4),
        _gate(
            "answerable_refusal_nonincrease",
            candidate["answerable_refusal_count"] <= baseline["answerable_refusal_count"],
        ),
        _gate(
            "unanswerable_false_answer_nonincrease",
            candidate["unanswerable_false_answer_count"]
            <= baseline["unanswerable_false_answer_count"],
        ),
        _gate("generation_context_gold_hit_unchanged", deltas["context_gold_hit_count"] == 0),
        _gate("candidate_p95_overhead_within_50ms", deltas["latency_p95_seconds"] <= 0.05),
        _gate(
            "no_retry_or_fallback",
            candidate["retry_action_count"] == 0 and candidate["fallback_action_count"] == 0,
        ),
    ]


def _frozen_protocol(
    specs: Sequence[CitationAwareCompositionSpec],
) -> dict[str, Any]:
    return {
        "split": "train_only",
        "outer_fold_count": _EXPECTED_FOLDS,
        "inner_fold_count_per_outer": 4,
        "policy_specs": [asdict(spec) for spec in specs],
        "model_heads": ["gold_document_logistic", "sentence_f1_ridge"],
        "expected_model_head_fit_count": _EXPECTED_MODEL_HEAD_FITS,
        "selection_order": [
            "maximum minimum inner-fold gold-citation delta",
            "maximum aggregate inner-OOF gold-citation delta",
            "maximum minimum inner-fold answerable F1 delta",
            "maximum aggregate inner-OOF answerable F1 delta",
            "minimum F1 regression count",
            "minimum changed verified count",
            "ascending stable policy id",
        ],
        "fallback_strategy_enabled": False,
        "development_and_test_closed": True,
        "runtime_registered_as_default": False,
    }


def _dataset_summary(cases: Sequence[Stage180Case]) -> dict[str, Any]:
    counts = [len(case.candidates) for case in cases]
    examples = [example for case in cases for example in case.examples]
    return {
        "question_count": len(cases),
        "answerable_question_count": sum(case.sample.answerable for case in cases),
        "candidate_row_count": len(examples),
        "questions_with_candidates": sum(bool(case.candidates) for case in cases),
        "questions_with_gold_document_candidate": sum(
            any(example.is_gold_document for example in case.examples) for case in cases
        ),
        "gold_document_candidate_row_count": sum(example.is_gold_document for example in examples),
        "candidate_count_distribution": _distribution(counts),
        "runtime_feature_names": sorted(examples[0].runtime_features if examples else {}),
    }


def _process_guards(
    *,
    report: Mapping[str, Any],
    cases: Sequence[Stage180Case],
    specs: Sequence[CitationAwareCompositionSpec],
    final: Mapping[str, Any],
    forbidden: Sequence[str],
) -> list[dict[str, Any]]:
    nested = report["nested_selection"]
    runtime = report["runtime"]
    boundaries = report["execution_boundaries"]
    provider = runtime["precomputed_score_provider"]
    return [
        _gate("exact_train_rows", len(cases) == _EXPECTED_ROWS),
        _gate(
            "exact_answerable_rows",
            sum(case.sample.answerable for case in cases) == _EXPECTED_ANSWERABLE,
        ),
        _gate("exact_five_folds", len({case.fold_id for case in cases}) == _EXPECTED_FOLDS),
        _gate("exact_policy_family", len(specs) == _EXPECTED_POLICY_SPECS),
        _gate(
            "exact_inner_partitions",
            nested["inner_partition_count"] == _EXPECTED_INNER_PARTITIONS,
        ),
        _gate(
            "exact_model_head_fits",
            nested["model_head_fit_count"] == _EXPECTED_MODEL_HEAD_FITS,
        ),
        _gate("one_runtime_resource_build", runtime["resource_factory_build_count"] == 1),
        _gate("complete_direct_agent_match", final["direct_match_count"] == _EXPECTED_ROWS),
        _gate("complete_context_match", final["context_match_count"] == _EXPECTED_ROWS),
        _gate(
            "stage178_candidate_profile_reproduced",
            report["stage178_candidate_profile_reproduced"] is True,
        ),
        _gate("three_agent_passes", boundaries["agent_turn_count"] == _EXPECTED_ROWS * 3),
        _gate("exact_score_provider_calls", provider["call_count"] == _EXPECTED_ROWS * 3),
        _gate("exact_score_provider_pairs", provider["pair_count"] == 9_714 * 3),
        _gate("development_closed", boundaries["development_loaded"] is False),
        _gate("test_closed", boundaries["test_loaded"] is False),
        _gate(
            "runtime_features_only",
            boundaries["runtime_policy_uses_only_runtime_features"] is True,
        ),
        _gate("no_retry", boundaries["retry_action_count"] == 0),
        _gate("no_fallback", boundaries["fallback_action_count"] == 0),
        _gate("default_runtime_unchanged", boundaries["runtime_registered_as_default"] is False),
        _gate("stage178b_not_run", boundaries["stage178b_run"] is False),
        _gate("public_report_safe", not forbidden),
    ]


def write_stage180_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage180Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    e2e = report["agent_e2e"]
    profiles = e2e["profiles"]
    folds = e2e["folds"]
    analysis = report["composition_analysis"]
    charts = {
        "selected_policy_counts.svg": _chart(
            "Stage 180 selected policy counts",
            tuple(
                BarDatum(name, float(value), str(value))
                for name, value in report["nested_selection"]["selected_policy_counts"].items()
            ),
            "outer folds",
        ),
        "fold_f1_delta.svg": _chart(
            "Stage 180 outer-fold answerable F1 deltas",
            tuple(
                BarDatum(
                    fold,
                    _fold_f1_delta(row),
                    f"{_fold_f1_delta(row):+.4f}",
                )
                for fold, row in folds.items()
            ),
            "candidate minus baseline F1",
        ),
        "fold_citation_delta.svg": _chart(
            "Stage 180 outer-fold gold-citation deltas",
            tuple(
                BarDatum(
                    fold,
                    float(
                        row["candidate"]["gold_citation_count"]
                        - row["baseline"]["gold_citation_count"]
                    ),
                    str(
                        row["candidate"]["gold_citation_count"]
                        - row["baseline"]["gold_citation_count"]
                    ),
                )
                for fold, row in folds.items()
            ),
            "candidate minus baseline citations",
        ),
        "aggregate_f1.svg": _chart(
            "Stage 180 answerable F1",
            tuple(
                BarDatum(
                    arm,
                    row["verified_metrics"]["average_token_f1"],
                    f"{row['verified_metrics']['average_token_f1']:.4f}",
                )
                for arm, row in profiles.items()
            ),
            "average token F1",
        ),
        "context_to_citation.svg": _chart(
            "Stage 180 context-to-citation conversion",
            (
                BarDatum(
                    "baseline",
                    analysis["baseline_context_to_citation_rate"],
                    f"{analysis['baseline_context_to_citation_rate']:.4f}",
                ),
                BarDatum(
                    "candidate",
                    analysis["candidate_context_to_citation_rate"],
                    f"{analysis['candidate_context_to_citation_rate']:.4f}",
                ),
            ),
            "gold citation rate among gold-hit contexts",
        ),
        "quality_gates.svg": _chart(
            "Stage 180 quality gates",
            tuple(
                BarDatum(row["name"], float(row["passed"]), "pass" if row["passed"] else "fail")
                for row in report["quality_gates"]
            ),
            "pass = 1",
        ),
    }
    written = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        written.append(Stage180Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _chart(title: str, bars: Sequence[BarDatum], x_label: str) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label=x_label,
        width=1480,
        margin_left=680,
    )


def _fold_f1_delta(fold_report: Mapping[str, Any]) -> float:
    return (
        fold_report["candidate"]["verified_metrics"]["average_token_f1"]
        - fold_report["baseline"]["verified_metrics"]["average_token_f1"]
    )


def _distribution(values: Sequence[int]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "minimum": float(ordered[0]),
        "median": float(statistics.median(ordered)),
        "p95": float(ordered[max(0, int(0.95 * len(ordered)) - 1)]),
        "maximum": float(ordered[-1]),
        "mean": round(statistics.fmean(ordered), 6),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_core(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "verified_metrics": profile["verified_metrics"],
        "context_gold_hit_count": profile["context_gold_hit_count"],
        "context_gold_hit_rate": profile["context_gold_hit_rate"],
        "gold_citation_count": profile["gold_citation_count"],
        "answerable_refusal_count": profile["answerable_refusal_count"],
        "unanswerable_false_answer_count": profile["unanswerable_false_answer_count"],
        "tool_call_count": profile["tool_call_count"],
        "retry_action_count": profile["retry_action_count"],
        "fallback_action_count": profile["fallback_action_count"],
    }


def _stage180_forbidden_keys_found(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in _FORBIDDEN_PUBLIC_KEYS:
                found.add(key_text)
            found.update(_stage180_forbidden_keys_found(child))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for child in value:
            found.update(_stage180_forbidden_keys_found(child))
    return found


def _emit(progress_sink: ProgressSink | None, **event: Any) -> None:
    if progress_sink is not None:
        progress_sink(event)
