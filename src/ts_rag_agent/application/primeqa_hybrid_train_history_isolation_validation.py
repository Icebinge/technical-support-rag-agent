from __future__ import annotations

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ts_rag_agent.application import (
    primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_protocol as stage160,
)
from ts_rag_agent.application.evidence_selection import tokenize_text
from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_runtime import (
    BoundedDynamicAgentRuntimeRun,
    PrimeQAHybridBoundedDynamicAgentRuntime,
)
from ts_rag_agent.application.primeqa_hybrid_bounded_dynamic_agent_service_entrypoint import (
    CanonicalBoundedDynamicAgentServicePaths,
    PrimeQAHybridBoundedDynamicAgentServiceEntrypoint,
)
from ts_rag_agent.application.primeqa_hybrid_structured_decision_router import (
    StructuredRouterPromptPolicy,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.domain.answer import GeneratedAnswer
from ts_rag_agent.domain.retrieval import RetrievalResult

from .primeqa_hybrid_train_history_isolation_protocol import (
    STAGE165_EXPECTED_ANSWERABLE_COUNT,
    STAGE165_EXPECTED_ARM_SCHEDULE_SHA256,
    STAGE165_EXPECTED_FOLD_ASSIGNMENT_SHA256,
    STAGE165_EXPECTED_GROUPING_SHA256,
    STAGE165_EXPECTED_ORDER_SHA256,
    STAGE165_EXPECTED_TRAIN_ROW_COUNT,
    STAGE165_EXPECTED_TRAIN_SHA256,
    STAGE165_EXPECTED_UNANSWERABLE_COUNT,
    STAGE165_FOLD_COUNT,
    Stage165Arm,
    Stage165ArmObservation,
    Stage165FoldAssignment,
    Stage165PairedWorkloadPlan,
    Stage165TrainSample,
    build_stage165_grouped_fold_assignment,
    build_stage165_paired_workload_plan,
    load_stage165_train_diagnostic_samples,
    stage165_private_report,
    summarize_stage165_pairs,
)

_STAGE = "Stage 165"
_CREATED_AT = "2026-07-19"
_ANALYSIS_ID = "primeqa_hybrid_train_history_isolation_paired_diagnostics_v1"
_PROTOCOL_ID = "primeqa_hybrid_stage165_full_train_paired_history_isolation_v1"
_EXPECTED_PAIR_COUNT = 562
_EXPECTED_ARM_ROW_COUNT = 1124
_EXPECTED_SYNTHETIC_THREAD_COUNT = 141
_EXPECTED_FULL_SYNTHETIC_THREAD_COUNT = 140
_EXPECTED_FIRST_TURN_COUNT = 141
_EXPECTED_POST_FIRST_COUNT = 421
_EXPECTED_MODEL_GENERATION_COUNT = 1125
_EXPECTED_SOURCE_HASHES = {
    "stage164_correction": "d80b786c32462cb9032e657ee1d1abc67f5cd995da66c1abd3831b3067c299fa",
    "train": STAGE165_EXPECTED_TRAIN_SHA256,
    "documents": "f93b5e2d8dcfb2c7d12676ef32ce22b7809692f14081aad98096099a5256722b",
    "stage157": "2351015d2c7447e6a5e1c2fe99b6583f0b9067e126ef2bfdd87b0b80c725c3e1",
    "router_source": "d9eeaff5fbb9c97a689efdee72d17f699cce47d1c94361047a74c90906442195",
    "runtime_source": "e3d38c5e81a86ac9454b2573fea455b94393f911238918df7b3247038273a071",
}
_EXPECTED_STAGE164_STATUS = "primeqa_hybrid_stage164_contract_correction_completed"
_EXPECTED_STAGE164_NEXT = "design_train_only_router_history_and_question_alignment_diagnostics"
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer",
        "answer_doc_id",
        "answer_text",
        "candidate_doc_ids",
        "cited_doc_ids",
        "document_id",
        "document_text",
        "document_title",
        "gold_answer",
        "matched_token_strings",
        "model_output",
        "question_id",
        "question_text",
        "question_title",
        "raw_answer_text",
        "raw_document_text",
        "raw_model_output",
        "raw_question_text",
        "retrieved_doc_ids",
        "sample_id",
    }
)

ProgressSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class Stage165ValidationRun:
    public_report: dict[str, Any]
    private_report: dict[str, Any]


@dataclass(frozen=True)
class Stage165Visualization:
    name: str
    path: str


class Stage165AgentSessionPort(Protocol):
    def open_thread(self, handle: str) -> None: ...

    def close_thread(self, handle: str) -> None: ...

    def measure_turn(
        self,
        *,
        handle: str,
        sample: Stage165TrainSample,
        fold_id: int,
        synthetic_thread_ordinal: int,
        synthetic_turn_position: int,
        arm: Stage165Arm,
        arm_order_position: int,
    ) -> Stage165ArmObservation: ...


class Stage165BoundedAgentSession:
    """Measure one train turn while joining gold only after runtime returns."""

    def __init__(
        self,
        *,
        runtime: PrimeQAHybridBoundedDynamicAgentRuntime,
        prompt_policy: StructuredRouterPromptPolicy,
    ) -> None:
        self._runtime = runtime
        self._prompt_policy = prompt_policy
        self._opened_handles: set[str] = set()
        self.open_count = 0
        self.close_count = 0

    @property
    def opened_thread_count(self) -> int:
        return len(self._opened_handles)

    def open_thread(self, handle: str) -> None:
        summary = self._runtime.open_thread(handle)
        if not summary.opened or summary.completed_turn_count != 0:
            raise RuntimeError("Stage165 thread did not open empty")
        self._opened_handles.add(handle)
        self.open_count += 1

    def close_thread(self, handle: str) -> None:
        summary = self._runtime.close_thread(handle)
        if summary.opened:
            raise RuntimeError("Stage165 thread remained open after close")
        self._opened_handles.remove(handle)
        self.close_count += 1

    def measure_turn(
        self,
        *,
        handle: str,
        sample: Stage165TrainSample,
        fold_id: int,
        synthetic_thread_ordinal: int,
        synthetic_turn_position: int,
        arm: Stage165Arm,
        arm_order_position: int,
    ) -> Stage165ArmObservation:
        before = self._runtime.thread_summary(handle)
        started_at = time.perf_counter()
        run = self._runtime.run_turn(
            opaque_thread_handle=handle,
            question=sample.runtime_query,
        )
        end_to_end_latency_ms = round((time.perf_counter() - started_at) * 1000, 3)
        after = self._runtime.thread_summary(handle)
        if after.completed_turn_count != before.completed_turn_count + 1:
            raise RuntimeError("Stage165 turn did not commit exactly once")
        return self._observation(
            sample=sample,
            run=run,
            fold_id=fold_id,
            synthetic_thread_ordinal=synthetic_thread_ordinal,
            synthetic_turn_position=synthetic_turn_position,
            arm=arm,
            arm_order_position=arm_order_position,
            history_turn_count_before=before.completed_turn_count,
            completed_turn_count_after=after.completed_turn_count,
            retained_state_bytes_after=after.retained_state_bytes,
            end_to_end_latency_ms=end_to_end_latency_ms,
        )

    def _observation(
        self,
        *,
        sample: Stage165TrainSample,
        run: BoundedDynamicAgentRuntimeRun,
        fold_id: int,
        synthetic_thread_ordinal: int,
        synthetic_turn_position: int,
        arm: Stage165Arm,
        arm_order_position: int,
        history_turn_count_before: int,
        completed_turn_count_after: int,
        retained_state_bytes_after: int,
        end_to_end_latency_ms: float,
    ) -> Stage165ArmObservation:
        state = run.workflow_run.final_state
        candidate_pool = tuple(run.candidate_pool_results)
        generation = tuple(state["generation_context_results"])
        verification = tuple(state["verification_context_results"])
        answer = run.verified_answer
        metrics = run.workflow_run.router_metrics
        if metrics is None:
            raise RuntimeError("Stage165 runtime turn returned no router metrics")
        gold_candidate = _first_document_result(candidate_pool, sample.gold_document_id)
        gold_generation = _first_document_result(generation, sample.gold_document_id)
        gold_verification = _first_document_result(verification, sample.gold_document_id)
        visibility = self._gold_prompt_visibility(sample=sample, result=gold_generation)
        trace = run.public_safe_trace
        return Stage165ArmObservation(
            private_identity_sha256=sample.private_identity_sha256,
            query_digest_sha256=sample.query_digest_sha256,
            diagnostic_group_sha256=sample.diagnostic_group_sha256,
            gold_document_sha256=sample.gold_document_sha256,
            fold_id=fold_id,
            synthetic_thread_ordinal=synthetic_thread_ordinal,
            synthetic_turn_position=synthetic_turn_position,
            arm=arm,
            arm_order_position=arm_order_position,
            answerable=sample.answerable,
            question_route=sample.question_route,
            split_subtype=sample.split_subtype,
            selected_action=str(trace.selected_action),
            terminal_state=trace.terminal_state,
            refused=answer.refused,
            history_turn_count_before=history_turn_count_before,
            completed_turn_count_after=completed_turn_count_after,
            retained_state_bytes_after=retained_state_bytes_after,
            candidate_pool_count=len(candidate_pool),
            generation_context_count=len(generation),
            verification_context_count=len(verification),
            candidate_context_sha256=_context_sha256(candidate_pool),
            generation_context_sha256=_context_sha256(generation),
            verification_context_sha256=_context_sha256(verification),
            output_sha256=_output_sha256(answer),
            gold_candidate_rank=(gold_candidate.rank if gold_candidate else None),
            gold_generation_rank=(gold_generation.rank if gold_generation else None),
            gold_verification_rank=(gold_verification.rank if gold_verification else None),
            gold_cited=(
                sample.gold_document_id is not None
                and any(
                    citation.document_id == sample.gold_document_id for citation in answer.citations
                )
            ),
            citation_count=len(answer.citations),
            answer_token_f1=(
                stage160.score_answer(answer.answer, sample.gold_answer, refused=answer.refused)
                if sample.answerable
                else None
            ),
            top_candidate_score=(candidate_pool[0].score if candidate_pool else None),
            gold_candidate_score=(gold_candidate.score if gold_candidate else None),
            question_token_recall_in_gold_prompt=visibility["question_recall"],
            answer_token_recall_in_gold_prompt=visibility["answer_recall"],
            answer_exact_span_visible=visibility["exact_visible"],
            router_input_token_count=metrics.input_token_count,
            router_output_token_count=metrics.output_token_count,
            router_generation_latency_ms=metrics.generation_latency_ms,
            end_to_end_latency_ms=end_to_end_latency_ms,
            retrieval_call_count=trace.retrieval_call_count,
            model_decision_count=trace.model_decision_count,
            composition_call_count=trace.composition_call_count,
            verification_call_count=trace.verification_call_count,
            diagnostic_observation_count=trace.diagnostic_observation_count,
            retry_action_count=trace.retry_action_count,
            fallback_action_count=trace.fallback_action_count,
        )

    def _gold_prompt_visibility(
        self,
        *,
        sample: Stage165TrainSample,
        result: RetrievalResult | None,
    ) -> dict[str, float | bool | None]:
        if result is None or not sample.answerable:
            return {
                "question_recall": None,
                "answer_recall": None,
                "exact_visible": None,
            }
        excerpt = result.document.text[: self._prompt_policy.max_evidence_chars_per_result]
        prompt_evidence = f"{result.document.title}\n{excerpt}"
        answer_tokens = tokenize_text(sample.gold_answer)
        question_tokens = tokenize_text(sample.runtime_query.full_question)
        prompt_tokens = tokenize_text(prompt_evidence)
        normalized_answer = _normalized_text(sample.gold_answer)
        normalized_prompt = _normalized_text(prompt_evidence)
        return {
            "question_recall": _multiset_recall(question_tokens, prompt_tokens),
            "answer_recall": _multiset_recall(answer_tokens, prompt_tokens),
            "exact_visible": bool(normalized_answer and normalized_answer in normalized_prompt),
        }


class Stage165PairedWorkloadExecutor:
    """Execute both arms once per sample without retry or recovery behavior."""

    def __init__(self, *, session: Stage165AgentSessionPort) -> None:
        self._session = session

    def execute(
        self,
        *,
        workload: Stage165PairedWorkloadPlan,
        folds: Stage165FoldAssignment,
        progress_sink: ProgressSink | None = None,
    ) -> tuple[Stage165ArmObservation, ...]:
        observations: list[Stage165ArmObservation] = []
        for thread in workload.threads:
            synthetic_handle = f"stage165-synthetic-{thread.ordinal:03d}"
            self._session.open_thread(synthetic_handle)
            try:
                for turn_position, sample in enumerate(thread.samples, start=1):
                    for arm_order_position, arm in enumerate(
                        workload.arm_order(sample),
                        start=1,
                    ):
                        if arm == "isolated":
                            isolated_handle = (
                                f"stage165-isolated-{thread.ordinal:03d}-{turn_position:02d}"
                            )
                            self._session.open_thread(isolated_handle)
                            try:
                                observation = self._session.measure_turn(
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
                                self._session.close_thread(isolated_handle)
                        else:
                            observation = self._session.measure_turn(
                                handle=synthetic_handle,
                                sample=sample,
                                fold_id=folds.fold_by_private_identity[
                                    sample.private_identity_sha256
                                ],
                                synthetic_thread_ordinal=thread.ordinal,
                                synthetic_turn_position=turn_position,
                                arm=arm,
                                arm_order_position=arm_order_position,
                            )
                        observations.append(observation)
            finally:
                self._session.close_thread(synthetic_handle)
            _emit(
                progress_sink,
                phase="paired_train_thread_completed",
                completed_thread_count=thread.ordinal,
                total_thread_count=len(workload.threads),
                completed_pair_count=sum(
                    len(item.samples) for item in workload.threads[: thread.ordinal]
                ),
                total_pair_count=len(workload.ordered_samples),
                completed_agent_turn_count=len(observations),
                total_agent_turn_count=len(workload.ordered_samples) * 2,
            )
        return tuple(observations)


def validate_primeqa_hybrid_train_history_isolation(
    *,
    settings: ProjectSettings,
    stage164_correction_path: Path,
    train_split_path: Path,
    user_confirmed_full_train_pairing: bool,
    confirmation_note: str,
    progress_sink: ProgressSink | None = None,
) -> Stage165ValidationRun:
    """Run the confirmed full-train paired Agent diagnostic once."""

    import torch

    if not user_confirmed_full_train_pairing:
        raise ValueError("Stage165 requires explicit full-train paired-run confirmation")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage165 formal paired diagnostics require CUDA")
    started_at = time.perf_counter()
    project_root = Path(__file__).resolve().parents[3]
    paths = CanonicalBoundedDynamicAgentServicePaths.from_settings(settings)
    source_authorization = _authorize_sources(
        project_root=project_root,
        stage164_correction_path=stage164_correction_path,
        train_split_path=train_split_path,
        paths=paths,
    )
    protocol = _frozen_protocol(confirmation_note=confirmation_note)
    protocol_sha = _canonical_json_sha256(protocol)
    current_source_before = _current_source_fingerprints(project_root)
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="sources_and_protocol_authorized")

    diagnostic_set = load_stage165_train_diagnostic_samples(train_split_path)
    workload = build_stage165_paired_workload_plan(diagnostic_set)
    folds = build_stage165_grouped_fold_assignment(diagnostic_set.samples)
    loaded_at = time.perf_counter()
    _emit(
        progress_sink,
        phase="train_paired_workload_loaded",
        total_pair_count=len(workload.ordered_samples),
        total_agent_turn_count=len(workload.ordered_samples) * 2,
        total_thread_count=len(workload.threads),
    )

    torch.cuda.reset_peak_memory_stats()
    entrypoint = PrimeQAHybridBoundedDynamicAgentServiceEntrypoint(
        settings=settings,
        paths=paths,
    )
    prepared = entrypoint.prepare()
    prepared_at = time.perf_counter()
    _emit(progress_sink, phase="single_runtime_prepared")

    session = Stage165BoundedAgentSession(
        runtime=prepared.runtime,
        prompt_policy=StructuredRouterPromptPolicy(),
    )
    observations = Stage165PairedWorkloadExecutor(session=session).execute(
        workload=workload,
        folds=folds,
        progress_sink=progress_sink,
    )
    executed_at = time.perf_counter()
    private_report = stage165_private_report(observations)
    private_sha = _canonical_json_sha256(private_report)
    diagnostics = summarize_stage165_pairs(observations)
    current_source_after = _current_source_fingerprints(project_root)
    finished_at = time.perf_counter()

    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "One confirmed full-train paired Agent run comparing isolated fresh threads "
            "with stable-hash synthetic four-turn history. Gold is joined only after each "
            "runtime turn. Grouped folds diagnose stability only; no model, threshold, or "
            "policy is fit or selected. Development and test are not loaded."
        ),
        "user_confirmation": {
            "full_train_pairing_confirmed": user_confirmed_full_train_pairing,
            "selected_option": "A",
            "confirmation_note": confirmation_note,
            "confirmed_unique_train_rows": _EXPECTED_PAIR_COUNT,
            "confirmed_agent_turns": _EXPECTED_ARM_ROW_COUNT,
        },
        "source_authorization": source_authorization,
        "frozen_protocol": protocol,
        "frozen_protocol_sha256": protocol_sha,
        "train_diagnostic_protocol": diagnostic_set.public_summary(),
        "workload_plan": workload.public_summary(),
        "grouped_fold_protocol": folds.public_summary(),
        "runtime": {
            "execution_mode": "direct_process_local_runtime_no_http_server",
            "single_runtime_instance": True,
            "resource_factory_build_count": prepared.resource_factory_build_count,
            "retrieval_encoder_device": prepared.retrieval_encoder_device,
            "model_generation_call_count": prepared.backend.generation_call_count,
            "warmup": prepared.warmup.to_public_dict(),
            "entrypoint_source_fingerprints": [
                fingerprint.to_public_dict() for fingerprint in prepared.source_fingerprints
            ],
            "gpu": {
                "available": bool(torch.cuda.is_available()),
                "device_name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            },
            "session": {
                "open_count": session.open_count,
                "close_count": session.close_count,
                "opened_thread_count_after_run": session.opened_thread_count,
            },
        },
        "paired_diagnostics": diagnostics,
        "private_diagnostic_artifact_contract": {
            "canonical_content_sha256": private_sha,
            "arm_row_count": len(observations),
            "pair_count": len(observations) // 2,
            "contains_hashed_sample_identity": True,
            "contains_raw_question": False,
            "contains_raw_answer": False,
            "contains_raw_document_id": False,
            "contains_raw_document_text": False,
            "contains_raw_model_output": False,
            "public_report_contains_case_rows": False,
            "git_policy": "ignored_local_artifact",
        },
        "execution_boundaries": {
            "train_loaded": True,
            "development_loaded": False,
            "test_loaded": False,
            "gold_projected_into_runtime": False,
            "agent_turns_run": len(observations),
            "retrieval_runs": len(observations),
            "model_fit": False,
            "threshold_tuned": False,
            "policy_selected": False,
            "runtime_registered_as_default": False,
            "remote_exposure": False,
            "http_server_started": False,
            "queue_actions_enabled": False,
            "retry_actions_enabled": False,
            "fallback_strategies_enabled": False,
            "query_rewrite_enabled": False,
            "second_retrieval_enabled": False,
        },
        "timing_seconds": {
            "source_authorization": round(authorized_at - started_at, 6),
            "train_load_and_plan": round(loaded_at - authorized_at, 6),
            "runtime_prepare": round(prepared_at - loaded_at, 6),
            "paired_agent_execution": round(executed_at - prepared_at, 6),
            "analysis_and_audit": round(finished_at - executed_at, 6),
            "total": round(finished_at - started_at, 6),
        },
        "current_source_fingerprints_before": current_source_before,
        "current_source_fingerprints_after": current_source_after,
    }
    report["guard_checks"] = _guard_checks(report, observations=observations)
    report["public_safe_contract"] = _public_safe_contract(report)
    report["decision"] = _decision(report)
    return Stage165ValidationRun(public_report=report, private_report=private_report)


def write_stage165_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage165Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = report["paired_diagnostics"]
    arms = diagnostics["arm_outcomes"]
    primary = diagnostics["primary_post_first_answerable_effect"]
    visible = diagnostics["gold_visible_post_first_answerable_effect"]
    safety = diagnostics["unanswerable_post_first_safety_effect"]
    first_turn = diagnostics["first_turn_negative_control"]
    folds = diagnostics["grouped_fold_stability"]
    alignment = diagnostics["question_alignment"]
    checks = report["guard_checks"]
    charts = {
        "stage165_arm_refusal_and_false_answer_rates.svg": _chart(
            "Stage165 paired arm outcome rates",
            [
                _bar(
                    "isolated answerable refusal",
                    arms["isolated"]["answerable_refusal_rate"],
                ),
                _bar(
                    "synthetic answerable refusal",
                    arms["synthetic_history"]["answerable_refusal_rate"],
                ),
                _bar(
                    "isolated unanswerable false answer",
                    arms["isolated"]["unanswerable_false_answer_rate"],
                ),
                _bar(
                    "synthetic unanswerable false answer",
                    arms["synthetic_history"]["unanswerable_false_answer_rate"],
                ),
            ],
            "rate",
        ),
        "stage165_post_first_refusal_transitions.svg": _chart(
            "Stage165 post-first answerable refusal transitions",
            [
                _bar(
                    "isolated answer -> synthetic refusal",
                    primary["isolated_answer_to_synthetic_refusal_count"],
                ),
                _bar(
                    "isolated refusal -> synthetic answer",
                    primary["isolated_refusal_to_synthetic_answer_count"],
                ),
                _bar("discordant pairs", primary["discordant_pair_count"]),
            ],
            "paired train rows",
        ),
        "stage165_gold_visible_refusal_transitions.svg": _chart(
            "Stage165 gold-visible post-first refusal transitions",
            [
                _bar(
                    "history harmed",
                    visible["isolated_answer_to_synthetic_refusal_count"],
                ),
                _bar(
                    "history helped",
                    visible["isolated_refusal_to_synthetic_answer_count"],
                ),
                _bar("visible pairs", visible["pair_count"]),
            ],
            "paired train rows",
        ),
        "stage165_unanswerable_safety_transitions.svg": _chart(
            "Stage165 post-first unanswerable safety transitions",
            [
                _bar(
                    "isolation worsened false answer",
                    safety["synthetic_refusal_to_isolated_false_answer_count"],
                ),
                _bar(
                    "isolation improved to refusal",
                    safety["synthetic_false_answer_to_isolated_refusal_count"],
                ),
                _bar("discordant pairs", safety["discordant_pair_count"]),
            ],
            "paired train rows",
        ),
        "stage165_first_turn_negative_control.svg": _chart(
            "Stage165 first-turn negative control",
            [
                _bar("pairs", first_turn["pair_count"]),
                _bar("context exact", first_turn["context_signature_exact_count"]),
                _bar("output exact", first_turn["output_exact_count"]),
                _bar("refusal disagreements", first_turn["refusal_disagreement_count"]),
            ],
            "paired train rows",
        ),
        "stage165_fold_answerable_refusal_deltas.svg": _chart(
            "Stage165 fold answerable refusal deltas",
            [
                _bar(
                    f"fold {fold_id}",
                    fold["post_first_answerable"][
                        "refusal_rate_difference_synthetic_minus_isolated"
                    ],
                )
                for fold_id, fold in folds["folds"].items()
            ],
            "synthetic minus isolated refusal rate",
        ),
        "stage165_fold_unanswerable_safety_deltas.svg": _chart(
            "Stage165 fold unanswerable false-answer deltas",
            [
                _bar(
                    f"fold {fold_id}",
                    fold["post_first_unanswerable"][
                        "false_answer_rate_difference_isolated_minus_synthetic"
                    ],
                )
                for fold_id, fold in folds["folds"].items()
            ],
            "isolated minus synthetic false-answer rate",
        ),
        "stage165_question_alignment_bins.svg": _chart(
            "Stage165 question-to-gold-prompt alignment bins",
            [
                _bar(
                    label,
                    values["refusal_rate_difference_synthetic_minus_isolated"],
                )
                for label, values in alignment["by_fixed_bin"].items()
            ],
            "synthetic minus isolated refusal rate",
        ),
        "stage165_arm_efficiency.svg": _chart(
            "Stage165 isolated-arm efficiency",
            [
                _bar(
                    "input token reduction",
                    primary["average_input_token_reduction_isolated_vs_synthetic"],
                ),
                _bar(
                    "generation ms reduction",
                    primary["average_generation_latency_reduction_isolated_vs_synthetic_ms"],
                ),
                _bar(
                    "answer F1 gain x1000",
                    primary["average_answer_f1_difference_isolated_minus_synthetic"] * 1000,
                ),
            ],
            "scaled diagnostic value",
        ),
        "stage165_guard_checks.svg": _chart(
            "Stage165 process guard checks",
            [_bar(str(check["name"]), bool(check["passed"])) for check in checks],
            "1 means passed",
            margin_left=650,
        ),
    }
    visualizations = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        visualizations.append(Stage165Visualization(name=filename, path=str(path)))
    return tuple(visualizations)


def _authorize_sources(
    *,
    project_root: Path,
    stage164_correction_path: Path,
    train_split_path: Path,
    paths: CanonicalBoundedDynamicAgentServicePaths,
) -> dict[str, Any]:
    source_paths = {
        "stage164_correction": stage164_correction_path,
        "train": train_split_path,
        "documents": paths.documents,
        "stage157": paths.stage157_validation,
        "router_source": paths.router_source,
        "runtime_source": paths.runtime_source,
    }
    fingerprints = {name: _fingerprint(path) for name, path in source_paths.items()}
    mismatches = [
        name
        for name, fingerprint in fingerprints.items()
        if fingerprint["sha256"] != _EXPECTED_SOURCE_HASHES[name]
    ]
    if mismatches:
        raise ValueError(f"Stage165 source authorization failed: {mismatches}")
    stage164 = _load_json_object(stage164_correction_path)
    decision = stage164.get("decision") or {}
    if not (
        decision.get("status") == _EXPECTED_STAGE164_STATUS
        and decision.get("next_direction") == _EXPECTED_STAGE164_NEXT
        and decision.get("policy_selected") is False
        and decision.get("test_gate_opened") is False
    ):
        raise ValueError("Stage165 Stage164 direction is not authorized")
    return {
        "authorized": True,
        "fingerprints": fingerprints,
        "stage164_status": decision.get("status"),
        "stage164_next_direction": decision.get("next_direction"),
        "train_only": True,
        "development_source_requested": False,
        "test_source_requested": False,
        "project_root": str(project_root),
    }


def _frozen_protocol(*, confirmation_note: str) -> dict[str, Any]:
    return {
        "protocol_id": _PROTOCOL_ID,
        "user_selected_option": "A_full_562_train_rows",
        "confirmation_note": confirmation_note,
        "cohort": {
            "split": "train",
            "unique_rows": _EXPECTED_PAIR_COUNT,
            "answerable_rows": STAGE165_EXPECTED_ANSWERABLE_COUNT,
            "unanswerable_rows": STAGE165_EXPECTED_UNANSWERABLE_COUNT,
            "arms_per_row": 2,
            "agent_turns": _EXPECTED_ARM_ROW_COUNT,
        },
        "arms": {
            "isolated": "fresh_explicit_thread_for_exactly_one_turn",
            "synthetic_history": "stable_hash_consecutive_groups_of_four",
        },
        "arm_order": "sha256_parity_balanced_per_sample",
        "negative_control": "turn_position_1_has_identical_empty_history",
        "primary_cohort": "answerable_synthetic_turn_position_2_through_4",
        "primary_metric": "refusal_rate_difference_synthetic_minus_isolated",
        "safety_cohort": "unanswerable_synthetic_turn_position_2_through_4",
        "safety_metric": "false_answer_rate_difference_isolated_minus_synthetic",
        "gold_visible_secondary_cohort": (
            "answerable_post_first_gold_generation_context_visible_in_both_arms"
        ),
        "alignment_metric": "question_token_recall_in_gold_prompt",
        "alignment_bins": [0.0, 0.25, 0.5, 0.75, 1.0],
        "alignment_low_high_boundary": 0.5,
        "grouped_folds": {
            "count": STAGE165_FOLD_COUNT,
            "group_key": "normalized_question_plus_answer_document_or_unanswerable",
            "purpose": "diagnostic_direction_stability_only",
        },
        "candidate_eligibility_requires": [
            "all_process_guards_pass",
            "first_turn_negative_control_exact",
            "paired_context_signatures_exact",
            "aggregate_post_first_answerable_refusal_reduction",
            "every_fold_post_first_answerable_refusal_nonregression",
            "aggregate_answer_f1_nonregression",
            "aggregate_gold_citation_nonregression",
            "aggregate_and_every_fold_unanswerable_false_answer_nonregression",
        ],
        "fit_models": False,
        "tune_thresholds": False,
        "select_runtime_policy": False,
        "development_loaded": False,
        "test_loaded": False,
        "runtime_registered_as_default": False,
        "remote_exposure": False,
        "queue": False,
        "retry": False,
        "fallback": False,
        "query_rewrite": False,
        "second_retrieval": False,
    }


def _guard_checks(
    report: Mapping[str, Any],
    *,
    observations: Sequence[Stage165ArmObservation],
) -> list[dict[str, Any]]:
    confirmation = report["user_confirmation"]
    source = report["source_authorization"]
    train = report["train_diagnostic_protocol"]
    workload = report["workload_plan"]
    folds = report["grouped_fold_protocol"]
    runtime = report["runtime"]
    diagnostics = report["paired_diagnostics"]
    overview = diagnostics["overview"]
    first_turn = diagnostics["first_turn_negative_control"]
    boundaries = report["execution_boundaries"]
    arm_counts = Counter(row.arm for row in observations)
    histories_exact = all(
        row.history_turn_count_before
        == (0 if row.arm == "isolated" else row.synthetic_turn_position - 1)
        for row in observations
    )
    trace_counts_exact = all(
        row.retrieval_call_count == 1
        and row.model_decision_count == 1
        and row.retry_action_count == 0
        and row.fallback_action_count == 0
        for row in observations
    )
    return [
        _check(
            "user_confirmed_option_a_full_train",
            confirmation.get("full_train_pairing_confirmed") is True
            and confirmation.get("selected_option") == "A",
        ),
        _check(
            "frozen_protocol_identity_exact",
            report.get("frozen_protocol_sha256")
            == _canonical_json_sha256(report.get("frozen_protocol")),
        ),
        _check(
            "upstream_sources_exact",
            source.get("authorized") is True
            and all(
                source["fingerprints"][name]["sha256"] == expected
                for name, expected in _EXPECTED_SOURCE_HASHES.items()
            ),
        ),
        _check(
            "stage164_direction_exact",
            source.get("stage164_status") == _EXPECTED_STAGE164_STATUS
            and source.get("stage164_next_direction") == _EXPECTED_STAGE164_NEXT,
        ),
        _check(
            "train_source_and_counts_exact",
            train.get("source_sha256") == STAGE165_EXPECTED_TRAIN_SHA256
            and train.get("train_row_count") == STAGE165_EXPECTED_TRAIN_ROW_COUNT
            and train.get("answerable_count") == STAGE165_EXPECTED_ANSWERABLE_COUNT
            and train.get("unanswerable_count") == STAGE165_EXPECTED_UNANSWERABLE_COUNT
            and train.get("stable_order_sha256") == STAGE165_EXPECTED_ORDER_SHA256,
        ),
        _check(
            "train_only_split_boundary_exact",
            train.get("assigned_split") == "train"
            and train.get("dev_loaded") is False
            and train.get("test_loaded") is False,
        ),
        _check(
            "paired_workload_shape_exact",
            workload.get("unique_sample_count") == _EXPECTED_PAIR_COUNT
            and workload.get("agent_turn_count") == _EXPECTED_ARM_ROW_COUNT
            and workload.get("thread_count") == _EXPECTED_SYNTHETIC_THREAD_COUNT
            and workload.get("full_four_turn_thread_count") == _EXPECTED_FULL_SYNTHETIC_THREAD_COUNT
            and workload.get("trailing_thread_turn_count") == 2
            and workload.get("first_turn_negative_control_count") == _EXPECTED_FIRST_TURN_COUNT
            and workload.get("post_first_turn_primary_count") == _EXPECTED_POST_FIRST_COUNT
            and workload.get("grouping_sha256") == STAGE165_EXPECTED_GROUPING_SHA256
            and workload.get("arm_schedule_sha256") == STAGE165_EXPECTED_ARM_SCHEDULE_SHA256,
        ),
        _check(
            "arm_schedule_contains_both_orders",
            all(int(value) > 0 for value in workload.get("arm_first_counts", {}).values()),
        ),
        _check(
            "grouped_five_fold_isolation",
            folds.get("fold_count") == STAGE165_FOLD_COUNT
            and folds.get("group_count") == sum(folds.get("group_counts", {}).values())
            and sum(folds.get("row_counts", {}).values()) == _EXPECTED_PAIR_COUNT
            and folds.get("fit_models") is False
            and folds.get("select_policy") is False
            and folds.get("tune_thresholds") is False
            and folds.get("assignment_sha256") == STAGE165_EXPECTED_FOLD_ASSIGNMENT_SHA256,
        ),
        _check(
            "arm_observations_and_pairs_exact",
            len(observations) == _EXPECTED_ARM_ROW_COUNT
            and arm_counts
            == {"isolated": _EXPECTED_PAIR_COUNT, "synthetic_history": _EXPECTED_PAIR_COUNT}
            and overview.get("pair_count") == _EXPECTED_PAIR_COUNT
            and overview.get("arm_observation_count") == _EXPECTED_ARM_ROW_COUNT,
        ),
        _check("history_state_assignment_exact", histories_exact),
        _check(
            "paired_retrieval_contexts_exact",
            overview.get("context_signature_exact_count") == _EXPECTED_PAIR_COUNT,
        ),
        _check(
            "first_turn_negative_control_exact",
            first_turn.get("pair_count") == _EXPECTED_FIRST_TURN_COUNT
            and first_turn.get("context_signature_exact_count") == _EXPECTED_FIRST_TURN_COUNT
            and first_turn.get("output_exact_count") == _EXPECTED_FIRST_TURN_COUNT
            and first_turn.get("refusal_disagreement_count") == 0
            and first_turn.get("average_input_token_difference_synthetic_minus_isolated") == 0.0,
        ),
        _check("one_retrieval_and_model_decision_per_turn", trace_counts_exact),
        _check(
            "single_runtime_resource_and_model_load",
            runtime.get("single_runtime_instance") is True
            and runtime.get("resource_factory_build_count") == 1
            and runtime.get("model_generation_call_count") == _EXPECTED_MODEL_GENERATION_COUNT,
        ),
        _check(
            "thread_lifecycle_closed_exact",
            runtime["session"]["open_count"] == runtime["session"]["close_count"]
            and runtime["session"]["open_count"]
            == _EXPECTED_PAIR_COUNT + _EXPECTED_SYNTHETIC_THREAD_COUNT
            and runtime["session"]["opened_thread_count_after_run"] == 0,
        ),
        _check(
            "current_sources_unchanged_during_run",
            report.get("current_source_fingerprints_before")
            == report.get("current_source_fingerprints_after"),
        ),
        _check(
            "private_artifact_content_free",
            report["private_diagnostic_artifact_contract"]["arm_row_count"]
            == _EXPECTED_ARM_ROW_COUNT
            and all(
                report["private_diagnostic_artifact_contract"][key] is False
                for key in (
                    "contains_raw_question",
                    "contains_raw_answer",
                    "contains_raw_document_id",
                    "contains_raw_document_text",
                    "contains_raw_model_output",
                    "public_report_contains_case_rows",
                )
            ),
        ),
        _check(
            "no_fit_tuning_or_policy_selection",
            boundaries.get("model_fit") is False
            and boundaries.get("threshold_tuned") is False
            and boundaries.get("policy_selected") is False,
        ),
        _check(
            "development_test_and_runtime_default_closed",
            boundaries.get("development_loaded") is False
            and boundaries.get("test_loaded") is False
            and boundaries.get("runtime_registered_as_default") is False
            and boundaries.get("remote_exposure") is False
            and boundaries.get("http_server_started") is False,
        ),
        _check(
            "queue_retry_fallback_rewrite_second_retrieval_closed",
            boundaries.get("queue_actions_enabled") is False
            and boundaries.get("retry_actions_enabled") is False
            and boundaries.get("fallback_strategies_enabled") is False
            and boundaries.get("query_rewrite_enabled") is False
            and boundaries.get("second_retrieval_enabled") is False,
        ),
    ]


def _decision(report: Mapping[str, Any]) -> dict[str, Any]:
    checks = report["guard_checks"]
    all_passed = all(check.get("passed") is True for check in checks)
    diagnostics = report["paired_diagnostics"]
    primary = diagnostics["primary_post_first_answerable_effect"]
    safety = diagnostics["unanswerable_post_first_safety_effect"]
    fold = diagnostics["grouped_fold_stability"]
    primary_fold = fold["primary_answerable_refusal_delta_direction"]
    safety_fold = fold["unanswerable_false_answer_delta_direction"]
    history_signal_stable = (
        float(primary["refusal_rate_difference_synthetic_minus_isolated"]) > 0
        and int(primary_fold["negative_count"]) == 0
        and int(primary_fold["positive_count"]) > 0
    )
    quality_nonregression = (
        float(primary["average_answer_f1_difference_isolated_minus_synthetic"]) >= 0
        and int(primary["gold_citation_difference_isolated_minus_synthetic"]) >= 0
    )
    safety_nonregression = (
        float(safety["false_answer_rate_difference_isolated_minus_synthetic"]) <= 0
        and int(safety_fold["positive_count"]) == 0
    )
    candidate_eligible = (
        all_passed and history_signal_stable and quality_nonregression and safety_nonregression
    )
    if not all_passed:
        status = "primeqa_hybrid_train_history_isolation_diagnostics_invalid"
        next_direction = "repair_stage165_process_without_using_dev_or_test"
    elif candidate_eligible:
        status = "primeqa_hybrid_train_history_isolation_candidate_train_safe"
        next_direction = "freeze_one_history_isolation_candidate_for_development_validation"
    else:
        status = "primeqa_hybrid_train_history_isolation_not_train_safe"
        next_direction = "stop_history_isolation_candidate_and_reassess_train_only_alignment"
    return {
        "status": status,
        "all_process_guards_passed": all_passed,
        "failed_process_guards": [
            check["name"] for check in checks if check.get("passed") is not True
        ],
        "history_contamination_signal_fold_stable": history_signal_stable,
        "isolated_answer_quality_nonregression": quality_nonregression,
        "isolated_unanswerable_safety_nonregression": safety_nonregression,
        "candidate_eligible_for_frozen_dev_validation": candidate_eligible,
        "diagnostic_only": True,
        "causal_scope": "synthetic_unrelated_history_only",
        "natural_conversation_claim": False,
        "policy_selected": False,
        "development_gate_opened": candidate_eligible,
        "test_gate_opened": False,
        "runtime_registered_as_default": False,
        "fallback_strategies_enabled": False,
        "next_direction": next_direction,
    }


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = sorted(_forbidden_keys_found(report))
    return {
        "forbidden_keys_found": forbidden,
        "contains_case_rows": False,
        "contains_raw_question": False,
        "contains_raw_answer": False,
        "contains_raw_document": False,
        "contains_raw_model_output": False,
        "raw_candidate_rows_written": False,
        "public_safe": not forbidden,
    }


def _forbidden_keys_found(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_PUBLIC_KEYS:
                found.add(normalized)
            found.update(_forbidden_keys_found(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            found.update(_forbidden_keys_found(item))
    return found


def _current_source_fingerprints(project_root: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "protocol": (
            project_root
            / "src"
            / "ts_rag_agent"
            / "application"
            / "primeqa_hybrid_train_history_isolation_protocol.py"
        ),
        "validation": Path(__file__).resolve(),
        "cli": (project_root / "scripts" / "analyze_primeqa_hybrid_train_history_isolation.py"),
    }
    return {name: _fingerprint(path) for name, path in paths.items()}


def _context_sha256(results: Sequence[RetrievalResult]) -> str:
    rows = [
        {
            "document_id": result.document.id,
            "rank": result.rank,
            "score": round(float(result.score), 12),
        }
        for result in results
    ]
    return _canonical_json_sha256(rows)


def _output_sha256(answer: GeneratedAnswer) -> str:
    return _canonical_json_sha256(
        {
            "answer": answer.answer,
            "refused": answer.refused,
            "citations": [citation.document_id for citation in answer.citations],
        }
    )


def _first_document_result(
    results: Sequence[RetrievalResult],
    document_id: str | None,
) -> RetrievalResult | None:
    if document_id is None:
        return None
    return next((result for result in results if result.document.id == document_id), None)


def _multiset_recall(needles: Sequence[str], haystack: Sequence[str]) -> float:
    if not needles:
        return 0.0
    required = Counter(needles)
    available = Counter(haystack)
    matched = sum(min(count, available[token]) for token, count in required.items())
    return round(matched / sum(required.values()), 6)


def _normalized_text(value: str) -> str:
    return " ".join(tokenize_text(value))


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


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _emit(progress_sink: ProgressSink | None, **event: Any) -> None:
    if progress_sink is not None:
        progress_sink({"stage": _STAGE, **event})


def _bar(label: str, value: int | float | bool) -> BarDatum:
    number = float(value)
    return BarDatum(label=label, value=number, value_label=str(value))


def _chart(
    title: str,
    rows: Sequence[BarDatum],
    x_label: str,
    *,
    margin_left: int = 430,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=rows,
        x_label=x_label,
        margin_left=margin_left,
    )
