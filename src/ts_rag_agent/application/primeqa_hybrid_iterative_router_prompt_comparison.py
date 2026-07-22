from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ts_rag_agent.application import primeqa_hybrid_iterative_router_calibration as stage169
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    _build_dense_channels,
    _build_lexical_channels,
    _build_train_fold_assignments,
    _load_json_object,
)
from ts_rag_agent.application.primeqa_hybrid_iterative_decision_router import (
    IterativeDecisionAction,
    IterativeDecisionPhase,
    IterativeRouterInstructionPolicy,
    IterativeRouterPromptBuilder,
    StrictIterativeStructuredDecisionRouter,
    stage170_challenger_instruction_policies,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector import (
    records_by_sample,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector_training import (
    Stage161TrainCandidateDatasetBuilder,
)
from ts_rag_agent.application.primeqa_hybrid_structured_decision_router import (
    Qwen3VLTransformersTextGenerationBackend,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    load_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)

_STAGE = "Stage 170"
_CREATED_AT = "2026-07-22"
_ANALYSIS_ID = "primeqa_hybrid_iterative_router_prompt_comparison_v1"
_EXPECTED_TRAIN_ROWS = 562
_EXPECTED_CANDIDATE_ROWS = 112_400
_EXPECTED_PROFILE_COUNT = 3
_TRAIN_FINALIST_COUNT = 2
_EXPECTED_MODEL_CALLS = 254
_SOURCE_HASHES = {
    "stage169": "aa1f66d64ecf901d811c8f4db436b88f3fd416f91f0d9078c8d37f2174b06ad1",
    "stage80": "2441bb1cb1e7888299d3f57962b18cd59df84e2086ac281105abcacfc144880f",
    "train": "cabd93e0b972c47384c4bf5cc2cd215a7fc519b2df4f81fba61db73c931aa155",
    "documents": "f93b5e2d8dcfb2c7d12676ef32ce22b7809692f14081aad98096099a5256722b",
    "model_config": "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b",
    "model_weights": "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0",
    "model_tokenizer": "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
}

ProgressSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class ProfileTrainOutcome:
    stratum: str
    fold_id: str
    initial: stage169.RouterCallObservation
    final: stage169.RouterCallObservation


@dataclass(frozen=True)
class Stage170Visualization:
    name: str
    path: str


def rank_synthetic_profiles(
    profile_results: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            profile_results,
            key=lambda profile_id: (
                -float(profile_results[profile_id]["phase_action_accuracy"]),
                -float(profile_results[profile_id]["clarification_kind_accuracy"]),
                -float(profile_results[profile_id]["exact_path_accuracy"]),
                profile_id,
            ),
        )
    )


def run_stage170_prompt_comparison(
    *,
    stage169_report_path: Path,
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
        "stage169": stage169_report_path,
        "stage80": stage80_report_path,
        "train": train_split_path,
        "documents": documents_path,
        "model_config": model_snapshot_path / "config.json",
        "model_weights": model_snapshot_path / "model.safetensors",
        "model_tokenizer": model_snapshot_path / "tokenizer.json",
    }
    fingerprints = {name: stage169._fingerprint(path) for name, path in source_paths.items()}
    _authorize_sources(fingerprints)
    baseline = _load_json_object(stage169_report_path)
    if baseline.get("decision", {}).get("status") != "stage169_router_requires_redesign":
        raise ValueError("Stage169 did not authorize prompt redesign")
    fingerprinted_at = time.perf_counter()
    tracker.capture("sources_fingerprinted")
    _emit(progress_sink, phase="sources_authorized")

    samples = load_primeqa_hybrid_split_samples(train_split_path)
    if len(samples) != _EXPECTED_TRAIN_ROWS or any(
        sample.assigned_split != "train" for sample in samples
    ):
        raise ValueError("Stage170 accepts only the exact 562-row train split")
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
        raise RuntimeError("Stage170 requires both authorized local dense channels")
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
        raise RuntimeError("Stage170 candidate replay row count drifted")
    train_cases = stage169.select_train_calibration_cases(
        samples=samples,
        grouped_records=records_by_sample(records),
        documents_by_id=documents_by_id,
    )
    evidence_ready_at = time.perf_counter()
    tracker.capture("train_evidence_ready")
    _emit(progress_sink, phase="train_evidence_ready", selected_cases=len(train_cases))

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage170 requires CUDA with bfloat16 support")
    backend = Qwen3VLTransformersTextGenerationBackend.load_local(snapshot_path=model_snapshot_path)
    model_loaded_at = time.perf_counter()
    tracker.capture("router_model_loaded")
    _emit(progress_sink, phase="router_model_loaded")

    policies = stage170_challenger_instruction_policies()
    if len(policies) != _EXPECTED_PROFILE_COUNT:
        raise RuntimeError("Stage170 challenger profile count drifted")
    synthetic_results: dict[str, dict[str, Any]] = {}
    synthetic_calls: dict[str, tuple[stage169.RouterCallObservation, ...]] = {}
    for index, policy in enumerate(policies, start=1):
        router = _router(backend, policy)
        result, calls = stage169._run_synthetic_calibration(router, tracker, None)
        synthetic_results[policy.profile_id] = result
        synthetic_calls[policy.profile_id] = calls
        _emit(
            progress_sink,
            phase="synthetic_profile_complete",
            completed=index,
            total=len(policies),
            profile_id=policy.profile_id,
        )
    synthetic_finished_at = time.perf_counter()
    ranked_profiles = rank_synthetic_profiles(synthetic_results)
    finalist_ids = ranked_profiles[:_TRAIN_FINALIST_COUNT]
    policies_by_id = {policy.profile_id: policy for policy in policies}
    _emit(progress_sink, phase="synthetic_screen_complete", finalists=list(finalist_ids))

    profile_reports: dict[str, dict[str, Any]] = {}
    all_calls: list[stage169.RouterCallObservation] = [
        call for profile_id in ranked_profiles for call in synthetic_calls[profile_id]
    ]
    for index, profile_id in enumerate(finalist_ids, start=1):
        outcomes, train_calls = _run_profile_train_cases(
            router=_router(backend, policies_by_id[profile_id]),
            tracker=tracker,
            cases=train_cases,
        )
        train_summary = _aggregate_train_outcomes(outcomes)
        combined_calls = (*synthetic_calls[profile_id], *train_calls)
        quality_metrics = stage169._quality_metrics(
            synthetic_results[profile_id], train_summary, combined_calls
        )
        quality_gates = stage169._quality_gates(quality_metrics)
        profile_reports[profile_id] = {
            "synthetic_rank": ranked_profiles.index(profile_id) + 1,
            "synthetic_calibration": synthetic_results[profile_id],
            "train_proxy_calibration": train_summary,
            "fold_stability": _fold_stability(outcomes),
            "quality_metrics": quality_metrics,
            "quality_gates": quality_gates,
            "quality_gate_pass_count": sum(gate["passed"] for gate in quality_gates),
            "all_quality_gates_passed": all(gate["passed"] for gate in quality_gates),
        }
        all_calls.extend(train_calls)
        _emit(
            progress_sink,
            phase="train_profile_complete",
            completed=index,
            total=len(finalist_ids),
            profile_id=profile_id,
        )
    comparison_finished_at = time.perf_counter()
    tracker.capture("comparison_complete")

    selected_profile_id = _select_final_profile(profile_reports)
    selected = profile_reports[selected_profile_id]
    process_guards = [
        _check("stage169_authorized_redesign", True),
        _check("exact_train_row_count", len(samples) == _EXPECTED_TRAIN_ROWS),
        _check("exact_candidate_row_count", len(records) == _EXPECTED_CANDIDATE_ROWS),
        _check("three_frozen_profiles", len(policies) == _EXPECTED_PROFILE_COUNT),
        _check("two_train_finalists", len(finalist_ids) == _TRAIN_FINALIST_COUNT),
        _check("fifty_train_cases", len(train_cases) == 50),
        _check("model_call_count_exact", backend.generation_call_count == _EXPECTED_MODEL_CALLS),
        _check("local_files_only", True),
        _check("development_not_loaded", True),
        _check("test_not_loaded", True),
        _check("answer_generation_not_run", True),
        _check("retry_count_zero", True),
        _check("default_runtime_unchanged", True),
    ]
    all_process_guards_passed = all(guard["passed"] for guard in process_guards)
    all_quality_gates_passed = selected["all_quality_gates_passed"]
    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "validation_scope": (
            "Synthetic-first comparison of three frozen instruction profiles followed by one "
            "train-only evaluation of the top two profiles on the exact Stage169 identities."
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
            "profile_ids": [policy.profile_id for policy in policies],
            "synthetic_screen_case_count_per_profile": 14,
            "synthetic_phase_calls_per_profile": 18,
            "train_finalist_count": _TRAIN_FINALIST_COUNT,
            "train_cases_per_finalist": 50,
            "train_phase_calls_per_finalist": 100,
            "expected_model_call_count": _EXPECTED_MODEL_CALLS,
            "selection_uses_synthetic_only": True,
            "train_used_once_after_synthetic_selection": True,
            "quality_thresholds": stage169._QUALITY_THRESHOLDS,
            "prompt_evidence_and_token_policy_held_constant": True,
        },
        "stage169_baseline": {
            "quality_metrics": baseline["quality_metrics"],
            "quality_gate_pass_count": sum(gate["passed"] for gate in baseline["quality_gates"]),
        },
        "synthetic_screen": {
            "ranked_profile_ids": list(ranked_profiles),
            "selected_train_finalists": list(finalist_ids),
            "profiles": synthetic_results,
        },
        "train_comparison": profile_reports,
        "selected_profile_id": selected_profile_id,
        "resource_consumption": stage169._resource_consumption_summary(
            tracker=tracker,
            calls=all_calls,
            wall_time_seconds=comparison_finished_at - started_at,
        ),
        "model_runtime": {
            "snapshot_revision": backend.snapshot_path.name,
            "generation_call_count": backend.generation_call_count,
            "peak_gpu_memory_bytes": max(call.gpu_peak_allocated_bytes for call in all_calls),
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
            "runtime_registered_as_default": False,
        },
        "timing_seconds": {
            "source_fingerprinting": round(fingerprinted_at - started_at, 6),
            "train_evidence_build": round(evidence_ready_at - fingerprinted_at, 6),
            "model_load": round(model_loaded_at - evidence_ready_at, 6),
            "synthetic_screen": round(synthetic_finished_at - model_loaded_at, 6),
            "train_comparison": round(comparison_finished_at - synthetic_finished_at, 6),
            "total_before_visualization": round(comparison_finished_at - started_at, 6),
        },
        "process_guards": process_guards,
        "decision": {
            "all_process_guards_passed": all_process_guards_passed,
            "all_quality_gates_passed": all_quality_gates_passed,
            "status": (
                "advance_to_stage171_train_only_iterative_runtime_e2e"
                if all_process_guards_passed and all_quality_gates_passed
                else "stage170_prompt_family_insufficient"
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
        report["decision"]["status"] = "stage170_process_invalid"
    _emit(progress_sink, phase="comparison_complete", decision=report["decision"])
    return report


def write_stage170_visualizations(
    *, report: Mapping[str, Any], output_dir: Path
) -> tuple[Stage170Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    screen = report["synthetic_screen"]
    comparison = report["train_comparison"]
    resources = report["resource_consumption"]
    charts = {
        "synthetic_profile_accuracy.svg": _chart(
            "Stage 170 synthetic profile action accuracy",
            tuple(
                _rate_bar(profile_id, screen["profiles"][profile_id]["phase_action_accuracy"])
                for profile_id in screen["ranked_profile_ids"]
            ),
        ),
        "quality_gate_comparison.svg": _chart(
            "Stage 170 frozen quality gates passed",
            (
                BarDatum(
                    "Stage169 baseline",
                    report["stage169_baseline"]["quality_gate_pass_count"],
                    f"{report['stage169_baseline']['quality_gate_pass_count']} / 8",
                ),
                *(
                    BarDatum(
                        profile_id,
                        comparison[profile_id]["quality_gate_pass_count"],
                        f"{comparison[profile_id]['quality_gate_pass_count']} / 8",
                    )
                    for profile_id in screen["selected_train_finalists"]
                ),
            ),
            x_label="Passed gates",
        ),
        "train_proxy_comparison.svg": _chart(
            "Stage 170 finalist train proxy rates",
            tuple(
                _rate_bar(
                    f"{profile_id}: {metric.replace('real_', '').replace('_rate', '')}",
                    comparison[profile_id]["quality_metrics"][metric],
                )
                for profile_id in screen["selected_train_finalists"]
                for metric in (
                    "real_initial_visible_compose_rate",
                    "real_alternate_only_inspect_rate",
                    "real_alternate_only_final_compose_rate",
                    "real_alternate_only_path_success_rate",
                    "real_insufficient_final_compose_rate",
                )
            ),
        ),
        "latency_comparison.svg": _chart(
            "Stage 170 finalist p95 generation latency",
            tuple(
                BarDatum(
                    profile_id,
                    comparison[profile_id]["quality_metrics"]["latency_ms"]["p95"],
                    f"{comparison[profile_id]['quality_metrics']['latency_ms']['p95']:.1f} ms",
                )
                for profile_id in screen["selected_train_finalists"]
            ),
            x_label="Milliseconds",
        ),
        "resource_peaks.svg": _chart(
            "Stage 170 process and GPU resource peaks",
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
        written.append(Stage170Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _router(
    backend: Qwen3VLTransformersTextGenerationBackend,
    policy: IterativeRouterInstructionPolicy,
) -> StrictIterativeStructuredDecisionRouter:
    return StrictIterativeStructuredDecisionRouter(
        backend=backend,
        prompt_builder=IterativeRouterPromptBuilder(instruction_policy=policy),
    )


def _run_profile_train_cases(
    *,
    router: StrictIterativeStructuredDecisionRouter,
    tracker: stage169.Stage169ResourceTracker,
    cases: Sequence[stage169.TrainCalibrationCase],
) -> tuple[tuple[ProfileTrainOutcome, ...], tuple[stage169.RouterCallObservation, ...]]:
    outcomes = []
    calls = []
    for case in cases:
        initial = stage169._invoke_router(
            router,
            phase=IterativeDecisionPhase.INITIAL,
            question=case.question,
            initial=case.initial_evidence,
            alternate=(),
            tracker=tracker,
        )
        final = stage169._invoke_router(
            router,
            phase=IterativeDecisionPhase.FINAL_AFTER_INSPECTION,
            question=case.question,
            initial=case.initial_evidence,
            alternate=case.alternate_evidence,
            tracker=tracker,
        )
        outcomes.append(ProfileTrainOutcome(case.stratum, case.fold_id, initial, final))
        calls.extend((initial, final))
    return tuple(outcomes), tuple(calls)


def _aggregate_train_outcomes(
    outcomes: Sequence[ProfileTrainOutcome],
) -> dict[str, dict[str, Any]]:
    report = {}
    for stratum in stage169._TRAIN_STRATA:
        rows = [outcome for outcome in outcomes if outcome.stratum == stratum]
        report[stratum] = {
            "case_count": len(rows),
            "initial_action_counts": _action_counts(row.initial.action for row in rows),
            "final_action_counts": _action_counts(row.final.action for row in rows),
            "initial_schema_valid_count": sum(row.initial.schema_valid for row in rows),
            "final_schema_valid_count": sum(row.final.schema_valid for row in rows),
            "inspect_then_compose_count": sum(
                row.initial.action == IterativeDecisionAction.INSPECT.value
                and row.final.action == IterativeDecisionAction.COMPOSE.value
                for row in rows
            ),
        }
    return report


def _fold_stability(outcomes: Sequence[ProfileTrainOutcome]) -> dict[str, Any]:
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
        folds[str(fold_id)] = {
            "case_count": len(rows),
            "schema_valid_rate": _rate(
                sum(row.initial.schema_valid + row.final.schema_valid for row in rows),
                len(rows) * 2,
            ),
            "initial_visible_compose_rate": _rate(
                sum(
                    row.initial.action == IterativeDecisionAction.COMPOSE.value
                    for row in initial_visible
                ),
                len(initial_visible),
            ),
            "alternate_only_inspect_rate": _rate(
                sum(
                    row.initial.action == IterativeDecisionAction.INSPECT.value
                    for row in alternate_only
                ),
                len(alternate_only),
            ),
            "alternate_only_final_compose_rate": _rate(
                sum(
                    row.final.action == IterativeDecisionAction.COMPOSE.value
                    for row in alternate_only
                ),
                len(alternate_only),
            ),
            "insufficient_final_compose_rate": _rate(
                sum(
                    row.final.action == IterativeDecisionAction.COMPOSE.value
                    for row in insufficient
                ),
                len(insufficient),
            ),
        }
    return {"fold_count": len(folds), "folds": folds}


def _select_final_profile(profile_reports: Mapping[str, Mapping[str, Any]]) -> str:
    return min(
        profile_reports,
        key=lambda profile_id: (
            -int(profile_reports[profile_id]["quality_gate_pass_count"]),
            -float(
                profile_reports[profile_id]["quality_metrics"]["synthetic_phase_action_accuracy"]
            ),
            -float(
                profile_reports[profile_id]["quality_metrics"]["real_initial_visible_compose_rate"]
            ),
            profile_id,
        ),
    )


def _action_counts(actions: Any) -> dict[str, int]:
    counts = Counter(action if action is not None else "schema_invalid" for action in actions)
    return dict(sorted(counts.items()))


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _authorize_sources(fingerprints: Mapping[str, Mapping[str, Any]]) -> None:
    for name, expected in _SOURCE_HASHES.items():
        observed = fingerprints.get(name, {}).get("sha256")
        if observed != expected:
            raise ValueError(f"Stage170 source hash mismatch for {name}: {observed}")


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
        margin_left=520,
        margin_right=180,
    )
