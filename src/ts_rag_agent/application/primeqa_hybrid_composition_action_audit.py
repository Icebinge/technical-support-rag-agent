from __future__ import annotations

import json
import statistics
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any

from ts_rag_agent.application import primeqa_hybrid_citation_aware_composition_cv as stage180
from ts_rag_agent.application import primeqa_hybrid_iterative_router_calibration as stage169
from ts_rag_agent.application import primeqa_hybrid_listwise_agent_e2e as stage178
from ts_rag_agent.application import primeqa_hybrid_semantic_evidence_cv as stage173
from ts_rag_agent.application.answer_verification import AnswerVerifier
from ts_rag_agent.application.composition_action_audit import (
    ActionAuditRow,
    build_action_runtime_features,
    classify_action_outcome,
    enumerate_atomic_composition_actions,
    run_action_predictability_oof,
    stage180_action_summary,
    summarize_action_rows,
)
from ts_rag_agent.application.evidence_selection import classify_question_route
from ts_rag_agent.application.listwise_runtime_reranker import (
    ListwiseUnionPrimaryContextSelectionPolicy,
    PrecomputedListwiseScoreProvider,
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
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)

_STAGE = "Stage 181"
_CREATED_AT = "2026-07-23"
_ANALYSIS_ID = "primeqa_hybrid_counterfactual_composition_action_audit_v1"
_EXPECTED_ROWS = 562
_EXPECTED_ANSWERABLE = 370
_EXPECTED_FOLDS = 5
_EXPECTED_STAGE180_MODEL_HEAD_FITS = 50
_EXPECTED_OOF_MODEL_FITS = 5
_MAX_SENTENCES = 3
_ALTERNATE_LIMIT = 12
_BASELINE_F1_REPRODUCTION_TOLERANCE = 0.001
_SOURCE_HASHES = {
    **stage180._SOURCE_HASHES,
    "stage180": "3605db66c11a3a9f527bfe44f9a442e6d139b114766c8d7d0edd2a0286f53be1",
}
_FORBIDDEN_PUBLIC_KEYS = stage180._FORBIDDEN_PUBLIC_KEYS | {
    "action_id",
    "question_key",
    "selected_indices",
}

ProgressSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class Stage181Visualization:
    name: str
    path: str


def run_stage181_composition_action_audit(
    *,
    stage180_report_path: Path,
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
    """Run the train-only Stage 181 counterfactual action audit."""

    started_at = time.perf_counter()
    source_paths = {
        "stage180": stage180_report_path,
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
    stage180_report = _load_json(stage180_report_path)
    stage178_private = _load_json(stage178_private_path)
    _authorize_stage180_report(stage180_report)
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="sources_authorized")

    samples = load_primeqa_hybrid_split_samples(train_split_path)
    _validate_samples(samples)
    fold_assignments = _build_train_fold_assignments(samples, fold_count=_EXPECTED_FOLDS)
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
    observation_sink = stage180._ObservationSink()
    workflow = stage178._workflow(
        candidate_pool_retriever=resources.candidate_pool_retriever,
        observation_adapter=PrimeQAHybridSidecarObservationAdapter(
            primary_context_selection_policy=ListwiseUnionPrimaryContextSelectionPolicy(
                score_provider=provider
            )
        ),
        observation_sink=observation_sink,
    )
    cases, collection_traces = stage180._collect_cases(
        samples=samples,
        fold_assignments=fold_assignments,
        workflow=workflow,
        progress_sink=progress_sink,
    )
    tracker.capture("composition_dataset_ready")
    collected_at = time.perf_counter()

    nested = stage180._run_nested_selection(
        cases=cases,
        specs=stage180.stage180_policy_specs(),
        progress_sink=progress_sink,
    )
    tracker.capture("stage180_actions_reconstructed")
    reconstructed_at = time.perf_counter()

    action_rows, build_diagnostics = _build_action_rows(
        cases=cases,
        policy_by_fold=nested["policy_by_fold"],
        progress_sink=progress_sink,
    )
    tracker.capture("action_catalog_ready")
    action_summary = summarize_action_rows(
        action_rows,
        total_question_count=_EXPECTED_ANSWERABLE,
    )
    stage180_summary = stage180_action_summary(action_rows)
    oof = run_action_predictability_oof(
        action_rows,
        total_question_count=_EXPECTED_ANSWERABLE,
    )
    oof_predictions = oof.pop("predictions")
    tracker.capture("oof_predictability_ready")
    analyzed_at = time.perf_counter()

    stage180_reproduced = _stage180_action_reproduced(
        reconstructed=stage180_summary,
        formal_report=stage180_report,
    )
    baseline_reproduction = _baseline_profile_reproduction(
        cases=cases,
        stage180_report=stage180_report,
    )
    snapshots = tracker.snapshots
    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train-only counterfactual audit of atomic answer-composition actions "
            "under a strict citation-and-F1 nonregression definition."
        ),
        "source_authorization": fingerprints,
        "frozen_protocol": _frozen_protocol(),
        "composition_dataset": {
            "question_count": len(cases),
            "answerable_question_count": sum(case.sample.answerable for case in cases),
            "answerable_with_candidates": sum(
                case.sample.answerable and bool(case.candidates) for case in cases
            ),
            "answerable_without_candidates": sum(
                case.sample.answerable and not case.candidates for case in cases
            ),
            "fold_question_counts": dict(sorted(Counter(case.fold_id for case in cases).items())),
            **build_diagnostics,
        },
        "action_audit": action_summary,
        "stage180_selected_action_audit": stage180_summary,
        "stage180_selected_action_reproduced": stage180_reproduced,
        "stage180_baseline_reproduction": baseline_reproduction,
        "oof_predictability": oof,
        "runtime": {
            "resource_factory_build_count": resource_factory.build_count,
            "precomputed_score_provider": asdict(provider.counters()),
            "observation_event_count": observation_sink.event_count,
            "workflow_counters": asdict(workflow.counters()),
        },
        "resource_consumption": _resource_summary(snapshots),
        "timing_seconds": {
            "source_authorization": round(authorized_at - started_at, 6),
            "load_train_and_protocol": round(loaded_at - authorized_at, 6),
            "resource_build_and_case_collection": round(collected_at - loaded_at, 6),
            "stage180_action_reconstruction": round(reconstructed_at - collected_at, 6),
            "action_audit_and_oof": round(analyzed_at - reconstructed_at, 6),
            "wall": round(analyzed_at - started_at, 6),
        },
        "execution_boundaries": {
            "train_loaded": True,
            "development_loaded": False,
            "test_loaded": False,
            "answerable_question_count": _EXPECTED_ANSWERABLE,
            "collection_agent_turn_count": len(collection_traces),
            "stage180_model_head_fit_count": nested["model_head_fit_count"],
            "oof_action_classifier_fit_count": oof["model"]["fit_count"],
            "gold_used_only_for_offline_labels_and_evaluation": True,
            "oof_model_uses_runtime_features_only": True,
            "runtime_policy_selected": False,
            "runtime_registered_as_default": False,
            "stage178b_run": False,
            "retry_action_count": sum(row.retry_action_count for row in collection_traces),
            "fallback_action_count": sum(row.fallback_action_count for row in collection_traces),
        },
    }
    forbidden = sorted(_forbidden_keys_found(report))
    report["public_safe_contract"] = {
        "forbidden_keys": sorted(_FORBIDDEN_PUBLIC_KEYS),
        "forbidden_keys_found": forbidden,
    }
    report["process_guards"] = _process_guards(
        report=report,
        cases=cases,
        action_rows=action_rows,
        oof_prediction_count=len(oof_predictions),
        forbidden=forbidden,
    )
    valid = all(row["passed"] for row in report["process_guards"])
    report["decision"] = {
        "status": (
            "stage181_counterfactual_action_audit_complete"
            if valid
            else "stage181_counterfactual_action_audit_invalid"
        ),
        "diagnostic_complete": valid,
        "policy_selected": False,
        "stage182_automatically_authorized": False,
        "development_opened": False,
        "test_opened": False,
        "stage178b_authorized": False,
        "default_runtime_activation": False,
    }
    _emit(progress_sink, phase="analysis_complete", decision=report["decision"])
    return report


def _build_action_rows(
    *,
    cases: Sequence[stage180.Stage180Case],
    policy_by_fold: Mapping[str, stage180.CitationAwareCompositionPolicy],
    progress_sink: ProgressSink | None,
) -> tuple[tuple[ActionAuditRow, ...], dict[str, Any]]:
    verifier = AnswerVerifier(min_evidence_score=7.0, max_citation_rank=200)
    rows = []
    unique_action_counts = []
    baseline_reproduction_count = 0
    answerable_cases = [case for case in cases if case.sample.answerable]
    for index, case in enumerate(answerable_cases, start=1):
        stage180_decision = policy_by_fold[case.fold_id].select(
            case.question,
            case.candidates,
            _MAX_SENTENCES,
        )
        candidate_index_by_identity = {
            id(candidate): candidate_index
            for candidate_index, candidate in enumerate(case.candidates)
        }
        stage180_indices = tuple(
            candidate_index_by_identity[id(candidate)]
            for candidate in stage180_decision.selected_candidates
        )
        actions = enumerate_atomic_composition_actions(
            candidates=case.candidates,
            stage180_selected_indices=stage180_indices,
            max_sentences=_MAX_SENTENCES,
            alternate_limit=_ALTERNATE_LIMIT,
        )
        unique_action_counts.append(len(actions))
        route = classify_question_route(case.question)
        candidate_features = [example.runtime_features for example in case.examples]
        for action in actions:
            selected = [case.candidates[position] for position in action.selected_indices]
            verified = stage180._compose_and_verify(
                question=case.question,
                selected=selected,
                verification_context=case.candidate_pool_results,
                verifier=verifier,
            )
            outcome = stage180._outcome(case=case, candidate=verified)
            citation_delta = int(outcome.candidate_gold_cited) - int(outcome.baseline_gold_cited)
            f1_delta = outcome.candidate_f1 - outcome.baseline_f1
            outcome_class, strict_expected = classify_action_outcome(
                citation_delta=citation_delta,
                f1_delta=f1_delta,
            )
            rows.append(
                ActionAuditRow(
                    question_key=case.sample.sample_id,
                    fold_id=case.fold_id,
                    route=route,
                    action=action,
                    runtime_features=build_action_runtime_features(
                        action=action,
                        candidates=case.candidates,
                        candidate_runtime_features=candidate_features,
                        route=route,
                        max_sentences=_MAX_SENTENCES,
                    ),
                    outcome_class=outcome_class,
                    strict_expected=strict_expected,
                    citation_delta=citation_delta,
                    f1_delta=f1_delta,
                )
            )
            if action.family == "baseline" and verified == case.baseline_verified:
                baseline_reproduction_count += 1
        if index % 25 == 0 or index == len(answerable_cases):
            _emit(
                progress_sink,
                phase="counterfactual_action_progress",
                completed=index,
                total=len(answerable_cases),
            )
    return tuple(rows), {
        "action_row_count_including_baseline": len(rows),
        "unique_action_count_distribution": _distribution(unique_action_counts),
        "baseline_direct_reproduction_count": baseline_reproduction_count,
        "alternate_sentence_limit": _ALTERNATE_LIMIT,
        "max_answer_sentences": _MAX_SENTENCES,
    }


def _authorize_sources(fingerprints: Mapping[str, Mapping[str, Any]]) -> None:
    mismatches = [
        name
        for name, expected in _SOURCE_HASHES.items()
        if fingerprints.get(name, {}).get("sha256") != expected
    ]
    if mismatches:
        raise ValueError(f"Stage181 source hash mismatch: {', '.join(sorted(mismatches))}")


def _authorize_stage180_report(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 180":
        raise ValueError("Stage181 requires the Stage180 report")
    if (
        report.get("decision", {}).get("status")
        != "stage180_citation_aware_composition_insufficient"
    ):
        raise ValueError("Stage181 requires the frozen insufficient Stage180 decision")
    boundaries = report.get("execution_boundaries", {})
    if (
        boundaries.get("development_loaded") is not False
        or boundaries.get("test_loaded") is not False
    ):
        raise ValueError("Stage180 development/test boundary drifted")
    if not all(row.get("passed") is True for row in report.get("process_guards", [])):
        raise ValueError("Stage180 process guards must all pass")


def _validate_samples(samples: Sequence[PrimeQAHybridSplitSample]) -> None:
    if len(samples) != _EXPECTED_ROWS or any(row.assigned_split != "train" for row in samples):
        raise ValueError("Stage181 accepts only the exact train split")
    if sum(row.answerable for row in samples) != _EXPECTED_ANSWERABLE:
        raise ValueError("Stage181 answerable row count drifted")


def _stage180_action_reproduced(
    *,
    reconstructed: Mapping[str, Any],
    formal_report: Mapping[str, Any],
) -> bool:
    expected = formal_report["agent_e2e"]
    return bool(
        reconstructed["question_count"] == _EXPECTED_ANSWERABLE
        and reconstructed["gold_citation_delta"] == expected["deltas"]["gold_citation_count"]
        and round(reconstructed["mean_answerable_f1_delta"], 6)
        == round(expected["paired_bootstrap"]["metrics"]["answer_f1"]["observed_delta"], 6)
    )


def _baseline_profile_reproduction(
    *,
    cases: Sequence[stage180.Stage180Case],
    stage180_report: Mapping[str, Any],
) -> dict[str, Any]:
    answerable = [case for case in cases if case.sample.answerable]
    baseline_f1 = statistics.fmean(
        stage178.stage160.score_answer(
            case.baseline_verified.answer,
            case.sample.answer,
            refused=case.baseline_verified.refused,
        )
        for case in answerable
    )
    gold_citations = sum(
        any(
            citation.document_id == case.sample.answer_doc_id
            for citation in case.baseline_verified.citations
        )
        for case in answerable
    )
    expected = stage180_report["agent_e2e"]["profiles"]["baseline"]
    expected_f1 = float(expected["verified_metrics"]["average_token_f1"])
    expected_citations = int(expected["gold_citation_count"])
    f1_absolute_delta = abs(baseline_f1 - expected_f1)
    f1_matches = _baseline_f1_within_tolerance(baseline_f1, expected_f1)
    citation_matches = gold_citations == expected_citations
    return {
        "actual_average_token_f1": round(baseline_f1, 6),
        "expected_average_token_f1": expected_f1,
        "absolute_f1_delta": round(f1_absolute_delta, 6),
        "absolute_f1_tolerance": _BASELINE_F1_REPRODUCTION_TOLERANCE,
        "f1_within_approved_tolerance": f1_matches,
        "actual_gold_citation_count": gold_citations,
        "expected_gold_citation_count": expected_citations,
        "gold_citation_count_matches": citation_matches,
        "passed": f1_matches and citation_matches,
    }


def _frozen_protocol() -> dict[str, Any]:
    return {
        "strict_expected_definition": {
            "gold_citation_nonregression": True,
            "answer_token_f1_nonregression": True,
            "at_least_one_strict_gain": True,
            "floating_equality_tolerance": 1e-12,
        },
        "historical_baseline_reproduction": {
            "gold_citation_count_exact": True,
            "paired_stage180_deltas_exact": True,
            "absolute_f1_tolerance": _BASELINE_F1_REPRODUCTION_TOLERANCE,
            "approved_after_two_stable_invalid_attempts": True,
        },
        "action_families": [
            "delete each baseline slot",
            "replace each baseline slot from first 12 alternates",
            "append when baseline has fewer than three sentences",
            "keep prefix 1 or 2",
            "document coverage",
            "lead-preserving document coverage",
            "Stage180 selected action",
        ],
        "equivalent_actions_deduplicated": True,
        "baseline_used_as_control_only": True,
        "oof_fold_count": _EXPECTED_FOLDS,
        "oof_model_family": "fixed class-balanced logistic regression",
        "coverage_points": [0.10, 0.25, 0.50, 1.00],
        "development_and_test_closed": True,
        "fallback_strategy_enabled": False,
        "runtime_policy_selection_enabled": False,
    }


def _resource_summary(snapshots: Sequence[Any]) -> dict[str, Any]:
    return {
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
    }


def _process_guards(
    *,
    report: Mapping[str, Any],
    cases: Sequence[stage180.Stage180Case],
    action_rows: Sequence[ActionAuditRow],
    oof_prediction_count: int,
    forbidden: Sequence[str],
) -> list[dict[str, Any]]:
    boundaries = report["execution_boundaries"]
    runtime = report["runtime"]
    provider = runtime["precomputed_score_provider"]
    nonbaseline_count = sum(row.action.family != "baseline" for row in action_rows)
    feature_keys = {str(key) for row in action_rows for key in row.runtime_features}
    feature_forbidden = {key for key in feature_keys if _is_forbidden_feature_name(key)}
    return [
        _gate("exact_train_rows", len(cases) == _EXPECTED_ROWS),
        _gate(
            "exact_answerable_rows",
            sum(case.sample.answerable for case in cases) == _EXPECTED_ANSWERABLE,
        ),
        _gate("exact_five_folds", len({case.fold_id for case in cases}) == _EXPECTED_FOLDS),
        _gate(
            "complete_baseline_reproduction",
            report["composition_dataset"]["baseline_direct_reproduction_count"]
            == _EXPECTED_ANSWERABLE,
        ),
        _gate(
            "stage180_baseline_profile_reproduced",
            report["stage180_baseline_reproduction"]["passed"],
        ),
        _gate("stage180_selected_action_reproduced", report["stage180_selected_action_reproduced"]),
        _gate(
            "exact_stage180_model_head_fits",
            boundaries["stage180_model_head_fit_count"] == _EXPECTED_STAGE180_MODEL_HEAD_FITS,
        ),
        _gate(
            "exact_oof_action_classifier_fits",
            boundaries["oof_action_classifier_fit_count"] == _EXPECTED_OOF_MODEL_FITS,
        ),
        _gate("complete_oof_action_predictions", oof_prediction_count == nonbaseline_count),
        _gate("runtime_feature_names_exclude_gold", not feature_forbidden),
        _gate("one_runtime_resource_build", runtime["resource_factory_build_count"] == 1),
        _gate(
            "one_collection_agent_pass", boundaries["collection_agent_turn_count"] == _EXPECTED_ROWS
        ),
        _gate("exact_score_provider_calls", provider["call_count"] == _EXPECTED_ROWS),
        _gate("exact_score_provider_pairs", provider["pair_count"] == 9_714),
        _gate("development_closed", boundaries["development_loaded"] is False),
        _gate("test_closed", boundaries["test_loaded"] is False),
        _gate("no_runtime_policy_selected", boundaries["runtime_policy_selected"] is False),
        _gate("default_runtime_unchanged", boundaries["runtime_registered_as_default"] is False),
        _gate("stage178b_not_run", boundaries["stage178b_run"] is False),
        _gate("no_retry", boundaries["retry_action_count"] == 0),
        _gate("no_fallback", boundaries["fallback_action_count"] == 0),
        _gate("public_report_safe", not forbidden),
    ]


def write_stage181_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage181Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = report["action_audit"]
    oof = report["oof_predictability"]
    family_summaries = audit["family_summaries"]
    charts = {
        "outcome_class_counts.svg": _chart(
            "Stage 181 counterfactual outcome classes",
            tuple(
                BarDatum(name, float(value), str(value))
                for name, value in audit["outcome_class_counts"].items()
            ),
            "unique nonbaseline actions",
        ),
        "family_expected_rates.svg": _chart(
            "Stage 181 strict expected rate by action family",
            tuple(
                BarDatum(name, row["strict_expected_rate"], f"{row['strict_expected_rate']:.3f}")
                for name, row in family_summaries.items()
            ),
            "strict expected action rate",
        ),
        "route_expected_rates.svg": _chart(
            "Stage 181 strict expected rate by question route",
            tuple(
                BarDatum(name, row["strict_expected_rate"], f"{row['strict_expected_rate']:.3f}")
                for name, row in audit["route_summaries"].items()
            ),
            "strict expected action rate",
        ),
        "oof_fold_roc_auc.svg": _chart(
            "Stage 181 OOF action classifier ROC AUC",
            tuple(
                BarDatum(name, row["roc_auc"] or 0.0, f"{(row['roc_auc'] or 0.0):.3f}")
                for name, row in oof["folds"].items()
            ),
            "held-out ROC AUC",
        ),
        "coverage_precision.svg": _chart(
            "Stage 181 OOF strict precision by question coverage",
            tuple(
                BarDatum(
                    f"{int(row['target_question_coverage'] * 100)}%",
                    row["strict_expected_precision"],
                    f"{row['strict_expected_precision']:.3f}",
                )
                for row in oof["coverage_curve"]
            ),
            "strict expected precision",
        ),
        "oracle_headroom.svg": _chart(
            "Stage 181 strict oracle headroom",
            (
                BarDatum(
                    "question coverage",
                    audit["oracle"]["question_coverage"],
                    f"{audit['oracle']['question_coverage']:.3f}",
                ),
                BarDatum(
                    "OOF top1 expected precision",
                    oof["question_ranking"]["top1_strict_expected_precision"],
                    f"{oof['question_ranking']['top1_strict_expected_precision']:.3f}",
                ),
            ),
            "rate",
        ),
        "family_citation_f1_scatter.svg": _family_scatter_svg(family_summaries),
    }
    written = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        written.append(Stage181Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _chart(title: str, bars: Sequence[BarDatum], x_label: str) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label=x_label,
        width=1480,
        margin_left=680,
    )


def _family_scatter_svg(families: Mapping[str, Mapping[str, Any]]) -> str:
    width = 1480
    height = 520
    left = 170
    right = 470
    top = 70
    bottom = 80
    rows = [
        (
            name,
            float(summary["mean_citation_delta"]),
            float(summary["mean_f1_delta"]),
            int(summary["action_count"]),
        )
        for name, summary in families.items()
    ]
    max_abs_x = max((abs(row[1]) for row in rows), default=1.0) or 1.0
    max_abs_y = max((abs(row[2]) for row in rows), default=1.0) or 1.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    center_x = left + plot_width / 2
    center_y = top + plot_height / 2
    points = []
    legend = []
    for index, (name, citation_delta, f1_delta, count) in enumerate(rows):
        x = center_x + (citation_delta / max_abs_x) * (plot_width * 0.45)
        y = center_y - (f1_delta / max_abs_y) * (plot_height * 0.45)
        radius = min(18.0, 5.0 + count**0.5 / 3.0)
        color = "#16803c" if citation_delta >= 0 and f1_delta >= 0 else "#dc2626"
        points.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" '
            f'opacity="0.78"><title>{escape(name)}</title></circle>'
        )
        legend_y = 82 + index * 42
        legend.append(
            f'<circle cx="{width - right + 28}" cy="{legend_y - 4}" r="6" '
            f'fill="{color}" opacity="0.78" />'
        )
        legend.append(
            f'<text x="{width - right + 44}" y="{legend_y}">{escape(name)} '
            f"(citation {citation_delta:+.4f}, F1 {f1_delta:+.4f})</text>"
        )
    return "\n".join(
        [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}" role="img">'
            ),
            '<rect width="100%" height="100%" fill="#ffffff" />',
            (
                "<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#111827}"
                ".title{font-size:18px;font-weight:700}.axis{stroke:#9ca3af}"
                ".label{fill:#4b5563}.legend-title{font-size:14px;font-weight:700}</style>"
            ),
            '<text x="24" y="32" class="title">Stage 181 action-family citation/F1 plane</text>',
            (
                f'<line x1="{center_x:.1f}" x2="{center_x:.1f}" y1="{top}" '
                f'y2="{height - bottom}" class="axis" />'
            ),
            (
                f'<line x1="{left}" x2="{width - right}" y1="{center_y:.1f}" '
                f'y2="{center_y:.1f}" class="axis" />'
            ),
            *points,
            f'<text x="{width - right + 18}" y="50" class="legend-title">action family</text>',
            *legend,
            (
                f'<text x="{center_x:.1f}" y="{height - 28}" text-anchor="middle" '
                'class="label">mean gold-citation delta</text>'
            ),
            f'<text x="28" y="{center_y:.1f}" class="label">mean F1 delta</text>',
            "</svg>",
        ]
    )


def _forbidden_keys_found(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        found = {str(key) for key in value if str(key) in _FORBIDDEN_PUBLIC_KEYS}
        for nested in value.values():
            found.update(_forbidden_keys_found(nested))
        return found
    if isinstance(value, (list, tuple)):
        found = set()
        for nested in value:
            found.update(_forbidden_keys_found(nested))
        return found
    return set()


def _is_forbidden_feature_name(name: str) -> bool:
    private_names = {
        "answer",
        "answer_doc_id",
        "document_id",
        "gold_answer",
        "question_id",
        "question_key",
        "selected_indices",
    }
    return bool(
        name in private_names
        or name.startswith("gold_")
        or "citation" in name
        or name == "f1"
        or name.endswith("_f1")
        or "_f1_" in name
    )


def _baseline_f1_within_tolerance(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= _BASELINE_F1_REPRODUCTION_TOLERANCE + 1e-12


def _distribution(values: Sequence[int]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "minimum": float(ordered[0]),
        "median": float(statistics.median(ordered)),
        "p95": float(ordered[max(0, int(0.95 * len(ordered)) - 1)]),
        "maximum": float(ordered[-1]),
        "mean": round(statistics.fmean(ordered), 6),
    }


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _emit(progress_sink: ProgressSink | None, **event: Any) -> None:
    if progress_sink is not None:
        progress_sink(event)
