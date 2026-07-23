from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application import primeqa_hybrid_citation_aware_composition_cv as stage180
from ts_rag_agent.application import primeqa_hybrid_composition_action_audit as stage181
from ts_rag_agent.application import primeqa_hybrid_iterative_router_calibration as stage169
from ts_rag_agent.application import primeqa_hybrid_listwise_agent_e2e as stage178
from ts_rag_agent.application import primeqa_hybrid_semantic_evidence_cv as stage173
from ts_rag_agent.application.composition_action_audit import (
    ActionAuditRow,
    stage180_action_summary,
    summarize_action_rows,
)
from ts_rag_agent.application.composition_dual_target_policy import (
    run_nested_dual_target_selection,
    stage182_policy_specs,
)
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
    load_primeqa_hybrid_split_samples,
)

_STAGE = "Stage 182"
_CREATED_AT = "2026-07-23"
_ANALYSIS_ID = "primeqa_hybrid_composition_dual_target_nested_cv_v1"
_EXPECTED_ROWS = 562
_EXPECTED_ANSWERABLE = 370
_EXPECTED_FOLDS = 5
_EXPECTED_STAGE180_MODEL_HEAD_FITS = 50
_EXPECTED_INNER_DUAL_TARGET_HEAD_FITS = 80
_SOURCE_HASHES = {
    **stage181._SOURCE_HASHES,
    "stage181": "a9c557d7346eb2b4958cddd2505937eba828556c7671d7e936bf883d80cfe88b",
}
_FORBIDDEN_PUBLIC_KEYS = stage181._FORBIDDEN_PUBLIC_KEYS | {
    "selected_actions",
    "runtime_features",
}

ProgressSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class Stage182Visualization:
    name: str
    path: str


def run_stage182_composition_dual_target_cv(
    *,
    stage181_report_path: Path,
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
    """Run the train-only Stage 182 nested dual-target policy experiment."""

    started_at = time.perf_counter()
    source_paths = {
        "stage181": stage181_report_path,
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
    stage181_report = _load_json(stage181_report_path)
    stage180_report = _load_json(stage180_report_path)
    stage178_private = _load_json(stage178_private_path)
    _authorize_stage181_report(stage181_report)
    stage181._authorize_stage180_report(stage180_report)
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="sources_authorized")

    samples = load_primeqa_hybrid_split_samples(train_split_path)
    stage181._validate_samples(samples)
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

    nested_stage180 = stage180._run_nested_selection(
        cases=cases,
        specs=stage180.stage180_policy_specs(),
        progress_sink=progress_sink,
    )
    action_rows, build_diagnostics = stage181._build_action_rows(
        cases=cases,
        policy_by_fold=nested_stage180["policy_by_fold"],
        progress_sink=progress_sink,
    )
    tracker.capture("action_dataset_reproduced")
    reproduced_at = time.perf_counter()

    action_summary = summarize_action_rows(
        action_rows,
        total_question_count=_EXPECTED_ANSWERABLE,
    )
    stage180_summary = stage180_action_summary(action_rows)
    reproduction = _stage181_reproduction(
        stage181_report=stage181_report,
        action_summary=action_summary,
        stage180_summary=stage180_summary,
        build_diagnostics=build_diagnostics,
    )
    if not reproduction["passed"]:
        raise ValueError("Stage182 action dataset did not reproduce Stage181")

    dual_target = run_nested_dual_target_selection(
        action_rows,
        specs=stage182_policy_specs(),
        total_question_count=_EXPECTED_ANSWERABLE,
    )
    selected_actions = dual_target.pop("selected_actions")
    tracker.capture("nested_dual_target_ready")
    analyzed_at = time.perf_counter()

    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train-only nested five-fold comparison of separate citation-gain and "
            "F1-regression action models with deliberate no-action abstention."
        ),
        "source_authorization": fingerprints,
        "frozen_protocol": _frozen_protocol(),
        "composition_dataset": {
            "question_count": len(cases),
            "answerable_question_count": sum(case.sample.answerable for case in cases),
            "fold_question_counts": dict(sorted(Counter(case.fold_id for case in cases).items())),
            **build_diagnostics,
        },
        "stage181_reproduction": reproduction,
        "dual_target_nested_cv": dual_target,
        "stage181_single_target_benchmark": {
            "top1_gold_citation_delta": stage181_report["oof_predictability"]["question_ranking"][
                "top1_gold_citation_delta"
            ],
            "top1_mean_answerable_f1_delta": stage181_report["oof_predictability"][
                "question_ranking"
            ]["top1_mean_answerable_f1_delta"],
            "top1_strict_expected_precision": stage181_report["oof_predictability"][
                "question_ranking"
            ]["top1_strict_expected_precision"],
        },
        "runtime": {
            "resource_factory_build_count": resource_factory.build_count,
            "precomputed_score_provider": asdict(provider.counters()),
            "observation_event_count": observation_sink.event_count,
            "workflow_counters": asdict(workflow.counters()),
        },
        "resource_consumption": stage181._resource_summary(tracker.snapshots),
        "timing_seconds": {
            "source_authorization": round(authorized_at - started_at, 6),
            "load_train_and_protocol": round(loaded_at - authorized_at, 6),
            "resource_build_and_case_collection": round(collected_at - loaded_at, 6),
            "stage180_and_action_reproduction": round(reproduced_at - collected_at, 6),
            "nested_dual_target_cv": round(analyzed_at - reproduced_at, 6),
            "wall": round(analyzed_at - started_at, 6),
        },
        "execution_boundaries": {
            "train_loaded": True,
            "development_loaded": False,
            "test_loaded": False,
            "collection_agent_turn_count": len(collection_traces),
            "stage180_model_head_fit_count": nested_stage180["model_head_fit_count"],
            "dual_target_model_head_fit_count": dual_target["model_head_fit_count"],
            "gold_used_only_for_training_labels_and_offline_evaluation": True,
            "models_use_runtime_features_only": True,
            "outer_folds_used_once_for_evaluation": True,
            "runtime_policy_selected": False,
            "runtime_registered_as_default": False,
            "stage178b_run": False,
            "retry_action_count": sum(row.retry_action_count for row in collection_traces),
            "fallback_action_count": sum(row.fallback_action_count for row in collection_traces),
        },
    }
    report["quality_gates"] = _quality_gates(report)
    forbidden = sorted(_forbidden_keys_found(report))
    report["public_safe_contract"] = {
        "forbidden_keys": sorted(_FORBIDDEN_PUBLIC_KEYS),
        "forbidden_keys_found": forbidden,
    }
    report["process_guards"] = _process_guards(
        report=report,
        cases=cases,
        action_rows=action_rows,
        selected_action_count=len(selected_actions),
        forbidden=forbidden,
    )
    valid = all(row["passed"] for row in report["process_guards"])
    selected = valid and all(row["passed"] for row in report["quality_gates"])
    report["decision"] = {
        "status": (
            "advance_to_stage183_dual_target_runtime_e2e"
            if selected
            else "stage182_dual_target_nested_cv_insufficient"
        ),
        "experiment_valid": valid,
        "candidate_selected": selected,
        "runtime_activation_authorized": False,
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
        raise ValueError(f"Stage182 source hash mismatch: {', '.join(sorted(mismatches))}")


def _authorize_stage181_report(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 181":
        raise ValueError("Stage182 requires the Stage181 report")
    if report.get("decision", {}).get("status") != "stage181_counterfactual_action_audit_complete":
        raise ValueError("Stage182 requires a completed Stage181 audit")
    if not all(row.get("passed") is True for row in report.get("process_guards", [])):
        raise ValueError("Stage181 process guards must all pass")
    boundaries = report.get("execution_boundaries", {})
    if (
        boundaries.get("development_loaded") is not False
        or boundaries.get("test_loaded") is not False
    ):
        raise ValueError("Stage181 development/test boundary drifted")


def _stage181_reproduction(
    *,
    stage181_report: Mapping[str, Any],
    action_summary: Mapping[str, Any],
    stage180_summary: Mapping[str, Any],
    build_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    expected_audit = stage181_report["action_audit"]
    expected_stage180 = stage181_report["stage180_selected_action_audit"]
    checks = {
        "action_row_count_including_baseline": (
            build_diagnostics["action_row_count_including_baseline"]
            == stage181_report["composition_dataset"]["action_row_count_including_baseline"]
        ),
        "nonbaseline_action_count": (
            action_summary["nonbaseline_action_count"] == expected_audit["nonbaseline_action_count"]
        ),
        "strict_expected_action_count": (
            action_summary["strict_expected_action_count"]
            == expected_audit["strict_expected_action_count"]
        ),
        "outcome_class_counts": (
            action_summary["outcome_class_counts"] == expected_audit["outcome_class_counts"]
        ),
        "oracle_gold_citation_delta": (
            action_summary["oracle"]["gold_citation_delta"]
            == expected_audit["oracle"]["gold_citation_delta"]
        ),
        "oracle_mean_f1_delta": (
            action_summary["oracle"]["mean_answerable_f1_delta"]
            == expected_audit["oracle"]["mean_answerable_f1_delta"]
        ),
        "stage180_gold_citation_delta": (
            stage180_summary["gold_citation_delta"] == expected_stage180["gold_citation_delta"]
        ),
        "stage180_mean_f1_delta": (
            stage180_summary["mean_answerable_f1_delta"]
            == expected_stage180["mean_answerable_f1_delta"]
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "actual_nonbaseline_action_count": action_summary["nonbaseline_action_count"],
        "actual_strict_expected_action_count": action_summary["strict_expected_action_count"],
    }


def _frozen_protocol() -> dict[str, Any]:
    return {
        "citation_target": "citation_delta > 0",
        "f1_risk_target": "f1_delta < -1e-12",
        "model_families": ["logistic", "hist_gradient_boosting"],
        "utility_modes": [
            "citation_only",
            "safe_product",
            "citation_minus_half_risk",
            "citation_minus_risk",
        ],
        "training_oof_target_coverages": [0.10, 0.25, 0.50, 1.00],
        "policy_candidate_count": len(stage182_policy_specs()),
        "outer_fold_count": 5,
        "inner_fold_count": 4,
        "inner_eligibility": (
            "aggregate strict A plus citation and F1 nonregression in all four inner folds"
        ),
        "outer_advancement": (
            "aggregate strict A, both bootstrap lower bounds nonnegative, both metrics "
            "nonregressing in at least four of five folds, and all outer folds select "
            "an inner-eligible policy"
        ),
        "no_action_behavior": "deliberate learned-threshold abstention retaining baseline",
        "development_and_test_closed": True,
        "fallback_strategy_enabled": False,
        "runtime_policy_activation_enabled": False,
    }


def _quality_gates(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    nested = report["dual_target_nested_cv"]
    aggregate = nested["aggregate"]
    bootstrap = nested["paired_bootstrap"]
    outer = nested["outer_folds"]
    return [
        _gate(
            "all_outer_folds_selected_inner_eligible_policy",
            all(fold["selected_spec"] is not None for fold in outer.values()),
        ),
        _gate("aggregate_strict_a_pass", aggregate["strict_aggregate_pass"]),
        _gate("gold_citation_strict_gain", aggregate["gold_citation_delta"] > 0),
        _gate("answer_f1_nonregression", aggregate["mean_f1_delta_all_questions"] >= 0),
        _gate(
            "gold_citation_bootstrap_ci_lower_nonnegative",
            bootstrap["gold_citation_delta"]["ci95_lower"] >= 0,
        ),
        _gate(
            "answer_f1_bootstrap_ci_lower_nonnegative",
            bootstrap["mean_f1_delta"]["ci95_lower"] >= 0,
        ),
        _gate(
            "gold_citation_nonregression_at_least_four_folds",
            aggregate["citation_nonregressing_fold_count"] >= 4,
        ),
        _gate(
            "answer_f1_nonregression_at_least_four_folds",
            aggregate["f1_nonregressing_fold_count"] >= 4,
        ),
    ]


def _process_guards(
    *,
    report: Mapping[str, Any],
    cases: Sequence[Any],
    action_rows: Sequence[ActionAuditRow],
    selected_action_count: int,
    forbidden: Sequence[str],
) -> list[dict[str, Any]]:
    boundaries = report["execution_boundaries"]
    runtime = report["runtime"]
    provider = runtime["precomputed_score_provider"]
    selected_fold_count = sum(
        fold["selected_spec"] is not None
        for fold in report["dual_target_nested_cv"]["outer_folds"].values()
    )
    expected_dual_head_fits = _EXPECTED_INNER_DUAL_TARGET_HEAD_FITS + 2 * selected_fold_count
    feature_keys = {str(key) for row in action_rows for key in row.runtime_features}
    forbidden_features = {key for key in feature_keys if stage181._is_forbidden_feature_name(key)}
    return [
        _gate("exact_train_rows", len(cases) == _EXPECTED_ROWS),
        _gate(
            "exact_answerable_rows",
            sum(case.sample.answerable for case in cases) == _EXPECTED_ANSWERABLE,
        ),
        _gate("exact_five_folds", len({case.fold_id for case in cases}) == _EXPECTED_FOLDS),
        _gate("stage181_action_dataset_reproduced", report["stage181_reproduction"]["passed"]),
        _gate(
            "exact_stage180_model_head_fits",
            boundaries["stage180_model_head_fit_count"] == _EXPECTED_STAGE180_MODEL_HEAD_FITS,
        ),
        _gate(
            "exact_nested_dual_target_head_fits",
            boundaries["dual_target_model_head_fit_count"] == expected_dual_head_fits,
        ),
        _gate(
            "outer_selected_actions_unique_by_question",
            selected_action_count
            == report["dual_target_nested_cv"]["aggregate"]["selected_question_count"],
        ),
        _gate("runtime_feature_names_exclude_gold", not forbidden_features),
        _gate("one_runtime_resource_build", runtime["resource_factory_build_count"] == 1),
        _gate(
            "one_collection_agent_pass",
            boundaries["collection_agent_turn_count"] == _EXPECTED_ROWS,
        ),
        _gate("exact_score_provider_calls", provider["call_count"] == _EXPECTED_ROWS),
        _gate("exact_score_provider_pairs", provider["pair_count"] == 9_714),
        _gate("development_closed", boundaries["development_loaded"] is False),
        _gate("test_closed", boundaries["test_loaded"] is False),
        _gate("outer_folds_evaluation_only", boundaries["outer_folds_used_once_for_evaluation"]),
        _gate("no_runtime_policy_selected", boundaries["runtime_policy_selected"] is False),
        _gate("default_runtime_unchanged", boundaries["runtime_registered_as_default"] is False),
        _gate("stage178b_not_run", boundaries["stage178b_run"] is False),
        _gate("no_retry", boundaries["retry_action_count"] == 0),
        _gate("no_fallback", boundaries["fallback_action_count"] == 0),
        _gate("public_report_safe", not forbidden),
    ]


def write_stage182_visualizations(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Stage182Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    nested = report["dual_target_nested_cv"]
    outer = nested["outer_folds"]
    aggregate = nested["aggregate"]
    benchmark = report["stage181_single_target_benchmark"]
    charts = {
        "outer_fold_citation_delta.svg": _chart(
            "Stage 182 held-out gold-citation delta by outer fold",
            tuple(
                BarDatum(
                    fold_id,
                    float(fold["heldout_policy_evaluation"]["gold_citation_delta"]),
                    str(fold["heldout_policy_evaluation"]["gold_citation_delta"]),
                )
                for fold_id, fold in outer.items()
            ),
            "gold-citation delta",
        ),
        "outer_fold_f1_delta.svg": _chart(
            "Stage 182 held-out answer-F1 delta by outer fold",
            tuple(
                BarDatum(
                    fold_id,
                    fold["heldout_policy_evaluation"]["mean_f1_delta_all_questions"],
                    f"{fold['heldout_policy_evaluation']['mean_f1_delta_all_questions']:+.4f}",
                )
                for fold_id, fold in outer.items()
            ),
            "mean F1 delta across fold questions",
        ),
        "outer_fold_coverage.svg": _chart(
            "Stage 182 held-out selected-question coverage",
            tuple(
                BarDatum(
                    fold_id,
                    fold["heldout_policy_evaluation"]["question_coverage"],
                    f"{fold['heldout_policy_evaluation']['question_coverage']:.3f}",
                )
                for fold_id, fold in outer.items()
            ),
            "question coverage",
        ),
        "outer_fold_strict_precision.svg": _chart(
            "Stage 182 held-out strict-action precision",
            tuple(
                BarDatum(
                    fold_id,
                    fold["heldout_policy_evaluation"]["strict_expected_precision"],
                    f"{fold['heldout_policy_evaluation']['strict_expected_precision']:.3f}",
                )
                for fold_id, fold in outer.items()
            ),
            "strict expected precision",
        ),
        "stage181_stage182_comparison.svg": _chart(
            "Stage 181 single-target versus Stage 182 nested dual-target",
            (
                BarDatum(
                    "Stage181 citation delta",
                    float(benchmark["top1_gold_citation_delta"]),
                    f"{benchmark['top1_gold_citation_delta']:+d}",
                ),
                BarDatum(
                    "Stage182 citation delta",
                    float(aggregate["gold_citation_delta"]),
                    f"{aggregate['gold_citation_delta']:+d}",
                ),
                BarDatum(
                    "Stage181 F1 delta x100",
                    benchmark["top1_mean_answerable_f1_delta"] * 100,
                    f"{benchmark['top1_mean_answerable_f1_delta']:+.4f}",
                ),
                BarDatum(
                    "Stage182 F1 delta x100",
                    aggregate["mean_f1_delta_all_questions"] * 100,
                    f"{aggregate['mean_f1_delta_all_questions']:+.4f}",
                ),
            ),
            "citation count or F1 delta x100",
        ),
        "quality_gates.svg": _chart(
            "Stage 182 strict advancement gates",
            tuple(
                BarDatum(
                    row["name"],
                    1.0 if row["passed"] else 0.0,
                    "pass" if row["passed"] else "fail",
                )
                for row in report["quality_gates"]
            ),
            "gate pass",
        ),
    }
    written = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        written.append(Stage182Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _chart(title: str, bars: Sequence[BarDatum], x_label: str) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label=x_label,
        width=1480,
        margin_left=660,
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


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _emit(progress_sink: ProgressSink | None, **event: Any) -> None:
    if progress_sink is not None:
        progress_sink(event)
