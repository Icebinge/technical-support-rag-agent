from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ts_rag_agent.application import primeqa_hybrid_iterative_router_calibration as stage169
from ts_rag_agent.application import primeqa_hybrid_train_history_evidence_gate_cv as stage167
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    _build_dense_channels,
    _build_lexical_channels,
    _build_train_fold_assignments,
    _load_json_object,
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
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)

_STAGE = "Stage 172"
_CREATED_AT = "2026-07-22"
_ANALYSIS_ID = "primeqa_hybrid_evidence_entailment_nested_cv_v1"
_EXPECTED_TRAIN_ROWS = 562
_EXPECTED_CANDIDATE_ROWS = 112_400
_EXPECTED_VIEW_CASES = 1_124
_MODEL_FAMILIES = ("logistic", "histogram_gbdt")
_THRESHOLDS = tuple(round(value / 100, 2) for value in range(10, 91, 5))
_MODEL_FEATURE_NAMES = (*stage167._EVIDENCE_FEATURE_NAMES, "phase_final", "visible_document_count")
_GATE_THRESHOLDS = {
    "initial_visible_compose_rate_min": 0.70,
    "alternate_only_inspect_rate_min": 0.50,
    "alternate_only_final_compose_rate_min": 0.70,
    "alternate_only_path_success_rate_min": 0.40,
    "insufficient_final_compose_rate_max": 0.20,
}
_SOURCE_HASHES = {
    "stage171": "cfb4dad9dd55587b058623c7d818f89ba5bcd8199bdf85f19c0d8df70d921e5d",
    "stage80": "2441bb1cb1e7888299d3f57962b18cd59df84e2086ac281105abcacfc144880f",
    "train": "cabd93e0b972c47384c4bf5cc2cd215a7fc519b2df4f81fba61db73c931aa155",
    "documents": "f93b5e2d8dcfb2c7d12676ef32ce22b7809692f14081aad98096099a5256722b",
}
_STRATA = stage169._TRAIN_STRATA
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer",
        "answer_doc_id",
        "answerable",
        "document_id",
        "gold_document",
        "gold_rank",
        "is_gold",
        "private_identity_sha256",
        "question_text",
        "sample_id",
    }
)

ProgressSink = Callable[[Mapping[str, Any]], None]
ModelFamily = Literal["logistic", "histogram_gbdt"]


@dataclass(frozen=True)
class EvidenceViewCase:
    private_identity: str
    group_identity: str
    fold_id: str
    phase: str
    stratum: str
    features: Mapping[str, float]
    sufficient_label: bool


@dataclass(frozen=True)
class EvidenceModelSpec:
    model_family: ModelFamily
    threshold: float

    @property
    def spec_id(self) -> str:
        material = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class EvidenceProxyMetrics:
    case_count: int
    positive_count: int
    negative_count: int
    predicted_sufficient_count: int
    true_positive_count: int
    false_positive_count: int
    true_negative_count: int
    false_negative_count: int
    balanced_accuracy: float
    roc_auc: float
    initial_visible_compose_rate: float
    alternate_only_inspect_rate: float
    alternate_only_final_compose_rate: float
    alternate_only_path_success_rate: float
    insufficient_final_compose_rate: float

    def public_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class Stage172Visualization:
    name: str
    path: str


class EvidenceEntailmentPredictor:
    """Balanced binary scorer over the frozen runtime-safe view summary."""

    def __init__(self, model_family: ModelFamily) -> None:
        self._model_family = model_family
        if model_family == "logistic":
            self._model: Any = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            max_iter=2_000,
                            random_state=172,
                        ),
                    ),
                ]
            )
        elif model_family == "histogram_gbdt":
            self._model = HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=150,
                max_leaf_nodes=9,
                l2_regularization=1.0,
                random_state=172,
            )
        else:
            raise ValueError(f"unsupported Stage172 model family: {model_family}")

    def fit(self, cases: Sequence[EvidenceViewCase]) -> None:
        matrix = _feature_matrix(cases)
        labels = np.asarray([case.sufficient_label for case in cases], dtype=int)
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("Stage172 training requires both evidence classes")
        if self._model_family == "histogram_gbdt":
            counts = np.bincount(labels, minlength=2)
            weights = np.asarray([len(labels) / (2 * counts[label]) for label in labels])
            self._model.fit(matrix, labels, sample_weight=weights)
            return
        self._model.fit(matrix, labels)

    def predict(self, cases: Sequence[EvidenceViewCase]) -> dict[str, float]:
        probabilities = self._model.predict_proba(_feature_matrix(cases))[:, 1]
        return {
            case.private_identity: float(probability)
            for case, probability in zip(cases, probabilities, strict=True)
        }


def build_stage172_specs() -> tuple[EvidenceModelSpec, ...]:
    return tuple(
        EvidenceModelSpec(family, threshold)
        for family in _MODEL_FAMILIES
        for threshold in _THRESHOLDS
    )


def summarize_evidence_view(
    *,
    records: Sequence[ContextCandidateRecord],
    visible_records: Sequence[ContextCandidateRecord],
    phase: str,
) -> dict[str, float]:
    if len(records) != 200:
        raise ValueError("Stage172 evidence summary requires exact Top200 records")
    if phase not in {"initial", "final"}:
        raise ValueError("Stage172 evidence phase must be initial or final")
    baseline = tuple(sorted(records, key=lambda record: record.baseline_rank)[:10])
    visible = _deduplicate_records(visible_records)
    if len(baseline) != 10 or not 1 <= len(visible) <= 20:
        raise ValueError("Stage172 evidence view must contain 1..20 unique records")
    rrf = [_value(record, "stage116_rrf_score") for record in baseline]
    visible_rrf = [_value(record, "stage116_rrf_score") for record in visible]
    overlap_scores = sorted(
        (_value(record, "current_query_overlap_combined_score") for record in visible),
        reverse=True,
    )
    baseline_ids = {record.document_id for record in baseline}
    values = {
        "rrf_top1": rrf[0],
        "rrf_top1_top2_margin": rrf[0] - rrf[1],
        "rrf_top1_top10_spread": rrf[0] - rrf[-1],
        "rrf_top10_mean": _mean(rrf),
        "top1_route_hits": _value(baseline[0], "route_hit_count"),
        "top1_lexical_hits": _value(baseline[0], "lexical_route_hit_count"),
        "top1_dense_hits": _value(baseline[0], "dense_route_hit_count"),
        "top1_best_route_inverse_rank": _value(baseline[0], "best_route_inverse_rank"),
        "top10_route_hits_mean": _mean_values(baseline, "route_hit_count"),
        "top10_dense_hits_mean": _mean_values(baseline, "dense_route_hit_count"),
        "selection_rrf_mean": _mean(visible_rrf),
        "selection_rrf_spread": max(visible_rrf) - min(visible_rrf),
        "selection_baseline_rank_mean": _mean([float(record.baseline_rank) for record in visible]),
        "selection_baseline_rank_max": float(max(record.baseline_rank for record in visible)),
        "selection_prefix_overlap_count": float(
            sum(record.document_id in baseline_ids for record in visible)
        ),
        "selection_overlap_score_top1_top2_margin": (
            overlap_scores[0] - overlap_scores[1] if len(overlap_scores) > 1 else 0.0
        ),
        "selection_overlap_count_mean": _mean_values(visible, "current_query_overlap_count"),
        "selection_overlap_count_max": _max_value(visible, "current_query_overlap_count"),
        "selection_overlap_ratio_mean": _mean_values(visible, "current_query_overlap_ratio"),
        "selection_query_coverage_mean": _mean_values(visible, "query_token_coverage"),
        "selection_query_coverage_max": _max_value(visible, "query_token_coverage"),
        "selection_body_coverage_mean": _mean_values(visible, "query_body_token_coverage"),
        "selection_title_overlap_mean": _mean_values(visible, "query_title_token_overlap"),
        "selection_heading_overlap_mean": _mean_values(visible, "query_section_heading_overlap"),
        "selection_special_match_mean": _mean_values(visible, "query_special_token_match_count"),
        "selection_special_match_max": _max_value(visible, "query_special_token_match_count"),
        "selection_bm25_top10_fraction": _mean_values(visible, "bm25_top10_indicator"),
        "phase_final": float(phase == "final"),
        "visible_document_count": float(len(visible)),
    }
    if tuple(values) != _MODEL_FEATURE_NAMES:
        raise RuntimeError("Stage172 evidence feature order drifted")
    return values


def build_evidence_view_cases(
    *,
    samples: Sequence[PrimeQAHybridSplitSample],
    grouped_records: Mapping[str, Sequence[ContextCandidateRecord]],
) -> tuple[EvidenceViewCase, ...]:
    cases = []
    for sample in samples:
        records = tuple(grouped_records[sample.sample_id])
        initial = select_current_query_overlap_top10(records).selected
        alternate = select_original_rrf_top10(records).selected
        final = _deduplicate_records((*initial, *alternate))
        initial_ids = {record.document_id for record in initial}
        final_ids = {record.document_id for record in final}
        candidate_ids = {record.document_id for record in records}
        if not sample.answerable:
            stratum = "unanswerable"
        elif sample.answer_doc_id in initial_ids:
            stratum = "initial_gold_visible"
        elif sample.answer_doc_id in final_ids:
            stratum = "alternate_only_gold_visible"
        elif sample.answer_doc_id in candidate_ids:
            stratum = "union_gold_missing_candidate_hit"
        else:
            stratum = "candidate_pool_gold_missing"
        fold_id = records[0].fold_id
        group_identity = _sha256_text(sample.sample_id)
        for phase, visible, sufficient in (
            (
                "initial",
                initial,
                bool(sample.answerable and sample.answer_doc_id in initial_ids),
            ),
            (
                "final",
                final,
                bool(sample.answerable and sample.answer_doc_id in final_ids),
            ),
        ):
            cases.append(
                EvidenceViewCase(
                    private_identity=_sha256_text(f"{sample.sample_id}:{phase}"),
                    group_identity=group_identity,
                    fold_id=fold_id,
                    phase=phase,
                    stratum=stratum,
                    features=summarize_evidence_view(
                        records=records,
                        visible_records=visible,
                        phase=phase,
                    ),
                    sufficient_label=sufficient,
                )
            )
    if len(cases) != len(samples) * 2:
        raise RuntimeError("Stage172 case construction did not create two views per sample")
    return tuple(cases)


def run_stage172_evidence_entailment_cv(
    *,
    stage171_report_path: Path,
    stage80_report_path: Path,
    train_split_path: Path,
    documents_path: Path,
    encoder_batch_size: int = 64,
    progress_sink: ProgressSink | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    started_cpu = time.process_time()
    start_memory = stage169._windows_process_memory()
    start_available = stage169._windows_available_memory_bytes()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    paths = {
        "stage171": stage171_report_path,
        "stage80": stage80_report_path,
        "train": train_split_path,
        "documents": documents_path,
    }
    fingerprints = {name: stage169._fingerprint(path) for name, path in paths.items()}
    _authorize_sources(fingerprints)
    stage171_report = _load_json_object(stage171_report_path)
    if stage171_report.get("decision", {}).get("status") != "stage171_hierarchy_requires_redesign":
        raise ValueError("Stage171 did not authorize evidence-entailment redesign")
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="sources_authorized")

    samples = load_primeqa_hybrid_split_samples(train_split_path)
    if len(samples) != _EXPECTED_TRAIN_ROWS or any(
        sample.assigned_split != "train" for sample in samples
    ):
        raise ValueError("Stage172 accepts only the exact 562-row train split")
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
        raise RuntimeError("Stage172 requires both authorized local dense channels")
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
        raise RuntimeError("Stage172 candidate replay row count drifted")
    cases = build_evidence_view_cases(
        samples=samples,
        grouped_records=records_by_sample(records),
    )
    if len(cases) != _EXPECTED_VIEW_CASES:
        raise RuntimeError("Stage172 view case count drifted")
    evidence_ready_at = time.perf_counter()
    _emit(progress_sink, phase="evidence_cases_ready", case_count=len(cases))

    specs = build_stage172_specs()
    outer_folds, oof_predictions, selected_specs = _run_nested_cv(
        cases=cases,
        specs=specs,
        progress_sink=progress_sink,
    )
    final_spec_row = _select_full_train_spec(cases=cases, specs=specs)
    final_spec: EvidenceModelSpec = final_spec_row["spec"]
    cv_finished_at = time.perf_counter()
    oof_metrics = evaluate_predictions(cases, oof_predictions, selected_specs)
    oof_gates = _quality_gates(oof_metrics)
    fold_metrics = _outer_fold_metrics(cases, oof_predictions, selected_specs)
    all_outer_safety_passed = all(
        metrics["insufficient_final_compose_rate"]
        <= _GATE_THRESHOLDS["insufficient_final_compose_rate_max"]
        for metrics in fold_metrics.values()
    )
    all_inner_selected_eligible = all(row["selected_spec_inner_eligible"] for row in outer_folds)
    candidate_selected = (
        all_inner_selected_eligible
        and bool(final_spec_row["eligible"])
        and all(gate["passed"] for gate in oof_gates)
        and all_outer_safety_passed
    )
    end_memory = stage169._windows_process_memory()
    end_available = stage169._windows_available_memory_bytes()
    finished_at = time.perf_counter()

    stratum_counts = {
        stratum: sum(case.phase == "initial" and case.stratum == stratum for case in cases)
        for stratum in _STRATA
    }
    process_guards = [
        _check("stage171_authorized_evidence_redesign", True),
        _check("exact_train_row_count", len(samples) == _EXPECTED_TRAIN_ROWS),
        _check("exact_candidate_row_count", len(records) == _EXPECTED_CANDIDATE_ROWS),
        _check("two_views_per_train_row", len(cases) == _EXPECTED_VIEW_CASES),
        _check("five_grouped_outer_folds", len(outer_folds) == 5),
        _check("complete_oof_prediction_coverage", len(oof_predictions) == len(cases)),
        _check("two_frozen_model_families", len(_MODEL_FAMILIES) == 2),
        _check("seventeen_frozen_thresholds", len(_THRESHOLDS) == 17),
        _check("runtime_feature_count_exact", len(_MODEL_FEATURE_NAMES) == 29),
        _check("development_not_loaded", True),
        _check("test_not_loaded", True),
        _check("model_generation_not_run", True),
        _check("agent_turns_not_run", True),
        _check("retry_count_zero", True),
        _check("fallback_count_zero", True),
        _check("default_runtime_unchanged", True),
    ]
    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train-only grouped nested five-fold evidence-entailment classification over "
            "runtime-safe initial and final evidence-view summaries."
        ),
        "source_authorization": fingerprints,
        "frozen_protocol": {
            "model_families": list(_MODEL_FAMILIES),
            "thresholds": list(_THRESHOLDS),
            "outer_fold_count": 5,
            "inner_selection": "grouped_leave_one_training_fold_out_predictions",
            "outer_evaluation": "one_shot_heldout_fold",
            "model_features": list(_MODEL_FEATURE_NAMES),
            "quality_thresholds": _GATE_THRESHOLDS,
            "selection_order": (
                "all-fold-safety, safe-fold-count, aggregate-safety, gate-count, exact-path, "
                "final-recall, initial-recall, lower-false-compose"
            ),
            "gold_labels_used_only_for_fit_and_evaluation": True,
            "development_and_test_closed": True,
        },
        "split_contract": {
            "loaded_split": "train",
            "fit_split": "outer_train_folds_only",
            "development_loaded": False,
            "test_loaded": False,
        },
        "case_summary": {
            "train_question_count": len(samples),
            "view_case_count": len(cases),
            "positive_view_count": sum(case.sufficient_label for case in cases),
            "negative_view_count": sum(not case.sufficient_label for case in cases),
            "stratum_question_counts": stratum_counts,
            "private_case_rows_written": False,
        },
        "feature_contract": {
            "feature_count": len(_MODEL_FEATURE_NAMES),
            "feature_names": list(_MODEL_FEATURE_NAMES),
            "feature_distribution": _feature_distribution(cases),
            "forbidden_runtime_inputs": [
                "answerable label",
                "gold document ID",
                "gold rank",
                "sample identity",
                "question text",
                "selected action",
            ],
        },
        "nested_cv": {
            "candidate_spec_count": len(specs),
            "outer_folds": outer_folds,
            "selected_spec_ids_by_fold": {
                fold_id: spec.spec_id for fold_id, spec in selected_specs.items()
            },
            "final_full_train_oof_selected_spec": asdict(final_spec),
            "final_full_train_oof_selected_spec_id": final_spec.spec_id,
            "final_full_train_oof_selected_spec_eligible": bool(final_spec_row["eligible"]),
            "final_full_train_oof_metrics": final_spec_row["metrics"].public_dict(),
            "final_full_train_oof_safe_fold_count": final_spec_row["safe_fold_count"],
            "all_inner_selected_specs_eligible": all_inner_selected_eligible,
            "oof_metrics": oof_metrics.public_dict(),
            "oof_quality_gates": oof_gates,
            "outer_fold_metrics": fold_metrics,
            "all_outer_folds_safety_passed": all_outer_safety_passed,
        },
        "stage171_comparison": {
            "stage171_evidence_gate_pass_count": 4,
            "stage172_evidence_gate_pass_count": sum(gate["passed"] for gate in oof_gates),
            "stage171_insufficient_final_compose_rate": 0.8,
            "stage172_insufficient_final_compose_rate": (
                oof_metrics.insufficient_final_compose_rate
            ),
        },
        "resource_consumption": {
            "sampling_mode": "process_boundary_snapshots_without_monitor_polling",
            "wall_time_seconds": round(finished_at - started_at, 6),
            "process_cpu_time_seconds": round(time.process_time() - started_cpu, 6),
            "process_peak_working_set_bytes": max(
                start_memory["peak_working_set_bytes"],
                end_memory["peak_working_set_bytes"],
            ),
            "process_private_usage_bytes_at_end": end_memory["private_usage_bytes"],
            "minimum_boundary_system_available_memory_bytes": min(start_available, end_available),
            "gpu_model_loaded": False,
            "model_generation_calls": 0,
        },
        "timing_seconds": {
            "source_authorization": round(authorized_at - started_at, 6),
            "evidence_replay_and_case_build": round(evidence_ready_at - authorized_at, 6),
            "nested_cv": round(cv_finished_at - evidence_ready_at, 6),
            "report_assembly": round(finished_at - cv_finished_at, 6),
        },
        "closed_boundaries": {
            "development_opened": False,
            "test_opened": False,
            "answer_generation_run": False,
            "agent_turns_run": False,
            "retry_action_count": 0,
            "fallback_action_count": 0,
            "runtime_registered_as_default": False,
        },
        "process_guards": process_guards,
        "decision": {
            "candidate_selected": candidate_selected,
            "status": (
                "advance_to_stage173_train_only_hierarchical_runtime_e2e"
                if candidate_selected
                else "stage172_no_grouped_oof_safe_evidence_classifier"
            ),
            "development_opened": False,
            "test_opened": False,
            "default_runtime_activation": False,
        },
    }
    forbidden = sorted(_forbidden_keys_found(report))
    report["public_safe_contract"] = {
        "forbidden_keys": sorted(_FORBIDDEN_PUBLIC_KEYS),
        "forbidden_keys_found": forbidden,
    }
    report["process_guards"].append(
        _check("public_report_contains_no_forbidden_keys", not forbidden)
    )
    all_process_guards = all(guard["passed"] for guard in report["process_guards"])
    report["decision"]["all_process_guards_passed"] = all_process_guards
    if not all_process_guards:
        report["decision"]["candidate_selected"] = False
        report["decision"]["status"] = "stage172_process_invalid"
    _emit(progress_sink, phase="analysis_complete", decision=report["decision"])
    return report


def write_stage172_visualizations(
    *, report: Mapping[str, Any], output_dir: Path
) -> tuple[Stage172Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    nested = report["nested_cv"]
    metrics = nested["oof_metrics"]
    gates = nested["oof_quality_gates"]
    folds = nested["outer_fold_metrics"]
    charts = {
        "oof_quality_gates.svg": _chart(
            "Stage 172 grouped OOF evidence gates",
            tuple(
                BarDatum(
                    gate["name"],
                    float(gate["passed"]),
                    f"{gate['observed']:.1%} / {gate['threshold']:.1%}",
                )
                for gate in gates
            ),
            x_label="1 means passed",
        ),
        "oof_proxy_rates.svg": _chart(
            "Stage 172 grouped OOF proxy rates",
            (
                _rate_bar("Initial-visible compose", metrics["initial_visible_compose_rate"]),
                _rate_bar("Alternate-only inspect", metrics["alternate_only_inspect_rate"]),
                _rate_bar(
                    "Alternate-only final compose",
                    metrics["alternate_only_final_compose_rate"],
                ),
                _rate_bar(
                    "Alternate-only exact path",
                    metrics["alternate_only_path_success_rate"],
                ),
                _rate_bar(
                    "Insufficient final compose",
                    metrics["insufficient_final_compose_rate"],
                ),
            ),
            x_label="Rate",
        ),
        "outer_fold_safety.svg": _chart(
            "Stage 172 outer-fold insufficient final compose",
            tuple(
                _rate_bar(fold_id, fold["insufficient_final_compose_rate"])
                for fold_id, fold in folds.items()
            ),
            x_label="False-compose rate",
        ),
        "outer_fold_path.svg": _chart(
            "Stage 172 outer-fold alternate exact path",
            tuple(
                _rate_bar(fold_id, fold["alternate_only_path_success_rate"])
                for fold_id, fold in folds.items()
            ),
            x_label="Exact-path rate",
        ),
        "inner_eligible_specs.svg": _chart(
            "Stage 172 inner-eligible specifications",
            tuple(
                BarDatum(
                    f"fold {row['heldout_fold']}",
                    row["inner_eligible_spec_count"],
                    str(row["inner_eligible_spec_count"]),
                )
                for row in nested["outer_folds"]
            ),
            x_label="Eligible spec count",
        ),
        "label_distribution.svg": _chart(
            "Stage 172 evidence-view label distribution",
            (
                BarDatum(
                    "Sufficient",
                    report["case_summary"]["positive_view_count"],
                    str(report["case_summary"]["positive_view_count"]),
                ),
                BarDatum(
                    "Insufficient",
                    report["case_summary"]["negative_view_count"],
                    str(report["case_summary"]["negative_view_count"]),
                ),
            ),
            x_label="View cases",
        ),
    }
    written = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        written.append(Stage172Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _run_nested_cv(
    *,
    cases: Sequence[EvidenceViewCase],
    specs: Sequence[EvidenceModelSpec],
    progress_sink: ProgressSink | None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, float],
    dict[str, EvidenceModelSpec],
]:
    folds = sorted({case.fold_id for case in cases})
    if len(folds) != 5:
        raise ValueError("Stage172 requires exactly five grouped folds")
    rows = []
    oof_predictions: dict[str, float] = {}
    selected_specs: dict[str, EvidenceModelSpec] = {}
    for index, outer_fold in enumerate(folds, start=1):
        outer_train = tuple(case for case in cases if case.fold_id != outer_fold)
        heldout = tuple(case for case in cases if case.fold_id == outer_fold)
        inner_predictions: dict[str, dict[str, float]] = {family: {} for family in _MODEL_FAMILIES}
        for family in _MODEL_FAMILIES:
            for inner_fold in sorted({case.fold_id for case in outer_train}):
                inner_train = tuple(case for case in outer_train if case.fold_id != inner_fold)
                inner_heldout = tuple(case for case in outer_train if case.fold_id == inner_fold)
                predictor = EvidenceEntailmentPredictor(family)
                predictor.fit(inner_train)
                inner_predictions[family].update(predictor.predict(inner_heldout))
        if any(len(predictions) != len(outer_train) for predictions in inner_predictions.values()):
            raise RuntimeError("Stage172 inner OOF prediction coverage is incomplete")
        spec_rows = [
            _spec_evaluation(outer_train, inner_predictions[spec.model_family], spec)
            for spec in specs
        ]
        eligible = [row for row in spec_rows if row["eligible"]]
        selected_row = max(spec_rows, key=_spec_selection_key)
        selected = selected_row["spec"]
        predictor = EvidenceEntailmentPredictor(selected.model_family)
        predictor.fit(outer_train)
        heldout_predictions = predictor.predict(heldout)
        oof_predictions.update(heldout_predictions)
        selected_specs[outer_fold] = selected
        heldout_metrics = evaluate_predictions(
            heldout,
            heldout_predictions,
            {outer_fold: selected},
        )
        rows.append(
            {
                "heldout_fold": outer_fold,
                "outer_train_case_count": len(outer_train),
                "heldout_case_count": len(heldout),
                "candidate_spec_count": len(specs),
                "inner_eligible_spec_count": len(eligible),
                "selected_spec": asdict(selected),
                "selected_spec_id": selected.spec_id,
                "selected_spec_inner_eligible": bool(selected_row["eligible"]),
                "selected_inner_metrics": selected_row["metrics"].public_dict(),
                "selected_inner_safe_fold_count": selected_row["safe_fold_count"],
                "heldout_metrics": heldout_metrics.public_dict(),
            }
        )
        _emit(progress_sink, phase="outer_fold_complete", completed=index, total=5)
    return rows, oof_predictions, selected_specs


def _select_full_train_spec(
    *,
    cases: Sequence[EvidenceViewCase],
    specs: Sequence[EvidenceModelSpec],
) -> dict[str, Any]:
    predictions_by_family: dict[str, dict[str, float]] = {family: {} for family in _MODEL_FAMILIES}
    folds = sorted({case.fold_id for case in cases})
    for family in _MODEL_FAMILIES:
        for heldout_fold in folds:
            train = tuple(case for case in cases if case.fold_id != heldout_fold)
            heldout = tuple(case for case in cases if case.fold_id == heldout_fold)
            predictor = EvidenceEntailmentPredictor(family)
            predictor.fit(train)
            predictions_by_family[family].update(predictor.predict(heldout))
    if any(len(predictions) != len(cases) for predictions in predictions_by_family.values()):
        raise RuntimeError("Stage172 full-train OOF prediction coverage is incomplete")
    rows = [
        _spec_evaluation(cases, predictions_by_family[spec.model_family], spec) for spec in specs
    ]
    return max(rows, key=_spec_selection_key)


def _spec_evaluation(
    cases: Sequence[EvidenceViewCase],
    predictions: Mapping[str, float],
    spec: EvidenceModelSpec,
) -> dict[str, Any]:
    specs_by_fold = {fold_id: spec for fold_id in {case.fold_id for case in cases}}
    metrics = evaluate_predictions(cases, predictions, specs_by_fold)
    gates = _quality_gates(metrics)
    fold_metrics = _fold_metrics_for_spec(cases, predictions, spec)
    safe_fold_count = sum(
        row.insufficient_final_compose_rate
        <= _GATE_THRESHOLDS["insufficient_final_compose_rate_max"]
        for row in fold_metrics.values()
    )
    eligible = all(gate["passed"] for gate in gates) and safe_fold_count == len(fold_metrics)
    return {
        "spec": spec,
        "metrics": metrics,
        "gates": gates,
        "safe_fold_count": safe_fold_count,
        "fold_count": len(fold_metrics),
        "eligible": eligible,
    }


def _spec_selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics: EvidenceProxyMetrics = row["metrics"]
    gates = row["gates"]
    return (
        int(row["safe_fold_count"] == row["fold_count"]),
        row["safe_fold_count"],
        int(
            metrics.insufficient_final_compose_rate
            <= _GATE_THRESHOLDS["insufficient_final_compose_rate_max"]
        ),
        sum(gate["passed"] for gate in gates),
        metrics.alternate_only_path_success_rate,
        metrics.alternate_only_final_compose_rate,
        metrics.initial_visible_compose_rate,
        -metrics.insufficient_final_compose_rate,
        metrics.balanced_accuracy,
        -abs(row["spec"].threshold - 0.5),
        row["spec"].spec_id,
    )


def evaluate_predictions(
    cases: Sequence[EvidenceViewCase],
    predictions: Mapping[str, float],
    specs_by_fold: Mapping[str, EvidenceModelSpec],
) -> EvidenceProxyMetrics:
    if len(predictions) != len(cases):
        raise ValueError("Stage172 evaluation requires one prediction per view case")
    predicted = {
        case.private_identity: predictions[case.private_identity]
        >= specs_by_fold[case.fold_id].threshold
        for case in cases
    }
    labels = np.asarray([case.sufficient_label for case in cases], dtype=int)
    scores = np.asarray([predictions[case.private_identity] for case in cases], dtype=float)
    choices = np.asarray([predicted[case.private_identity] for case in cases], dtype=int)
    tp = int(np.sum((labels == 1) & (choices == 1)))
    fp = int(np.sum((labels == 0) & (choices == 1)))
    tn = int(np.sum((labels == 0) & (choices == 0)))
    fn = int(np.sum((labels == 1) & (choices == 0)))
    tpr = _rate(tp, tp + fn)
    tnr = _rate(tn, tn + fp)
    groups = _cases_by_group(cases)
    initial_visible = [
        rows for rows in groups.values() if rows["initial"].stratum == "initial_gold_visible"
    ]
    alternate_only = [
        rows for rows in groups.values() if rows["initial"].stratum == "alternate_only_gold_visible"
    ]
    insufficient = [
        rows
        for rows in groups.values()
        if rows["initial"].stratum
        in {
            "union_gold_missing_candidate_hit",
            "candidate_pool_gold_missing",
            "unanswerable",
        }
    ]
    return EvidenceProxyMetrics(
        case_count=len(cases),
        positive_count=int(labels.sum()),
        negative_count=int(len(labels) - labels.sum()),
        predicted_sufficient_count=int(choices.sum()),
        true_positive_count=tp,
        false_positive_count=fp,
        true_negative_count=tn,
        false_negative_count=fn,
        balanced_accuracy=round((tpr + tnr) / 2, 6),
        roc_auc=round(float(roc_auc_score(labels, scores)), 6),
        initial_visible_compose_rate=_group_rate(
            initial_visible,
            lambda rows: predicted[rows["initial"].private_identity],
        ),
        alternate_only_inspect_rate=_group_rate(
            alternate_only,
            lambda rows: not predicted[rows["initial"].private_identity],
        ),
        alternate_only_final_compose_rate=_group_rate(
            alternate_only,
            lambda rows: predicted[rows["final"].private_identity],
        ),
        alternate_only_path_success_rate=_group_rate(
            alternate_only,
            lambda rows: (
                not predicted[rows["initial"].private_identity]
                and predicted[rows["final"].private_identity]
            ),
        ),
        insufficient_final_compose_rate=_group_rate(
            insufficient,
            lambda rows: predicted[rows["final"].private_identity],
        ),
    )


def _quality_gates(metrics: EvidenceProxyMetrics) -> list[dict[str, Any]]:
    rows = (
        (
            "initial_visible_compose_rate",
            metrics.initial_visible_compose_rate,
            _GATE_THRESHOLDS["initial_visible_compose_rate_min"],
            "min",
        ),
        (
            "alternate_only_inspect_rate",
            metrics.alternate_only_inspect_rate,
            _GATE_THRESHOLDS["alternate_only_inspect_rate_min"],
            "min",
        ),
        (
            "alternate_only_final_compose_rate",
            metrics.alternate_only_final_compose_rate,
            _GATE_THRESHOLDS["alternate_only_final_compose_rate_min"],
            "min",
        ),
        (
            "alternate_only_path_success_rate",
            metrics.alternate_only_path_success_rate,
            _GATE_THRESHOLDS["alternate_only_path_success_rate_min"],
            "min",
        ),
        (
            "insufficient_final_compose_rate",
            metrics.insufficient_final_compose_rate,
            _GATE_THRESHOLDS["insufficient_final_compose_rate_max"],
            "max",
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
        for name, observed, threshold, direction in rows
    ]


def _outer_fold_metrics(
    cases: Sequence[EvidenceViewCase],
    predictions: Mapping[str, float],
    specs_by_fold: Mapping[str, EvidenceModelSpec],
) -> dict[str, dict[str, int | float]]:
    result = {}
    for fold_id in sorted(specs_by_fold):
        fold_cases = tuple(case for case in cases if case.fold_id == fold_id)
        result[fold_id] = evaluate_predictions(
            fold_cases,
            {case.private_identity: predictions[case.private_identity] for case in fold_cases},
            {fold_id: specs_by_fold[fold_id]},
        ).public_dict()
    return result


def _fold_metrics_for_spec(
    cases: Sequence[EvidenceViewCase],
    predictions: Mapping[str, float],
    spec: EvidenceModelSpec,
) -> dict[str, EvidenceProxyMetrics]:
    result = {}
    for fold_id in sorted({case.fold_id for case in cases}):
        fold_cases = tuple(case for case in cases if case.fold_id == fold_id)
        result[fold_id] = evaluate_predictions(
            fold_cases,
            {case.private_identity: predictions[case.private_identity] for case in fold_cases},
            {fold_id: spec},
        )
    return result


def _feature_matrix(cases: Sequence[EvidenceViewCase]) -> np.ndarray:
    return np.asarray(
        [[float(case.features[name]) for name in _MODEL_FEATURE_NAMES] for case in cases],
        dtype=float,
    )


def _feature_distribution(
    cases: Sequence[EvidenceViewCase],
) -> dict[str, dict[str, float | int]]:
    result = {}
    for name in _MODEL_FEATURE_NAMES:
        values = np.asarray([case.features[name] for case in cases], dtype=float)
        result[name] = {
            "minimum": round(float(values.min()), 12),
            "median": round(float(np.median(values)), 12),
            "maximum": round(float(values.max()), 12),
            "unique_count": len(set(values.tolist())),
        }
    return result


def _cases_by_group(
    cases: Sequence[EvidenceViewCase],
) -> dict[str, dict[str, EvidenceViewCase]]:
    grouped: dict[str, dict[str, EvidenceViewCase]] = {}
    for case in cases:
        grouped.setdefault(case.group_identity, {})[case.phase] = case
    if any(set(rows) != {"initial", "final"} for rows in grouped.values()):
        raise ValueError("Stage172 grouped cases require initial and final views")
    return grouped


def _deduplicate_records(
    records: Sequence[ContextCandidateRecord],
) -> tuple[ContextCandidateRecord, ...]:
    seen = set()
    selected = []
    for record in records:
        if record.document_id not in seen:
            selected.append(record)
            seen.add(record.document_id)
    return tuple(selected)


def _group_rate(groups: Sequence[Mapping[str, EvidenceViewCase]], predicate: Any) -> float:
    if not groups:
        raise ValueError("Stage172 proxy metric group cannot be empty")
    return round(sum(bool(predicate(group)) for group in groups) / len(groups), 6)


def _value(record: ContextCandidateRecord, name: str) -> float:
    return float(record.features.get(name, 0.0))


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values))


def _mean_values(records: Sequence[ContextCandidateRecord], name: str) -> float:
    return _mean([_value(record, name) for record in records])


def _max_value(records: Sequence[ContextCandidateRecord], name: str) -> float:
    return max(_value(record, name) for record in records)


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("Stage172 required rate denominator must be positive")
    return numerator / denominator


def _authorize_sources(fingerprints: Mapping[str, Mapping[str, Any]]) -> None:
    for name, expected in _SOURCE_HASHES.items():
        if fingerprints[name]["sha256"] != expected:
            raise ValueError(f"Stage172 source hash mismatch: {name}")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _emit(sink: ProgressSink | None, **event: Any) -> None:
    if sink is not None:
        sink({"stage": _STAGE, **event})


def _rate_bar(label: str, value: float) -> BarDatum:
    return BarDatum(label, value, f"{value:.1%}")


def _chart(
    title: str,
    bars: Sequence[BarDatum],
    *,
    x_label: str,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=bars,
        x_label=x_label,
        width=1200,
        margin_left=440,
        margin_right=200,
    )
