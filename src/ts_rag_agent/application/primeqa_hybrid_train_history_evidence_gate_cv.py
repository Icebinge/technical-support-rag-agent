from __future__ import annotations

import hashlib
import json
import os
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector_training import (
    Stage161TrainCandidateDatasetBuilder,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    load_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)

_STAGE = "Stage 167"
_CREATED_AT = "2026-07-22"
_ANALYSIS_ID = "primeqa_hybrid_train_history_evidence_gate_nested_cv_v1"
_FOLDS = (0, 1, 2, 3, 4)
_EXPECTED_TRAIN_ROWS = 562
_EXPECTED_POST_FIRST_CASES = 421
_EXPECTED_CANDIDATE_ROWS = 112_400
_CONTEXT_DEPTH = 10
_FLOAT_TOLERANCE = 1e-12
_MODEL_FAMILIES = ("logistic", "histogram_gbdt")
_BENEFIT_THRESHOLDS = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85)
_HARM_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
_ROUTES = (
    "error_or_log",
    "how_to_or_lookup",
    "install_upgrade_config",
    "limitation_or_restriction",
    "other",
    "security_bulletin_post_fix_behavior",
    "security_bulletin_remediation",
    "security_bulletin_vulnerability_detail",
)
_SOURCE_HASHES = {
    "stage161": "a13b8ee5538581f0eb87a649c48fdf4ae715b6cfa8a43a97b5115001f9cd1197",
    "stage165_correction": "589b65959069d4f12aacc0ff95c2a5f65df173ea0e8c67ca426c15026c01e29d",
    "stage165_private": "ce4b5b281093319696a51251d475a3fc5fa6b7dac2e7f9659464fe1d8e55ad1b",
    "stage166": "8430cb37aa8bc0764f607b050d6c8007472ffc1a4f2d94853333ece652e4892b",
    "stage80": "2441bb1cb1e7888299d3f57962b18cd59df84e2086ac281105abcacfc144880f",
    "train": "cabd93e0b972c47384c4bf5cc2cd215a7fc519b2df4f81fba61db73c931aa155",
    "documents": "f93b5e2d8dcfb2c7d12676ef32ce22b7809692f14081aad98096099a5256722b",
}

ProgressSink = Callable[[Mapping[str, Any]], None]
ModelFamily = Literal["logistic", "histogram_gbdt"]

_EVIDENCE_FEATURE_NAMES = (
    "rrf_top1",
    "rrf_top1_top2_margin",
    "rrf_top1_top10_spread",
    "rrf_top10_mean",
    "top1_route_hits",
    "top1_lexical_hits",
    "top1_dense_hits",
    "top1_best_route_inverse_rank",
    "top10_route_hits_mean",
    "top10_dense_hits_mean",
    "selection_rrf_mean",
    "selection_rrf_spread",
    "selection_baseline_rank_mean",
    "selection_baseline_rank_max",
    "selection_prefix_overlap_count",
    "selection_overlap_score_top1_top2_margin",
    "selection_overlap_count_mean",
    "selection_overlap_count_max",
    "selection_overlap_ratio_mean",
    "selection_query_coverage_mean",
    "selection_query_coverage_max",
    "selection_body_coverage_mean",
    "selection_title_overlap_mean",
    "selection_heading_overlap_mean",
    "selection_special_match_mean",
    "selection_special_match_max",
    "selection_bm25_top10_fraction",
)


@dataclass(frozen=True)
class Stage167EvidenceSummary:
    sample_id: str
    private_identity_sha256: str
    values: Mapping[str, float]


@dataclass(frozen=True)
class Stage167PairCase:
    private_identity_sha256: str
    diagnostic_group_sha256: str
    fold_id: int
    question_route: str
    turn_position: int
    answerable: bool
    evidence: Mapping[str, float]
    isolated_refused: bool
    synthetic_refused: bool
    isolated_f1: float
    synthetic_f1: float
    isolated_gold_cited: bool
    synthetic_gold_cited: bool

    @property
    def beneficial_label(self) -> bool:
        if self.answerable:
            nonregression = (
                int(self.isolated_refused) <= int(self.synthetic_refused)
                and self.isolated_f1 + _FLOAT_TOLERANCE >= self.synthetic_f1
                and int(self.isolated_gold_cited) >= int(self.synthetic_gold_cited)
            )
            strict_gain = (
                int(self.isolated_refused) < int(self.synthetic_refused)
                or self.isolated_f1 > self.synthetic_f1 + _FLOAT_TOLERANCE
                or int(self.isolated_gold_cited) > int(self.synthetic_gold_cited)
            )
            return nonregression and strict_gain
        return int(not self.isolated_refused) < int(not self.synthetic_refused)

    @property
    def harmful_label(self) -> bool:
        if self.answerable:
            return (
                int(self.isolated_refused) > int(self.synthetic_refused)
                or self.isolated_f1 + _FLOAT_TOLERANCE < self.synthetic_f1
                or int(self.isolated_gold_cited) < int(self.synthetic_gold_cited)
            )
        return int(not self.isolated_refused) > int(not self.synthetic_refused)


@dataclass(frozen=True)
class Stage167GateSpec:
    model_family: ModelFamily
    benefit_threshold: float
    harm_threshold: float

    @property
    def spec_id(self) -> str:
        material = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Stage167Prediction:
    benefit_probability: float
    harm_probability: float


@dataclass(frozen=True)
class Stage167PolicyMetrics:
    case_count: int
    isolated_selection_count: int
    answerable_count: int
    unanswerable_count: int
    answerable_refusal_count: int
    answerable_f1_sum: float
    answerable_average_f1: float
    answerable_gold_citation_count: int
    unanswerable_false_answer_count: int

    def public_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class Stage167Visualization:
    name: str
    path: str


class _DualRiskPredictor:
    """Fit independent benefit and harm classifiers over runtime-visible features."""

    def __init__(self, model_family: ModelFamily) -> None:
        self._model_family = model_family
        self._benefit_model = self._new_model()
        self._harm_model = self._new_model()

    def fit(self, cases: Sequence[Stage167PairCase]) -> None:
        matrix = _feature_matrix(cases)
        benefit = np.asarray([case.beneficial_label for case in cases], dtype=int)
        harm = np.asarray([case.harmful_label for case in cases], dtype=int)
        if len(set(benefit.tolist())) != 2 or len(set(harm.tolist())) != 2:
            raise ValueError("Stage167 model training requires both classes for both targets")
        self._fit_one(self._benefit_model, matrix, benefit)
        self._fit_one(self._harm_model, matrix, harm)

    def predict(self, cases: Sequence[Stage167PairCase]) -> dict[str, Stage167Prediction]:
        matrix = _feature_matrix(cases)
        benefit = self._benefit_model.predict_proba(matrix)[:, 1]
        harm = self._harm_model.predict_proba(matrix)[:, 1]
        return {
            case.private_identity_sha256: Stage167Prediction(float(b), float(h))
            for case, b, h in zip(cases, benefit, harm, strict=True)
        }

    def _new_model(self) -> Any:
        if self._model_family == "logistic":
            return Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=167,
                        ),
                    ),
                ]
            )
        if self._model_family == "histogram_gbdt":
            return HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=100,
                max_leaf_nodes=7,
                l2_regularization=1.0,
                random_state=167,
            )
        raise ValueError(f"Unsupported Stage167 model family: {self._model_family}")

    def _fit_one(self, model: Any, matrix: np.ndarray, labels: np.ndarray) -> None:
        if self._model_family == "histogram_gbdt":
            counts = np.bincount(labels, minlength=2)
            weights = np.asarray([len(labels) / (2 * counts[label]) for label in labels])
            model.fit(matrix, labels, sample_weight=weights)
            return
        model.fit(matrix, labels)


def build_stage167_gate_specs() -> tuple[Stage167GateSpec, ...]:
    return tuple(
        Stage167GateSpec(family, benefit, harm)
        for family in _MODEL_FAMILIES
        for benefit in _BENEFIT_THRESHOLDS
        for harm in _HARM_THRESHOLDS
    )


def summarize_candidate_evidence(
    *,
    sample_id: str,
    records: Sequence[ContextCandidateRecord],
) -> Stage167EvidenceSummary:
    if len(records) != 200:
        raise ValueError("Stage167 evidence summary requires exact Top200 records")
    baseline = tuple(sorted(records, key=lambda record: record.baseline_rank)[:_CONTEXT_DEPTH])
    selected = select_current_query_overlap_top10(records).selected
    if len(baseline) != _CONTEXT_DEPTH or len(selected) != _CONTEXT_DEPTH:
        raise ValueError("Stage167 evidence summary requires exact Top10 contexts")
    rrf = [_value(record, "stage116_rrf_score") for record in baseline]
    selected_rrf = [_value(record, "stage116_rrf_score") for record in selected]
    overlap_score = [_value(record, "current_query_overlap_combined_score") for record in selected]
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
        "selection_rrf_mean": _mean(selected_rrf),
        "selection_rrf_spread": max(selected_rrf) - min(selected_rrf),
        "selection_baseline_rank_mean": _mean([float(row.baseline_rank) for row in selected]),
        "selection_baseline_rank_max": float(max(row.baseline_rank for row in selected)),
        "selection_prefix_overlap_count": float(
            sum(record.document_id in baseline_ids for record in selected)
        ),
        "selection_overlap_score_top1_top2_margin": overlap_score[0] - overlap_score[1],
        "selection_overlap_count_mean": _mean_values(selected, "current_query_overlap_count"),
        "selection_overlap_count_max": _max_value(selected, "current_query_overlap_count"),
        "selection_overlap_ratio_mean": _mean_values(selected, "current_query_overlap_ratio"),
        "selection_query_coverage_mean": _mean_values(selected, "query_token_coverage"),
        "selection_query_coverage_max": _max_value(selected, "query_token_coverage"),
        "selection_body_coverage_mean": _mean_values(selected, "query_body_token_coverage"),
        "selection_title_overlap_mean": _mean_values(selected, "query_title_token_overlap"),
        "selection_heading_overlap_mean": _mean_values(selected, "query_section_heading_overlap"),
        "selection_special_match_mean": _mean_values(selected, "query_special_token_match_count"),
        "selection_special_match_max": _max_value(selected, "query_special_token_match_count"),
        "selection_bm25_top10_fraction": _mean_values(selected, "bm25_top10_indicator"),
    }
    if tuple(values) != _EVIDENCE_FEATURE_NAMES:
        raise RuntimeError("Stage167 evidence feature order drifted")
    return Stage167EvidenceSummary(
        sample_id=sample_id,
        private_identity_sha256=_sha256_text(sample_id),
        values=values,
    )


def run_stage167_evidence_gate_nested_cv(
    *,
    stage161_report_path: Path,
    stage165_correction_path: Path,
    stage165_private_path: Path,
    stage166_report_path: Path,
    stage80_report_path: Path,
    train_split_path: Path,
    documents_path: Path,
    user_confirmed_stage167: bool,
    confirmation_note: str,
    encoder_batch_size: int = 64,
    encoder_device: str | None = None,
    progress_sink: ProgressSink | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    paths = {
        "stage161": stage161_report_path,
        "stage165_correction": stage165_correction_path,
        "stage165_private": stage165_private_path,
        "stage166": stage166_report_path,
        "stage80": stage80_report_path,
        "train": train_split_path,
        "documents": documents_path,
    }
    fingerprints = {name: _fingerprint(path) for name, path in paths.items()}
    _authorize_sources(fingerprints)
    stage161_report = _load_json_object(stage161_report_path)
    correction = _load_json_object(stage165_correction_path)
    private = _load_json_object(stage165_private_path)
    stage166_report = _load_json_object(stage166_report_path)
    _authorize_reports(stage161_report, correction, private, stage166_report)
    protocol = _frozen_protocol()
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="sources_and_protocol_authorized")

    samples = load_primeqa_hybrid_split_samples(train_split_path)
    if len(samples) != _EXPECTED_TRAIN_ROWS or any(
        row.assigned_split != "train" for row in samples
    ):
        raise ValueError("Stage167 accepts only the exact 562-row train split")
    documents_by_id = load_primeqa_documents(documents_path)
    sections_by_document = load_primeqa_document_sections(documents_path)
    documents = list(documents_by_id.values())
    fold_assignments = _build_train_fold_assignments(samples, fold_count=5)
    loaded_at = time.perf_counter()
    _emit(progress_sink, phase="train_and_documents_loaded", train_rows=len(samples))

    stage80_report = _load_json_object(stage80_report_path)
    dense_channels, dense_summary = _build_dense_channels(
        include_dense_channels=True,
        stage80_report=stage80_report,
        stage80_report_path=stage80_report_path,
        documents=documents,
        document_ids=tuple(document.id for document in documents),
        encoder_batch_size=encoder_batch_size,
        encoder_device=encoder_device,
        encoder_factory=None,
    )
    if dense_summary["status"] != "dense_channels_ready":
        raise RuntimeError("Stage167 requires the two authorized local dense channels")
    lexical_channels = _build_lexical_channels(
        documents=documents,
        sections_by_document=sections_by_document,
        bm25_k1=1.5,
        bm25_b=0.75,
        component_depth=200,
    )
    channels = tuple([*lexical_channels, *dense_channels])
    channels_at = time.perf_counter()
    _emit(progress_sink, phase="retrieval_channels_ready", channel_count=len(channels))

    records = Stage161TrainCandidateDatasetBuilder(
        documents_by_id=documents_by_id,
        sections_by_document=sections_by_document,
        channels=channels,
        fold_assignments=fold_assignments,
        progress_sink=progress_sink,
        progress_stage=_STAGE,
        progress_phase="train_evidence_replay",
    ).build(samples)
    if len(records) != _EXPECTED_CANDIDATE_ROWS:
        raise RuntimeError("Stage167 candidate replay row count drifted")
    grouped = records_by_sample(records)
    evidence = {
        summary.private_identity_sha256: summary
        for sample in samples
        for summary in [
            summarize_candidate_evidence(
                sample_id=sample.sample_id, records=grouped[sample.sample_id]
            )
        ]
    }
    evidence_at = time.perf_counter()
    _emit(progress_sink, phase="evidence_summaries_ready", summary_count=len(evidence))

    cases = _build_cases(private["rows"], evidence)
    specs = build_stage167_gate_specs()
    outer_folds, oof_predictions, oof_specs = _run_nested_cv(cases, specs, progress_sink)
    baseline = evaluate_policy(cases, {}, None)
    oof = _evaluate_per_case_specs(cases, oof_predictions, oof_specs)
    oof_by_fold = _fold_metrics(cases, oof_predictions, oof_specs)
    analyzed_at = time.perf_counter()

    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train-only nested five-fold history-isolation gate analysis. It deterministically "
            "replays the frozen Stage161 retrieval contract to derive runtime-visible Top10 "
            "evidence summaries, joins Stage165 paired outcomes only as training labels, and "
            "loads neither development nor test."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_stage167),
            "confirmation_note": confirmation_note,
        },
        "source_authorization": fingerprints,
        "frozen_protocol": protocol,
        "frozen_protocol_sha256": _canonical_json_sha256(protocol),
        "evidence_provenance": {
            "kind": "deterministic_train_only_recomputation",
            "not_direct_stage165_candidate_capture": True,
            "reason": "Stage165 persisted context hashes but no per-candidate feature rows",
            "retrieval_contract": "stage161_exact_stage116_top200_plus_current_query_overlap_top10",
            "online_candidate_pool_rebuild_added": False,
            "runtime_feature_computation_scope": "existing_retrieval_results_only",
        },
        "split_contract": {
            "loaded_split": "train",
            "fit_split": "train_outer_four_folds_only",
            "inner_selection": "leave_one_training_fold_out_predictions",
            "outer_evaluation": "one_shot_heldout_fold",
            "development_loaded": False,
            "test_loaded": False,
        },
        "feature_contract": {
            "numeric_evidence_features": list(_EVIDENCE_FEATURE_NAMES),
            "categorical_runtime_features": ["question_route", "synthetic_turn_position"],
            "model_input_feature_count": len(_EVIDENCE_FEATURE_NAMES) + len(_ROUTES) + 3,
            "decision_time": "after_existing_retrieval_before_router_generation",
            "forbidden_runtime_features": [
                "answerable_label",
                "gold_document",
                "gold_rank",
                "selected_action",
                "refused_outcome",
                "answer_f1",
                "citation_outcome",
                "beneficial_label",
                "harmful_label",
            ],
            "feature_distribution": _feature_distribution(cases),
        },
        "case_summary": {
            "case_count": len(cases),
            "answerable_count": sum(case.answerable for case in cases),
            "unanswerable_count": sum(not case.answerable for case in cases),
            "beneficial_label_count": sum(case.beneficial_label for case in cases),
            "harmful_label_count": sum(case.harmful_label for case in cases),
            "neutral_label_count": sum(
                not case.beneficial_label and not case.harmful_label for case in cases
            ),
            "evidence_summary_count": len(evidence),
            "candidate_record_count_in_memory": len(records),
            "private_case_rows_written": False,
        },
        "candidate_family": {
            "model_families": list(_MODEL_FAMILIES),
            "benefit_thresholds": list(_BENEFIT_THRESHOLDS),
            "harm_thresholds": list(_HARM_THRESHOLDS),
            "spec_count": len(specs),
            "family_sha256": _canonical_json_sha256([asdict(spec) for spec in specs]),
        },
        "nested_cv": {
            "outer_folds": outer_folds,
            "oof_candidate": oof.public_dict(),
            "oof_baseline": baseline.public_dict(),
            "oof_delta": _metric_delta(oof, baseline),
            "oof_by_fold": oof_by_fold,
            "strict_nonregression_fold_count": sum(
                values["strict_nonregression"] for values in oof_by_fold.values()
            ),
        },
        "controls": {
            "always_synthetic_history": baseline.public_dict(),
            "always_isolated": _evaluate_choices(
                cases,
                {case.private_identity_sha256: True for case in cases},
            ).public_dict(),
            "offline_pareto_oracle": _evaluate_oracle(cases).public_dict(),
        },
        "execution_counts": {
            "retrieval_channel_builds": 1,
            "train_query_retrievals": len(samples),
            "candidate_feature_rows_computed": len(records),
            "model_generation_calls": 0,
            "agent_turns": 0,
            "development_rows_loaded": 0,
            "test_rows_loaded": 0,
            "fallback_actions": 0,
        },
        "dense_channel_preflight": dense_summary,
        "timing_seconds": {
            "authorize": round(authorized_at - started_at, 6),
            "load_train_and_documents": round(loaded_at - authorized_at, 6),
            "build_retrieval_channels": round(channels_at - loaded_at, 6),
            "replay_candidates_and_summarize_evidence": round(evidence_at - channels_at, 6),
            "nested_cv_and_report": round(analyzed_at - evidence_at, 6),
            "total": round(analyzed_at - started_at, 6),
        },
    }
    report["guard_checks"] = _guard_checks(report)
    report["public_safe_contract"] = _public_safe_contract(report)
    passed = all(check["passed"] for check in report["guard_checks"])
    delta = report["nested_cv"]["oof_delta"]
    selected = (
        passed
        and _strict_nonregression_values(delta)
        and any(
            abs(float(delta[key])) > _FLOAT_TOLERANCE
            for key in (
                "answerable_refusal_count",
                "answerable_f1_sum",
                "answerable_gold_citation_count",
                "unanswerable_false_answer_count",
            )
        )
    )
    report["decision"] = {
        "status": (
            "primeqa_hybrid_stage167_evidence_gate_train_nested_cv_passed"
            if selected
            else "primeqa_hybrid_stage167_evidence_gate_not_train_safe"
        ),
        "all_process_guards_passed": passed,
        "candidate_selected": selected,
        "development_gate_opened": False,
        "test_gate_opened": False,
        "runtime_registered_as_default": False,
        "fallback_strategies_enabled": False,
        "next_direction": (
            "freeze_selected_gate_for_independent_dev_validation"
            if selected
            else "stop_history_isolation_gate_and_begin_agent_capability_work"
        ),
    }
    return report


def evaluate_policy(
    cases: Sequence[Stage167PairCase],
    predictions: Mapping[str, Stage167Prediction],
    spec: Stage167GateSpec | None,
) -> Stage167PolicyMetrics:
    selections = {
        case.private_identity_sha256: _select_isolated(case, predictions, spec) for case in cases
    }
    return _evaluate_choices(cases, selections)


def write_stage167_visualizations(
    *, report: Mapping[str, Any], output_dir: Path
) -> tuple[Stage167Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    nested = report["nested_cv"]
    folds = nested["outer_folds"]
    delta = nested["oof_delta"]
    charts = {
        "stage167_oof_metric_deltas.svg": _chart(
            "Stage167 OOF metric deltas vs always synthetic",
            [
                _bar("answerable refusals", delta["answerable_refusal_count"]),
                _bar("answerable F1 sum", delta["answerable_f1_sum"]),
                _bar("gold citations", delta["answerable_gold_citation_count"]),
                _bar("unanswerable false answers", delta["unanswerable_false_answer_count"]),
            ],
            "delta; lower is better for refusals and false answers",
        ),
        "stage167_outer_selected_counts.svg": _chart(
            "Stage167 held-out isolated selections",
            [
                _bar(
                    f"fold {row['heldout_fold']}",
                    row["heldout_metrics"]["isolated_selection_count"],
                )
                for row in folds
            ],
            "cases",
        ),
        "stage167_inner_eligible_specs.svg": _chart(
            "Stage167 inner-OOF eligible specifications",
            [
                _bar(f"fold {row['heldout_fold']}", row["inner_eligible_spec_count"])
                for row in folds
            ],
            "specifications",
        ),
        "stage167_label_distribution.svg": _chart(
            "Stage167 paired training label distribution",
            [
                _bar("beneficial", report["case_summary"]["beneficial_label_count"]),
                _bar("harmful", report["case_summary"]["harmful_label_count"]),
                _bar("neutral", report["case_summary"]["neutral_label_count"]),
            ],
            "cases",
        ),
        "stage167_fold_safety_deltas.svg": _chart(
            "Stage167 held-out unanswerable false-answer deltas",
            [
                _bar(
                    f"fold {row['heldout_fold']}",
                    row["heldout_delta"]["unanswerable_false_answer_count"],
                )
                for row in folds
            ],
            "false-answer count delta",
        ),
        "stage167_guard_status.svg": _chart(
            "Stage167 process guards",
            [_bar(str(row["name"]), bool(row["passed"])) for row in report["guard_checks"]],
            "1 means passed",
            margin_left=560,
        ),
    }
    visualizations = []
    for name, svg in charts.items():
        path = output_dir / name
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        visualizations.append(Stage167Visualization(name=name, path=str(path)))
    return tuple(visualizations)


def _run_nested_cv(
    cases: Sequence[Stage167PairCase],
    specs: Sequence[Stage167GateSpec],
    progress_sink: ProgressSink | None,
) -> tuple[list[dict[str, Any]], dict[str, Stage167Prediction], dict[str, Stage167GateSpec | None]]:
    rows = []
    oof_predictions: dict[str, Stage167Prediction] = {}
    oof_specs: dict[str, Stage167GateSpec | None] = {}
    for outer_fold in _FOLDS:
        train = tuple(case for case in cases if case.fold_id != outer_fold)
        heldout = tuple(case for case in cases if case.fold_id == outer_fold)
        inner_predictions: dict[str, dict[str, Stage167Prediction]] = {
            family: {} for family in _MODEL_FAMILIES
        }
        for family in _MODEL_FAMILIES:
            for inner_fold in sorted(set(case.fold_id for case in train)):
                inner_train = tuple(case for case in train if case.fold_id != inner_fold)
                inner_heldout = tuple(case for case in train if case.fold_id == inner_fold)
                predictor = _DualRiskPredictor(family)
                predictor.fit(inner_train)
                inner_predictions[family].update(predictor.predict(inner_heldout))
        eligible = [
            spec
            for spec in specs
            if _strict_inner_eligible(train, inner_predictions[spec.model_family], spec)
        ]
        selected = _select_spec(train, inner_predictions, eligible)
        heldout_predictions: dict[str, Stage167Prediction] = {}
        if selected is not None:
            predictor = _DualRiskPredictor(selected.model_family)
            predictor.fit(train)
            heldout_predictions = predictor.predict(heldout)
        candidate = evaluate_policy(heldout, heldout_predictions, selected)
        baseline = evaluate_policy(heldout, {}, None)
        oof_predictions.update(heldout_predictions)
        for case in heldout:
            oof_specs[case.private_identity_sha256] = selected
        rows.append(
            {
                "heldout_fold": outer_fold,
                "train_case_count": len(train),
                "heldout_case_count": len(heldout),
                "candidate_spec_count": len(specs),
                "inner_eligible_spec_count": len(eligible),
                "selected_spec": asdict(selected) if selected else None,
                "heldout_metrics": candidate.public_dict(),
                "heldout_baseline": baseline.public_dict(),
                "heldout_delta": _metric_delta(candidate, baseline),
                "heldout_strict_nonregression": _strict_nonregression(candidate, baseline),
            }
        )
        _emit(progress_sink, phase="outer_fold_completed", completed=outer_fold + 1, total=5)
    return rows, oof_predictions, oof_specs


def _strict_inner_eligible(
    cases: Sequence[Stage167PairCase],
    predictions: Mapping[str, Stage167Prediction],
    spec: Stage167GateSpec,
) -> bool:
    candidate = evaluate_policy(cases, predictions, spec)
    baseline = evaluate_policy(cases, {}, None)
    delta = _metric_delta(candidate, baseline)
    if not _strict_nonregression_values(delta) or not _has_strict_gain(delta):
        return False
    for fold in sorted(set(case.fold_id for case in cases)):
        fold_cases = tuple(case for case in cases if case.fold_id == fold)
        if not _strict_nonregression(
            evaluate_policy(fold_cases, predictions, spec),
            evaluate_policy(fold_cases, {}, None),
        ):
            return False
    return True


def _select_spec(
    cases: Sequence[Stage167PairCase],
    predictions: Mapping[str, Mapping[str, Stage167Prediction]],
    eligible: Sequence[Stage167GateSpec],
) -> Stage167GateSpec | None:
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda spec: _selection_key(
            evaluate_policy(cases, predictions[spec.model_family], spec),
            evaluate_policy(cases, {}, None),
            spec,
        ),
    )


def _selection_key(
    candidate: Stage167PolicyMetrics,
    baseline: Stage167PolicyMetrics,
    spec: Stage167GateSpec,
) -> tuple[Any, ...]:
    delta = _metric_delta(candidate, baseline)
    return (
        delta["answerable_f1_sum"],
        delta["answerable_gold_citation_count"],
        -delta["answerable_refusal_count"],
        -delta["unanswerable_false_answer_count"],
        -candidate.isolated_selection_count,
        spec.spec_id,
    )


def _evaluate_per_case_specs(
    cases: Sequence[Stage167PairCase],
    predictions: Mapping[str, Stage167Prediction],
    specs: Mapping[str, Stage167GateSpec | None],
) -> Stage167PolicyMetrics:
    return _evaluate_choices(
        cases,
        {
            case.private_identity_sha256: _select_isolated(
                case, predictions, specs[case.private_identity_sha256]
            )
            for case in cases
        },
    )


def _fold_metrics(
    cases: Sequence[Stage167PairCase],
    predictions: Mapping[str, Stage167Prediction],
    specs: Mapping[str, Stage167GateSpec | None],
) -> dict[str, dict[str, Any]]:
    result = {}
    for fold in _FOLDS:
        fold_cases = tuple(case for case in cases if case.fold_id == fold)
        candidate = _evaluate_per_case_specs(fold_cases, predictions, specs)
        baseline = evaluate_policy(fold_cases, {}, None)
        result[str(fold)] = {
            "candidate": candidate.public_dict(),
            "baseline": baseline.public_dict(),
            "delta": _metric_delta(candidate, baseline),
            "strict_nonregression": _strict_nonregression(candidate, baseline),
        }
    return result


def _select_isolated(
    case: Stage167PairCase,
    predictions: Mapping[str, Stage167Prediction],
    spec: Stage167GateSpec | None,
) -> bool:
    if spec is None:
        return False
    prediction = predictions.get(case.private_identity_sha256)
    if prediction is None:
        raise ValueError("Stage167 gate prediction is missing for a selected specification")
    return (
        prediction.benefit_probability >= spec.benefit_threshold
        and prediction.harm_probability <= spec.harm_threshold
    )


def _evaluate_choices(
    cases: Sequence[Stage167PairCase], choices: Mapping[str, bool]
) -> Stage167PolicyMetrics:
    refusal = 0
    f1_sum = 0.0
    citations = 0
    false_answers = 0
    selected_count = 0
    for case in cases:
        isolated = choices.get(case.private_identity_sha256, False)
        selected_count += isolated
        selected_refused = case.isolated_refused if isolated else case.synthetic_refused
        if case.answerable:
            refusal += selected_refused
            f1_sum += case.isolated_f1 if isolated else case.synthetic_f1
            citations += case.isolated_gold_cited if isolated else case.synthetic_gold_cited
        else:
            false_answers += not selected_refused
    answerable_count = sum(case.answerable for case in cases)
    return Stage167PolicyMetrics(
        case_count=len(cases),
        isolated_selection_count=selected_count,
        answerable_count=answerable_count,
        unanswerable_count=len(cases) - answerable_count,
        answerable_refusal_count=refusal,
        answerable_f1_sum=round(f1_sum, 12),
        answerable_average_f1=round(f1_sum / max(1, answerable_count), 12),
        answerable_gold_citation_count=citations,
        unanswerable_false_answer_count=false_answers,
    )


def _evaluate_oracle(cases: Sequence[Stage167PairCase]) -> Stage167PolicyMetrics:
    return _evaluate_choices(
        cases, {case.private_identity_sha256: case.beneficial_label for case in cases}
    )


def _metric_delta(
    candidate: Stage167PolicyMetrics, baseline: Stage167PolicyMetrics
) -> dict[str, int | float]:
    return {
        "isolated_selection_count": candidate.isolated_selection_count,
        "answerable_refusal_count": candidate.answerable_refusal_count
        - baseline.answerable_refusal_count,
        "answerable_f1_sum": round(candidate.answerable_f1_sum - baseline.answerable_f1_sum, 12),
        "answerable_average_f1": round(
            candidate.answerable_average_f1 - baseline.answerable_average_f1, 12
        ),
        "answerable_gold_citation_count": candidate.answerable_gold_citation_count
        - baseline.answerable_gold_citation_count,
        "unanswerable_false_answer_count": candidate.unanswerable_false_answer_count
        - baseline.unanswerable_false_answer_count,
    }


def _strict_nonregression(
    candidate: Stage167PolicyMetrics, baseline: Stage167PolicyMetrics
) -> bool:
    return _strict_nonregression_values(_metric_delta(candidate, baseline))


def _strict_nonregression_values(delta: Mapping[str, Any]) -> bool:
    return (
        int(delta["answerable_refusal_count"]) <= 0
        and float(delta["answerable_f1_sum"]) >= -_FLOAT_TOLERANCE
        and int(delta["answerable_gold_citation_count"]) >= 0
        and int(delta["unanswerable_false_answer_count"]) <= 0
    )


def _has_strict_gain(delta: Mapping[str, Any]) -> bool:
    return (
        int(delta["answerable_refusal_count"]) < 0
        or float(delta["answerable_f1_sum"]) > _FLOAT_TOLERANCE
        or int(delta["answerable_gold_citation_count"]) > 0
        or int(delta["unanswerable_false_answer_count"]) < 0
    )


def _build_cases(
    rows: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Stage167EvidenceSummary],
) -> tuple[Stage167PairCase, ...]:
    by_identity: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if int(row["synthetic_turn_position"]) == 1:
            continue
        by_identity.setdefault(str(row["private_identity_sha256"]), {})[str(row["arm"])] = row
    cases = []
    for identity, arms in sorted(by_identity.items()):
        isolated = arms["isolated"]
        synthetic = arms["synthetic_history"]
        summary = evidence.get(identity)
        if summary is None:
            raise ValueError("Stage167 could not align a Stage165 case with train evidence")
        cases.append(
            Stage167PairCase(
                private_identity_sha256=identity,
                diagnostic_group_sha256=str(isolated["diagnostic_group_sha256"]),
                fold_id=int(isolated["fold_id"]),
                question_route=str(isolated["question_route"]),
                turn_position=int(isolated["synthetic_turn_position"]),
                answerable=bool(isolated["answerable"]),
                evidence=summary.values,
                isolated_refused=bool(isolated["refused"]),
                synthetic_refused=bool(synthetic["refused"]),
                isolated_f1=float(isolated["answer_token_f1"] or 0.0),
                synthetic_f1=float(synthetic["answer_token_f1"] or 0.0),
                isolated_gold_cited=bool(isolated["gold_cited"]),
                synthetic_gold_cited=bool(synthetic["gold_cited"]),
            )
        )
    if len(cases) != _EXPECTED_POST_FIRST_CASES:
        raise ValueError("Stage167 post-first case count drifted")
    return tuple(cases)


def _feature_matrix(cases: Sequence[Stage167PairCase]) -> np.ndarray:
    rows = []
    for case in cases:
        numeric = [float(case.evidence[name]) for name in _EVIDENCE_FEATURE_NAMES]
        routes = [float(case.question_route == route) for route in _ROUTES]
        positions = [float(case.turn_position == position) for position in (2, 3, 4)]
        rows.append([*numeric, *routes, *positions])
    return np.asarray(rows, dtype=float)


def _feature_distribution(cases: Sequence[Stage167PairCase]) -> dict[str, dict[str, float | int]]:
    result = {}
    for name in _EVIDENCE_FEATURE_NAMES:
        values = np.asarray([case.evidence[name] for case in cases], dtype=float)
        result[name] = {
            "minimum": round(float(values.min()), 12),
            "median": round(float(np.median(values)), 12),
            "maximum": round(float(values.max()), 12),
            "unique_count": len(set(values.tolist())),
        }
    return result


def _frozen_protocol() -> dict[str, Any]:
    return {
        "protocol_version": "primeqa_hybrid_stage167_v1",
        "evidence_source": "frozen_stage161_train_replay",
        "candidate_pool_depth": 200,
        "generation_context_depth": 10,
        "model_families": list(_MODEL_FAMILIES),
        "benefit_thresholds": list(_BENEFIT_THRESHOLDS),
        "harm_thresholds": list(_HARM_THRESHOLDS),
        "benefit_label": "per_case_pareto_nonregression_with_strict_gain",
        "harm_label": "per_case_any_quality_or_safety_regression",
        "selection": "inner_oof_strict_all_fold_nonregression_quality_first",
        "outer_cv": "five_fixed_stage165_grouped_folds",
        "dev_test_closed": True,
        "fallback_enabled": False,
    }


def _authorize_sources(fingerprints: Mapping[str, Mapping[str, Any]]) -> None:
    for name, expected in _SOURCE_HASHES.items():
        if fingerprints[name]["sha256"] != expected:
            raise ValueError(f"Stage167 source hash mismatch: {name}")


def _authorize_reports(
    stage161_report: Mapping[str, Any],
    correction: Mapping[str, Any],
    private: Mapping[str, Any],
    stage166_report: Mapping[str, Any],
) -> None:
    checks = (
        stage161_report.get("decision", {}).get("status")
        == "primeqa_hybrid_protected_context_selector_no_train_cv_safe_config",
        correction.get("decision", {}).get("status")
        == "primeqa_hybrid_stage165_transition_correction_completed",
        private.get("arm_row_count") == 1124,
        stage166_report.get("decision", {}).get("status")
        == "primeqa_hybrid_stage166_runtime_feature_family_insufficient",
    )
    if not all(checks):
        raise ValueError("Stage167 source report contract mismatch")


def _guard_checks(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    split = report["split_contract"]
    counts = report["execution_counts"]
    features = report["feature_contract"]
    checks = [
        ("user_confirmed", report["user_confirmation"]["confirmed"]),
        ("exact_train_case_count", report["case_summary"]["case_count"] == 421),
        (
            "exact_candidate_replay",
            report["case_summary"]["candidate_record_count_in_memory"] == 112400,
        ),
        (
            "all_evidence_features_nonconstant",
            all(row["unique_count"] > 1 for row in features["feature_distribution"].values()),
        ),
        ("nested_outer_five_fold_complete", len(report["nested_cv"]["outer_folds"]) == 5),
        ("development_closed", split["development_loaded"] is False),
        ("test_closed", split["test_loaded"] is False),
        ("no_generation_or_agent", counts["model_generation_calls"] == counts["agent_turns"] == 0),
        ("no_fallback", counts["fallback_actions"] == 0),
        (
            "no_private_case_rows_written",
            report["case_summary"]["private_case_rows_written"] is False,
        ),
        (
            "runtime_does_not_rebuild_candidate_pool",
            report["evidence_provenance"]["online_candidate_pool_rebuild_added"] is False,
        ),
    ]
    return [{"name": name, "passed": bool(passed)} for name, passed in checks]


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = {"sample_id", "private_identity_sha256", "question_text", "document_id", "answer"}
    found = sorted(_find_keys(report) & forbidden)
    return {
        "forbidden_keys_found": found,
        "contains_case_rows": False,
        "contains_raw_question": False,
        "contains_raw_answer": False,
        "contains_raw_document_id": False,
        "public_safe": not found,
    }


def _find_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_find_keys(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            keys.update(_find_keys(child))
    return keys


def _fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _value(record: ContextCandidateRecord, name: str) -> float:
    return float(record.features.get(name, 0.0))


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _mean_values(records: Sequence[ContextCandidateRecord], name: str) -> float:
    return _mean([_value(record, name) for record in records])


def _max_value(records: Sequence[ContextCandidateRecord], name: str) -> float:
    return max(_value(record, name) for record in records)


def _emit(progress_sink: ProgressSink | None, *, phase: str, **values: Any) -> None:
    if progress_sink is not None:
        progress_sink({"stage": _STAGE, "phase": phase, **values})


def _bar(label: str, value: int | float | bool) -> BarDatum:
    return BarDatum(label=label, value=float(value), value_label=str(value))


def _chart(
    title: str,
    data: Sequence[BarDatum],
    x_label: str,
    *,
    margin_left: int = 300,
) -> str:
    return render_horizontal_bar_chart_svg(
        title=title,
        bars=data,
        x_label=x_label,
        width=980,
        margin_left=margin_left,
    )
