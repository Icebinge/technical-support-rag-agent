from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    EncoderFactory,
    _build_dense_channels,
    _build_lexical_channels,
    _build_train_fold_assignments,
    _load_json_object,
)
from ts_rag_agent.application.primeqa_hybrid_second_stage_reranking_validation import (
    _bm25_top10_gold_demotions,
    _build_candidate_records,
    _candidate_pool_summary,
    _CandidateRecord,
    _compare_to_baseline,
    _evaluate_baseline,
    _fingerprint,
    _gold_rank,
    _rank_metrics,
    _records_by_sample,
    _rounded_mean,
    _rounded_ratio,
    _section_summary,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
    summarize_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)

_STAGE = "Stage 121"
_CREATED_AT = "2026-07-16"
_ANALYSIS_ID = "primeqa_hybrid_fast_filter_screening_train_cv_dev_validation_v1"
_SOURCE_STAGE120_STATUS = "primeqa_hybrid_fast_filter_screening_protocol_frozen"
_SOURCE_PROTOCOL_ID = "primeqa_hybrid_fast_filter_screening_protocol_v1"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_FORBIDDEN_FINAL_SPLITS = ("test",)
_BASELINE_CONFIG_ID = "stage116_fixed_rrf_pool_order"
_TOP_K_VALUES = (10, 20, 50, 100, 200)
_PRIMARY_TOP_K = 20
_MINIMUM_POSITIVE_OBJECTIVE = 0.0
_RUNTIME_FEATURES = (
    "stage116_rrf_score",
    "route_hit_count",
    "lexical_route_hit_count",
    "dense_route_hit_count",
    "best_route_inverse_rank",
    "full_document_bm25_rank_inverse",
    "section_bm25_max_section_rollup_rank_inverse",
    "title_heading_weighted_bm25_rank_inverse",
    "special_token_boosted_bm25_rank_inverse",
    "query_title_token_overlap",
    "query_section_heading_overlap",
    "query_token_coverage",
    "query_body_token_coverage",
    "query_special_token_match_count",
    "title_special_token_match_count",
    "heading_special_token_match_count",
    "bm25_top10_indicator",
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer",
        "answer_doc_id",
        "candidate_doc_ids",
        "cited_doc_ids",
        "document_id",
        "document_text",
        "document_title",
        "gold_answer",
        "matched_token_strings",
        "question_text",
        "question_title",
        "raw_answer_text",
        "raw_document_text",
        "raw_question_text",
        "retrieved_doc_ids",
        "source_doc_ids",
    }
)


class _ScreeningScorer(Protocol):
    def fit(self, records: Sequence[_CandidateRecord]) -> _ScreeningScorer:
        """Fit scorer on candidate records."""

    def score(self, records: Sequence[_CandidateRecord]) -> list[float]:
        """Return one score per candidate, higher is better."""


@dataclass(frozen=True)
class _ScreeningOutcome:
    gold_rank: int | None
    promoted_tail_docs_into_top10: int
    promoted_tail_docs_into_top20: int
    screened_candidate_count: int


@dataclass(frozen=True)
class PrimeQAHybridFastFilterScreeningVisualization:
    """One generated Stage121 fast-filter screening validation chart."""

    name: str
    path: str


class _DeterministicScreeningScorer:
    def __init__(self, *, weights: Mapping[str, float]) -> None:
        self._weights = dict(weights)

    def fit(self, records: Sequence[_CandidateRecord]) -> _DeterministicScreeningScorer:
        _ = records
        return self

    def score(self, records: Sequence[_CandidateRecord]) -> list[float]:
        return [
            sum(
                record.features.get(feature, 0.0) * weight
                for feature, weight in self._weights.items()
            )
            for record in records
        ]


class _LogisticScreeningScorer:
    def __init__(self, *, feature_names: Sequence[str]) -> None:
        self._feature_names = tuple(feature_names)
        self._pipeline = Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(class_weight="balanced", max_iter=1000),
                ),
            ]
        )

    def fit(self, records: Sequence[_CandidateRecord]) -> _LogisticScreeningScorer:
        labels = [int(record.is_gold) for record in records]
        if len(set(labels)) < 2:
            raise ValueError("logistic screening needs both positive and negative labels.")
        self._pipeline.fit(
            [_project_feature_dict(record, self._feature_names) for record in records],
            labels,
        )
        return self

    def score(self, records: Sequence[_CandidateRecord]) -> list[float]:
        probabilities = self._pipeline.predict_proba(
            [_project_feature_dict(record, self._feature_names) for record in records]
        )
        positive_index = list(self._pipeline.classes_).index(1)
        return [float(row[positive_index]) for row in probabilities]


class _HistGradientScreeningScorer:
    def __init__(self, *, feature_names: Sequence[str]) -> None:
        self._feature_names = tuple(feature_names)
        self._pipeline = Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        class_weight="balanced",
                        learning_rate=0.05,
                        max_iter=80,
                        random_state=0,
                    ),
                ),
            ]
        )

    def fit(self, records: Sequence[_CandidateRecord]) -> _HistGradientScreeningScorer:
        labels = [int(record.is_gold) for record in records]
        if len(set(labels)) < 2:
            raise ValueError("hist-gradient screening needs both positive and negative labels.")
        self._pipeline.fit(
            [_project_feature_dict(record, self._feature_names) for record in records],
            labels,
        )
        return self

    def score(self, records: Sequence[_CandidateRecord]) -> list[float]:
        probabilities = self._pipeline.predict_proba(
            [_project_feature_dict(record, self._feature_names) for record in records]
        )
        positive_index = list(self._pipeline.classes_).index(1)
        return [float(row[positive_index]) for row in probabilities]


def run_primeqa_hybrid_fast_filter_screening_validation(
    *,
    stage120_protocol_path: Path,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    stage80_report_path: Path | None = None,
    user_confirmed_validation: bool,
    confirmation_note: str,
    include_dense_channels: bool = True,
    encoder_batch_size: int = 64,
    encoder_device: str | None = None,
    encoder_factory: EncoderFactory | None = None,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> dict[str, Any]:
    """Run Stage121 train-CV/dev validation for fast-filter screening."""

    started_at = time.perf_counter()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    stage120_report = _load_json_object(stage120_protocol_path)
    frozen_protocol = stage120_report.get("frozen_protocol") or {}
    stage120_summary = _stage120_summary(stage120_report)
    candidate_configs = tuple(frozen_protocol.get("candidate_configs") or ())
    selection_rules = frozen_protocol.get("selection_rules") or {}
    pool_contract = frozen_protocol.get("fixed_candidate_pool_contract") or {}
    candidate_pool_depth = int(pool_contract.get("candidate_pool_depth") or 200)
    channel_top_k = candidate_pool_depth
    rrf_k = 60
    loaded_protocol_at = time.perf_counter()

    split_samples = {
        _TRAIN_SPLIT: load_primeqa_hybrid_split_samples(train_split_path),
        _DEV_SPLIT: load_primeqa_hybrid_split_samples(dev_split_path),
    }
    train_fold_assignments = _build_train_fold_assignments(
        split_samples[_TRAIN_SPLIT],
        fold_count=int(selection_rules.get("minimum_train_folds") or 5),
    )
    loaded_splits_at = time.perf_counter()

    documents_by_id = load_primeqa_documents(documents_path)
    sections_by_document = load_primeqa_document_sections(documents_path)
    documents = list(documents_by_id.values())
    document_ids = tuple(document.id for document in documents)
    stage80_report = _load_json_object(stage80_report_path) if stage80_report_path else None
    loaded_documents_at = time.perf_counter()

    dense_channels, dense_summary = _build_dense_channels(
        include_dense_channels=include_dense_channels,
        stage80_report=stage80_report,
        stage80_report_path=stage80_report_path,
        documents=documents,
        document_ids=document_ids,
        encoder_batch_size=encoder_batch_size,
        encoder_device=encoder_device,
        encoder_factory=encoder_factory,
    )
    lexical_channels = _build_lexical_channels(
        documents=documents,
        sections_by_document=sections_by_document,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        component_depth=channel_top_k,
    )
    channels = lexical_channels + dense_channels
    built_channels_at = time.perf_counter()

    records_by_split = {
        split: _build_candidate_records(
            split=split,
            samples=samples,
            documents_by_id=documents_by_id,
            sections_by_document=sections_by_document,
            channels=channels,
            fold_assignments=train_fold_assignments if split == _TRAIN_SPLIT else None,
            channel_top_k=channel_top_k,
            candidate_pool_depth=candidate_pool_depth,
            rrf_k=rrf_k,
        )
        for split, samples in split_samples.items()
    }
    built_records_at = time.perf_counter()

    baseline_metrics = {
        "train_cv": _evaluate_baseline(records_by_split[_TRAIN_SPLIT]),
        "train_full": _evaluate_baseline(records_by_split[_TRAIN_SPLIT]),
        "dev": _evaluate_baseline(records_by_split[_DEV_SPLIT]),
    }
    config_results = [
        _evaluate_config(
            config=config,
            train_records=records_by_split[_TRAIN_SPLIT],
            dev_records=records_by_split[_DEV_SPLIT],
            baseline_metrics=baseline_metrics,
            selection_rules=selection_rules,
        )
        for config in candidate_configs
    ]
    evaluated_at = time.perf_counter()

    train_cv_selection = _select_train_cv_config(
        config_results=config_results,
        selection_rules=selection_rules,
    )
    dev_validation = _dev_validation_summary(
        config_results=config_results,
        selected_config_id=train_cv_selection.get("selected_config_id"),
    )
    guard_checks = _guard_checks(
        stage120_summary=stage120_summary,
        user_confirmed_validation=user_confirmed_validation,
        confirmation_note=confirmation_note,
        split_samples=split_samples,
        dense_summary=dense_summary,
        candidate_pool_depth=candidate_pool_depth,
        records_by_split=records_by_split,
        train_cv_selection=train_cv_selection,
        config_results=config_results,
    )
    checked_at = time.perf_counter()

    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train-CV/dev validation for the frozen Stage120 fast-filter plus "
            "alternate-screening protocol over the fixed Stage116 top200 pool. "
            "This stage rebuilds train/dev candidate records in memory, keeps "
            "a protected Stage116 prefix, screens only a constrained window, "
            "limits tail promotions, keeps dev report-only, keeps test locked, "
            "does not run answer or final metrics, does not change runtime "
            "defaults, and does not add fallback strategies."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_validation),
            "confirmation_note": confirmation_note,
        },
        "split_contract": {
            "split_name": _SPLIT_NAME,
            "protocol_version": _PROTOCOL_VERSION,
            "development_splits": list(_ALLOWED_DEVELOPMENT_SPLITS),
            "selection_split": "train",
            "selection_mode": "train_grouped_cross_validation_then_full_train_refit",
            "validation_split": "dev",
            "dev_validation_mode": "single_pass_report_only_no_retuning",
            "forbidden_final_splits": list(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "stage120_protocol": _fingerprint(stage120_protocol_path),
            "train_split": _fingerprint(train_split_path),
            "dev_split": _fingerprint(dev_split_path),
            "documents": _fingerprint(documents_path),
            "stage80_report": _fingerprint(stage80_report_path)
            if stage80_report_path is not None
            else None,
        },
        "stage120_summary": stage120_summary,
        "analysis_config": {
            "candidate_pool_depth": candidate_pool_depth,
            "channel_top_k": channel_top_k,
            "rrf_k": rrf_k,
            "include_dense_channels": include_dense_channels,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "candidate_config_count": len(candidate_configs),
            "baseline_config_id": _BASELINE_CONFIG_ID,
            "minimum_positive_objective_for_selection": _MINIMUM_POSITIVE_OBJECTIVE,
        },
        "loaded_data_summary": {
            "split_samples": summarize_primeqa_hybrid_split_samples(split_samples),
            "document_count": len(documents_by_id),
            **_section_summary(sections_by_document),
            "test_split_loaded": False,
        },
        "candidate_pool_summary": _candidate_pool_summary(records_by_split),
        "dense_channel_preflight": dense_summary,
        "baseline_metrics": baseline_metrics,
        "config_results": config_results,
        "train_cv_selection": train_cv_selection,
        "dev_validation": dev_validation,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks=guard_checks, train_cv_selection=train_cv_selection),
        "timing_seconds": {
            "load_protocol": round(loaded_protocol_at - started_at, 3),
            "load_splits_and_build_folds": round(loaded_splits_at - loaded_protocol_at, 3),
            "load_documents_and_reports": round(loaded_documents_at - loaded_splits_at, 3),
            "build_retrieval_channels": round(built_channels_at - loaded_documents_at, 3),
            "build_candidate_records": round(built_records_at - built_channels_at, 3),
            "evaluate_screening_configs": round(evaluated_at - built_records_at, 3),
            "selection_and_guard_checks": round(checked_at - evaluated_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_fast_filter_screening_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridFastFilterScreeningVisualization]:
    """Write Stage121 SVG visualizations."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage121_train_cv_objective_scores.svg": render_horizontal_bar_chart_svg(
            title="Stage121 train-CV objective scores",
            bars=_objective_score_bars(report),
            x_label="objective score",
            width=1540,
            margin_left=780,
        ),
        "stage121_train_cv_mrr_at_20_delta.svg": render_horizontal_bar_chart_svg(
            title="Stage121 train-CV MRR@20 delta",
            bars=_metric_delta_bars(report, split="train_cv", metric="mrr_at_20_delta"),
            x_label="delta",
            width=1540,
            margin_left=780,
        ),
        "stage121_train_cv_hit_at_10_delta.svg": render_horizontal_bar_chart_svg(
            title="Stage121 train-CV hit@10 delta",
            bars=_metric_delta_bars(report, split="train_cv", metric="hit@10_delta"),
            x_label="delta",
            width=1540,
            margin_left=780,
        ),
        "stage121_train_cv_guard_pass_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage121 train-CV selection guard pass counts",
            bars=_config_guard_pass_bars(report),
            x_label="passed guards",
            width=1540,
            margin_left=780,
        ),
        "stage121_train_cv_top10_tail_promotions.svg": render_horizontal_bar_chart_svg(
            title="Stage121 train-CV tail docs promoted into top10",
            bars=_promotion_average_bars(report),
            x_label="average promoted tail docs",
            width=1540,
            margin_left=780,
        ),
        "stage121_dev_selected_config_deltas.svg": render_horizontal_bar_chart_svg(
            title="Stage121 dev selected config deltas",
            bars=_dev_selected_delta_bars(report),
            x_label="delta",
            width=1180,
            margin_left=520,
        ),
        "stage121_selection_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage121 decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1320,
            margin_left=660,
        ),
        "stage121_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage121 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1660,
            margin_left=920,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridFastFilterScreeningVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _evaluate_config(
    *,
    config: Mapping[str, Any],
    train_records: Sequence[_CandidateRecord],
    dev_records: Sequence[_CandidateRecord],
    baseline_metrics: Mapping[str, Mapping[str, Any]],
    selection_rules: Mapping[str, Any],
) -> dict[str, Any]:
    config_id = str(config["config_id"])
    try:
        train_cv_outcomes = _evaluate_train_cv_config(
            config=config,
            train_records=train_records,
        )
        train_full_scorer = _fit_scorer_for_config(
            config,
            _training_records_for_config(config, train_records),
        )
    except ValueError as error:
        return _failed_config_result(
            config=config,
            training_error=error,
            baseline_metrics=baseline_metrics,
        )

    train_full_outcomes = _screen_records(train_records, train_full_scorer, config)
    dev_outcomes = _screen_records(dev_records, train_full_scorer, config)
    outcomes_by_split = {
        "train_cv": train_cv_outcomes,
        "train_full": train_full_outcomes,
        "dev": dev_outcomes,
    }
    metrics_by_split = {
        split: _screening_metrics(outcomes)
        for split, outcomes in outcomes_by_split.items()
    }
    comparisons = {
        split: _compare_to_baseline(metrics_by_split[split], baseline_metrics[split])
        for split in ("train_cv", "train_full", "dev")
    }
    train_cv_guards = _train_cv_guard_results(
        config_id=config_id,
        train_records=train_records,
        outcomes=train_cv_outcomes,
        baseline_metrics=baseline_metrics["train_cv"],
        comparison=comparisons["train_cv"],
        guard_thresholds=selection_rules["guard_thresholds"],
    )
    objective = _objective_score(
        comparison=comparisons["train_cv"],
        train_cv_guards=train_cv_guards,
    )
    guard_passed = all(guard["passed"] for guard in train_cv_guards)
    positive_objective = objective > _MINIMUM_POSITIVE_OBJECTIVE
    return {
        "config_id": config_id,
        "family_id": config["family_id"],
        "selector_algorithm": (config.get("screening_selector") or {}).get("algorithm"),
        "filter_rule": (config.get("fast_filter") or {}).get("filter_rule"),
        "selection_eligible": bool(config.get("selection_eligible")),
        "protected_prefix_depth": int(config["fast_filter"]["protected_prefix_depth"]),
        "screened_window_size": int(config["fast_filter"]["screened_window_size"]),
        "promotion_budget_top10": int(config["safety_constraints"]["promotion_budget_top10"]),
        "promotion_budget_top20": int(config["safety_constraints"]["promotion_budget_top20"]),
        "full_top200_rerank_allowed": bool(
            config["safety_constraints"]["full_top200_rerank_allowed"]
        ),
        "metrics_by_split": metrics_by_split,
        "comparisons_to_baseline": comparisons,
        "train_cv_selection_guards": train_cv_guards,
        "train_cv_objective_score": round(objective, 6),
        "train_cv_guard_passed": guard_passed,
        "train_cv_positive_objective": positive_objective,
        "train_cv_selectable": bool(config.get("selection_eligible"))
        and guard_passed
        and positive_objective,
        "training_status": "succeeded",
        "training_error": None,
    }


def _failed_config_result(
    *,
    config: Mapping[str, Any],
    training_error: ValueError,
    baseline_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    config_id = str(config["config_id"])
    comparisons = {
        split: _compare_to_baseline(baseline_metrics[split], baseline_metrics[split])
        for split in ("train_cv", "train_full", "dev")
    }
    return {
        "config_id": config_id,
        "family_id": config["family_id"],
        "selector_algorithm": (config.get("screening_selector") or {}).get("algorithm"),
        "filter_rule": (config.get("fast_filter") or {}).get("filter_rule"),
        "selection_eligible": bool(config.get("selection_eligible")),
        "protected_prefix_depth": int(config["fast_filter"]["protected_prefix_depth"]),
        "screened_window_size": int(config["fast_filter"]["screened_window_size"]),
        "promotion_budget_top10": int(config["safety_constraints"]["promotion_budget_top10"]),
        "promotion_budget_top20": int(config["safety_constraints"]["promotion_budget_top20"]),
        "full_top200_rerank_allowed": bool(
            config["safety_constraints"]["full_top200_rerank_allowed"]
        ),
        "metrics_by_split": {
            split: _baseline_like_screening_metrics(baseline_metrics[split])
            for split in ("train_cv", "train_full", "dev")
        },
        "comparisons_to_baseline": comparisons,
        "train_cv_selection_guards": [
            _config_guard(
                config_id=config_id,
                name="train_cv_config_training_succeeded",
                passed=False,
                observed=str(training_error),
                expected="config trains successfully on every train-CV fold",
            )
        ],
        "train_cv_objective_score": -999.0,
        "train_cv_guard_passed": False,
        "train_cv_positive_objective": False,
        "train_cv_selectable": False,
        "training_status": "failed",
        "training_error": str(training_error),
    }


def _evaluate_train_cv_config(
    *,
    config: Mapping[str, Any],
    train_records: Sequence[_CandidateRecord],
) -> dict[str, _ScreeningOutcome]:
    grouped = _records_by_sample(train_records)
    fold_ids = sorted({record.fold_id for record in train_records if record.fold_id is not None})
    outcomes: dict[str, _ScreeningOutcome] = {}
    for fold_id in fold_ids:
        fit_records = [
            record
            for record in train_records
            if record.fold_id != fold_id
        ]
        validation_records = [
            record
            for record in train_records
            if record.fold_id == fold_id
        ]
        scorer = _fit_scorer_for_config(config, _training_records_for_config(config, fit_records))
        outcomes.update(_screen_records(validation_records, scorer, config))
    missing_samples = set(grouped) - set(outcomes)
    if missing_samples:
        scorer = _fit_scorer_for_config(config, _training_records_for_config(config, train_records))
        for sample_id in missing_samples:
            sample_records = grouped[sample_id]
            outcomes[sample_id] = _screen_one_sample(sample_records, scorer, config)
    return outcomes


def _fit_scorer_for_config(
    config: Mapping[str, Any],
    records: Sequence[_CandidateRecord],
) -> _ScreeningScorer:
    algorithm = str((config.get("screening_selector") or {}).get("algorithm"))
    config_id = str(config["config_id"])
    if algorithm in {
        "calibrated_route_consensus_score",
        "deterministic_evidence_margin_selector",
    }:
        return _DeterministicScreeningScorer(
            weights=_deterministic_weights(config_id=config_id, algorithm=algorithm)
        ).fit(records)
    feature_names = _feature_names_for_config(config_id=config_id, algorithm=algorithm)
    if algorithm == "pairwise_logistic_preference":
        return _LogisticScreeningScorer(feature_names=feature_names).fit(records)
    if algorithm == "pairwise_hist_gradient_boosting_preference":
        return _HistGradientScreeningScorer(feature_names=feature_names).fit(records)
    raise ValueError(f"Unknown screening selector algorithm: {algorithm}")


def _training_records_for_config(
    config: Mapping[str, Any],
    records: Sequence[_CandidateRecord],
) -> list[_CandidateRecord]:
    algorithm = str((config.get("screening_selector") or {}).get("algorithm"))
    if algorithm in {
        "calibrated_route_consensus_score",
        "deterministic_evidence_margin_selector",
    }:
        return list(records)
    grouped = _records_by_sample(records)
    selected: list[_CandidateRecord] = []
    for sample_records in grouped.values():
        eligible = _screened_candidates(sample_records, config)
        gold_records = [record for record in eligible if record.is_gold]
        if not gold_records:
            continue
        non_gold = [record for record in eligible if not record.is_gold]
        selected.extend(gold_records)
        selected.extend(sorted(non_gold, key=lambda record: record.baseline_rank)[:40])
    return selected


def _screen_records(
    records: Sequence[_CandidateRecord],
    scorer: _ScreeningScorer,
    config: Mapping[str, Any],
) -> dict[str, _ScreeningOutcome]:
    return {
        sample_id: _screen_one_sample(sample_records, scorer, config)
        for sample_id, sample_records in _records_by_sample(records).items()
    }


def _screen_one_sample(
    sample_records: Sequence[_CandidateRecord],
    scorer: _ScreeningScorer,
    config: Mapping[str, Any],
) -> _ScreeningOutcome:
    ordered_baseline = sorted(sample_records, key=lambda record: record.baseline_rank)
    protected_prefix_depth = int(config["fast_filter"]["protected_prefix_depth"])
    promotion_budget_top10 = int(config["safety_constraints"]["promotion_budget_top10"])
    promotion_budget_top20 = int(config["safety_constraints"]["promotion_budget_top20"])
    screened = _screened_candidates(ordered_baseline, config)
    scores = scorer.score(screened) if screened else []
    scored_screened = sorted(
        zip(scores, screened, strict=True),
        key=lambda item: (-item[0], item[1].baseline_rank),
    )

    top10_budget = max(0, min(promotion_budget_top10, 10 - protected_prefix_depth))
    top10_promotions = [
        record
        for _, record in scored_screened
        if record.baseline_rank > 10
    ][:top10_budget]
    promoted_top10_ids = {record.doc_id for record in top10_promotions}
    top20_promotions = [
        record
        for _, record in scored_screened
        if record.baseline_rank > 20 and record.doc_id not in promoted_top10_ids
    ][:promotion_budget_top20]
    promoted_ids = promoted_top10_ids | {record.doc_id for record in top20_promotions}

    protected_prefix = ordered_baseline[:protected_prefix_depth]
    top10_remainder = [
        record
        for record in ordered_baseline[protected_prefix_depth:10]
        if protected_prefix_depth < 10 and record.doc_id not in promoted_ids
    ]
    top20_remainder_start = max(protected_prefix_depth, 10)
    top20_remainder = [
        record
        for record in ordered_baseline[top20_remainder_start:20]
        if top20_remainder_start < 20 and record.doc_id not in promoted_ids
    ]
    rest = [
        record
        for record in ordered_baseline[20:]
        if record.doc_id not in promoted_ids
    ]
    screened_order = [
        *protected_prefix,
        *top10_promotions,
        *top10_remainder,
        *top20_promotions,
        *top20_remainder,
        *rest,
    ]
    return _ScreeningOutcome(
        gold_rank=_gold_rank(screened_order),
        promoted_tail_docs_into_top10=sum(
            1 for record in screened_order[:10] if record.baseline_rank > 10
        ),
        promoted_tail_docs_into_top20=sum(
            1 for record in screened_order[:20] if record.baseline_rank > 20
        ),
        screened_candidate_count=len(screened),
    )


def _screened_candidates(
    sample_records: Sequence[_CandidateRecord],
    config: Mapping[str, Any],
) -> list[_CandidateRecord]:
    protected_prefix_depth = int(config["fast_filter"]["protected_prefix_depth"])
    window_size = int(config["fast_filter"]["screened_window_size"])
    filter_rule = str(config["fast_filter"]["filter_rule"])
    return [
        record
        for record in sample_records
        if protected_prefix_depth < record.baseline_rank <= window_size
        and _passes_filter_rule(record, filter_rule)
    ]


def _passes_filter_rule(record: _CandidateRecord, filter_rule: str) -> bool:
    features = record.features
    if filter_rule == "route_vote_or_best_secondary_rank":
        return features.get("route_hit_count", 0.0) >= 2 or features.get(
            "best_route_inverse_rank",
            0.0,
        ) >= 1 / 20
    if filter_rule == "strong_route_consensus_and_margin":
        return features.get("route_hit_count", 0.0) >= 3 or (
            features.get("route_hit_count", 0.0) >= 2
            and features.get("best_route_inverse_rank", 0.0) >= 1 / 20
        )
    if filter_rule == "low_confidence_tail_screen_only":
        return record.baseline_rank > 20
    if filter_rule == "title_heading_body_evidence_density":
        return (
            features.get("query_title_token_overlap", 0.0)
            + features.get("query_section_heading_overlap", 0.0)
            + features.get("query_token_coverage", 0.0)
            + features.get("query_body_token_coverage", 0.0)
        ) > 0.25
    if filter_rule == "special_token_exact_or_title_heading_match":
        return (
            features.get("special_token_match_count", 0.0) > 0
            or features.get("query_title_token_overlap", 0.0) > 0
            or features.get("query_section_heading_overlap", 0.0) > 0
        )
    if filter_rule == "route_vote_plus_evidence_density":
        return (
            features.get("route_hit_count", 0.0) >= 2
            or features.get("special_token_match_count", 0.0) > 0
            or features.get("query_token_coverage", 0.0) >= 0.3
        )
    raise ValueError(f"Unknown fast-filter rule: {filter_rule}")


def _screening_metrics(outcomes: Mapping[str, _ScreeningOutcome]) -> dict[str, Any]:
    ranks = {sample_id: outcome.gold_rank for sample_id, outcome in outcomes.items()}
    metrics = _rank_metrics(ranks)
    metrics.update(
        {
            "average_screened_candidate_count": _rounded_mean(
                [outcome.screened_candidate_count for outcome in outcomes.values()]
            ),
            "average_promoted_tail_docs_into_top10": _rounded_mean(
                [outcome.promoted_tail_docs_into_top10 for outcome in outcomes.values()]
            ),
            "average_promoted_tail_docs_into_top20": _rounded_mean(
                [outcome.promoted_tail_docs_into_top20 for outcome in outcomes.values()]
            ),
        }
    )
    return metrics


def _baseline_like_screening_metrics(baseline: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(baseline),
        "average_screened_candidate_count": 0.0,
        "average_promoted_tail_docs_into_top10": 0.0,
        "average_promoted_tail_docs_into_top20": 0.0,
    }


def _train_cv_guard_results(
    *,
    config_id: str,
    train_records: Sequence[_CandidateRecord],
    outcomes: Mapping[str, _ScreeningOutcome],
    baseline_metrics: Mapping[str, Any],
    comparison: Mapping[str, Any],
    guard_thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    screened_ranks = {
        sample_id: outcome.gold_rank for sample_id, outcome in outcomes.items()
    }
    baseline_ranks = {
        sample_id: _gold_rank(sorted(sample_records, key=lambda record: record.baseline_rank))
        for sample_id, sample_records in _records_by_sample(train_records).items()
    }
    top20_regression_count = sum(
        1
        for sample_id, baseline_rank in baseline_ranks.items()
        if baseline_rank is not None
        and baseline_rank <= 20
        and (screened_ranks.get(sample_id) is None or screened_ranks[sample_id] > 20)
    )
    top10_regression_count = sum(
        1
        for sample_id, baseline_rank in baseline_ranks.items()
        if baseline_rank is not None
        and baseline_rank <= 10
        and (screened_ranks.get(sample_id) is None or screened_ranks[sample_id] > 10)
    )
    evaluated = int(baseline_metrics["evaluated_questions"])
    top20_regression_rate = _rounded_ratio(top20_regression_count, evaluated)
    hit200_loss_count = max(0, -int(comparison["hit@200_count_delta"]))
    promoted_top10_average = _rounded_mean(
        [outcome.promoted_tail_docs_into_top10 for outcome in outcomes.values()]
    )
    bm25_top10_demotions = _bm25_top10_gold_demotions(
        train_records=train_records,
        reranked_ranks=screened_ranks,
    )
    return [
        _config_guard(
            config_id=config_id,
            name="train_cv_hit_at_200_loss_count_within_guard",
            passed=hit200_loss_count
            <= int(guard_thresholds["maximum_train_cv_hit_at_200_loss_count"]),
            observed=hit200_loss_count,
            expected=f"<= {guard_thresholds['maximum_train_cv_hit_at_200_loss_count']}",
        ),
        _config_guard(
            config_id=config_id,
            name="train_cv_top10_regression_count_within_guard",
            passed=top10_regression_count
            <= int(guard_thresholds["maximum_train_cv_top10_regression_count"]),
            observed=top10_regression_count,
            expected=f"<= {guard_thresholds['maximum_train_cv_top10_regression_count']}",
        ),
        _config_guard(
            config_id=config_id,
            name="train_cv_hit_at_20_regression_rate_within_guard",
            passed=top20_regression_rate
            <= float(guard_thresholds["maximum_train_cv_hit_at_20_regression_rate"]),
            observed=top20_regression_rate,
            expected=f"<= {guard_thresholds['maximum_train_cv_hit_at_20_regression_rate']}",
        ),
        _config_guard(
            config_id=config_id,
            name="train_cv_bm25_top10_gold_demotions_to_below_50_within_guard",
            passed=bm25_top10_demotions
            <= int(guard_thresholds["maximum_train_cv_bm25_top10_gold_demotions_to_below_50"]),
            observed=bm25_top10_demotions,
            expected=(
                "<= "
                f"{guard_thresholds['maximum_train_cv_bm25_top10_gold_demotions_to_below_50']}"
            ),
        ),
        _config_guard(
            config_id=config_id,
            name="train_cv_hit_at_10_delta_non_negative",
            passed=float(comparison["hit@10_delta"])
            >= float(guard_thresholds["minimum_train_cv_hit_at_10_delta"]),
            observed=comparison["hit@10_delta"],
            expected=f">= {guard_thresholds['minimum_train_cv_hit_at_10_delta']}",
        ),
        _config_guard(
            config_id=config_id,
            name="train_cv_mrr_at_20_delta_non_negative",
            passed=float(comparison["mrr_at_20_delta"])
            >= float(guard_thresholds["minimum_train_cv_mrr_at_20_delta"]),
            observed=comparison["mrr_at_20_delta"],
            expected=f">= {guard_thresholds['minimum_train_cv_mrr_at_20_delta']}",
        ),
        _config_guard(
            config_id=config_id,
            name="train_cv_promoted_tail_docs_into_top10_average_within_guard",
            passed=promoted_top10_average
            <= float(guard_thresholds["maximum_train_cv_promoted_tail_docs_into_top10_average"]),
            observed=promoted_top10_average,
            expected=(
                "<= "
                f"{guard_thresholds['maximum_train_cv_promoted_tail_docs_into_top10_average']}"
            ),
        ),
    ]


def _objective_score(
    *,
    comparison: Mapping[str, Any],
    train_cv_guards: Sequence[Mapping[str, Any]],
) -> float:
    guard_observed = {guard["name"]: guard["observed"] for guard in train_cv_guards}
    return (
        float(comparison["mrr_at_20_delta"]) * 5.0
        + float(comparison["hit@10_delta"]) * 4.0
        + float(comparison["hit@20_delta"]) * 2.0
        - float(guard_observed["train_cv_top10_regression_count_within_guard"]) * 2.0
        - float(
            guard_observed[
                "train_cv_bm25_top10_gold_demotions_to_below_50_within_guard"
            ]
        )
        * 2.0
        - max(0, -int(comparison["hit@200_count_delta"])) * 10.0
    )


def _select_train_cv_config(
    *,
    config_results: Sequence[Mapping[str, Any]],
    selection_rules: Mapping[str, Any],
) -> dict[str, Any]:
    guard_passed = [result for result in config_results if result["train_cv_guard_passed"]]
    selectable = [result for result in config_results if result["train_cv_selectable"]]
    if not selectable:
        return {
            "selection_split": "train",
            "selection_source": "train_cv_only",
            "selection_mode": selection_rules.get("selection_mode"),
            "selected_config_id": None,
            "selected_family_id": None,
            "guard_passed_config_count": len(guard_passed),
            "selectable_config_count": 0,
            "config_count": len(config_results),
            "status": "no_positive_train_cv_selectable_config",
            "dev_used_for_selection": False,
            "dev_used_for_retuning": False,
        }
    ordered = sorted(
        selectable,
        key=lambda result: (
            -float(result["train_cv_objective_score"]),
            float(
                result["metrics_by_split"]["train_cv"][
                    "average_promoted_tail_docs_into_top10"
                ]
            ),
            -float(result["comparisons_to_baseline"]["train_cv"]["mrr_at_20_delta"]),
            str(result["config_id"]),
        ),
    )
    selected = ordered[0]
    return {
        "selection_split": "train",
        "selection_source": "train_cv_only",
        "selection_mode": selection_rules.get("selection_mode"),
        "selected_config_id": selected["config_id"],
        "selected_family_id": selected["family_id"],
        "selected_train_cv_objective_score": selected["train_cv_objective_score"],
        "selected_train_cv_comparison": selected["comparisons_to_baseline"]["train_cv"],
        "guard_passed_config_count": len(guard_passed),
        "selectable_config_count": len(selectable),
        "config_count": len(config_results),
        "status": "train_cv_selected_positive_config",
        "dev_used_for_selection": False,
        "dev_used_for_retuning": False,
    }


def _dev_validation_summary(
    *,
    config_results: Sequence[Mapping[str, Any]],
    selected_config_id: str | None,
) -> dict[str, Any]:
    if selected_config_id is None:
        return {
            "validation_split": "dev",
            "selected_config_id": None,
            "status": "no_positive_train_cv_selectable_config",
            "dev_used_for_selection": False,
            "dev_used_for_retuning": False,
        }
    selected = next(
        result
        for result in config_results
        if result["config_id"] == selected_config_id
    )
    return {
        "validation_split": "dev",
        "selected_config_id": selected_config_id,
        "status": "reported_not_used_for_selection",
        "dev_comparison_to_baseline": selected["comparisons_to_baseline"]["dev"],
        "dev_metrics": selected["metrics_by_split"]["dev"],
        "dev_used_for_selection": False,
        "dev_used_for_retuning": False,
    }


def _guard_checks(
    *,
    stage120_summary: Mapping[str, Any],
    user_confirmed_validation: bool,
    confirmation_note: str,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    dense_summary: Mapping[str, Any],
    candidate_pool_depth: int,
    records_by_split: Mapping[str, Sequence[_CandidateRecord]],
    train_cv_selection: Mapping[str, Any],
    config_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    full_rerank_configs = [
        result["config_id"]
        for result in config_results
        if result["full_top200_rerank_allowed"]
    ]
    return [
        _check(
            name="user_confirmed_stage121_validation",
            passed=user_confirmed_validation,
            observed=confirmation_note,
            expected="user confirmed Stage121 fast-filter screening validation",
        ),
        _check(
            name="stage120_protocol_frozen",
            passed=stage120_summary.get("decision_status") == _SOURCE_STAGE120_STATUS,
            observed=stage120_summary.get("decision_status"),
            expected=_SOURCE_STAGE120_STATUS,
        ),
        _check(
            name="stage120_protocol_id_matches",
            passed=stage120_summary.get("protocol_id") == _SOURCE_PROTOCOL_ID,
            observed=stage120_summary.get("protocol_id"),
            expected=_SOURCE_PROTOCOL_ID,
        ),
        _check(
            name="stage121_uses_only_train_dev_splits",
            passed=set(split_samples) == set(_ALLOWED_DEVELOPMENT_SPLITS),
            observed=sorted(split_samples),
            expected=list(_ALLOWED_DEVELOPMENT_SPLITS),
        ),
        _check(
            name="stage121_test_split_not_loaded",
            passed=True,
            observed="test split path is not accepted by Stage121 runner",
            expected="no test split load",
        ),
        _check(
            name="stage121_candidate_pool_depth_matches_protocol",
            passed=candidate_pool_depth == int(stage120_summary.get("candidate_pool_depth") or 200),
            observed=candidate_pool_depth,
            expected=stage120_summary.get("candidate_pool_depth"),
        ),
        _check(
            name="stage121_dense_channels_ready_or_disabled",
            passed=bool(dense_summary.get("can_run_without_download")),
            observed=dense_summary.get("status"),
            expected="dense ready or explicitly disabled without download",
        ),
        _check(
            name="stage121_no_model_download_attempted",
            passed=bool(dense_summary.get("no_model_download_attempted")),
            observed=dense_summary.get("no_model_download_attempted"),
            expected=True,
        ),
        _check(
            name="stage121_no_full_top200_rerank_configs",
            passed=not full_rerank_configs,
            observed=full_rerank_configs,
            expected=[],
        ),
        _check(
            name="stage121_candidate_rows_not_written",
            passed=True,
            observed="candidate records built in memory only",
            expected="no candidate row artifact written",
        ),
        _check(
            name="stage121_dev_not_used_for_selection_or_retuning",
            passed=train_cv_selection.get("dev_used_for_selection") is False
            and train_cv_selection.get("dev_used_for_retuning") is False,
            observed={
                "dev_used_for_selection": train_cv_selection.get("dev_used_for_selection"),
                "dev_used_for_retuning": train_cv_selection.get("dev_used_for_retuning"),
            },
            expected=False,
        ),
        _check(
            name="stage121_runtime_defaults_unchanged",
            passed=True,
            observed="offline screening validation only",
            expected="runtime defaults unchanged",
        ),
        _check(
            name="stage121_fallback_strategies_not_added",
            passed=True,
            observed="fixed candidate-pool screening validation only",
            expected="no fallback strategies",
        ),
        _check(
            name="stage121_public_report_has_no_raw_candidate_ids",
            passed=True,
            observed={
                split: len(records)
                for split, records in records_by_split.items()
            },
            expected="records counted but not serialized",
        ),
    ]


def _decision(
    *,
    guard_checks: Sequence[Mapping[str, Any]],
    train_cv_selection: Mapping[str, Any],
) -> dict[str, Any]:
    if not all(check["passed"] for check in guard_checks):
        return {
            "status": "primeqa_hybrid_fast_filter_screening_validation_blocked",
            "recommended_next_direction": "fix_stage121_validation_blockers",
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    if train_cv_selection.get("selected_config_id") is None:
        return {
            "status": (
                "primeqa_hybrid_fast_filter_screening_completed_"
                "no_positive_train_cv_selectable_config"
            ),
            "recommended_next_direction": "analyze_fast_filter_screening_failure_patterns",
            "can_continue_train_dev_development": True,
            "selected_config_id": None,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    return {
        "status": (
            "primeqa_hybrid_fast_filter_screening_completed_"
            "train_cv_selected_dev_reported"
        ),
        "recommended_next_direction": "review_fast_filter_screening_changed_cases",
        "can_continue_train_dev_development": True,
        "selected_config_id": train_cv_selection.get("selected_config_id"),
        "selected_family_id": train_cv_selection.get("selected_family_id"),
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
    }


def _stage120_summary(stage120_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage120_report.get("decision") or {}
    frozen = stage120_report.get("frozen_protocol") or {}
    pool_contract = frozen.get("fixed_candidate_pool_contract") or {}
    selection_rules = frozen.get("selection_rules") or {}
    return {
        "stage": stage120_report.get("stage"),
        "protocol_id": stage120_report.get("protocol_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "candidate_pool_depth": pool_contract.get("candidate_pool_depth"),
        "candidate_config_count": len(frozen.get("candidate_configs") or []),
        "candidate_family_count": len(frozen.get("candidate_families") or []),
        "selection_rules": selection_rules,
        "guard_thresholds": selection_rules.get("guard_thresholds") or {},
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


def _feature_names_for_config(*, config_id: str, algorithm: str) -> tuple[str, ...]:
    if algorithm == "pairwise_logistic_preference":
        if "evidence_density" in config_id:
            return (
                "stage116_rrf_score",
                "query_title_token_overlap",
                "query_section_heading_overlap",
                "query_token_coverage",
                "query_body_token_coverage",
                "special_token_match_count",
                "bm25_top10_indicator",
            )
        return (
            "stage116_rrf_score",
            "route_hit_count",
            "lexical_route_hit_count",
            "dense_route_hit_count",
            "best_route_inverse_rank",
            "bm25_top10_indicator",
        )
    if algorithm == "pairwise_hist_gradient_boosting_preference":
        return _RUNTIME_FEATURES
    return ("stage116_rrf_score",)


def _deterministic_weights(*, config_id: str, algorithm: str) -> dict[str, float]:
    if algorithm == "calibrated_route_consensus_score":
        return {
            "stage116_rrf_score": 1.0,
            "route_hit_count": 1.2,
            "best_route_inverse_rank": 1.5,
            "lexical_route_hit_count": 0.8,
            "bm25_top10_indicator": 0.4,
        }
    if config_id == "special_token_exact_window40_rule_selector_v1":
        return {
            "stage116_rrf_score": 1.0,
            "special_token_match_count": 2.0,
            "title_special_token_match_count": 1.0,
            "heading_special_token_match_count": 1.0,
            "query_title_token_overlap": 0.6,
            "query_section_heading_overlap": 0.6,
        }
    return {"stage116_rrf_score": 1.0}


def _project_feature_dict(
    record: _CandidateRecord,
    feature_names: Sequence[str],
) -> dict[str, float]:
    return {feature_name: record.features.get(feature_name, 0.0) for feature_name in feature_names}


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    forbidden_keys = sorted(_find_forbidden_public_keys(report))
    return {
        "public_safe_summary_only": True,
        "raw_question_text_written": False,
        "raw_answer_text_written": False,
        "raw_document_text_written": False,
        "raw_document_ids_written": False,
        "raw_candidate_rows_written": False,
        "forbidden_keys_found": forbidden_keys,
    }


def _find_forbidden_public_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_string = str(key)
            if key_string in _FORBIDDEN_PUBLIC_KEYS:
                found.add(key_string)
            found.update(_find_forbidden_public_keys(child))
    elif isinstance(value, list | tuple):
        for child in value:
            found.update(_find_forbidden_public_keys(child))
    return found


def _config_guard(
    *,
    config_id: str,
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
) -> dict[str, Any]:
    return {
        "config_id": config_id,
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def _check(
    *,
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def _objective_score_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return sorted(
        [
            BarDatum(
                label=str(result["config_id"]),
                value=float(result["train_cv_objective_score"]),
                value_label=f"{float(result['train_cv_objective_score']):+.4f}",
            )
            for result in report["config_results"]
        ],
        key=lambda bar: (-bar.value, bar.label),
    )


def _metric_delta_bars(report: Mapping[str, Any], *, split: str, metric: str) -> list[BarDatum]:
    return sorted(
        [
            BarDatum(
                label=str(result["config_id"]),
                value=float(result["comparisons_to_baseline"][split][metric]),
                value_label=f"{float(result['comparisons_to_baseline'][split][metric]):+.4f}",
            )
            for result in report["config_results"]
        ],
        key=lambda bar: (-bar.value, bar.label),
    )


def _config_guard_pass_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return sorted(
        [
            BarDatum(
                label=str(result["config_id"]),
                value=sum(1 for guard in result["train_cv_selection_guards"] if guard["passed"]),
                value_label=(
                    f"{sum(1 for guard in result['train_cv_selection_guards'] if guard['passed'])}"
                    f" / {len(result['train_cv_selection_guards'])}"
                ),
            )
            for result in report["config_results"]
        ],
        key=lambda bar: (-bar.value, bar.label),
    )


def _promotion_average_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return sorted(
        [
            BarDatum(
                label=str(result["config_id"]),
                value=float(
                    result["metrics_by_split"]["train_cv"][
                        "average_promoted_tail_docs_into_top10"
                    ]
                ),
                value_label=str(
                    result["metrics_by_split"]["train_cv"][
                        "average_promoted_tail_docs_into_top10"
                    ]
                ),
            )
            for result in report["config_results"]
        ],
        key=lambda bar: (bar.value, bar.label),
    )


def _dev_selected_delta_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    selected_config_id = report["train_cv_selection"].get("selected_config_id")
    if selected_config_id is None:
        return []
    selected = next(
        result for result in report["config_results"] if result["config_id"] == selected_config_id
    )
    comparison = selected["comparisons_to_baseline"]["dev"]
    metrics = ("mrr_at_20_delta", "hit@10_delta", "hit@20_delta", "hit@200_delta")
    return [
        BarDatum(
            label=metric,
            value=float(comparison[metric]),
            value_label=f"{float(comparison[metric]):+.4f}",
        )
        for metric in metrics
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report["decision"]
    flags = (
        "can_continue_train_dev_development",
        "can_open_final_test_gate_now",
        "can_run_final_test_metrics_now",
        "can_use_test_for_tuning",
        "fallback_strategies_enabled",
    )
    return [
        BarDatum(
            label=flag,
            value=1.0 if decision.get(flag) else 0.0,
            value_label=str(bool(decision.get(flag))).lower(),
        )
        for flag in flags
    ]


def _guard_check_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(check["name"]),
            value=1.0 if check["passed"] else 0.0,
            value_label="pass" if check["passed"] else "fail",
        )
        for check in report["guard_checks"]
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
