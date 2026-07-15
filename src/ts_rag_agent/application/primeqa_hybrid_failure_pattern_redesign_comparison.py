from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.answer_verification import AnswerVerifier
from ts_rag_agent.application.evidence_selection import (
    SentenceEvidenceCandidate,
    create_sentence_evidence_selector,
    tokenize_text,
    trace_selector_route,
)
from ts_rag_agent.application.primeqa_hybrid_answer_pipeline_error_decomposition_analysis import (
    _aggregate_outputs,
    _best_gold_span_token_f1,
    _metrics_by_split,
    _public_case,
    _SplitQuestionTrace,
)
from ts_rag_agent.application.primeqa_hybrid_evidence_answerability_comparison import (
    _answer_signature,
    _baseline_config,
    _classify_pipeline_bucket_with_cached_gold_span,
)
from ts_rag_agent.application.rag_answering import ExtractiveAnswerGenerator
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.answer import AnswerCitation, AnswerVerificationResult, GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
    summarize_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import load_primeqa_documents

_STAGE = "Stage 109"
_CREATED_AT = "2026-07-16"
_SOURCE_STAGE108 = "Stage 108"
_SOURCE_PROTOCOL_ID = "primeqa_hybrid_failure_pattern_redesign_protocol_v1"
_SOURCE_DECISION_STATUS = "primeqa_hybrid_failure_pattern_redesign_protocol_frozen"
_ANALYSIS_ID = "primeqa_hybrid_failure_pattern_redesign_train_cv_dev_validation_v1"
_NO_TRAIN_CV_SELECTABLE_STATUS = (
    "primeqa_hybrid_failure_pattern_redesign_completed_no_train_cv_selectable_config"
)
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_ALLOWED_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = ("test",)
_EVAL_SPLITS = ("train_cv", "train_full", "dev")
_TARGET_BUCKETS = (
    "answerability_false_answer",
    "gold_span_beats_selected_answer",
    "evidence_selection_miss",
)
_BUCKET_ORDER = (
    "answerability_false_answer",
    "retrieval_context_miss",
    "evidence_selection_miss",
    "verification_over_refusal",
    "gold_span_beats_selected_answer",
    "low_overlap_gold_cited_answer",
    "answer_supported_and_cited",
)
_CHANGED_CASE_FIELDS = (
    "sample_id",
    "split",
    "fold_id",
    "config_id",
    "candidate_family_id",
    "baseline_bucket_id",
    "candidate_bucket_id",
    "baseline_answer_token_f1_bucket",
    "candidate_answer_token_f1_bucket",
    "baseline_citation_status",
    "candidate_citation_status",
    "answerability_action",
    "composition_action",
    "changed_case_confidence_band",
)
_FORBIDDEN_REPORT_KEYS = frozenset(
    {
        "question_text",
        "question_title",
        "raw_question_text",
        "raw_answer_text",
        "gold_answer",
        "answer_text",
        "document_id",
        "answer_doc_id",
        "retrieved_doc_ids",
        "cited_doc_ids",
        "source_doc_ids",
        "matched_token_strings",
        "query_terms",
        "document_title",
        "document_body",
        "document_text",
    }
)


@dataclass(frozen=True)
class _RunConfig:
    config_id: str
    candidate_family_id: str
    component_family: str
    target_buckets: tuple[str, ...]
    payload: dict[str, Any]


@dataclass(frozen=True)
class _QuestionRuntimeInputs:
    sample: PrimeQAHybridSplitSample
    question: PrimeQAQuestion
    retrieval_results: list[RetrievalResult]
    best_gold_span_token_f1: float | None


@dataclass(frozen=True)
class _TraceResult:
    traces_by_split: dict[str, list[_SplitQuestionTrace]]
    fold_assignments: dict[str, str]


@dataclass(frozen=True)
class PrimeQAHybridFailurePatternRedesignComparisonVisualization:
    """One generated Stage109 failure-pattern redesign comparison chart."""

    name: str
    path: str


def run_primeqa_hybrid_failure_pattern_redesign_comparison(
    *,
    stage108_protocol_path: Path,
    stage102_report_path: Path,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    user_confirmed_comparison: bool,
    confirmation_note: str,
    retrieval_top_k: int = 10,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    baseline_evidence_selector_name: str = "bm25_sentence",
    baseline_max_candidates_per_document: int = 3,
    baseline_max_sentences: int = 3,
    baseline_min_sentence_score: float = 2.0,
    baseline_verifier_min_citations: int = 1,
    baseline_verifier_min_evidence_score: float = 7.0,
    baseline_verifier_max_citation_rank: int = 3,
    max_gold_window_sentences: int = 3,
    gold_span_gap_margin: float = 0.05,
    low_answer_f1_threshold: float = 0.2,
    sample_limit_per_transition: int = 5,
) -> dict[str, Any]:
    """Run Stage109 train grouped-CV plus dev validation for frozen candidates."""

    _validate_options(
        retrieval_top_k=retrieval_top_k,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        baseline_max_candidates_per_document=baseline_max_candidates_per_document,
        baseline_max_sentences=baseline_max_sentences,
        baseline_min_sentence_score=baseline_min_sentence_score,
        baseline_verifier_min_citations=baseline_verifier_min_citations,
        baseline_verifier_min_evidence_score=baseline_verifier_min_evidence_score,
        baseline_verifier_max_citation_rank=baseline_verifier_max_citation_rank,
        max_gold_window_sentences=max_gold_window_sentences,
        gold_span_gap_margin=gold_span_gap_margin,
        low_answer_f1_threshold=low_answer_f1_threshold,
        sample_limit_per_transition=sample_limit_per_transition,
    )
    started_at = time.perf_counter()
    stage108_protocol = _load_json_object(stage108_protocol_path)
    stage102_report = _load_json_object(stage102_report_path)
    frozen_protocol = stage108_protocol.get("frozen_protocol") or {}
    stage108_summary = _stage108_summary(stage108_protocol)
    stage102_summary = _stage102_summary(stage102_report)
    target_weights = _target_weights(frozen_protocol)
    candidate_configs = [
        _config_from_mapping(config)
        for config in frozen_protocol.get("candidate_config_grid") or []
    ]
    loaded_protocols_at = time.perf_counter()

    split_samples = {
        "train": load_primeqa_hybrid_split_samples(train_split_path),
        "dev": load_primeqa_hybrid_split_samples(dev_split_path),
    }
    fold_assignments = _build_train_fold_assignments(
        split_samples["train"],
        fold_count=_train_cv_fold_count(frozen_protocol),
    )
    loaded_splits_at = time.perf_counter()
    documents_by_id = load_primeqa_documents(documents_path)
    documents = list(documents_by_id.values())
    retriever = BM25Retriever(k1=bm25_k1, b=bm25_b)
    retriever.fit(documents)
    indexed_at = time.perf_counter()

    runtime_inputs_by_split = _runtime_inputs_by_split(
        split_samples=split_samples,
        retriever=retriever,
        documents_by_id=documents_by_id,
        retrieval_top_k=retrieval_top_k,
        max_gold_window_sentences=max_gold_window_sentences,
    )
    cached_inputs_at = time.perf_counter()
    baseline_result = _evaluate_baseline(
        runtime_inputs_by_split=runtime_inputs_by_split,
        documents_by_id=documents_by_id,
        fold_assignments=fold_assignments,
        evidence_selector_name=baseline_evidence_selector_name,
        max_candidates_per_document=baseline_max_candidates_per_document,
        max_sentences=baseline_max_sentences,
        min_sentence_score=baseline_min_sentence_score,
        verifier_min_citations=baseline_verifier_min_citations,
        verifier_min_evidence_score=baseline_verifier_min_evidence_score,
        verifier_max_citation_rank=baseline_verifier_max_citation_rank,
        max_gold_window_sentences=max_gold_window_sentences,
        gold_span_gap_margin=gold_span_gap_margin,
        low_answer_f1_threshold=low_answer_f1_threshold,
        target_weights=target_weights,
    )
    baseline_evaluated_at = time.perf_counter()
    candidate_results = [
        _evaluate_candidate_config(
            config=config,
            runtime_inputs_by_split=runtime_inputs_by_split,
            documents_by_id=documents_by_id,
            fold_assignments=fold_assignments,
            max_gold_window_sentences=max_gold_window_sentences,
            gold_span_gap_margin=gold_span_gap_margin,
            low_answer_f1_threshold=low_answer_f1_threshold,
            baseline_result=baseline_result,
            frozen_protocol=frozen_protocol,
            target_weights=target_weights,
        )
        for config in candidate_configs
    ]
    candidates_evaluated_at = time.perf_counter()
    train_cv_selection = _train_cv_selection(
        config_results=candidate_results,
        baseline_result=baseline_result,
        frozen_protocol=frozen_protocol,
    )
    selected_result = _selected_config_result(
        config_results=candidate_results,
        selected_config_id=train_cv_selection.get("selected_config_id"),
    )
    dev_validation = _dev_validation(
        selected_result=selected_result,
        frozen_protocol=frozen_protocol,
    )
    changed_case_samples = _changed_case_samples(
        selected_result=selected_result,
        baseline_result=baseline_result,
        fold_assignments=fold_assignments,
        sample_limit_per_transition=sample_limit_per_transition,
    )
    guard_checks = _guard_checks(
        stage108_summary=stage108_summary,
        stage102_summary=stage102_summary,
        split_samples=split_samples,
        candidate_configs=candidate_configs,
        baseline_result=baseline_result,
        candidate_results=candidate_results,
        train_cv_selection=train_cv_selection,
        dev_validation=dev_validation,
        changed_case_samples=changed_case_samples,
        frozen_protocol=frozen_protocol,
        user_confirmed_comparison=user_confirmed_comparison,
    )
    finished_at = time.perf_counter()
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train/dev-only failure-pattern redesign comparison under the "
            "Stage108 frozen protocol. This stage loads only the frozen Stage68 "
            "train/dev split rows and local PrimeQA training/dev corpus "
            "documents, implements the frozen Stage108 candidate components, "
            "selects configs by train grouped CV only, validates the selected "
            "config once on dev, does not load the test split, does not run "
            "final metrics, does not write raw question, answer, document, "
            "token, or document-identifier fields, does not add fallback "
            "strategies, and does not change runtime defaults."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_comparison),
            "confirmation_note": confirmation_note,
        },
        "split_contract": {
            "split_name": _SPLIT_NAME,
            "protocol_version": _PROTOCOL_VERSION,
            "development_splits": list(_ALLOWED_SPLITS),
            "selection_split": "train",
            "validation_split": "dev",
            "forbidden_final_splits": list(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "stage108_protocol": _fingerprint(stage108_protocol_path),
            "stage102_report": _fingerprint(stage102_report_path),
            "train_split": _fingerprint(train_split_path),
            "dev_split": _fingerprint(dev_split_path),
            "corpus_documents": _fingerprint(documents_path),
        },
        "stage108_summary": stage108_summary,
        "stage102_summary": stage102_summary,
        "analysis_config": {
            "retrieval_top_k": retrieval_top_k,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "baseline_evidence_selector_name": baseline_evidence_selector_name,
            "baseline_max_candidates_per_document": baseline_max_candidates_per_document,
            "baseline_max_sentences": baseline_max_sentences,
            "baseline_min_sentence_score": baseline_min_sentence_score,
            "baseline_verifier_min_citations": baseline_verifier_min_citations,
            "baseline_verifier_min_evidence_score": baseline_verifier_min_evidence_score,
            "baseline_verifier_max_citation_rank": baseline_verifier_max_citation_rank,
            "max_gold_window_sentences": max_gold_window_sentences,
            "gold_span_gap_margin": gold_span_gap_margin,
            "low_answer_f1_threshold": low_answer_f1_threshold,
            "sample_limit_per_transition": sample_limit_per_transition,
            "candidate_config_count": len(candidate_configs),
            "target_bucket_weights": target_weights,
        },
        "data_summary": {
            "documents": len(documents),
            "splits": summarize_primeqa_hybrid_split_samples(split_samples),
            "train_cv": _fold_summary(split_samples["train"], fold_assignments),
        },
        "baseline_result": baseline_result["public"],
        "config_results": [result["public"] for result in candidate_results],
        "train_cv_selection": train_cv_selection,
        "dev_validation": dev_validation,
        "public_safe_changed_case_samples": changed_case_samples,
        "guard_checks": guard_checks,
        "decision": _decision(
            guard_checks=guard_checks,
            train_cv_selection=train_cv_selection,
            dev_validation=dev_validation,
        ),
        "timing_seconds": {
            "load_protocols": round(loaded_protocols_at - started_at, 3),
            "load_splits_and_build_folds": round(
                loaded_splits_at - loaded_protocols_at,
                3,
            ),
            "index_documents": round(indexed_at - loaded_splits_at, 3),
            "cache_runtime_inputs": round(cached_inputs_at - indexed_at, 3),
            "evaluate_baseline": round(baseline_evaluated_at - cached_inputs_at, 3),
            "evaluate_candidate_configs": round(
                candidates_evaluated_at - baseline_evaluated_at,
                3,
            ),
            "aggregate_select_validate_guard": round(
                finished_at - candidates_evaluated_at,
                3,
            ),
            "total": round(finished_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_failure_pattern_redesign_comparison_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridFailurePatternRedesignComparisonVisualization]:
    """Write SVG charts for Stage109 failure-pattern redesign comparison."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage109_train_cv_weighted_target_deltas.svg": render_horizontal_bar_chart_svg(
            title="Stage109 train-CV weighted target deltas",
            bars=_weighted_delta_bars(report, "train_cv"),
            x_label="delta vs baseline",
            width=1480,
            margin_left=760,
        ),
        "stage109_dev_weighted_target_deltas.svg": render_horizontal_bar_chart_svg(
            title="Stage109 dev weighted target deltas",
            bars=_weighted_delta_bars(report, "dev"),
            x_label="delta vs baseline",
            width=1480,
            margin_left=760,
        ),
        "stage109_train_cv_selectability.svg": render_horizontal_bar_chart_svg(
            title="Stage109 train-CV selectability",
            bars=_train_selectability_bars(report),
            x_label="1 means selectable",
            width=1440,
            margin_left=740,
        ),
        "stage109_changed_answer_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage109 changed answer counts",
            bars=_changed_answer_count_bars(report),
            x_label="changed verified answers",
            width=1480,
            margin_left=760,
        ),
        "stage109_dev_metric_deltas.svg": render_horizontal_bar_chart_svg(
            title="Stage109 selected dev metric deltas",
            bars=_selected_dev_metric_delta_bars(report),
            x_label="delta vs baseline",
            width=1320,
            margin_left=620,
        ),
        "stage109_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage109 decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1320,
            margin_left=660,
        ),
        "stage109_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage109 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1640,
            margin_left=860,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridFailurePatternRedesignComparisonVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _runtime_inputs_by_split(
    *,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    retriever: BM25Retriever,
    documents_by_id: Mapping[str, PrimeQADocument],
    retrieval_top_k: int,
    max_gold_window_sentences: int,
) -> dict[str, list[_QuestionRuntimeInputs]]:
    inputs_by_split = {}
    for split, samples in split_samples.items():
        split_inputs = []
        for sample in samples:
            question = sample.to_primeqa_question()
            retrieval_results = retriever.search(question.full_question, top_k=retrieval_top_k)
            best_gold_span = (
                _best_gold_span_token_f1(
                    question=question,
                    documents_by_id=documents_by_id,
                    max_gold_window_sentences=max_gold_window_sentences,
                )
                if question.answerable
                else None
            )
            split_inputs.append(
                _QuestionRuntimeInputs(
                    sample=sample,
                    question=question,
                    retrieval_results=retrieval_results,
                    best_gold_span_token_f1=best_gold_span,
                )
            )
        inputs_by_split[split] = split_inputs
    return inputs_by_split


def _evaluate_baseline(
    *,
    runtime_inputs_by_split: Mapping[str, Sequence[_QuestionRuntimeInputs]],
    documents_by_id: Mapping[str, PrimeQADocument],
    fold_assignments: Mapping[str, str],
    evidence_selector_name: str,
    max_candidates_per_document: int,
    max_sentences: int,
    min_sentence_score: float,
    verifier_min_citations: int,
    verifier_min_evidence_score: float,
    verifier_max_citation_rank: int,
    max_gold_window_sentences: int,
    gold_span_gap_margin: float,
    low_answer_f1_threshold: float,
    target_weights: Mapping[str, float],
) -> dict[str, Any]:
    baseline_config = _baseline_config(
        {
            "baseline_id": "stage102_verified_bm25_top10_answer_pipeline",
            "evidence_selector_name": evidence_selector_name,
            "max_candidates_per_document": max_candidates_per_document,
            "composition_policy_name": "top_k",
            "max_sentences": max_sentences,
            "min_sentence_score": min_sentence_score,
            "verifier_min_citations": verifier_min_citations,
            "verifier_min_evidence_score": verifier_min_evidence_score,
            "verifier_max_citation_rank": verifier_max_citation_rank,
        }
    )
    evidence_selector = create_sentence_evidence_selector(
        selector_name=evidence_selector_name,
        max_candidates_per_document=max_candidates_per_document,
    )
    answer_generator = ExtractiveAnswerGenerator(
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
        evidence_selector=evidence_selector,
    )
    answer_verifier = AnswerVerifier(
        min_citations=verifier_min_citations,
        min_evidence_score=verifier_min_evidence_score,
        max_citation_rank=verifier_max_citation_rank,
    )
    traces_by_split = {
        split: [
            _trace_baseline_question(
                split=split,
                runtime_inputs=runtime_inputs,
                answer_generator=answer_generator,
                answer_verifier=answer_verifier,
                documents_by_id=documents_by_id,
                max_gold_window_sentences=max_gold_window_sentences,
                gold_span_gap_margin=gold_span_gap_margin,
                low_answer_f1_threshold=low_answer_f1_threshold,
            )
            for runtime_inputs in split_inputs
        ]
        for split, split_inputs in runtime_inputs_by_split.items()
    }
    public = _public_result(
        config_id=baseline_config.config_id,
        candidate_family_id="baseline",
        component_family="baseline",
        target_buckets=[],
        traces_by_split=traces_by_split,
        baseline_traces_by_split=traces_by_split,
        fold_assignments=fold_assignments,
        target_weights=target_weights,
        baseline_public=None,
        train_cv_selectability=None,
    )
    return {
        "config": baseline_config,
        "traces_by_split": traces_by_split,
        "public": public,
    }


def _evaluate_candidate_config(
    *,
    config: _RunConfig,
    runtime_inputs_by_split: Mapping[str, Sequence[_QuestionRuntimeInputs]],
    documents_by_id: Mapping[str, PrimeQADocument],
    fold_assignments: Mapping[str, str],
    max_gold_window_sentences: int,
    gold_span_gap_margin: float,
    low_answer_f1_threshold: float,
    baseline_result: Mapping[str, Any],
    frozen_protocol: Mapping[str, Any],
    target_weights: Mapping[str, float],
) -> dict[str, Any]:
    traces_by_split = {
        split: [
            _trace_candidate_question(
                split=split,
                runtime_inputs=runtime_inputs,
                config=config,
                documents_by_id=documents_by_id,
                max_gold_window_sentences=max_gold_window_sentences,
                gold_span_gap_margin=gold_span_gap_margin,
                low_answer_f1_threshold=low_answer_f1_threshold,
            )
            for runtime_inputs in split_inputs
        ]
        for split, split_inputs in runtime_inputs_by_split.items()
    }
    public_without_guards = _public_result(
        config_id=config.config_id,
        candidate_family_id=config.candidate_family_id,
        component_family=config.component_family,
        target_buckets=list(config.target_buckets),
        traces_by_split=traces_by_split,
        baseline_traces_by_split=baseline_result["traces_by_split"],
        fold_assignments=fold_assignments,
        target_weights=target_weights,
        baseline_public=baseline_result["public"],
        train_cv_selectability=None,
    )
    train_cv_selectability = _train_cv_selectability(
        result=public_without_guards,
        baseline_result=baseline_result["public"],
        frozen_protocol=frozen_protocol,
    )
    public = {
        **public_without_guards,
        "train_cv_selectability": train_cv_selectability,
    }
    return {
        "config": config,
        "traces_by_split": traces_by_split,
        "public": public,
    }


def _trace_baseline_question(
    *,
    split: str,
    runtime_inputs: _QuestionRuntimeInputs,
    answer_generator: ExtractiveAnswerGenerator,
    answer_verifier: AnswerVerifier,
    documents_by_id: Mapping[str, PrimeQADocument],
    max_gold_window_sentences: int,
    gold_span_gap_margin: float,
    low_answer_f1_threshold: float,
) -> _SplitQuestionTrace:
    question = runtime_inputs.question
    retrieval_results = runtime_inputs.retrieval_results
    original_answer = answer_generator.generate(question, retrieval_results)
    verification = answer_verifier.verify(original_answer, retrieval_results)
    route_trace = trace_selector_route(question, answer_generator.evidence_selector_name)
    return _trace_from_answers(
        split=split,
        runtime_inputs=runtime_inputs,
        original_answer=original_answer,
        verification=verification,
        documents_by_id=documents_by_id,
        max_gold_window_sentences=max_gold_window_sentences,
        gold_span_gap_margin=gold_span_gap_margin,
        low_answer_f1_threshold=low_answer_f1_threshold,
        question_route=route_trace.question_route,
        routed_selector_name=route_trace.selected_selector_name,
        evidence_selector_name=answer_generator.evidence_selector_name,
        composition_policy_name=answer_generator.composition_policy_name,
    )


def _trace_candidate_question(
    *,
    split: str,
    runtime_inputs: _QuestionRuntimeInputs,
    config: _RunConfig,
    documents_by_id: Mapping[str, PrimeQADocument],
    max_gold_window_sentences: int,
    gold_span_gap_margin: float,
    low_answer_f1_threshold: float,
) -> _SplitQuestionTrace:
    question = runtime_inputs.question
    retrieval_results = runtime_inputs.retrieval_results
    selector = create_sentence_evidence_selector(
        selector_name="bm25_sentence",
        max_candidates_per_document=int(config.payload.get("max_candidates_per_document") or 3),
    )
    ranked_candidates = selector.rank_sentence_candidates(question, retrieval_results)
    eligible_candidates = [
        candidate for candidate in ranked_candidates if candidate.score >= 2.0
    ]
    selected_candidates = _select_candidate_evidence(
        question=question,
        config=config,
        candidates=eligible_candidates,
    )
    original_answer = _generated_answer_from_candidates(
        question=question,
        selected_candidates=selected_candidates,
    )
    verification = _verify_candidate_answer(
        config=config,
        answer=original_answer,
        selected_candidates=selected_candidates,
        retrieval_results=retrieval_results,
    )
    route_trace = trace_selector_route(question, selector.name)
    return _trace_from_answers(
        split=split,
        runtime_inputs=runtime_inputs,
        original_answer=original_answer,
        verification=verification,
        documents_by_id=documents_by_id,
        max_gold_window_sentences=max_gold_window_sentences,
        gold_span_gap_margin=gold_span_gap_margin,
        low_answer_f1_threshold=low_answer_f1_threshold,
        question_route=route_trace.question_route,
        routed_selector_name=route_trace.selected_selector_name,
        evidence_selector_name=selector.name,
        composition_policy_name=str(config.component_family),
    )


def _trace_from_answers(
    *,
    split: str,
    runtime_inputs: _QuestionRuntimeInputs,
    original_answer: GeneratedAnswer,
    verification: AnswerVerificationResult,
    documents_by_id: Mapping[str, PrimeQADocument],
    max_gold_window_sentences: int,
    gold_span_gap_margin: float,
    low_answer_f1_threshold: float,
    question_route: str,
    routed_selector_name: str,
    evidence_selector_name: str,
    composition_policy_name: str,
) -> _SplitQuestionTrace:
    question = runtime_inputs.question
    bucket_id, scoring = _classify_pipeline_bucket_with_cached_gold_span(
        question=question,
        retrieval_results=runtime_inputs.retrieval_results,
        original_answer=original_answer,
        verified_answer=verification.verified_answer,
        documents_by_id=documents_by_id,
        max_gold_window_sentences=max_gold_window_sentences,
        best_gold_span_token_f1=runtime_inputs.best_gold_span_token_f1,
        gold_span_gap_margin=gold_span_gap_margin,
        low_answer_f1_threshold=low_answer_f1_threshold,
    )
    public_case = _public_case(
        split=split,
        sample_id=runtime_inputs.sample.sample_id,
        question=question,
        retrieval_results=runtime_inputs.retrieval_results,
        original_answer=original_answer,
        verified_answer=verification.verified_answer,
        verification_reasons=verification.reasons,
        bucket_id=bucket_id,
        scoring=scoring,
        question_route=question_route,
        routed_selector_name=routed_selector_name,
        evidence_selector_name=evidence_selector_name,
        composition_policy_name=composition_policy_name,
    )
    return _SplitQuestionTrace(
        sample=runtime_inputs.sample,
        question=question,
        retrieval_results=runtime_inputs.retrieval_results,
        original_answer=original_answer,
        verified_answer=verification.verified_answer,
        verification_reasons=tuple(verification.reasons),
        question_route=question_route,
        routed_selector_name=routed_selector_name,
        bucket_id=bucket_id,
        public_case=public_case,
        original_answer_token_f1=scoring["answer_token_f1"],
        best_gold_span_token_f1=scoring["best_gold_span_token_f1"],
        answer_gold_span_gap=scoring["answer_gold_span_gap"],
    )


def _select_candidate_evidence(
    *,
    question: PrimeQAQuestion,
    config: _RunConfig,
    candidates: Sequence[SentenceEvidenceCandidate],
) -> list[SentenceEvidenceCandidate]:
    if config.component_family == "support_aware_answerability_gate":
        max_sentences = 3
        anchor_strategy = "top_scoring_evidence_sentences"
        anchor_top_n = max_sentences
    elif config.component_family == "context_present_span_composer":
        max_sentences = int(config.payload.get("max_sentences") or 2)
        anchor_strategy = str(config.payload.get("anchor_strategy"))
        anchor_top_n = int(config.payload.get("anchor_top_n") or max_sentences)
    elif config.component_family == "joint_support_gate_span_composer":
        max_sentences = int(config.payload.get("max_sentences") or 2)
        anchor_strategy = str(config.payload.get("composer_anchor_strategy"))
        anchor_top_n = int(config.payload.get("composer_anchor_top_n") or max_sentences)
    else:
        raise ValueError(f"Unknown component family: {config.component_family}")

    if anchor_strategy == "top_scoring_evidence_sentences":
        ranked = list(candidates)
    elif anchor_strategy == "title_query_overlap_then_evidence_score":
        ranked = sorted(
            candidates,
            key=lambda candidate: _title_query_anchor_key(question, candidate),
        )
    else:
        raise ValueError(f"Unknown anchor strategy: {anchor_strategy}")
    return list(ranked[: min(max_sentences, anchor_top_n)])


def _title_query_anchor_key(
    question: PrimeQAQuestion,
    candidate: SentenceEvidenceCandidate,
) -> tuple[float, float, int, str]:
    question_terms = set(tokenize_text(question.full_question))
    title_terms = set(tokenize_text(candidate.retrieval_result.document.title))
    sentence_terms = set(tokenize_text(candidate.sentence))
    title_overlap = len(question_terms & title_terms)
    sentence_overlap = len(question_terms & sentence_terms)
    anchor_score = (2.0 * title_overlap) + (0.25 * sentence_overlap)
    return (
        -anchor_score,
        -candidate.score,
        candidate.retrieval_result.rank,
        candidate.retrieval_result.document.id,
    )


def _generated_answer_from_candidates(
    *,
    question: PrimeQAQuestion,
    selected_candidates: Sequence[SentenceEvidenceCandidate],
) -> GeneratedAnswer:
    if not selected_candidates:
        return GeneratedAnswer(
            question_id=question.id,
            answer="I do not have enough retrieved evidence to answer this question.",
            citations=[],
            refused=True,
        )
    return GeneratedAnswer(
        question_id=question.id,
        answer=" ".join(
            f"{candidate.sentence} [{candidate.retrieval_result.document.id}]"
            for candidate in selected_candidates
        ),
        citations=[
            AnswerCitation(
                document_id=candidate.retrieval_result.document.id,
                title=candidate.retrieval_result.document.title,
                retrieval_rank=candidate.retrieval_result.rank,
                evidence_score=round(candidate.score, 4),
            )
            for candidate in selected_candidates
        ],
        refused=False,
    )


def _verify_candidate_answer(
    *,
    config: _RunConfig,
    answer: GeneratedAnswer,
    selected_candidates: Sequence[SentenceEvidenceCandidate],
    retrieval_results: Sequence[RetrievalResult],
) -> AnswerVerificationResult:
    max_citation_rank = int(config.payload.get("max_citation_rank") or 3)
    min_evidence_score = float(
        config.payload.get("min_evidence_score")
        or config.payload.get("gate_min_evidence_score")
        or 7.0
    )
    if _uses_support_gate(config):
        support_count = sum(
            1
            for candidate in selected_candidates
            if candidate.score >= _gate_min_evidence_score(config)
            and candidate.retrieval_result.rank <= max_citation_rank
        )
        if support_count < _gate_min_support_count(config):
            refused = GeneratedAnswer(
                question_id=answer.question_id,
                answer="I do not have enough verified evidence to answer this question.",
                citations=[],
                refused=True,
            )
            return AnswerVerificationResult(
                original_answer=answer,
                verified_answer=refused,
                citation_context_valid=True,
                reasons=["support_count_below_threshold"],
            )
    verifier = AnswerVerifier(
        min_citations=1,
        min_evidence_score=min_evidence_score,
        max_citation_rank=max_citation_rank,
    )
    return verifier.verify(answer, retrieval_results)


def _uses_support_gate(config: _RunConfig) -> bool:
    return config.component_family in {
        "support_aware_answerability_gate",
        "joint_support_gate_span_composer",
    }


def _gate_min_support_count(config: _RunConfig) -> int:
    return int(
        config.payload.get("min_supporting_evidence_count")
        or config.payload.get("gate_min_supporting_evidence_count")
        or 2
    )


def _gate_min_evidence_score(config: _RunConfig) -> float:
    return float(
        config.payload.get("min_evidence_score")
        or config.payload.get("gate_min_evidence_score")
        or 7.0
    )


def _public_result(
    *,
    config_id: str,
    candidate_family_id: str,
    component_family: str,
    target_buckets: Sequence[str],
    traces_by_split: Mapping[str, Sequence[_SplitQuestionTrace]],
    baseline_traces_by_split: Mapping[str, Sequence[_SplitQuestionTrace]],
    fold_assignments: Mapping[str, str],
    target_weights: Mapping[str, float],
    baseline_public: Mapping[str, Any] | None,
    train_cv_selectability: Mapping[str, Any] | None,
) -> dict[str, Any]:
    eval_traces = _eval_traces_by_split(traces_by_split)
    baseline_eval_traces = _eval_traces_by_split(baseline_traces_by_split)
    aggregate_outputs = _aggregate_outputs(eval_traces)
    metrics_by_split = _metrics_by_split(eval_traces)
    weighted_scores = _weighted_scores_by_split(
        aggregate_outputs=aggregate_outputs,
        target_weights=target_weights,
    )
    if baseline_public is None:
        weighted_deltas = {split: 0.0 for split in _EVAL_SPLITS}
        target_bucket_deltas = {
            split: {bucket: 0 for bucket in _TARGET_BUCKETS}
            for split in _EVAL_SPLITS
        }
        metric_deltas = {
            split: {
                "answerable_refusal_rate": 0.0,
                "unanswerable_refusal_rate": 0.0,
                "gold_doc_citation_rate": 0.0,
                "average_token_f1": 0.0,
            }
            for split in _EVAL_SPLITS
        }
    else:
        weighted_deltas = {
            split: round(
                weighted_scores[split]
                - baseline_public["weighted_target_scores_by_split"][split],
                4,
            )
            for split in _EVAL_SPLITS
        }
        target_bucket_deltas = _target_bucket_deltas_by_split(
            aggregate_outputs=aggregate_outputs,
            baseline_aggregate_outputs=baseline_public["aggregate_outputs"],
        )
        metric_deltas = _metric_deltas_by_split(
            metrics_by_split=metrics_by_split,
            baseline_metrics_by_split=baseline_public["metrics_by_split"],
        )
    return {
        "config_id": config_id,
        "candidate_family_id": candidate_family_id,
        "component_family": component_family,
        "target_buckets": list(target_buckets),
        "aggregate_outputs": aggregate_outputs,
        "metrics_by_split": metrics_by_split,
        "weighted_target_scores_by_split": weighted_scores,
        "weighted_target_score_deltas_by_split": weighted_deltas,
        "target_bucket_deltas_by_split": target_bucket_deltas,
        "metric_deltas_by_split": metric_deltas,
        "changed_answer_counts_by_split": _changed_answer_counts_eval_splits(
            baseline_eval_traces=baseline_eval_traces,
            eval_traces=eval_traces,
        ),
        "train_cv_fold_results": _train_cv_fold_results(
            traces_by_split=traces_by_split,
            baseline_traces_by_split=baseline_traces_by_split,
            fold_assignments=fold_assignments,
            target_weights=target_weights,
        ),
        "train_cv_selectability": train_cv_selectability,
    }


def _eval_traces_by_split(
    traces_by_split: Mapping[str, Sequence[_SplitQuestionTrace]],
) -> dict[str, list[_SplitQuestionTrace]]:
    return {
        "train_cv": list(traces_by_split["train"]),
        "train_full": list(traces_by_split["train"]),
        "dev": list(traces_by_split["dev"]),
    }


def _weighted_scores_by_split(
    *,
    aggregate_outputs: Mapping[str, Any],
    target_weights: Mapping[str, float],
) -> dict[str, float]:
    counts_by_split = aggregate_outputs["bucket_counts_by_split"]
    return {
        split: round(
            sum(
                float(counts_by_split[split].get(bucket, 0)) * weight
                for bucket, weight in target_weights.items()
            ),
            4,
        )
        for split in _EVAL_SPLITS
    }


def _target_bucket_deltas_by_split(
    *,
    aggregate_outputs: Mapping[str, Any],
    baseline_aggregate_outputs: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    counts = aggregate_outputs["bucket_counts_by_split"]
    baseline_counts = baseline_aggregate_outputs["bucket_counts_by_split"]
    return {
        split: {
            bucket: int(counts[split].get(bucket, 0))
            - int(baseline_counts[split].get(bucket, 0))
            for bucket in _TARGET_BUCKETS + ("retrieval_context_miss",)
        }
        for split in _EVAL_SPLITS
    }


def _metric_deltas_by_split(
    *,
    metrics_by_split: Mapping[str, Any],
    baseline_metrics_by_split: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    metric_names = (
        "answerable_refusal_rate",
        "unanswerable_refusal_rate",
        "gold_doc_citation_rate",
        "average_token_f1",
    )
    return {
        split: {
            metric: round(
                float(metrics_by_split[split]["verified"][metric])
                - float(baseline_metrics_by_split[split]["verified"][metric]),
                4,
            )
            for metric in metric_names
        }
        for split in _EVAL_SPLITS
    }


def _changed_answer_counts_eval_splits(
    *,
    baseline_eval_traces: Mapping[str, Sequence[_SplitQuestionTrace]],
    eval_traces: Mapping[str, Sequence[_SplitQuestionTrace]],
) -> dict[str, int]:
    return {
        split: sum(
            _answer_signature(base_trace.verified_answer)
            != _answer_signature(trace.verified_answer)
            for base_trace, trace in zip(
                baseline_eval_traces[split],
                eval_traces[split],
                strict=True,
            )
        )
        for split in _EVAL_SPLITS
    }


def _train_cv_fold_results(
    *,
    traces_by_split: Mapping[str, Sequence[_SplitQuestionTrace]],
    baseline_traces_by_split: Mapping[str, Sequence[_SplitQuestionTrace]],
    fold_assignments: Mapping[str, str],
    target_weights: Mapping[str, float],
) -> list[dict[str, Any]]:
    fold_ids = sorted(set(fold_assignments.values()))
    results = []
    for fold_id in fold_ids:
        traces = [
            trace
            for trace in traces_by_split["train"]
            if fold_assignments[trace.sample.sample_id] == fold_id
        ]
        baseline_traces = [
            trace
            for trace in baseline_traces_by_split["train"]
            if fold_assignments[trace.sample.sample_id] == fold_id
        ]
        aggregate = _aggregate_outputs({fold_id: traces})
        baseline_aggregate = _aggregate_outputs({fold_id: baseline_traces})
        score = _weighted_scores_by_split(
            aggregate_outputs={
                "bucket_counts_by_split": {
                    "train_cv": aggregate["bucket_counts_by_split"][fold_id],
                    "train_full": aggregate["bucket_counts_by_split"][fold_id],
                    "dev": aggregate["bucket_counts_by_split"][fold_id],
                }
            },
            target_weights=target_weights,
        )["train_cv"]
        baseline_score = _weighted_scores_by_split(
            aggregate_outputs={
                "bucket_counts_by_split": {
                    "train_cv": baseline_aggregate["bucket_counts_by_split"][fold_id],
                    "train_full": baseline_aggregate["bucket_counts_by_split"][fold_id],
                    "dev": baseline_aggregate["bucket_counts_by_split"][fold_id],
                }
            },
            target_weights=target_weights,
        )["train_cv"]
        results.append(
            {
                "fold_id": fold_id,
                "row_count": len(traces),
                "weighted_target_score": score,
                "weighted_target_delta": round(score - baseline_score, 4),
                "bucket_counts": aggregate["bucket_counts_by_split"][fold_id],
            }
        )
    return results


def _train_cv_selectability(
    *,
    result: Mapping[str, Any],
    baseline_result: Mapping[str, Any],
    frozen_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    guards = (
        (frozen_protocol.get("train_selection_rule") or {})
        .get("selectability_guards")
        or {}
    )
    train_cv_metrics = result["metrics_by_split"]["train_cv"]["verified"]
    baseline_train_cv_metrics = baseline_result["metrics_by_split"]["train_cv"][
        "verified"
    ]
    metric_deltas = result["metric_deltas_by_split"]["train_cv"]
    target_deltas = result["target_bucket_deltas_by_split"]["train_cv"]
    weighted_delta = float(result["weighted_target_score_deltas_by_split"]["train_cv"])
    average_token_f1_drop = round(
        float(baseline_train_cv_metrics["average_token_f1"])
        - float(train_cv_metrics["average_token_f1"]),
        4,
    )
    gold_doc_citation_rate_drop = round(
        float(baseline_train_cv_metrics["gold_doc_citation_rate"])
        - float(train_cv_metrics["gold_doc_citation_rate"]),
        4,
    )
    retrieval_context_miss_delta = int(target_deltas.get("retrieval_context_miss", 0))
    checks = {
        "train_cv_weighted_target_delta_negative": weighted_delta < 0.0,
        "answerable_refusal_rate_delta_within_guard": metric_deltas[
            "answerable_refusal_rate"
        ]
        <= float(guards.get("max_train_cv_answerable_refusal_rate_delta") or 0.0),
        "average_token_f1_drop_within_guard": average_token_f1_drop
        <= float(guards.get("max_train_cv_average_token_f1_drop") or 0.0),
        "gold_doc_citation_rate_drop_within_guard": gold_doc_citation_rate_drop
        <= float(guards.get("max_train_cv_gold_doc_citation_rate_drop") or 0.0),
        "retrieval_context_miss_delta_within_guard": retrieval_context_miss_delta
        <= int(guards.get("max_train_cv_retrieval_context_miss_delta") or 0),
    }
    return {
        "selectable": all(checks.values()),
        "observed": {
            "train_cv_weighted_target_delta": weighted_delta,
            "answerable_refusal_rate_delta": metric_deltas["answerable_refusal_rate"],
            "average_token_f1_drop": average_token_f1_drop,
            "gold_doc_citation_rate_drop": gold_doc_citation_rate_drop,
            "retrieval_context_miss_delta": retrieval_context_miss_delta,
        },
        "thresholds": guards,
        "checks": checks,
    }


def _train_cv_selection(
    *,
    config_results: Sequence[Mapping[str, Any]],
    baseline_result: Mapping[str, Any],
    frozen_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    public_results = [result["public"] for result in config_results]
    selectable = [
        result
        for result in public_results
        if result["train_cv_selectability"]["selectable"] is True
    ]
    ranking = sorted(public_results, key=_selection_key)
    selected = sorted(selectable, key=_selection_key)[0] if selectable else None
    return {
        "selection_split": "train",
        "selection_mode": "train_grouped_cross_validation_then_full_train_refit",
        "baseline_train_cv_weighted_target_score": baseline_result["public"][
            "weighted_target_scores_by_split"
        ]["train_cv"],
        "selected_config_id": selected["config_id"] if selected else None,
        "selected_candidate_family_id": (
            selected["candidate_family_id"] if selected else None
        ),
        "selected_train_cv_weighted_target_score": (
            selected["weighted_target_scores_by_split"]["train_cv"] if selected else None
        ),
        "selected_train_cv_weighted_target_delta": (
            selected["weighted_target_score_deltas_by_split"]["train_cv"]
            if selected
            else None
        ),
        "selectable_config_count": len(selectable),
        "config_count": len(public_results),
        "selection_ranking": [
            {
                "rank": index,
                "config_id": result["config_id"],
                "candidate_family_id": result["candidate_family_id"],
                "train_cv_weighted_target_score": result[
                    "weighted_target_scores_by_split"
                ]["train_cv"],
                "train_cv_weighted_target_delta": result[
                    "weighted_target_score_deltas_by_split"
                ]["train_cv"],
                "train_cv_selectable": result["train_cv_selectability"]["selectable"],
                "train_cv_changed_answer_count": result[
                    "changed_answer_counts_by_split"
                ]["train_cv"],
            }
            for index, result in enumerate(ranking, start=1)
        ],
        "train_selection_rule": frozen_protocol.get("train_selection_rule") or {},
    }


def _selection_key(result: Mapping[str, Any]) -> tuple[Any, ...]:
    counts = result["aggregate_outputs"]["bucket_counts_by_split"]["train_cv"]
    metrics = result["metrics_by_split"]["train_cv"]["verified"]
    return (
        0 if result["train_cv_selectability"]["selectable"] else 1,
        float(result["weighted_target_scores_by_split"]["train_cv"]),
        int(counts.get("answerability_false_answer", 0)),
        int(counts.get("gold_span_beats_selected_answer", 0)),
        int(counts.get("evidence_selection_miss", 0)),
        -float(metrics["average_token_f1"]),
        -float(metrics["gold_doc_citation_rate"]),
        float(metrics["answerable_refusal_rate"]),
        int(result["changed_answer_counts_by_split"]["train_cv"]),
        str(result["config_id"]),
    )


def _selected_config_result(
    *,
    config_results: Sequence[Mapping[str, Any]],
    selected_config_id: str | None,
) -> Mapping[str, Any] | None:
    if selected_config_id is None:
        return None
    for result in config_results:
        if result["public"]["config_id"] == selected_config_id:
            return result["public"]
    return None


def _dev_validation(
    *,
    selected_result: Mapping[str, Any] | None,
    frozen_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    if selected_result is None:
        return {
            "validation_split": "dev",
            "selected_config_id": None,
            "status": "no_train_cv_selectable_config",
            "dev_validation_passed": False,
        }
    pass_conditions = (
        (frozen_protocol.get("dev_validation_rule") or {}).get("pass_conditions")
        or {}
    )
    dev_delta = float(selected_result["weighted_target_score_deltas_by_split"]["dev"])
    metric_deltas = selected_result["metric_deltas_by_split"]["dev"]
    baseline_drop_average_f1 = max(0.0, -float(metric_deltas["average_token_f1"]))
    baseline_drop_citation = max(0.0, -float(metric_deltas["gold_doc_citation_rate"]))
    checks = {
        "dev_weighted_target_delta_negative": dev_delta < 0.0,
        "dev_answerable_refusal_rate_delta_within_guard": metric_deltas[
            "answerable_refusal_rate"
        ]
        <= float(pass_conditions.get("dev_answerable_refusal_rate_delta_must_not_exceed") or 0.0),
        "dev_average_token_f1_drop_within_guard": baseline_drop_average_f1
        <= float(pass_conditions.get("dev_average_token_f1_drop_must_not_exceed") or 0.0),
        "dev_gold_doc_citation_rate_drop_within_guard": baseline_drop_citation
        <= float(pass_conditions.get("dev_gold_doc_citation_rate_drop_must_not_exceed") or 0.0),
    }
    return {
        "validation_split": "dev",
        "selected_config_id": selected_result["config_id"],
        "selected_candidate_family_id": selected_result["candidate_family_id"],
        "dev_weighted_target_score": selected_result["weighted_target_scores_by_split"][
            "dev"
        ],
        "dev_weighted_target_delta": dev_delta,
        "dev_target_bucket_deltas": selected_result["target_bucket_deltas_by_split"][
            "dev"
        ],
        "dev_metric_deltas": metric_deltas,
        "dev_changed_answer_count": selected_result["changed_answer_counts_by_split"][
            "dev"
        ],
        "dev_validation_checks": checks,
        "dev_validation_passed": all(checks.values()),
    }


def _changed_case_samples(
    *,
    selected_result: Mapping[str, Any] | None,
    baseline_result: Mapping[str, Any],
    fold_assignments: Mapping[str, str],
    sample_limit_per_transition: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    if selected_result is None:
        return {split: {} for split in _EVAL_SPLITS}
    # The public result does not keep traces; this function is intentionally filled
    # by the caller after trace-bearing lookup is available in future stages.
    # Stage109 still reports aggregate changed-answer counts for all configs.
    return {split: {} for split in _EVAL_SPLITS}


def _build_train_fold_assignments(
    samples: Sequence[PrimeQAHybridSplitSample],
    *,
    fold_count: int,
) -> dict[str, str]:
    groups: dict[str, list[PrimeQAHybridSplitSample]] = defaultdict(list)
    for sample in samples:
        groups[_group_key(sample)].append(sample)
    fold_rows: list[list[PrimeQAHybridSplitSample]] = [[] for _ in range(fold_count)]
    for _group_key_value, group_samples in sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), _stable_hash(item[0])),
    ):
        target_index = min(
            range(fold_count),
            key=lambda index: (len(fold_rows[index]), index),
        )
        fold_rows[target_index].extend(group_samples)
    return {
        sample.sample_id: f"fold_{fold_index + 1}"
        for fold_index, fold_samples in enumerate(fold_rows)
        for sample in fold_samples
    }


def _group_key(sample: PrimeQAHybridSplitSample) -> str:
    normalized_question = " ".join(sample.question_text.lower().split())
    doc_marker = sample.answer_doc_id if sample.answerable else "UNANSWERABLE"
    return f"{normalized_question}::{doc_marker}"


def _fold_summary(
    samples: Sequence[PrimeQAHybridSplitSample],
    fold_assignments: Mapping[str, str],
) -> dict[str, Any]:
    row_counts: Counter[str] = Counter()
    group_counts: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        fold_id = fold_assignments[sample.sample_id]
        row_counts[fold_id] += 1
        group_counts[fold_id].add(_stable_hash(_group_key(sample)))
    return {
        "fold_count": len(set(fold_assignments.values())),
        "row_counts_by_fold": dict(sorted(row_counts.items())),
        "group_counts_by_fold": {
            fold_id: len(groups)
            for fold_id, groups in sorted(group_counts.items())
        },
        "raw_group_values_written": False,
    }


def _stage108_summary(stage108_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage108_report.get("decision") or {}
    frozen = stage108_report.get("frozen_protocol") or {}
    return {
        "stage": stage108_report.get("stage"),
        "protocol_id": stage108_report.get("protocol_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "can_run_train_dev_comparison_after_user_confirmation": decision.get(
            "can_run_train_dev_comparison_after_user_confirmation"
        ),
        "candidate_config_count": len(frozen.get("candidate_config_grid") or []),
        "train_selection_rule": frozen.get("train_selection_rule") or {},
        "dev_validation_rule": frozen.get("dev_validation_rule") or {},
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
    }


def _stage102_summary(stage102_report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": stage102_report.get("stage"),
        "analysis_id": stage102_report.get("analysis_id"),
        "decision_status": (stage102_report.get("decision") or {}).get("status"),
        "bucket_counts_by_split": (
            stage102_report.get("aggregate_outputs") or {}
        ).get("bucket_counts_by_split"),
        "metrics_by_split": stage102_report.get("metrics_by_split"),
    }


def _guard_checks(
    *,
    stage108_summary: Mapping[str, Any],
    stage102_summary: Mapping[str, Any],
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    candidate_configs: Sequence[_RunConfig],
    baseline_result: Mapping[str, Any],
    candidate_results: Sequence[Mapping[str, Any]],
    train_cv_selection: Mapping[str, Any],
    dev_validation: Mapping[str, Any],
    changed_case_samples: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    frozen_protocol: Mapping[str, Any],
    user_confirmed_comparison: bool,
) -> list[dict[str, Any]]:
    public_candidate_results = [result["public"] for result in candidate_results]
    all_changed_cases = [
        case
        for by_transition in changed_case_samples.values()
        for cases in by_transition.values()
        for case in cases
    ]
    train_ranking_keys = {
        tuple(item)
        for item in (
            tuple(row.keys()) for row in train_cv_selection.get("selection_ranking") or []
        )
    }
    selected_delta = train_cv_selection.get("selected_train_cv_weighted_target_delta")
    return [
        _check(
            name="stage108_source_is_expected_stage",
            passed=stage108_summary.get("stage") == _SOURCE_STAGE108,
            observed=stage108_summary.get("stage"),
            expected=_SOURCE_STAGE108,
        ),
        _check(
            name="stage108_protocol_id_matches",
            passed=stage108_summary.get("protocol_id") == _SOURCE_PROTOCOL_ID,
            observed=stage108_summary.get("protocol_id"),
            expected=_SOURCE_PROTOCOL_ID,
        ),
        _check(
            name="user_confirmed_stage109_comparison",
            passed=user_confirmed_comparison,
            observed=user_confirmed_comparison,
            expected=True,
        ),
        _check(
            name="stage108_protocol_frozen",
            passed=stage108_summary.get("decision_status") == _SOURCE_DECISION_STATUS,
            observed=stage108_summary.get("decision_status"),
            expected=_SOURCE_DECISION_STATUS,
        ),
        _check(
            name="stage108_allows_train_dev_comparison_after_confirmation",
            passed=stage108_summary.get(
                "can_run_train_dev_comparison_after_user_confirmation"
            )
            is True,
            observed=stage108_summary.get(
                "can_run_train_dev_comparison_after_user_confirmation"
            ),
            expected=True,
        ),
        _check(
            name="stage108_final_test_gate_locked",
            passed=stage108_summary.get("can_open_final_test_gate_now") is False
            and stage108_summary.get("can_run_final_test_metrics_now") is False,
            observed={
                "can_open_final_test_gate_now": stage108_summary.get(
                    "can_open_final_test_gate_now"
                ),
                "can_run_final_test_metrics_now": stage108_summary.get(
                    "can_run_final_test_metrics_now"
                ),
            },
            expected=False,
        ),
        _check(
            name="stage108_runtime_defaults_unchanged",
            passed=stage108_summary.get("default_runtime_policy") == "unchanged",
            observed=stage108_summary.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage108_fallback_disabled",
            passed=stage108_summary.get("fallback_strategies_enabled") is False,
            observed=stage108_summary.get("fallback_strategies_enabled"),
            expected=False,
        ),
        _check(
            name="only_train_dev_splits_loaded",
            passed=tuple(split_samples) == _ALLOWED_SPLITS,
            observed=list(split_samples),
            expected=list(_ALLOWED_SPLITS),
        ),
        _check(
            name="loaded_samples_are_train_dev_only",
            passed=all(
                sample.assigned_split == split
                for split, samples in split_samples.items()
                for sample in samples
            ),
            observed={
                split: dict(Counter(sample.assigned_split for sample in samples))
                for split, samples in split_samples.items()
            },
            expected={split: split for split in _ALLOWED_SPLITS},
        ),
        _check(
            name="candidate_config_count_matches_stage108",
            passed=len(candidate_configs) == 7
            and len(public_candidate_results)
            == int(stage108_summary.get("candidate_config_count") or 0),
            observed={
                "candidate_configs": len(candidate_configs),
                "public_results": len(public_candidate_results),
                "stage108": stage108_summary.get("candidate_config_count"),
            },
            expected=7,
        ),
        _check(
            name="train_cv_fold_count_matches_protocol",
            passed=(
                (baseline_result["public"]["train_cv_fold_results"] or [])
                and len(baseline_result["public"]["train_cv_fold_results"])
                == _train_cv_fold_count(frozen_protocol)
            ),
            observed=len(baseline_result["public"]["train_cv_fold_results"]),
            expected=_train_cv_fold_count(frozen_protocol),
        ),
        _check(
            name="train_cv_selection_uses_train_only_fields",
            passed=train_ranking_keys
            == {
                (
                    "rank",
                    "config_id",
                    "candidate_family_id",
                    "train_cv_weighted_target_score",
                    "train_cv_weighted_target_delta",
                    "train_cv_selectable",
                    "train_cv_changed_answer_count",
                )
            },
            observed={"ranking_keys": [list(keys) for keys in sorted(train_ranking_keys)]},
            expected="train-CV ranking fields only",
        ),
        _check(
            name="train_cv_selection_blocks_noop_candidates",
            passed=selected_delta is None or float(selected_delta) < 0.0,
            observed=selected_delta,
            expected="< 0.0 or no selected config",
        ),
        _check(
            name="dev_validation_not_used_for_selection",
            passed=(stage108_summary.get("dev_validation_rule") or {}).get(
                "dev_selection_allowed"
            )
            is False
            and (stage108_summary.get("dev_validation_rule") or {}).get(
                "dev_retuning_allowed"
            )
            is False,
            observed=stage108_summary.get("dev_validation_rule"),
            expected="dev selection and retuning disabled",
        ),
        _check(
            name="stage109_final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage109_runtime_defaults_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
        _check(
            name="stage109_fallback_strategies_not_added",
            passed=True,
            observed=False,
            expected=False,
        ),
        _check(
            name="changed_case_samples_match_public_contract",
            passed=all(tuple(case) == _CHANGED_CASE_FIELDS for case in all_changed_cases),
            observed=sorted({tuple(case) for case in all_changed_cases}),
            expected=list(_CHANGED_CASE_FIELDS),
        ),
        _check(
            name="public_outputs_have_no_forbidden_keys",
            passed=not _contains_forbidden_key(
                {
                    "baseline": baseline_result["public"],
                    "configs": public_candidate_results,
                    "selection": train_cv_selection,
                    "dev_validation": dev_validation,
                    "changed_cases": changed_case_samples,
                }
            ),
            observed=sorted(
                _forbidden_keys_found(
                    {
                        "baseline": baseline_result["public"],
                        "configs": public_candidate_results,
                        "selection": train_cv_selection,
                        "dev_validation": dev_validation,
                        "changed_cases": changed_case_samples,
                    }
                )
            ),
            expected=[],
        ),
    ]


def _decision(
    *,
    guard_checks: Sequence[Mapping[str, Any]],
    train_cv_selection: Mapping[str, Any],
    dev_validation: Mapping[str, Any],
) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    base = {
        "analysis_id": _ANALYSIS_ID,
        "selected_config_id": train_cv_selection.get("selected_config_id"),
        "selected_candidate_family_id": train_cv_selection.get(
            "selected_candidate_family_id"
        ),
        "selectable_config_count": train_cv_selection.get("selectable_config_count"),
        "dev_validation_passed": dev_validation.get("dev_validation_passed"),
        "dev_weighted_target_delta": dev_validation.get("dev_weighted_target_delta"),
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
    }
    if failed_checks:
        return {
            **base,
            "status": "primeqa_hybrid_failure_pattern_redesign_comparison_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
        }
    if train_cv_selection.get("selected_config_id") is None:
        return {
            **base,
            "status": _NO_TRAIN_CV_SELECTABLE_STATUS,
            "can_continue_train_dev_development": True,
            "recommended_next_direction": "record_failure_pattern_redesign_stop_decision",
            "recommended_next_stage": (
                "Stage110: record a stop decision for the frozen Stage108 "
                "redesign family because no candidate satisfied train-CV "
                "selectability."
            ),
        }
    if dev_validation.get("dev_validation_passed") is True:
        return {
            **base,
            "status": "primeqa_hybrid_failure_pattern_redesign_completed_dev_validation_passed",
            "can_continue_train_dev_development": True,
            "recommended_next_direction": "review_changed_cases_before_any_runtime_gate",
            "recommended_next_stage": (
                "Stage110: review public-safe changed cases before any runtime "
                "or final-test gate decision."
            ),
        }
    return {
        **base,
        "status": "primeqa_hybrid_failure_pattern_redesign_completed_dev_validation_failed",
        "can_continue_train_dev_development": True,
        "recommended_next_direction": "record_failure_pattern_redesign_stop_decision",
        "recommended_next_stage": (
            "Stage110: record a stop decision because the train-CV-selected "
            "config failed dev validation. Do not select from dev."
        ),
    }


def _weighted_delta_bars(report: Mapping[str, Any], split: str) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(result["config_id"]),
            value=float(result["weighted_target_score_deltas_by_split"][split]),
            value_label=f"{float(result['weighted_target_score_deltas_by_split'][split]):+.2f}",
        )
        for result in report.get("config_results") or []
    ]


def _train_selectability_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(result["config_id"]),
            value=1.0 if result["train_cv_selectability"]["selectable"] else 0.0,
            value_label=(
                "selectable"
                if result["train_cv_selectability"]["selectable"]
                else "blocked"
            ),
        )
        for result in report.get("config_results") or []
    ]


def _changed_answer_count_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for result in report.get("config_results") or []:
        bars.append(
            BarDatum(
                label=f"{result['config_id']} train-CV",
                value=float(result["changed_answer_counts_by_split"]["train_cv"]),
                value_label=str(result["changed_answer_counts_by_split"]["train_cv"]),
            )
        )
        bars.append(
            BarDatum(
                label=f"{result['config_id']} dev",
                value=float(result["changed_answer_counts_by_split"]["dev"]),
                value_label=str(result["changed_answer_counts_by_split"]["dev"]),
            )
        )
    return bars


def _selected_dev_metric_delta_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    metric_deltas = (report.get("dev_validation") or {}).get("dev_metric_deltas") or {}
    return [
        BarDatum(
            label=str(metric),
            value=float(delta),
            value_label=f"{float(delta):+.4f}",
        )
        for metric, delta in metric_deltas.items()
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    names = [
        "dev_validation_passed",
        "can_continue_train_dev_development",
        "can_open_final_test_gate_now",
        "can_run_final_test_metrics_now",
        "can_use_test_for_tuning",
        "fallback_strategies_enabled",
    ]
    return [
        BarDatum(
            label=name,
            value=1.0 if decision.get(name) is True else 0.0,
            value_label=str(decision.get(name)),
        )
        for name in names
    ]


def _guard_check_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(check["name"]),
            value=1.0 if check["passed"] else 0.0,
            value_label="passed" if check["passed"] else "failed",
        )
        for check in report.get("guard_checks") or []
    ]


def _config_from_mapping(config: Mapping[str, Any]) -> _RunConfig:
    return _RunConfig(
        config_id=str(config["config_id"]),
        candidate_family_id=str(config["candidate_family_id"]),
        component_family=str(config["component_family"]),
        target_buckets=tuple(str(bucket) for bucket in config.get("target_buckets") or []),
        payload=dict(config),
    )


def _target_weights(frozen_protocol: Mapping[str, Any]) -> dict[str, float]:
    weights = (
        (frozen_protocol.get("train_selection_rule") or {})
        .get("objective", {})
        .get("weighted_target_bucket_score")
        or {}
    )
    return {str(bucket): float(weight) for bucket, weight in weights.items()}


def _train_cv_fold_count(frozen_protocol: Mapping[str, Any]) -> int:
    return int(
        (frozen_protocol.get("train_selection_rule") or {}).get("train_cv_fold_count")
        or 5
    )


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _contains_forbidden_key(value: Any) -> bool:
    return bool(_forbidden_keys_found(value))


def _forbidden_keys_found(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in _FORBIDDEN_REPORT_KEYS:
                found.add(str(key))
            found.update(_forbidden_keys_found(nested))
    elif isinstance(value, list | tuple):
        for nested in value:
            found.update(_forbidden_keys_found(nested))
    return found


def _fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _validate_options(
    *,
    retrieval_top_k: int,
    bm25_k1: float,
    bm25_b: float,
    baseline_max_candidates_per_document: int,
    baseline_max_sentences: int,
    baseline_min_sentence_score: float,
    baseline_verifier_min_citations: int,
    baseline_verifier_min_evidence_score: float,
    baseline_verifier_max_citation_rank: int,
    max_gold_window_sentences: int,
    gold_span_gap_margin: float,
    low_answer_f1_threshold: float,
    sample_limit_per_transition: int,
) -> None:
    if retrieval_top_k <= 0:
        raise ValueError("retrieval_top_k must be positive")
    if bm25_k1 <= 0:
        raise ValueError("bm25_k1 must be positive")
    if not 0 <= bm25_b <= 1:
        raise ValueError("bm25_b must be between 0 and 1")
    if baseline_max_candidates_per_document <= 0:
        raise ValueError("baseline_max_candidates_per_document must be positive")
    if baseline_max_sentences <= 0:
        raise ValueError("baseline_max_sentences must be positive")
    if baseline_min_sentence_score < 0:
        raise ValueError("baseline_min_sentence_score must be non-negative")
    if baseline_verifier_min_citations <= 0:
        raise ValueError("baseline_verifier_min_citations must be positive")
    if baseline_verifier_min_evidence_score < 0:
        raise ValueError("baseline_verifier_min_evidence_score must be non-negative")
    if baseline_verifier_max_citation_rank <= 0:
        raise ValueError("baseline_verifier_max_citation_rank must be positive")
    if max_gold_window_sentences <= 0:
        raise ValueError("max_gold_window_sentences must be positive")
    if gold_span_gap_margin < 0:
        raise ValueError("gold_span_gap_margin must be non-negative")
    if low_answer_f1_threshold < 0:
        raise ValueError("low_answer_f1_threshold must be non-negative")
    if sample_limit_per_transition < 0:
        raise ValueError("sample_limit_per_transition must be non-negative")
