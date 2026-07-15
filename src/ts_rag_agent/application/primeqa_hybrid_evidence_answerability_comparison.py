from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.answer_composition import create_answer_composition_policy
from ts_rag_agent.application.answer_verification import AnswerVerifier
from ts_rag_agent.application.evidence_selection import (
    create_sentence_evidence_selector,
    trace_selector_route,
)
from ts_rag_agent.application.primeqa_hybrid_answer_pipeline_error_decomposition_analysis import (
    _aggregate_outputs,
    _best_gold_span_token_f1,
    _metrics_by_split,
    _public_case,
    _SplitQuestionTrace,
)
from ts_rag_agent.application.rag_answering import ExtractiveAnswerGenerator
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.application.text_metrics import token_f1
from ts_rag_agent.domain.answer import GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
    summarize_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import load_primeqa_documents

_STAGE = "Stage 105"
_CREATED_AT = "2026-07-15"
_SOURCE_STAGE104 = "Stage 104"
_SOURCE_STAGE102 = "Stage 102"
_ANALYSIS_ID = "evidence_answerability_candidate_train_dev_comparison_v1"
_STAGE104_PROTOCOL_ID = "evidence_answerability_candidate_train_dev_comparison_v1"
_EXPECTED_STAGE104_STATUS = (
    "primeqa_hybrid_evidence_answerability_comparison_protocol_frozen"
)
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_ALLOWED_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = ("test",)
_TARGET_BUCKET_WEIGHTS = {
    "answerability_false_answer": 1.55,
    "gold_span_beats_selected_answer": 1.45,
    "evidence_selection_miss": 1.70,
}
_TARGET_BUCKETS = tuple(_TARGET_BUCKET_WEIGHTS)
_BUCKET_ORDER = (
    "answerability_false_answer",
    "retrieval_context_miss",
    "evidence_selection_miss",
    "verification_over_refusal",
    "gold_span_beats_selected_answer",
    "low_overlap_gold_cited_answer",
    "answer_supported_and_cited",
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
_CHANGED_CASE_FIELDS = (
    "sample_id",
    "split",
    "config_id",
    "candidate_id",
    "baseline_bucket_id",
    "candidate_bucket_id",
    "baseline_answer_token_f1_bucket",
    "candidate_answer_token_f1_bucket",
    "baseline_citation_status",
    "candidate_citation_status",
    "answerability_action",
    "evidence_selection_action",
    "changed_case_confidence_band",
)


@dataclass(frozen=True)
class _RunConfig:
    config_id: str
    candidate_id: str
    selector_name: str
    composition_policy_name: str
    max_candidates_per_document: int
    max_sentences: int
    min_sentence_score: float
    verifier_min_citations: int
    verifier_min_evidence_score: float
    verifier_max_citation_rank: int


@dataclass(frozen=True)
class _QuestionRuntimeInputs:
    sample: PrimeQAHybridSplitSample
    question: PrimeQAQuestion
    retrieval_results: list[RetrievalResult]
    best_gold_span_token_f1: float | None


@dataclass(frozen=True)
class PrimeQAHybridEvidenceAnswerabilityComparisonVisualization:
    """One generated Stage105 evidence/answerability comparison chart."""

    name: str
    path: str


def run_primeqa_hybrid_evidence_answerability_comparison(
    *,
    stage104_protocol_path: Path,
    stage102_report_path: Path,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    user_confirmed_comparison: bool,
    confirmation_note: str,
    max_gold_window_sentences: int = 3,
    gold_span_gap_margin: float = 0.05,
    low_answer_f1_threshold: float = 0.2,
    sample_limit_per_bucket_transition: int = 5,
) -> dict[str, Any]:
    """Run the Stage105 frozen train/dev evidence-answerability comparison."""

    _validate_options(
        max_gold_window_sentences=max_gold_window_sentences,
        gold_span_gap_margin=gold_span_gap_margin,
        low_answer_f1_threshold=low_answer_f1_threshold,
        sample_limit_per_bucket_transition=sample_limit_per_bucket_transition,
    )
    started_at = time.perf_counter()
    stage104_protocol = _load_json_object(stage104_protocol_path)
    stage102_report = _load_json_object(stage102_report_path)
    stage104_summary = _stage104_public_summary(stage104_protocol)
    stage102_summary = _stage102_public_summary(stage102_report)
    frozen_protocol = stage104_protocol.get("frozen_protocol") or {}
    baseline_config = _baseline_config(frozen_protocol.get("baseline_reference") or {})
    candidate_configs = [
        _config_from_mapping(config)
        for config in frozen_protocol.get("candidate_config_grid") or []
    ]
    loaded_protocols_at = time.perf_counter()

    split_samples = {
        "train": load_primeqa_hybrid_split_samples(train_split_path),
        "dev": load_primeqa_hybrid_split_samples(dev_split_path),
    }
    loaded_splits_at = time.perf_counter()
    documents_by_id = load_primeqa_documents(documents_path)
    documents = list(documents_by_id.values())
    retriever = BM25Retriever(
        k1=float(baseline_config_payload(baseline_config)["bm25_k1"]),
        b=float(baseline_config_payload(baseline_config)["bm25_b"]),
    )
    retriever.fit(documents)
    indexed_at = time.perf_counter()

    runtime_inputs_by_split = _runtime_inputs_by_split(
        split_samples=split_samples,
        retriever=retriever,
        documents_by_id=documents_by_id,
        retrieval_top_k=int(baseline_config_payload(baseline_config)["retrieval_top_k"]),
        max_gold_window_sentences=max_gold_window_sentences,
    )
    cached_inputs_at = time.perf_counter()
    baseline_result = _evaluate_config(
        config=baseline_config,
        runtime_inputs_by_split=runtime_inputs_by_split,
        documents_by_id=documents_by_id,
        max_gold_window_sentences=max_gold_window_sentences,
        gold_span_gap_margin=gold_span_gap_margin,
        low_answer_f1_threshold=low_answer_f1_threshold,
        baseline_traces_by_split=None,
    )
    baseline_evaluated_at = time.perf_counter()
    config_results = [
        _evaluate_config(
            config=config,
            runtime_inputs_by_split=runtime_inputs_by_split,
            documents_by_id=documents_by_id,
            max_gold_window_sentences=max_gold_window_sentences,
            gold_span_gap_margin=gold_span_gap_margin,
            low_answer_f1_threshold=low_answer_f1_threshold,
            baseline_traces_by_split=baseline_result["traces_by_split"],
        )
        for config in candidate_configs
    ]
    evaluated_configs_at = time.perf_counter()
    public_config_results = [
        _public_config_result(
            result=result,
            baseline_result=baseline_result,
            frozen_protocol=frozen_protocol,
        )
        for result in config_results
    ]
    public_baseline_result = _public_baseline_result(baseline_result)
    train_selection = _train_selection(
        config_results=public_config_results,
        baseline_result=public_baseline_result,
        frozen_protocol=frozen_protocol,
    )
    selected_result = _selected_config_result(
        config_results=public_config_results,
        selected_config_id=train_selection.get("selected_config_id"),
    )
    dev_validation = _dev_validation(selected_result)
    changed_case_samples = _changed_case_samples(
        config_results=config_results,
        baseline_result=baseline_result,
        selected_config_id=train_selection.get("selected_config_id"),
        sample_limit_per_bucket_transition=sample_limit_per_bucket_transition,
    )
    guard_checks = _guard_checks(
        stage104_summary=stage104_summary,
        stage102_summary=stage102_summary,
        split_samples=split_samples,
        candidate_configs=candidate_configs,
        baseline_result=public_baseline_result,
        config_results=public_config_results,
        train_selection=train_selection,
        changed_case_samples=changed_case_samples,
        user_confirmed_comparison=user_confirmed_comparison,
    )
    finished_at = time.perf_counter()
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train/dev-only evidence-answerability candidate comparison under "
            "the Stage104 frozen grid. This stage loads the frozen Stage68 "
            "train/dev split rows and local PrimeQA corpus documents, runs the "
            "Stage102 verified baseline and all Stage104 candidate configs, "
            "selects configs on train only, validates once on dev, does not "
            "load the test split, does not run final metrics, does not write "
            "raw question, answer, document, token, or document-identifier "
            "fields, does not add fallback strategies, and does not change "
            "runtime defaults."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_comparison),
            "confirmation_note": confirmation_note,
        },
        "split_contract": {
            "split_name": _SPLIT_NAME,
            "protocol_version": _PROTOCOL_VERSION,
            "development_splits": list(_ALLOWED_SPLITS),
            "forbidden_final_splits": list(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "stage104_protocol": _fingerprint(stage104_protocol_path),
            "stage102_report": _fingerprint(stage102_report_path),
            "train_split": _fingerprint(train_split_path),
            "dev_split": _fingerprint(dev_split_path),
            "corpus_documents": _fingerprint(documents_path),
        },
        "stage104_summary": stage104_summary,
        "stage102_summary": stage102_summary,
        "analysis_config": {
            "max_gold_window_sentences": max_gold_window_sentences,
            "gold_span_gap_margin": gold_span_gap_margin,
            "low_answer_f1_threshold": low_answer_f1_threshold,
            "sample_limit_per_bucket_transition": sample_limit_per_bucket_transition,
            "config_count": len(candidate_configs),
            "baseline_config_id": baseline_config.config_id,
        },
        "data_summary": {
            "documents": len(documents),
            "splits": summarize_primeqa_hybrid_split_samples(split_samples),
        },
        "baseline_result": public_baseline_result,
        "config_results": public_config_results,
        "train_selection": train_selection,
        "dev_validation": dev_validation,
        "public_safe_changed_case_samples": changed_case_samples,
        "guard_checks": guard_checks,
        "decision": _decision(
            guard_checks=guard_checks,
            train_selection=train_selection,
            dev_validation=dev_validation,
        ),
        "timing_seconds": {
            "load_protocols": round(loaded_protocols_at - started_at, 3),
            "load_splits": round(loaded_splits_at - loaded_protocols_at, 3),
            "index_documents": round(indexed_at - loaded_splits_at, 3),
            "cache_runtime_inputs": round(cached_inputs_at - indexed_at, 3),
            "evaluate_baseline": round(baseline_evaluated_at - cached_inputs_at, 3),
            "evaluate_candidate_configs": round(
                evaluated_configs_at - baseline_evaluated_at,
                3,
            ),
            "aggregate_and_guard": round(finished_at - evaluated_configs_at, 3),
            "total": round(finished_at - started_at, 3),
        },
    }
    return report


def write_primeqa_hybrid_evidence_answerability_comparison_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridEvidenceAnswerabilityComparisonVisualization]:
    """Write SVG charts for the Stage105 comparison report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage105_train_weighted_target_scores.svg": render_horizontal_bar_chart_svg(
            title="Stage105 train weighted target scores",
            bars=_weighted_score_bars(report, "train"),
            x_label="weighted target score",
            width=1380,
            margin_left=700,
        ),
        "stage105_dev_weighted_target_scores.svg": render_horizontal_bar_chart_svg(
            title="Stage105 dev weighted target scores",
            bars=_weighted_score_bars(report, "dev"),
            x_label="weighted target score",
            width=1380,
            margin_left=700,
        ),
        "stage105_train_target_score_deltas.svg": render_horizontal_bar_chart_svg(
            title="Stage105 train weighted target deltas",
            bars=_weighted_delta_bars(report, "train"),
            x_label="delta vs baseline",
            width=1380,
            margin_left=700,
        ),
        "stage105_dev_target_score_deltas.svg": render_horizontal_bar_chart_svg(
            title="Stage105 dev weighted target deltas",
            bars=_weighted_delta_bars(report, "dev"),
            x_label="delta vs baseline",
            width=1380,
            margin_left=700,
        ),
        "stage105_train_selectability_guards.svg": render_horizontal_bar_chart_svg(
            title="Stage105 train selectability guards",
            bars=_train_selectability_bars(report),
            x_label="1 means selectable",
            width=1320,
            margin_left=660,
        ),
        "stage105_changed_answer_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage105 changed answer counts",
            bars=_changed_answer_count_bars(report),
            x_label="changed verified answers",
            width=1380,
            margin_left=700,
        ),
        "stage105_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage105 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1560,
            margin_left=820,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridEvidenceAnswerabilityComparisonVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def baseline_config_payload(config: _RunConfig) -> dict[str, Any]:
    return {
        "retrieval_top_k": 10,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "max_candidates_per_document": config.max_candidates_per_document,
        "max_sentences": config.max_sentences,
        "min_sentence_score": config.min_sentence_score,
        "verifier_min_citations": config.verifier_min_citations,
        "verifier_min_evidence_score": config.verifier_min_evidence_score,
        "verifier_max_citation_rank": config.verifier_max_citation_rank,
    }


def _baseline_config(baseline: Mapping[str, Any]) -> _RunConfig:
    return _RunConfig(
        config_id=str(
            baseline.get("baseline_id") or "stage102_verified_bm25_top10_answer_pipeline"
        ),
        candidate_id="baseline",
        selector_name=str(baseline.get("evidence_selector_name") or "bm25_sentence"),
        composition_policy_name=str(
            baseline.get("composition_policy_name") or "top_k"
        ),
        max_candidates_per_document=int(
            baseline.get("max_candidates_per_document") or 3
        ),
        max_sentences=int(baseline.get("max_sentences") or 3),
        min_sentence_score=float(baseline.get("min_sentence_score") or 2.0),
        verifier_min_citations=int(baseline.get("verifier_min_citations") or 1),
        verifier_min_evidence_score=float(
            baseline.get("verifier_min_evidence_score") or 7.0
        ),
        verifier_max_citation_rank=int(
            baseline.get("verifier_max_citation_rank") or 3
        ),
    )


def _config_from_mapping(config: Mapping[str, Any]) -> _RunConfig:
    return _RunConfig(
        config_id=str(config["config_id"]),
        candidate_id=str(config["candidate_id"]),
        selector_name=str(config["selector_name"]),
        composition_policy_name=str(config["composition_policy_name"]),
        max_candidates_per_document=int(config["max_candidates_per_document"]),
        max_sentences=int(config["max_sentences"]),
        min_sentence_score=float(config["min_sentence_score"]),
        verifier_min_citations=int(config["verifier_min_citations"]),
        verifier_min_evidence_score=float(config["verifier_min_evidence_score"]),
        verifier_max_citation_rank=int(config["verifier_max_citation_rank"]),
    )


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
            retrieval_results = retriever.search(
                question.full_question,
                top_k=retrieval_top_k,
            )
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


def _evaluate_config(
    *,
    config: _RunConfig,
    runtime_inputs_by_split: Mapping[str, Sequence[_QuestionRuntimeInputs]],
    documents_by_id: Mapping[str, PrimeQADocument],
    max_gold_window_sentences: int,
    gold_span_gap_margin: float,
    low_answer_f1_threshold: float,
    baseline_traces_by_split: Mapping[str, Sequence[_SplitQuestionTrace]] | None,
) -> dict[str, Any]:
    evidence_selector = create_sentence_evidence_selector(
        selector_name=config.selector_name,
        max_candidates_per_document=config.max_candidates_per_document,
    )
    composition_policy = create_answer_composition_policy(config.composition_policy_name)
    answer_generator = ExtractiveAnswerGenerator(
        max_sentences=config.max_sentences,
        min_sentence_score=config.min_sentence_score,
        evidence_selector=evidence_selector,
        composition_policy=composition_policy,
    )
    answer_verifier = AnswerVerifier(
        min_citations=config.verifier_min_citations,
        min_evidence_score=config.verifier_min_evidence_score,
        max_citation_rank=config.verifier_max_citation_rank,
    )
    traces_by_split = {
        split: [
            _trace_question(
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
    aggregate_outputs = _aggregate_outputs(traces_by_split)
    metrics_by_split = _metrics_by_split(traces_by_split)
    return {
        "config": config,
        "traces_by_split": traces_by_split,
        "aggregate_outputs": aggregate_outputs,
        "metrics_by_split": metrics_by_split,
        "changed_answer_counts_by_split": _changed_answer_counts_by_split(
            baseline_traces_by_split=baseline_traces_by_split,
            traces_by_split=traces_by_split,
        ),
    }


def _trace_question(
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
    bucket_id, scoring = _classify_pipeline_bucket_with_cached_gold_span(
        question=question,
        retrieval_results=retrieval_results,
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
        retrieval_results=retrieval_results,
        original_answer=original_answer,
        verified_answer=verification.verified_answer,
        verification_reasons=verification.reasons,
        bucket_id=bucket_id,
        scoring=scoring,
        question_route=route_trace.question_route,
        routed_selector_name=route_trace.selected_selector_name,
        evidence_selector_name=answer_generator.evidence_selector_name,
        composition_policy_name=answer_generator.composition_policy_name,
    )
    return _SplitQuestionTrace(
        sample=runtime_inputs.sample,
        question=question,
        retrieval_results=retrieval_results,
        original_answer=original_answer,
        verified_answer=verification.verified_answer,
        verification_reasons=tuple(verification.reasons),
        question_route=route_trace.question_route,
        routed_selector_name=route_trace.selected_selector_name,
        bucket_id=bucket_id,
        public_case=public_case,
        original_answer_token_f1=scoring["answer_token_f1"],
        best_gold_span_token_f1=scoring["best_gold_span_token_f1"],
        answer_gold_span_gap=scoring["answer_gold_span_gap"],
    )


def _classify_pipeline_bucket_with_cached_gold_span(
    *,
    question: PrimeQAQuestion,
    retrieval_results: Sequence[RetrievalResult],
    original_answer: GeneratedAnswer,
    verified_answer: GeneratedAnswer,
    documents_by_id: Mapping[str, PrimeQADocument],
    max_gold_window_sentences: int,
    best_gold_span_token_f1: float | None,
    gold_span_gap_margin: float,
    low_answer_f1_threshold: float,
) -> tuple[str, dict[str, float | None]]:
    answer_token_f1 = (
        round(token_f1(original_answer.answer, question.answer), 4)
        if question.answerable
        else None
    )
    if best_gold_span_token_f1 is None and question.answerable:
        best_gold_span_token_f1 = _best_gold_span_token_f1(
            question=question,
            documents_by_id=documents_by_id,
            max_gold_window_sentences=max_gold_window_sentences,
        )
    answer_gold_span_gap = (
        round(max(0.0, best_gold_span_token_f1 - answer_token_f1), 4)
        if answer_token_f1 is not None and best_gold_span_token_f1 is not None
        else None
    )
    scoring = {
        "answer_token_f1": answer_token_f1,
        "best_gold_span_token_f1": best_gold_span_token_f1,
        "answer_gold_span_gap": answer_gold_span_gap,
    }
    if not question.answerable:
        if not verified_answer.refused:
            return "answerability_false_answer", scoring
        return "answer_supported_and_cited", scoring
    if _gold_context_absent(question, retrieval_results):
        return "retrieval_context_miss", scoring
    if _gold_not_cited(question, original_answer):
        return "evidence_selection_miss", scoring
    if verified_answer.refused:
        return "verification_over_refusal", scoring
    if answer_gold_span_gap is not None and answer_gold_span_gap >= gold_span_gap_margin:
        return "gold_span_beats_selected_answer", scoring
    if answer_token_f1 is not None and answer_token_f1 < low_answer_f1_threshold:
        return "low_overlap_gold_cited_answer", scoring
    return "answer_supported_and_cited", scoring


def _gold_context_absent(
    question: PrimeQAQuestion,
    retrieval_results: Sequence[RetrievalResult],
) -> bool:
    if not question.answerable or not question.answer_doc_id:
        return False
    return question.answer_doc_id not in {result.document.id for result in retrieval_results}


def _gold_not_cited(question: PrimeQAQuestion, answer: GeneratedAnswer) -> bool:
    if not question.answerable or not question.answer_doc_id:
        return False
    if answer.refused:
        return True
    return question.answer_doc_id not in {
        citation.document_id for citation in answer.citations
    }


def _public_baseline_result(baseline_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "config_id": baseline_result["config"].config_id,
        "candidate_id": baseline_result["config"].candidate_id,
        "aggregate_outputs": baseline_result["aggregate_outputs"],
        "metrics_by_split": baseline_result["metrics_by_split"],
        "weighted_target_scores_by_split": _weighted_scores_by_split(
            baseline_result["aggregate_outputs"]
        ),
    }


def _public_config_result(
    *,
    result: Mapping[str, Any],
    baseline_result: Mapping[str, Any],
    frozen_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    config = result["config"]
    aggregate_outputs = result["aggregate_outputs"]
    metrics_by_split = result["metrics_by_split"]
    weighted_scores = _weighted_scores_by_split(aggregate_outputs)
    baseline_weighted_scores = _weighted_scores_by_split(
        baseline_result["aggregate_outputs"]
    )
    train_guards = _train_selectability_guard_results(
        result=result,
        baseline_result=baseline_result,
        frozen_protocol=frozen_protocol,
    )
    return {
        "config_id": config.config_id,
        "candidate_id": config.candidate_id,
        "selector_name": config.selector_name,
        "composition_policy_name": config.composition_policy_name,
        "max_candidates_per_document": config.max_candidates_per_document,
        "verifier_min_evidence_score": config.verifier_min_evidence_score,
        "verifier_max_citation_rank": config.verifier_max_citation_rank,
        "aggregate_outputs": aggregate_outputs,
        "metrics_by_split": metrics_by_split,
        "weighted_target_scores_by_split": weighted_scores,
        "weighted_target_score_deltas_by_split": {
            split: round(weighted_scores[split] - baseline_weighted_scores[split], 4)
            for split in _ALLOWED_SPLITS
        },
        "target_bucket_deltas_by_split": _target_bucket_deltas_by_split(
            result=result,
            baseline_result=baseline_result,
        ),
        "metric_deltas_by_split": _metric_deltas_by_split(
            metrics_by_split,
            baseline_result["metrics_by_split"],
        ),
        "changed_answer_counts_by_split": result["changed_answer_counts_by_split"],
        "train_selectability": train_guards,
    }


def _weighted_scores_by_split(aggregate_outputs: Mapping[str, Any]) -> dict[str, float]:
    counts_by_split = aggregate_outputs["bucket_counts_by_split"]
    return {
        split: round(
            sum(
                float(counts_by_split[split].get(bucket, 0)) * weight
                for bucket, weight in _TARGET_BUCKET_WEIGHTS.items()
            ),
            4,
        )
        for split in _ALLOWED_SPLITS
    }


def _target_bucket_deltas_by_split(
    *,
    result: Mapping[str, Any],
    baseline_result: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    counts = result["aggregate_outputs"]["bucket_counts_by_split"]
    baseline_counts = baseline_result["aggregate_outputs"]["bucket_counts_by_split"]
    return {
        split: {
            bucket: int(counts[split].get(bucket, 0))
            - int(baseline_counts[split].get(bucket, 0))
            for bucket in _TARGET_BUCKETS
        }
        for split in _ALLOWED_SPLITS
    }


def _metric_deltas_by_split(
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
        for split in _ALLOWED_SPLITS
    }


def _train_selectability_guard_results(
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
    train_metrics = result["metrics_by_split"]["train"]["verified"]
    baseline_train_metrics = baseline_result["metrics_by_split"]["train"]["verified"]
    answerable_refusal_delta = round(
        float(train_metrics["answerable_refusal_rate"])
        - float(baseline_train_metrics["answerable_refusal_rate"]),
        4,
    )
    average_token_f1_drop = round(
        float(baseline_train_metrics["average_token_f1"])
        - float(train_metrics["average_token_f1"]),
        4,
    )
    gold_doc_citation_rate_drop = round(
        float(baseline_train_metrics["gold_doc_citation_rate"])
        - float(train_metrics["gold_doc_citation_rate"]),
        4,
    )
    checks = {
        "answerable_refusal_rate_delta_within_guard": answerable_refusal_delta
        <= float(guards.get("max_train_answerable_refusal_rate_delta") or 0.0),
        "average_token_f1_drop_within_guard": average_token_f1_drop
        <= float(guards.get("max_train_average_token_f1_drop") or 0.0),
        "gold_doc_citation_rate_drop_within_guard": gold_doc_citation_rate_drop
        <= float(guards.get("max_train_gold_doc_citation_rate_drop") or 0.0),
    }
    return {
        "selectable": all(checks.values()),
        "observed": {
            "answerable_refusal_rate_delta": answerable_refusal_delta,
            "average_token_f1_drop": average_token_f1_drop,
            "gold_doc_citation_rate_drop": gold_doc_citation_rate_drop,
        },
        "thresholds": guards,
        "checks": checks,
    }


def _train_selection(
    *,
    config_results: Sequence[Mapping[str, Any]],
    baseline_result: Mapping[str, Any],
    frozen_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    selectable = [
        result
        for result in config_results
        if result["train_selectability"]["selectable"]
    ]
    ranking = sorted(
        config_results,
        key=lambda result: _selection_key(result),
    )
    selectable_ranking = [
        {
            "rank": rank,
            "config_id": result["config_id"],
            "candidate_id": result["candidate_id"],
            "train_weighted_target_score": result[
                "weighted_target_scores_by_split"
            ]["train"],
            "train_weighted_target_delta": result[
                "weighted_target_score_deltas_by_split"
            ]["train"],
            "train_selectable": result["train_selectability"]["selectable"],
            "train_changed_answer_count": result["changed_answer_counts_by_split"][
                "train"
            ],
        }
        for rank, result in enumerate(ranking, start=1)
    ]
    selected = sorted(selectable, key=_selection_key)[0] if selectable else None
    return {
        "selection_split": "train",
        "baseline_train_weighted_target_score": baseline_result[
            "weighted_target_scores_by_split"
        ]["train"],
        "selected_config_id": selected["config_id"] if selected else None,
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "selected_train_weighted_target_score": (
            selected["weighted_target_scores_by_split"]["train"] if selected else None
        ),
        "selected_train_weighted_target_delta": (
            selected["weighted_target_score_deltas_by_split"]["train"]
            if selected
            else None
        ),
        "selectable_config_count": len(selectable),
        "config_count": len(config_results),
        "selection_ranking": selectable_ranking,
        "train_selection_rule": frozen_protocol.get("train_selection_rule") or {},
    }


def _selection_key(result: Mapping[str, Any]) -> tuple[Any, ...]:
    train_counts = result["aggregate_outputs"]["bucket_counts_by_split"]["train"]
    train_metrics = result["metrics_by_split"]["train"]["verified"]
    return (
        0 if result["train_selectability"]["selectable"] else 1,
        float(result["weighted_target_scores_by_split"]["train"]),
        int(train_counts.get("answerability_false_answer", 0)),
        int(train_counts.get("gold_span_beats_selected_answer", 0)),
        int(train_counts.get("evidence_selection_miss", 0)),
        -float(train_metrics["average_token_f1"]),
        -float(train_metrics["gold_doc_citation_rate"]),
        int(result["changed_answer_counts_by_split"]["train"]),
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
        if result["config_id"] == selected_config_id:
            return result
    return None


def _dev_validation(selected_result: Mapping[str, Any] | None) -> dict[str, Any]:
    if selected_result is None:
        return {
            "validation_split": "dev",
            "selected_config_id": None,
            "status": "no_train_selectable_config",
        }
    dev_delta = selected_result["weighted_target_score_deltas_by_split"]["dev"]
    metric_deltas = selected_result["metric_deltas_by_split"]["dev"]
    dev_guard_checks = {
        "weighted_target_score_improved": dev_delta < 0,
        "answerable_refusal_rate_delta_reported": (
            "answerable_refusal_rate" in metric_deltas
        ),
        "average_token_f1_delta_reported": "average_token_f1" in metric_deltas,
        "gold_doc_citation_rate_delta_reported": (
            "gold_doc_citation_rate" in metric_deltas
        ),
    }
    return {
        "validation_split": "dev",
        "selected_config_id": selected_result["config_id"],
        "selected_candidate_id": selected_result["candidate_id"],
        "dev_weighted_target_score": selected_result[
            "weighted_target_scores_by_split"
        ]["dev"],
        "dev_weighted_target_delta": dev_delta,
        "dev_target_bucket_deltas": selected_result["target_bucket_deltas_by_split"][
            "dev"
        ],
        "dev_metric_deltas": metric_deltas,
        "dev_changed_answer_count": selected_result["changed_answer_counts_by_split"][
            "dev"
        ],
        "dev_validation_checks": dev_guard_checks,
        "dev_validation_passed": all(dev_guard_checks.values()),
    }


def _changed_answer_counts_by_split(
    *,
    baseline_traces_by_split: Mapping[str, Sequence[_SplitQuestionTrace]] | None,
    traces_by_split: Mapping[str, Sequence[_SplitQuestionTrace]],
) -> dict[str, int]:
    if baseline_traces_by_split is None:
        return {split: 0 for split in traces_by_split}
    return {
        split: sum(
            _answer_signature(base_trace.verified_answer)
            != _answer_signature(trace.verified_answer)
            for base_trace, trace in zip(
                baseline_traces_by_split[split],
                traces_by_split[split],
                strict=True,
            )
        )
        for split in traces_by_split
    }


def _changed_case_samples(
    *,
    config_results: Sequence[Mapping[str, Any]],
    baseline_result: Mapping[str, Any],
    selected_config_id: str | None,
    sample_limit_per_bucket_transition: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    if selected_config_id is None:
        return {split: {} for split in _ALLOWED_SPLITS}
    result = None
    for candidate_result in config_results:
        if candidate_result["config"].config_id == selected_config_id:
            result = candidate_result
            break
    if result is None:
        return {split: {} for split in _ALLOWED_SPLITS}
    samples: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split in _ALLOWED_SPLITS:
        by_transition: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for baseline_trace, candidate_trace in zip(
            baseline_result["traces_by_split"][split],
            result["traces_by_split"][split],
            strict=True,
        ):
            if _answer_signature(baseline_trace.verified_answer) == _answer_signature(
                candidate_trace.verified_answer
            ):
                continue
            transition = f"{baseline_trace.bucket_id}->{candidate_trace.bucket_id}"
            by_transition[transition].append(
                _changed_case(
                    config=result["config"],
                    baseline_trace=baseline_trace,
                    candidate_trace=candidate_trace,
                )
            )
        samples[split] = {
            transition: cases[:sample_limit_per_bucket_transition]
            for transition, cases in sorted(by_transition.items())
        }
    return samples


def _changed_case(
    *,
    config: _RunConfig,
    baseline_trace: _SplitQuestionTrace,
    candidate_trace: _SplitQuestionTrace,
) -> dict[str, Any]:
    case = {
        "sample_id": candidate_trace.sample.sample_id,
        "split": candidate_trace.public_case["split"],
        "config_id": config.config_id,
        "candidate_id": config.candidate_id,
        "baseline_bucket_id": baseline_trace.bucket_id,
        "candidate_bucket_id": candidate_trace.bucket_id,
        "baseline_answer_token_f1_bucket": baseline_trace.public_case[
            "answer_token_f1_bucket"
        ],
        "candidate_answer_token_f1_bucket": candidate_trace.public_case[
            "answer_token_f1_bucket"
        ],
        "baseline_citation_status": baseline_trace.public_case["citation_status"],
        "candidate_citation_status": candidate_trace.public_case["citation_status"],
        "answerability_action": _answerability_action(
            baseline_trace=baseline_trace,
            candidate_trace=candidate_trace,
        ),
        "evidence_selection_action": _evidence_selection_action(
            baseline_trace=baseline_trace,
            candidate_trace=candidate_trace,
        ),
        "changed_case_confidence_band": _changed_case_confidence_band(
            baseline_trace,
            candidate_trace,
        ),
    }
    if tuple(case) != _CHANGED_CASE_FIELDS:
        raise ValueError("changed case fields do not match Stage104 contract")
    return case


def _answerability_action(
    *,
    baseline_trace: _SplitQuestionTrace,
    candidate_trace: _SplitQuestionTrace,
) -> str:
    if baseline_trace.verified_answer.refused == candidate_trace.verified_answer.refused:
        return "unchanged_refusal_status"
    if candidate_trace.verified_answer.refused:
        return "candidate_refused"
    return "candidate_answered"


def _evidence_selection_action(
    *,
    baseline_trace: _SplitQuestionTrace,
    candidate_trace: _SplitQuestionTrace,
) -> str:
    baseline_status = baseline_trace.public_case["citation_status"]
    candidate_status = candidate_trace.public_case["citation_status"]
    if baseline_status == candidate_status:
        return "unchanged_citation_status"
    return f"{baseline_status}_to_{candidate_status}"


def _changed_case_confidence_band(
    baseline_trace: _SplitQuestionTrace,
    candidate_trace: _SplitQuestionTrace,
) -> str:
    if baseline_trace.bucket_id != candidate_trace.bucket_id:
        return "high"
    if baseline_trace.public_case["citation_status"] != candidate_trace.public_case[
        "citation_status"
    ]:
        return "medium"
    return "low"


def _answer_signature(answer: GeneratedAnswer) -> tuple[Any, ...]:
    return (
        answer.refused,
        answer.answer,
        tuple(
            (
                citation.document_id,
                citation.retrieval_rank,
                round(citation.evidence_score, 4),
            )
            for citation in answer.citations
        ),
    )


def _stage104_public_summary(stage104_report: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage104_report.get("decision") or {}
    frozen_protocol = stage104_report.get("frozen_protocol") or {}
    return {
        "stage": stage104_report.get("stage"),
        "protocol_id": stage104_report.get("protocol_id"),
        "decision_status": decision.get("status"),
        "recommended_direction": decision.get("recommended_direction"),
        "can_run_train_dev_candidate_comparison_after_user_confirmation": decision.get(
            "can_run_train_dev_candidate_comparison_after_user_confirmation"
        ),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get(
            "can_run_final_test_metrics_now"
        ),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "config_count": len(frozen_protocol.get("candidate_config_grid") or []),
        "protocol_status": frozen_protocol.get("protocol_status"),
    }


def _stage102_public_summary(stage102_report: Mapping[str, Any]) -> dict[str, Any]:
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
    stage104_summary: Mapping[str, Any],
    stage102_summary: Mapping[str, Any],
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    candidate_configs: Sequence[_RunConfig],
    baseline_result: Mapping[str, Any],
    config_results: Sequence[Mapping[str, Any]],
    train_selection: Mapping[str, Any],
    changed_case_samples: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    user_confirmed_comparison: bool,
) -> list[dict[str, Any]]:
    all_changed_cases = [
        case
        for split_samples_by_transition in changed_case_samples.values()
        for cases in split_samples_by_transition.values()
        for case in cases
    ]
    return [
        _check(
            name="stage104_source_is_expected_stage",
            passed=stage104_summary.get("stage") == _SOURCE_STAGE104,
            observed=stage104_summary.get("stage"),
            expected=_SOURCE_STAGE104,
        ),
        _check(
            name="user_confirmed_stage105_comparison",
            passed=user_confirmed_comparison,
            observed=user_confirmed_comparison,
            expected=True,
        ),
        _check(
            name="stage104_protocol_is_frozen",
            passed=stage104_summary.get("decision_status")
            == _EXPECTED_STAGE104_STATUS,
            observed=stage104_summary.get("decision_status"),
            expected=_EXPECTED_STAGE104_STATUS,
        ),
        _check(
            name="stage104_protocol_id_matches",
            passed=stage104_summary.get("protocol_id") == _STAGE104_PROTOCOL_ID,
            observed=stage104_summary.get("protocol_id"),
            expected=_STAGE104_PROTOCOL_ID,
        ),
        _check(
            name="stage104_allows_comparison_after_confirmation",
            passed=stage104_summary.get(
                "can_run_train_dev_candidate_comparison_after_user_confirmation"
            )
            is True,
            observed=stage104_summary.get(
                "can_run_train_dev_candidate_comparison_after_user_confirmation"
            ),
            expected=True,
        ),
        _check(
            name="stage104_final_test_gate_locked",
            passed=stage104_summary.get("can_run_final_test_metrics_now") is False
            and stage104_summary.get("can_open_final_test_gate_now") is False,
            observed={
                "can_open_final_test_gate_now": stage104_summary.get(
                    "can_open_final_test_gate_now"
                ),
                "can_run_final_test_metrics_now": stage104_summary.get(
                    "can_run_final_test_metrics_now"
                ),
            },
            expected=False,
        ),
        _check(
            name="stage104_forbids_test_tuning",
            passed=stage104_summary.get("can_use_test_for_tuning") is False,
            observed=stage104_summary.get("can_use_test_for_tuning"),
            expected=False,
        ),
        _check(
            name="stage104_runtime_defaults_unchanged",
            passed=stage104_summary.get("default_runtime_policy") == "unchanged",
            observed=stage104_summary.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage104_fallback_disabled",
            passed=stage104_summary.get("fallback_strategies_enabled") is False,
            observed=stage104_summary.get("fallback_strategies_enabled"),
            expected=False,
        ),
        _check(
            name="stage102_source_is_expected_stage",
            passed=stage102_summary.get("stage") == _SOURCE_STAGE102,
            observed=stage102_summary.get("stage"),
            expected=_SOURCE_STAGE102,
        ),
        _check(
            name="stage102_analysis_completed",
            passed=stage102_summary.get("decision_status")
            == "primeqa_hybrid_answer_pipeline_error_decomposition_completed",
            observed=stage102_summary.get("decision_status"),
            expected="primeqa_hybrid_answer_pipeline_error_decomposition_completed",
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
                split: _counter_dict(sample.assigned_split for sample in samples)
                for split, samples in split_samples.items()
            },
            expected={split: split for split in _ALLOWED_SPLITS},
        ),
        _check(
            name="test_split_not_loaded",
            passed="test" not in split_samples,
            observed=list(split_samples),
            expected="no test split",
        ),
        _check(
            name="all_stage104_configs_were_run",
            passed=len(config_results) == len(candidate_configs)
            == int(stage104_summary.get("config_count") or 0),
            observed={
                "candidate_configs": len(candidate_configs),
                "config_results": len(config_results),
                "stage104_config_count": stage104_summary.get("config_count"),
            },
            expected="all Stage104 configs run exactly once",
        ),
        _check(
            name="baseline_bucket_counts_match_stage102",
            passed=baseline_result["aggregate_outputs"]["bucket_counts_by_split"]
            == stage102_summary.get("bucket_counts_by_split"),
            observed=baseline_result["aggregate_outputs"]["bucket_counts_by_split"],
            expected=stage102_summary.get("bucket_counts_by_split"),
        ),
        _check(
            name="baseline_verified_metrics_match_stage102",
            passed=_baseline_verified_metrics_match_stage102(
                baseline_result,
                stage102_summary,
            ),
            observed={
                split: baseline_result["metrics_by_split"][split]["verified"]
                for split in _ALLOWED_SPLITS
            },
            expected={
                split: (stage102_summary.get("metrics_by_split") or {})[split][
                    "verified"
                ]
                for split in _ALLOWED_SPLITS
            },
        ),
        _check(
            name="train_selection_uses_train_split",
            passed=train_selection.get("selection_split") == "train",
            observed=train_selection.get("selection_split"),
            expected="train",
        ),
        _check(
            name="dev_validation_not_used_for_selection",
            passed=_train_selection_is_dev_blind(train_selection),
            observed={
                "selection_split": train_selection.get("selection_split"),
                "ranking_keys": [
                    list(entry)
                    for entry in train_selection.get("selection_ranking", [])[:3]
                ],
            },
            expected={
                "selection_split": "train",
                "ranking_keys": list(_TRAIN_SELECTION_RANKING_FIELDS),
            },
        ),
        _check(
            name="candidate_results_include_dev_for_all_configs",
            passed=all("dev" in result["metrics_by_split"] for result in config_results),
            observed=[result["config_id"] for result in config_results],
            expected="dev metrics reported for all configs",
        ),
        _check(
            name="changed_case_samples_use_stage104_fields",
            passed=all(tuple(case) == _CHANGED_CASE_FIELDS for case in all_changed_cases),
            observed=[list(case) for case in all_changed_cases[:3]],
            expected=list(_CHANGED_CASE_FIELDS),
        ),
        _check(
            name="changed_case_samples_exclude_forbidden_keys",
            passed=not _contains_forbidden_keys(all_changed_cases),
            observed=_find_forbidden_keys(all_changed_cases),
            expected="no raw text, document id, source DOC_IDS, or token fields",
        ),
        _check(
            name="stage105_final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage105_default_runtime_policy_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
        _check(
            name="stage105_fallback_strategies_not_added",
            passed=True,
            observed=False,
            expected=False,
        ),
    ]


def _baseline_verified_metrics_match_stage102(
    baseline_result: Mapping[str, Any],
    stage102_summary: Mapping[str, Any],
) -> bool:
    stage102_metrics = stage102_summary.get("metrics_by_split") or {}
    for split in _ALLOWED_SPLITS:
        if baseline_result["metrics_by_split"][split]["verified"] != stage102_metrics[
            split
        ]["verified"]:
            return False
    return True


_TRAIN_SELECTION_RANKING_FIELDS = (
    "rank",
    "config_id",
    "candidate_id",
    "train_weighted_target_score",
    "train_weighted_target_delta",
    "train_selectable",
    "train_changed_answer_count",
)


def _train_selection_is_dev_blind(train_selection: Mapping[str, Any]) -> bool:
    if train_selection.get("selection_split") != "train":
        return False
    for entry in train_selection.get("selection_ranking", []):
        if tuple(entry) != _TRAIN_SELECTION_RANKING_FIELDS:
            return False
        if any("dev" in key.lower() for key in entry):
            return False
    return True


def _decision(
    *,
    guard_checks: Sequence[Mapping[str, Any]],
    train_selection: Mapping[str, Any],
    dev_validation: Mapping[str, Any],
) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    base = {
        "analysis_id": _ANALYSIS_ID,
        "selected_config_id": train_selection.get("selected_config_id"),
        "selected_candidate_id": train_selection.get("selected_candidate_id"),
        "selectable_config_count": train_selection.get("selectable_config_count"),
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
            "status": "primeqa_hybrid_evidence_answerability_comparison_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
        }
    if train_selection.get("selected_config_id") is None:
        return {
            **base,
            "status": "primeqa_hybrid_evidence_answerability_comparison_no_selectable_config",
            "can_continue_train_dev_development": True,
            "recommended_next_direction": "evidence_answerability_stop_decision",
            "recommended_next_stage": (
                "Stage106: record a stop decision because no Stage104 config "
                "passed the train selectability guards; keep test locked and "
                "runtime defaults unchanged."
            ),
        }
    if dev_validation.get("dev_validation_passed") is True:
        return {
            **base,
            "status": "primeqa_hybrid_evidence_answerability_comparison_completed",
            "can_continue_train_dev_development": True,
            "recommended_next_direction": "evidence_answerability_changed_case_review",
            "recommended_next_stage": (
                "Stage106: review the selected config's public-safe changed "
                "cases and guard metrics before any runtime/default or final "
                "test gate discussion."
            ),
        }
    return {
        **base,
        "status": "primeqa_hybrid_evidence_answerability_comparison_completed_dev_guard_failed",
        "can_continue_train_dev_development": True,
        "recommended_next_direction": "evidence_answerability_stop_decision",
        "recommended_next_stage": (
            "Stage106: record a stop or redesign decision because the "
            "train-selected config did not pass dev validation; keep test "
            "locked and runtime defaults unchanged."
        ),
    }


def _weighted_score_bars(report: Mapping[str, Any], split: str) -> list[BarDatum]:
    rows = [
        ("baseline", report["baseline_result"]["weighted_target_scores_by_split"][split])
    ]
    rows.extend(
        (
            result["config_id"],
            result["weighted_target_scores_by_split"][split],
        )
        for result in report["config_results"]
    )
    return [
        BarDatum(label=label, value=float(value), value_label=f"{float(value):.2f}")
        for label, value in rows
    ]


def _weighted_delta_bars(report: Mapping[str, Any], split: str) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(result["config_id"]),
            value=float(result["weighted_target_score_deltas_by_split"][split]),
            value_label=f"{float(result['weighted_target_score_deltas_by_split'][split]):.2f}",
        )
        for result in report["config_results"]
    ]


def _train_selectability_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(result["config_id"]),
            value=1.0 if result["train_selectability"]["selectable"] else 0.0,
            value_label="selectable"
            if result["train_selectability"]["selectable"]
            else "blocked",
        )
        for result in report["config_results"]
    ]


def _changed_answer_count_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=f"{split}::{result['config_id']}",
            value=float(result["changed_answer_counts_by_split"][split]),
            value_label=str(result["changed_answer_counts_by_split"][split]),
        )
        for result in report["config_results"]
        for split in _ALLOWED_SPLITS
    ]


def _guard_check_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(check["name"]),
            value=1.0 if check["passed"] else 0.0,
            value_label="passed" if check["passed"] else "failed",
        )
        for check in report["guard_checks"]
    ]


def _validate_options(
    *,
    max_gold_window_sentences: int,
    gold_span_gap_margin: float,
    low_answer_f1_threshold: float,
    sample_limit_per_bucket_transition: int,
) -> None:
    if max_gold_window_sentences <= 0:
        raise ValueError("max_gold_window_sentences must be positive")
    if gold_span_gap_margin < 0:
        raise ValueError("gold_span_gap_margin must be non-negative")
    if low_answer_f1_threshold < 0:
        raise ValueError("low_answer_f1_threshold must be non-negative")
    if sample_limit_per_bucket_transition < 0:
        raise ValueError("sample_limit_per_bucket_transition must be non-negative")


def _counter_dict(values) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _contains_forbidden_keys(value: Any) -> bool:
    return bool(_find_forbidden_keys(value))


def _find_forbidden_keys(value: Any) -> list[str]:
    found = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                normalized_key = str(key).lower()
                if normalized_key in _FORBIDDEN_REPORT_KEYS:
                    found.add(normalized_key)
                visit(child)
        elif isinstance(item, list | tuple):
            for child in item:
                visit(child)

    visit(value)
    return sorted(found)


def _load_json_object(path: Path) -> dict[str, Any]:
    _ensure_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _fingerprint(path: Path) -> dict[str, Any]:
    _ensure_file(path)
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")


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
