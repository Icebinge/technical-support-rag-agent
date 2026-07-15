from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.answer_composition import create_answer_composition_policy
from ts_rag_agent.application.answer_verification import AnswerVerifier
from ts_rag_agent.application.evidence_selection import (
    create_sentence_evidence_selector,
    split_sentences,
    trace_selector_route,
)
from ts_rag_agent.application.rag_answering import ExtractiveAnswerGenerator, evaluate_answers
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.application.text_metrics import token_f1
from ts_rag_agent.domain.answer import GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import load_primeqa_documents

_STAGE = "Stage 102"
_CREATED_AT = "2026-07-15"
_SOURCE_STAGE101 = "Stage 101"
_PROTOCOL_ID = "answer_pipeline_error_decomposition_train_dev_v1"
_ANALYSIS_ID = "answer_pipeline_error_decomposition_train_dev_analysis_v1"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_ALLOWED_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = ("test",)
_EXPECTED_PUBLIC_CASE_FIELDS = (
    "sample_id",
    "split",
    "answerability_label",
    "pipeline_bucket_id",
    "pipeline_stage",
    "retrieval_rank_bucket",
    "retrieval_context_status",
    "citation_status",
    "evidence_selection_status",
    "answer_token_f1_bucket",
    "best_gold_span_f1_bucket",
    "answer_gold_span_gap_bucket",
    "verifier_decision",
    "refusal_reason_code",
    "question_route",
    "evidence_selector_name",
    "composition_policy_id",
    "bucket_confidence_band",
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
_BUCKET_TO_STAGE = {
    "answerability_false_answer": "answerability",
    "retrieval_context_miss": "retrieval",
    "evidence_selection_miss": "evidence_selection",
    "verification_over_refusal": "verification",
    "gold_span_beats_selected_answer": "answer_composition",
    "low_overlap_gold_cited_answer": "answer_composition",
    "answer_supported_and_cited": "non_error_reference",
}
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
    }
)


@dataclass(frozen=True)
class _SplitQuestionTrace:
    sample: PrimeQAHybridSplitSample
    question: PrimeQAQuestion
    retrieval_results: list[RetrievalResult]
    original_answer: GeneratedAnswer
    verified_answer: GeneratedAnswer
    verification_reasons: tuple[str, ...]
    question_route: str
    routed_selector_name: str
    bucket_id: str
    public_case: dict[str, Any]
    original_answer_token_f1: float | None
    best_gold_span_token_f1: float | None
    answer_gold_span_gap: float | None


@dataclass(frozen=True)
class PrimeQAHybridAnswerPipelineErrorDecompositionVisualization:
    """One generated Stage102 answer-pipeline error decomposition chart."""

    name: str
    path: str


def run_primeqa_hybrid_answer_pipeline_error_decomposition(
    *,
    stage101_protocol_path: Path,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    user_confirmed_analysis: bool,
    confirmation_note: str,
    retrieval_top_k: int = 10,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    evidence_selector_name: str = "bm25-sentence",
    max_candidates_per_document: int = 3,
    composition_policy_name: str = "top-k",
    max_sentences: int = 3,
    min_sentence_score: float = 2.0,
    min_evidence_score: float = 7.0,
    max_citation_rank: int = 3,
    min_citations: int = 1,
    max_gold_window_sentences: int = 3,
    gold_span_gap_margin: float = 0.05,
    low_answer_f1_threshold: float = 0.2,
    sample_limit_per_bucket: int = 5,
) -> dict[str, Any]:
    """Run Stage102 train/dev-only answer-pipeline error decomposition."""

    _validate_options(
        retrieval_top_k=retrieval_top_k,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        max_candidates_per_document=max_candidates_per_document,
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
        min_evidence_score=min_evidence_score,
        max_citation_rank=max_citation_rank,
        min_citations=min_citations,
        max_gold_window_sentences=max_gold_window_sentences,
        gold_span_gap_margin=gold_span_gap_margin,
        low_answer_f1_threshold=low_answer_f1_threshold,
        sample_limit_per_bucket=sample_limit_per_bucket,
    )
    started_at = time.perf_counter()
    stage101_protocol = _load_json_object(stage101_protocol_path)
    stage101_summary = _stage101_public_summary(stage101_protocol)
    split_samples = {
        "train": load_primeqa_hybrid_split_samples(train_split_path),
        "dev": load_primeqa_hybrid_split_samples(dev_split_path),
    }
    loaded_splits_at = time.perf_counter()
    documents_by_id = load_primeqa_documents(documents_path)
    documents = list(documents_by_id.values())
    retriever = BM25Retriever(k1=bm25_k1, b=bm25_b)
    retriever.fit(documents)
    indexed_at = time.perf_counter()

    evidence_selector = create_sentence_evidence_selector(
        selector_name=evidence_selector_name,
        max_candidates_per_document=max_candidates_per_document,
    )
    composition_policy = create_answer_composition_policy(composition_policy_name)
    answer_generator = ExtractiveAnswerGenerator(
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
        evidence_selector=evidence_selector,
        composition_policy=composition_policy,
    )
    answer_verifier = AnswerVerifier(
        min_citations=min_citations,
        min_evidence_score=min_evidence_score,
        max_citation_rank=max_citation_rank,
    )

    traces_by_split = {
        split: _analyze_split(
            split=split,
            samples=samples,
            retriever=retriever,
            documents_by_id=documents_by_id,
            answer_generator=answer_generator,
            answer_verifier=answer_verifier,
            retrieval_top_k=retrieval_top_k,
            max_gold_window_sentences=max_gold_window_sentences,
            gold_span_gap_margin=gold_span_gap_margin,
            low_answer_f1_threshold=low_answer_f1_threshold,
        )
        for split, samples in split_samples.items()
    }
    analyzed_at = time.perf_counter()
    aggregate_outputs = _aggregate_outputs(traces_by_split)
    public_safe_case_samples = _public_safe_case_samples(
        traces_by_split=traces_by_split,
        sample_limit_per_bucket=sample_limit_per_bucket,
    )
    metrics_by_split = _metrics_by_split(traces_by_split)
    decision_inputs = _decision_inputs(
        aggregate_outputs=aggregate_outputs,
        metrics_by_split=metrics_by_split,
    )
    guard_checks = _guard_checks(
        stage101_summary=stage101_summary,
        split_samples=split_samples,
        aggregate_outputs=aggregate_outputs,
        public_safe_case_samples=public_safe_case_samples,
        user_confirmed_analysis=user_confirmed_analysis,
    )
    finished_at = time.perf_counter()
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train/dev-only answer-pipeline error decomposition under the "
            "Stage101 public-safe field contract. This stage loads train/dev "
            "frozen split rows and local corpus documents to compute bucketed "
            "diagnostics, does not load the test split, does not run final "
            "metrics, does not write raw question, answer, document, token, or "
            "document-identifier fields, does not add fallback strategies, and "
            "does not change runtime defaults."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_analysis),
            "confirmation_note": confirmation_note,
        },
        "split_contract": {
            "split_name": _SPLIT_NAME,
            "protocol_version": _PROTOCOL_VERSION,
            "development_splits": list(_ALLOWED_SPLITS),
            "forbidden_final_splits": list(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "stage101_protocol": _fingerprint(stage101_protocol_path),
            "train_split": _fingerprint(train_split_path),
            "dev_split": _fingerprint(dev_split_path),
            "corpus_documents": _fingerprint(documents_path),
        },
        "stage101_summary": stage101_summary,
        "analysis_config": {
            "diagnostic_profile": (
                "stage102_bm25_top10_bm25_sentence_mcpd3_topk_verified_evidence7"
            ),
            "retriever": "BM25",
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "retrieval_top_k": retrieval_top_k,
            "answer_generator": "extractive_sentence_baseline",
            "evidence_selector": answer_generator.evidence_selector_name,
            "max_candidates_per_document": max_candidates_per_document,
            "composition_policy": answer_generator.composition_policy_name,
            "max_sentences": max_sentences,
            "min_sentence_score": min_sentence_score,
            "answer_verifier": "citation_and_evidence_gate",
            "min_evidence_score": min_evidence_score,
            "max_citation_rank": max_citation_rank,
            "min_citations": min_citations,
            "max_gold_window_sentences": max_gold_window_sentences,
            "gold_span_gap_margin": gold_span_gap_margin,
            "low_answer_f1_threshold": low_answer_f1_threshold,
            "sample_limit_per_bucket": sample_limit_per_bucket,
        },
        "data_summary": {
            "documents": len(documents),
            "splits": {
                split: _split_sample_summary(samples)
                for split, samples in split_samples.items()
            },
        },
        "metrics_by_split": metrics_by_split,
        "aggregate_outputs": aggregate_outputs,
        "public_safe_case_samples": public_safe_case_samples,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks=guard_checks, decision_inputs=decision_inputs),
        "timing_seconds": {
            "load_stage101_and_splits": round(loaded_splits_at - started_at, 3),
            "load_documents_and_bm25_index": round(indexed_at - loaded_splits_at, 3),
            "answer_pipeline_decomposition": round(analyzed_at - indexed_at, 3),
            "aggregate_and_guard": round(finished_at - analyzed_at, 3),
            "total": round(finished_at - started_at, 3),
        },
    }
    return report


def write_primeqa_hybrid_answer_pipeline_decomposition_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridAnswerPipelineErrorDecompositionVisualization]:
    """Write SVG charts for the Stage102 decomposition report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage102_bucket_counts_by_split.svg": render_horizontal_bar_chart_svg(
            title="Stage102 bucket counts by split",
            bars=_bucket_count_bars(report),
            x_label="case count",
            width=1320,
            margin_left=560,
        ),
        "stage102_pipeline_stage_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage102 pipeline-stage counts",
            bars=_pipeline_stage_bars(report),
            x_label="case count",
            width=1200,
            margin_left=420,
        ),
        "stage102_answerability_bucket_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage102 answerability and bucket counts",
            bars=_answerability_bucket_bars(report),
            x_label="case count",
            width=1340,
            margin_left=620,
        ),
        "stage102_verified_metric_rates.svg": render_horizontal_bar_chart_svg(
            title="Stage102 verified metric rates",
            bars=_verified_metric_bars(report),
            x_label="rate or average F1",
            width=1280,
            margin_left=560,
        ),
        "stage102_public_case_sample_counts.svg": render_horizontal_bar_chart_svg(
            title="Stage102 public-safe sample counts",
            bars=_case_sample_bars(report),
            x_label="sample count",
            width=1320,
            margin_left=560,
        ),
        "stage102_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage102 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1460,
            margin_left=760,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridAnswerPipelineErrorDecompositionVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _analyze_split(
    *,
    split: str,
    samples: Sequence[PrimeQAHybridSplitSample],
    retriever: BM25Retriever,
    documents_by_id: Mapping[str, PrimeQADocument],
    answer_generator: ExtractiveAnswerGenerator,
    answer_verifier: AnswerVerifier,
    retrieval_top_k: int,
    max_gold_window_sentences: int,
    gold_span_gap_margin: float,
    low_answer_f1_threshold: float,
) -> list[_SplitQuestionTrace]:
    traces = []
    for sample in samples:
        question = sample.to_primeqa_question()
        retrieval_results = retriever.search(question.full_question, top_k=retrieval_top_k)
        original_answer = answer_generator.generate(question, retrieval_results)
        verification = answer_verifier.verify(original_answer, retrieval_results)
        route_trace = trace_selector_route(question, answer_generator.evidence_selector_name)
        bucket_id, scoring = _classify_pipeline_bucket(
            question=question,
            retrieval_results=retrieval_results,
            original_answer=original_answer,
            verified_answer=verification.verified_answer,
            documents_by_id=documents_by_id,
            max_gold_window_sentences=max_gold_window_sentences,
            gold_span_gap_margin=gold_span_gap_margin,
            low_answer_f1_threshold=low_answer_f1_threshold,
        )
        public_case = _public_case(
            split=split,
            sample_id=sample.sample_id,
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
        traces.append(
            _SplitQuestionTrace(
                sample=sample,
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
        )
    return traces


def _classify_pipeline_bucket(
    *,
    question: PrimeQAQuestion,
    retrieval_results: Sequence[RetrievalResult],
    original_answer: GeneratedAnswer,
    verified_answer: GeneratedAnswer,
    documents_by_id: Mapping[str, PrimeQADocument],
    max_gold_window_sentences: int,
    gold_span_gap_margin: float,
    low_answer_f1_threshold: float,
) -> tuple[str, dict[str, float | None]]:
    answer_token_f1 = (
        round(token_f1(original_answer.answer, question.answer), 4)
        if question.answerable
        else None
    )
    best_gold_span_token_f1 = (
        _best_gold_span_token_f1(
            question=question,
            documents_by_id=documents_by_id,
            max_gold_window_sentences=max_gold_window_sentences,
        )
        if question.answerable
        else None
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

    gold_context_status = _gold_context_status(question, retrieval_results)
    if gold_context_status == "gold_context_absent":
        return "retrieval_context_miss", scoring

    citation_status = _citation_status(question, original_answer)
    if citation_status != "gold_cited":
        return "evidence_selection_miss", scoring

    if verified_answer.refused:
        return "verification_over_refusal", scoring

    if answer_gold_span_gap is not None and answer_gold_span_gap >= gold_span_gap_margin:
        return "gold_span_beats_selected_answer", scoring

    if answer_token_f1 is not None and answer_token_f1 < low_answer_f1_threshold:
        return "low_overlap_gold_cited_answer", scoring

    return "answer_supported_and_cited", scoring


def _public_case(
    *,
    split: str,
    sample_id: str,
    question: PrimeQAQuestion,
    retrieval_results: Sequence[RetrievalResult],
    original_answer: GeneratedAnswer,
    verified_answer: GeneratedAnswer,
    verification_reasons: Sequence[str],
    bucket_id: str,
    scoring: Mapping[str, float | None],
    question_route: str,
    routed_selector_name: str,
    evidence_selector_name: str,
    composition_policy_name: str,
) -> dict[str, Any]:
    answer_token_f1 = scoring["answer_token_f1"]
    best_gold_span_token_f1 = scoring["best_gold_span_token_f1"]
    answer_gold_span_gap = scoring["answer_gold_span_gap"]
    case = {
        "sample_id": sample_id,
        "split": split,
        "answerability_label": "answerable" if question.answerable else "unanswerable",
        "pipeline_bucket_id": bucket_id,
        "pipeline_stage": _BUCKET_TO_STAGE[bucket_id],
        "retrieval_rank_bucket": _retrieval_rank_bucket(question, retrieval_results),
        "retrieval_context_status": _gold_context_status(question, retrieval_results),
        "citation_status": _citation_status(question, original_answer),
        "evidence_selection_status": _evidence_selection_status(
            question,
            retrieval_results,
            original_answer,
        ),
        "answer_token_f1_bucket": _f1_bucket(answer_token_f1),
        "best_gold_span_f1_bucket": _f1_bucket(best_gold_span_token_f1),
        "answer_gold_span_gap_bucket": _gap_bucket(answer_gold_span_gap),
        "verifier_decision": "refused" if verified_answer.refused else "answered",
        "refusal_reason_code": "+".join(sorted(verification_reasons)),
        "question_route": question_route,
        "evidence_selector_name": routed_selector_name or evidence_selector_name,
        "composition_policy_id": composition_policy_name,
        "bucket_confidence_band": _bucket_confidence_band(
            bucket_id=bucket_id,
            answer_gold_span_gap=answer_gold_span_gap,
            answer_token_f1=answer_token_f1,
        ),
    }
    if tuple(case) != _EXPECTED_PUBLIC_CASE_FIELDS:
        raise ValueError("public case fields do not match Stage101 contract")
    return case


def _aggregate_outputs(
    traces_by_split: Mapping[str, Sequence[_SplitQuestionTrace]],
) -> dict[str, Any]:
    bucket_counts_by_split = {
        split: _ordered_counts(trace.bucket_id for trace in traces)
        for split, traces in traces_by_split.items()
    }
    row_counts = {
        split: len(traces)
        for split, traces in traces_by_split.items()
    }
    return {
        "bucket_counts_by_split": bucket_counts_by_split,
        "bucket_rates_by_split": {
            split: {
                bucket: _safe_rate(count, row_counts[split])
                for bucket, count in counts.items()
            }
            for split, counts in bucket_counts_by_split.items()
        },
        "pipeline_stage_counts_by_split": {
            split: _counter_dict(trace.public_case["pipeline_stage"] for trace in traces)
            for split, traces in traces_by_split.items()
        },
        "route_bucket_cross_tab": _route_bucket_cross_tab(traces_by_split),
        "answerability_bucket_cross_tab": _answerability_bucket_cross_tab(
            traces_by_split
        ),
        "token_f1_bucket_distributions": {
            split: _counter_dict(
                trace.public_case["answer_token_f1_bucket"] for trace in traces
            )
            for split, traces in traces_by_split.items()
        },
        "retrieval_rank_bucket_distributions": {
            split: _counter_dict(
                trace.public_case["retrieval_rank_bucket"] for trace in traces
            )
            for split, traces in traces_by_split.items()
        },
        "verification_decision_distributions": {
            split: _counter_dict(
                trace.public_case["verifier_decision"] for trace in traces
            )
            for split, traces in traces_by_split.items()
        },
        "top_priority_buckets": {
            split: _top_priority_buckets(bucket_counts_by_split[split])
            for split in traces_by_split
        },
    }


def _public_safe_case_samples(
    *,
    traces_by_split: Mapping[str, Sequence[_SplitQuestionTrace]],
    sample_limit_per_bucket: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    samples: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split, traces in traces_by_split.items():
        samples[split] = {}
        traces_by_bucket: dict[str, list[_SplitQuestionTrace]] = defaultdict(list)
        for trace in traces:
            traces_by_bucket[trace.bucket_id].append(trace)
        for bucket in _BUCKET_ORDER:
            ranked = sorted(
                traces_by_bucket.get(bucket, []),
                key=lambda trace: (
                    _gap_sort_value(trace),
                    trace.public_case["answer_token_f1_bucket"],
                    trace.public_case["sample_id"],
                ),
            )
            samples[split][bucket] = [
                trace.public_case for trace in ranked[:sample_limit_per_bucket]
            ]
    return samples


def _metrics_by_split(
    traces_by_split: Mapping[str, Sequence[_SplitQuestionTrace]],
) -> dict[str, dict[str, Any]]:
    metrics = {}
    for split, traces in traces_by_split.items():
        questions = [trace.question for trace in traces]
        original_answers = [trace.original_answer for trace in traces]
        verified_answers = [trace.verified_answer for trace in traces]
        answerable_count = sum(question.answerable for question in questions)
        answerable_gold_context = sum(
            1
            for trace in traces
            if trace.question.answerable
            and _gold_context_status(trace.question, trace.retrieval_results)
            == "gold_context_present"
        )
        metrics[split] = {
            "original": asdict(evaluate_answers(questions, original_answers)),
            "verified": asdict(evaluate_answers(questions, verified_answers)),
            "answerable_gold_context_count": answerable_gold_context,
            "answerable_gold_context_rate": _safe_rate(
                answerable_gold_context,
                answerable_count,
            ),
        }
    return metrics


def _decision_inputs(
    *,
    aggregate_outputs: Mapping[str, Any],
    metrics_by_split: Mapping[str, Any],
) -> dict[str, Any]:
    train_counts = aggregate_outputs["bucket_counts_by_split"].get("train", {})
    dev_counts = aggregate_outputs["bucket_counts_by_split"].get("dev", {})
    train_top = aggregate_outputs["top_priority_buckets"].get("train", [])
    dev_top = aggregate_outputs["top_priority_buckets"].get("dev", [])
    return {
        "train_top_bucket": train_top[0]["bucket_id"] if train_top else None,
        "dev_top_bucket": dev_top[0]["bucket_id"] if dev_top else None,
        "train_evidence_selection_miss": train_counts.get("evidence_selection_miss", 0),
        "dev_evidence_selection_miss": dev_counts.get("evidence_selection_miss", 0),
        "train_answerability_false_answer": train_counts.get(
            "answerability_false_answer",
            0,
        ),
        "dev_answerability_false_answer": dev_counts.get(
            "answerability_false_answer",
            0,
        ),
        "train_verified_average_token_f1": metrics_by_split["train"]["verified"][
            "average_token_f1"
        ],
        "dev_verified_average_token_f1": metrics_by_split["dev"]["verified"][
            "average_token_f1"
        ],
    }


def _guard_checks(
    *,
    stage101_summary: Mapping[str, Any],
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    aggregate_outputs: Mapping[str, Any],
    public_safe_case_samples: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    user_confirmed_analysis: bool,
) -> list[dict[str, Any]]:
    all_public_cases = [
        case
        for split_samples_by_bucket in public_safe_case_samples.values()
        for cases in split_samples_by_bucket.values()
        for case in cases
    ]
    return [
        _check(
            name="stage101_source_is_expected_stage",
            passed=stage101_summary.get("stage") == _SOURCE_STAGE101,
            observed=stage101_summary.get("stage"),
            expected=_SOURCE_STAGE101,
        ),
        _check(
            name="stage101_protocol_is_frozen",
            passed=stage101_summary.get("decision_status")
            == "primeqa_hybrid_answer_pipeline_error_decomposition_protocol_frozen",
            observed=stage101_summary.get("decision_status"),
            expected="primeqa_hybrid_answer_pipeline_error_decomposition_protocol_frozen",
        ),
        _check(
            name="stage101_protocol_id_matches",
            passed=stage101_summary.get("protocol_id") == _PROTOCOL_ID,
            observed=stage101_summary.get("protocol_id"),
            expected=_PROTOCOL_ID,
        ),
        _check(
            name="user_confirmed_stage102_analysis",
            passed=user_confirmed_analysis,
            observed=user_confirmed_analysis,
            expected=True,
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
            expected="no test split key",
        ),
        _check(
            name="stage101_allows_train_dev_error_decomposition",
            passed=stage101_summary.get(
                "can_run_train_dev_error_decomposition_after_user_confirmation"
            )
            is True,
            observed=stage101_summary.get(
                "can_run_train_dev_error_decomposition_after_user_confirmation"
            ),
            expected=True,
        ),
        _check(
            name="stage101_final_metrics_locked",
            passed=stage101_summary.get("can_run_final_test_metrics_now") is False,
            observed=stage101_summary.get("can_run_final_test_metrics_now"),
            expected=False,
        ),
        _check(
            name="stage101_forbids_test_tuning",
            passed=stage101_summary.get("can_use_test_for_tuning") is False,
            observed=stage101_summary.get("can_use_test_for_tuning"),
            expected=False,
        ),
        _check(
            name="stage101_runtime_default_unchanged",
            passed=stage101_summary.get("default_runtime_policy") == "unchanged",
            observed=stage101_summary.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage101_fallback_strategies_disabled",
            passed=stage101_summary.get("fallback_strategies_enabled") is False,
            observed=stage101_summary.get("fallback_strategies_enabled"),
            expected=False,
        ),
        _check(
            name="expected_bucket_order_present",
            passed=all(
                tuple(aggregate_outputs["bucket_counts_by_split"][split])
                == _BUCKET_ORDER
                for split in _ALLOWED_SPLITS
            ),
            observed=aggregate_outputs["bucket_counts_by_split"],
            expected=list(_BUCKET_ORDER),
        ),
        _check(
            name="public_case_samples_use_stage101_fields",
            passed=all(tuple(case) == _EXPECTED_PUBLIC_CASE_FIELDS for case in all_public_cases),
            observed=[list(case) for case in all_public_cases[:3]],
            expected=list(_EXPECTED_PUBLIC_CASE_FIELDS),
        ),
        _check(
            name="public_case_samples_do_not_exceed_limit",
            passed=all(
                len(cases) <= 5
                for by_bucket in public_safe_case_samples.values()
                for cases in by_bucket.values()
            ),
            observed={
                split: {bucket: len(cases) for bucket, cases in by_bucket.items()}
                for split, by_bucket in public_safe_case_samples.items()
            },
            expected="<= 5 per split and bucket",
        ),
        _check(
            name="public_case_samples_exclude_forbidden_keys",
            passed=not _contains_forbidden_keys(all_public_cases),
            observed=_find_forbidden_keys(all_public_cases),
            expected="no raw text, document id, source DOC_IDS, or token fields",
        ),
        _check(
            name="stage102_runs_train_dev_analysis_only",
            passed=True,
            observed="train_dev_only",
            expected="train_dev_only",
        ),
        _check(
            name="stage102_final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage102_default_runtime_policy_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
        _check(
            name="stage102_fallback_strategies_not_added",
            passed=True,
            observed=False,
            expected=False,
        ),
    ]


def _decision(
    *,
    guard_checks: Sequence[Mapping[str, Any]],
    decision_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_answer_pipeline_error_decomposition_blocked",
            "analysis_id": _ANALYSIS_ID,
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        }
    return {
        "status": "primeqa_hybrid_answer_pipeline_error_decomposition_completed",
        "analysis_id": _ANALYSIS_ID,
        "train_top_bucket": decision_inputs["train_top_bucket"],
        "dev_top_bucket": decision_inputs["dev_top_bucket"],
        "train_evidence_selection_miss": decision_inputs[
            "train_evidence_selection_miss"
        ],
        "dev_evidence_selection_miss": decision_inputs["dev_evidence_selection_miss"],
        "train_answerability_false_answer": decision_inputs[
            "train_answerability_false_answer"
        ],
        "dev_answerability_false_answer": decision_inputs[
            "dev_answerability_false_answer"
        ],
        "train_verified_average_token_f1": decision_inputs[
            "train_verified_average_token_f1"
        ],
        "dev_verified_average_token_f1": decision_inputs[
            "dev_verified_average_token_f1"
        ],
        "recommended_next_direction": "evidence_selection_and_answerability_candidate_design",
        "requires_user_confirmation_before_next_protocol": True,
        "can_continue_train_dev_development": True,
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage103: design a train/dev-only candidate intervention from the "
            "Stage102 bucket evidence, prioritizing shared train/dev bottlenecks "
            "without using test or changing runtime defaults."
        ),
    }


def _stage101_public_summary(stage101_protocol: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage101_protocol.get("decision") or {}
    frozen_protocol = stage101_protocol.get("frozen_protocol") or {}
    return {
        "stage": stage101_protocol.get("stage"),
        "decision_status": decision.get("status"),
        "protocol_id": decision.get("protocol_id"),
        "recommended_direction": decision.get("recommended_direction"),
        "can_run_train_dev_error_decomposition_after_user_confirmation": decision.get(
            "can_run_train_dev_error_decomposition_after_user_confirmation"
        ),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get(
            "can_run_final_test_metrics_now"
        ),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "public_case_field_count": len(
            (
                frozen_protocol.get("public_safe_output_contract") or {}
            ).get("case_sample_fields")
            or []
        ),
    }


def _split_sample_summary(samples: Sequence[PrimeQAHybridSplitSample]) -> dict[str, Any]:
    answerable_count = sum(sample.answerable for sample in samples)
    return {
        "row_count": len(samples),
        "answerable_count": answerable_count,
        "unanswerable_count": len(samples) - answerable_count,
        "split_subtype_counts": _counter_dict(sample.split_subtype for sample in samples),
        "source_split_counts": _counter_dict(sample.source_split for sample in samples),
    }


def _best_gold_span_token_f1(
    *,
    question: PrimeQAQuestion,
    documents_by_id: Mapping[str, PrimeQADocument],
    max_gold_window_sentences: int,
) -> float | None:
    if not question.answer_doc_id:
        return None
    document = documents_by_id.get(question.answer_doc_id)
    if document is None:
        return None
    sentences = [
        " ".join(sentence.split())
        for sentence in split_sentences(document.text)
        if len(" ".join(sentence.split())) >= 24
    ]
    if not sentences:
        return None
    best_score = 0.0
    max_size = min(max_gold_window_sentences, len(sentences))
    for window_size in range(1, max_size + 1):
        for start_index in range(0, len(sentences) - window_size + 1):
            text = " ".join(sentences[start_index : start_index + window_size])
            best_score = max(best_score, token_f1(text, question.answer))
    return round(best_score, 4)


def _gold_context_status(
    question: PrimeQAQuestion,
    retrieval_results: Sequence[RetrievalResult],
) -> str:
    if not question.answerable or not question.answer_doc_id:
        return "not_applicable"
    if question.answer_doc_id in {result.document.id for result in retrieval_results}:
        return "gold_context_present"
    return "gold_context_absent"


def _citation_status(question: PrimeQAQuestion, answer: GeneratedAnswer) -> str:
    if not question.answerable or not question.answer_doc_id:
        return "not_applicable"
    if answer.refused:
        return "no_citation"
    if question.answer_doc_id in {citation.document_id for citation in answer.citations}:
        return "gold_cited"
    return "gold_not_cited"


def _evidence_selection_status(
    question: PrimeQAQuestion,
    retrieval_results: Sequence[RetrievalResult],
    answer: GeneratedAnswer,
) -> str:
    if not question.answerable:
        return "not_applicable"
    gold_context_status = _gold_context_status(question, retrieval_results)
    if gold_context_status == "gold_context_absent":
        return "not_applicable_context_absent"
    if _citation_status(question, answer) == "gold_cited":
        return "selected_gold_evidence"
    return "gold_present_not_selected"


def _retrieval_rank_bucket(
    question: PrimeQAQuestion,
    retrieval_results: Sequence[RetrievalResult],
) -> str:
    if not question.answerable or not question.answer_doc_id:
        return "not_applicable"
    for result in retrieval_results:
        if result.document.id == question.answer_doc_id:
            if result.rank == 1:
                return "rank_1"
            if result.rank <= 3:
                return "rank_2_to_3"
            if result.rank <= 5:
                return "rank_4_to_5"
            if result.rank <= 10:
                return "rank_6_to_10"
            return "rank_over_10"
    return "not_found_top_k"


def _f1_bucket(value: float | None) -> str:
    if value is None:
        return "not_applicable"
    if value < 0.2:
        return "f1_0_00_to_0_19"
    if value < 0.4:
        return "f1_0_20_to_0_39"
    if value < 0.6:
        return "f1_0_40_to_0_59"
    if value < 0.8:
        return "f1_0_60_to_0_79"
    return "f1_0_80_to_1_00"


def _gap_bucket(value: float | None) -> str:
    if value is None:
        return "not_applicable"
    if value < 0.05:
        return "gap_0_00_to_0_04"
    if value < 0.2:
        return "gap_0_05_to_0_19"
    if value < 0.4:
        return "gap_0_20_to_0_39"
    return "gap_0_40_plus"


def _bucket_confidence_band(
    *,
    bucket_id: str,
    answer_gold_span_gap: float | None,
    answer_token_f1: float | None,
) -> str:
    if bucket_id in {
        "answerability_false_answer",
        "retrieval_context_miss",
        "evidence_selection_miss",
    }:
        return "high"
    if bucket_id == "verification_over_refusal":
        return "medium"
    if bucket_id == "gold_span_beats_selected_answer":
        if answer_gold_span_gap is not None and answer_gold_span_gap >= 0.2:
            return "high"
        return "medium"
    if bucket_id == "low_overlap_gold_cited_answer":
        if answer_token_f1 is not None and answer_token_f1 < 0.1:
            return "high"
        return "medium"
    return "reference"


def _route_bucket_cross_tab(
    traces_by_split: Mapping[str, Sequence[_SplitQuestionTrace]],
) -> dict[str, dict[str, int]]:
    rows: dict[str, Counter[str]] = defaultdict(Counter)
    for split, traces in traces_by_split.items():
        for trace in traces:
            rows[f"{split}::{trace.question_route}"][trace.bucket_id] += 1
    return {
        route_key: _ordered_counts(counter.elements())
        for route_key, counter in sorted(rows.items())
    }


def _answerability_bucket_cross_tab(
    traces_by_split: Mapping[str, Sequence[_SplitQuestionTrace]],
) -> dict[str, dict[str, int]]:
    rows: dict[str, Counter[str]] = defaultdict(Counter)
    for split, traces in traces_by_split.items():
        for trace in traces:
            label = "answerable" if trace.question.answerable else "unanswerable"
            rows[f"{split}::{label}"][trace.bucket_id] += 1
    return {
        key: _ordered_counts(counter.elements())
        for key, counter in sorted(rows.items())
    }


def _top_priority_buckets(bucket_counts: Mapping[str, int]) -> list[dict[str, Any]]:
    priority_weight = {
        "answerability_false_answer": 1.55,
        "retrieval_context_miss": 1.35,
        "evidence_selection_miss": 1.70,
        "verification_over_refusal": 1.20,
        "gold_span_beats_selected_answer": 1.45,
        "low_overlap_gold_cited_answer": 1.10,
        "answer_supported_and_cited": 0.25,
    }
    rows = [
        {
            "bucket_id": bucket,
            "case_count": int(count),
            "priority_weight": priority_weight[bucket],
            "priority_score": round(int(count) * priority_weight[bucket], 4),
        }
        for bucket, count in bucket_counts.items()
    ]
    return sorted(
        rows,
        key=lambda row: (-row["priority_score"], row["bucket_id"]),
    )


def _ordered_counts(values) -> dict[str, int]:
    counter = Counter(values)
    return {bucket: counter.get(bucket, 0) for bucket in _BUCKET_ORDER}


def _counter_dict(values) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _gap_sort_value(trace: _SplitQuestionTrace) -> float:
    if trace.answer_gold_span_gap is None:
        return 0.0
    return -trace.answer_gold_span_gap


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


def _bucket_count_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    counts_by_split = report["aggregate_outputs"]["bucket_counts_by_split"]
    return [
        BarDatum(
            label=f"{split}::{bucket}",
            value=float(count),
            value_label=str(count),
        )
        for split in _ALLOWED_SPLITS
        for bucket, count in counts_by_split[split].items()
    ]


def _pipeline_stage_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    counts_by_split = report["aggregate_outputs"]["pipeline_stage_counts_by_split"]
    return [
        BarDatum(
            label=f"{split}::{stage}",
            value=float(count),
            value_label=str(count),
        )
        for split, stage_counts in counts_by_split.items()
        for stage, count in sorted(stage_counts.items())
    ]


def _answerability_bucket_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    rows = report["aggregate_outputs"]["answerability_bucket_cross_tab"]
    return [
        BarDatum(
            label=f"{key}::{bucket}",
            value=float(count),
            value_label=str(count),
        )
        for key, counts in rows.items()
        for bucket, count in counts.items()
        if count
    ]


def _verified_metric_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    metric_names = [
        "gold_doc_citation_rate",
        "answerable_refusal_rate",
        "unanswerable_refusal_rate",
        "average_token_f1",
    ]
    return [
        BarDatum(
            label=f"{split}::{metric}",
            value=float(report["metrics_by_split"][split]["verified"][metric]),
            value_label=f"{float(report['metrics_by_split'][split]['verified'][metric]):.4f}",
        )
        for split in _ALLOWED_SPLITS
        for metric in metric_names
    ]


def _case_sample_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    samples = report["public_safe_case_samples"]
    return [
        BarDatum(
            label=f"{split}::{bucket}",
            value=float(len(cases)),
            value_label=str(len(cases)),
        )
        for split in _ALLOWED_SPLITS
        for bucket, cases in samples[split].items()
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
    retrieval_top_k: int,
    bm25_k1: float,
    bm25_b: float,
    max_candidates_per_document: int,
    max_sentences: int,
    min_sentence_score: float,
    min_evidence_score: float,
    max_citation_rank: int,
    min_citations: int,
    max_gold_window_sentences: int,
    gold_span_gap_margin: float,
    low_answer_f1_threshold: float,
    sample_limit_per_bucket: int,
) -> None:
    if retrieval_top_k <= 0:
        raise ValueError("retrieval_top_k must be positive")
    if bm25_k1 <= 0:
        raise ValueError("bm25_k1 must be positive")
    if not 0 <= bm25_b <= 1:
        raise ValueError("bm25_b must be between 0 and 1")
    if max_candidates_per_document <= 0:
        raise ValueError("max_candidates_per_document must be positive")
    if max_sentences <= 0:
        raise ValueError("max_sentences must be positive")
    if min_sentence_score < 0:
        raise ValueError("min_sentence_score must be non-negative")
    if min_evidence_score < 0:
        raise ValueError("min_evidence_score must be non-negative")
    if max_citation_rank <= 0:
        raise ValueError("max_citation_rank must be positive")
    if min_citations <= 0:
        raise ValueError("min_citations must be positive")
    if max_gold_window_sentences <= 0:
        raise ValueError("max_gold_window_sentences must be positive")
    if gold_span_gap_margin < 0:
        raise ValueError("gold_span_gap_margin must be non-negative")
    if low_answer_f1_threshold < 0:
        raise ValueError("low_answer_f1_threshold must be non-negative")
    if sample_limit_per_bucket < 0:
        raise ValueError("sample_limit_per_bucket must be non-negative")


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
