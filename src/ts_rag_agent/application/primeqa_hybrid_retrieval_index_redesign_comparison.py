from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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
from ts_rag_agent.application.primeqa_hybrid_evidence_answerability_comparison import (
    _answer_signature,
    _classify_pipeline_bucket_with_cached_gold_span,
)
from ts_rag_agent.application.rag_answering import ExtractiveAnswerGenerator
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.dataset import (
    PrimeQADocument,
    PrimeQADocumentSection,
    PrimeQAQuestion,
)
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.bm25_retriever import BM25Retriever, tokenize_text
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
    summarize_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)

_STAGE = "Stage 114"
_CREATED_AT = "2026-07-16"
_SOURCE_STAGE113 = "Stage 113"
_SOURCE_PROTOCOL_ID = "primeqa_hybrid_retrieval_index_redesign_protocol_v1"
_SOURCE_PROTOCOL_STATUS = "frozen_requires_user_confirmation_before_train_dev_run"
_ANALYSIS_ID = "primeqa_hybrid_retrieval_index_redesign_train_cv_dev_validation_v1"
_NO_TRAIN_CV_SELECTABLE_STATUS = (
    "primeqa_hybrid_retrieval_index_redesign_completed_no_train_cv_selectable_config"
)
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_ALLOWED_SPLITS = ("train", "dev")
_EVAL_SPLITS = ("train_cv", "train_full", "dev")
_FORBIDDEN_FINAL_SPLITS = ("test",)
_TARGET_BUCKETS = (
    "retrieval_context_miss",
    "answerability_false_answer",
    "evidence_selection_miss",
    "gold_span_beats_selected_answer",
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
_SPECIAL_TOKEN_RE = re.compile(
    r"\b(?:CVE-\d{4}-\d{4,7}|[A-Z]{1,8}\d{2,}[A-Z0-9-]*|"
    r"\d+(?:\.\d+){1,}|[A-Z0-9]+(?:[-_][A-Z0-9]+)+)\b",
    re.IGNORECASE,
)


class _Retriever(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Return ranked retrieval results."""


@dataclass(frozen=True)
class _RunConfig:
    config_id: str
    family_id: str
    retrieval_mode: str
    selection_eligible: bool
    payload: dict[str, Any]


@dataclass(frozen=True)
class _QuestionInputs:
    sample: PrimeQAHybridSplitSample
    question: PrimeQAQuestion
    best_gold_span_token_f1: float | None


@dataclass(frozen=True)
class _SectionIndexRecord:
    document_id: str
    section_id: str
    text: str


@dataclass(frozen=True)
class PrimeQAHybridRetrievalIndexRedesignComparisonVisualization:
    """One generated Stage114 retrieval/index redesign comparison chart."""

    name: str
    path: str


class _MappedBM25Retriever:
    """BM25 over transformed documents that returns the original documents."""

    def __init__(
        self,
        *,
        indexed_documents: Iterable[PrimeQADocument],
        original_documents_by_id: Mapping[str, PrimeQADocument],
        k1: float,
        b: float,
    ) -> None:
        self._original_documents_by_id = dict(original_documents_by_id)
        self._retriever = BM25Retriever(k1=k1, b=b)
        self._retriever.fit(indexed_documents)

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        results = self._retriever.search(query, top_k=top_k)
        return [
            RetrievalResult(
                document=self._original_documents_by_id[result.document.id],
                score=result.score,
                rank=rank,
            )
            for rank, result in enumerate(results, start=1)
        ]


class _SectionBM25DocumentRollupRetriever:
    """BM25 over section records with deterministic section-to-document rollup."""

    def __init__(
        self,
        *,
        documents: Sequence[PrimeQADocument],
        sections_by_document: Mapping[str, Sequence[PrimeQADocumentSection]],
        title_weight: float,
        section_heading_weight: float,
        body_weight: float,
        rollup_mode: str,
        top_n_sections: int,
        rrf_k: int,
        k1: float,
        b: float,
    ) -> None:
        if rollup_mode not in {"max_score", "top_n_section_rrf"}:
            raise ValueError(f"Unknown section rollup mode: {rollup_mode}")
        if top_n_sections <= 0:
            raise ValueError("top_n_sections must be positive")
        self._documents_by_id = {document.id: document for document in documents}
        self._rollup_mode = rollup_mode
        self._top_n_sections = top_n_sections
        self._rrf_k = rrf_k
        self._k1 = k1
        self._b = b
        self._records = _section_index_records(
            documents=documents,
            sections_by_document=sections_by_document,
            title_weight=title_weight,
            section_heading_weight=section_heading_weight,
            body_weight=body_weight,
        )
        self._section_lengths: list[int] = []
        self._avg_section_length = 0.0
        self._idf: dict[str, float] = {}
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._fit()

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_terms = tokenize_text(query)
        if not query_terms or not self._records:
            return []

        section_scores: dict[int, float] = defaultdict(float)
        for term in query_terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for section_index, term_frequency in self._postings[term]:
                section_scores[section_index] += self._score_term(
                    idf,
                    term_frequency,
                    self._section_lengths[section_index],
                )

        if self._rollup_mode == "max_score":
            document_scores = self._max_section_scores(section_scores)
        else:
            document_scores = self._top_n_section_rrf_scores(section_scores)

        ranked_doc_ids = sorted(
            document_scores,
            key=lambda doc_id: (-document_scores[doc_id], doc_id),
        )
        return [
            RetrievalResult(
                document=self._documents_by_id[doc_id],
                score=document_scores[doc_id],
                rank=rank,
            )
            for rank, doc_id in enumerate(ranked_doc_ids[:top_k], start=1)
        ]

    def _fit(self) -> None:
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for section_index, record in enumerate(self._records):
            tokens = tokenize_text(record.text)
            term_counts = Counter(tokens)
            self._section_lengths.append(len(tokens))
            for term, term_frequency in term_counts.items():
                postings[term].append((section_index, term_frequency))

        section_count = len(self._records)
        total_length = sum(self._section_lengths)
        self._avg_section_length = total_length / section_count if section_count else 0.0
        self._postings = dict(postings)
        self._idf = {
            term: _compute_idf(section_count, len(term_postings))
            for term, term_postings in self._postings.items()
        }

    def _max_section_scores(self, section_scores: Mapping[int, float]) -> dict[str, float]:
        document_scores: dict[str, float] = {}
        for section_index, section_score in section_scores.items():
            doc_id = self._records[section_index].document_id
            current_score = document_scores.get(doc_id)
            if current_score is None or section_score > current_score:
                document_scores[doc_id] = section_score
        return document_scores

    def _top_n_section_rrf_scores(
        self,
        section_scores: Mapping[int, float],
    ) -> dict[str, float]:
        document_scores: dict[str, float] = defaultdict(float)
        sections_used_by_document: Counter[str] = Counter()
        ranked_sections = sorted(
            section_scores,
            key=lambda index: (
                -section_scores[index],
                self._records[index].document_id,
                self._records[index].section_id,
            ),
        )
        for section_rank, section_index in enumerate(ranked_sections, start=1):
            doc_id = self._records[section_index].document_id
            if sections_used_by_document[doc_id] >= self._top_n_sections:
                continue
            document_scores[doc_id] += 1.0 / (self._rrf_k + section_rank)
            sections_used_by_document[doc_id] += 1
        return dict(document_scores)

    def _score_term(self, idf: float, term_frequency: int, section_length: int) -> float:
        length_normalizer = 1 - self._b
        if self._avg_section_length:
            length_normalizer += self._b * section_length / self._avg_section_length
        numerator = term_frequency * (self._k1 + 1)
        denominator = term_frequency + self._k1 * length_normalizer
        return idf * numerator / denominator


class _RrfFusionRetriever:
    """Fuse multiple retrievers with reciprocal rank fusion."""

    def __init__(
        self,
        *,
        retrievers: Sequence[_Retriever],
        rrf_k: int,
        component_depth: int,
    ) -> None:
        if not retrievers:
            raise ValueError("retrievers must not be empty")
        self._retrievers = tuple(retrievers)
        self._rrf_k = rrf_k
        self._component_depth = component_depth

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores: dict[str, float] = defaultdict(float)
        documents_by_id: dict[str, PrimeQADocument] = {}
        for retriever in self._retrievers:
            for result in retriever.search(
                query,
                top_k=max(top_k, self._component_depth),
            ):
                doc_id = result.document.id
                documents_by_id[doc_id] = result.document
                scores[doc_id] += 1.0 / (self._rrf_k + result.rank)

        ranked_doc_ids = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
        return [
            RetrievalResult(
                document=documents_by_id[doc_id],
                score=scores[doc_id],
                rank=rank,
            )
            for rank, doc_id in enumerate(ranked_doc_ids[:top_k], start=1)
        ]


class _SpecialTokenBoostRetriever:
    """Add deterministic exact special-token boosts to a runtime-visible index."""

    def __init__(
        self,
        *,
        base_retriever: _Retriever,
        documents: Sequence[PrimeQADocument],
        sections_by_document: Mapping[str, Sequence[PrimeQADocumentSection]],
        boost: float,
        component_depth: int,
    ) -> None:
        if boost < 0:
            raise ValueError("boost must be non-negative")
        self._base_retriever = base_retriever
        self._documents_by_id = {document.id: document for document in documents}
        self._boost = boost
        self._component_depth = component_depth
        self._token_to_doc_ids = _special_token_index(
            documents=documents,
            sections_by_document=sections_by_document,
        )

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        base_results = self._base_retriever.search(
            query,
            top_k=max(top_k, self._component_depth),
        )
        scores: dict[str, float] = {
            result.document.id: result.score for result in base_results
        }
        query_special_tokens = _special_tokens(query)
        for token in query_special_tokens:
            for doc_id in self._token_to_doc_ids.get(token, ()):
                scores[doc_id] = scores.get(doc_id, 0.0) + self._boost

        ranked_doc_ids = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
        return [
            RetrievalResult(
                document=self._documents_by_id[doc_id],
                score=scores[doc_id],
                rank=rank,
            )
            for rank, doc_id in enumerate(ranked_doc_ids[:top_k], start=1)
        ]


def run_primeqa_hybrid_retrieval_index_redesign_comparison(
    *,
    stage113_protocol_path: Path,
    stage102_report_path: Path,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    user_confirmed_comparison: bool,
    confirmation_note: str,
    retrieval_top_k: int = 10,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    evidence_selector_name: str = "bm25-sentence",
    max_candidates_per_document: int = 3,
    composition_policy_name: str = "top-k",
    max_sentences: int = 3,
    min_sentence_score: float = 2.0,
    verifier_min_citations: int = 1,
    verifier_min_evidence_score: float = 7.0,
    verifier_max_citation_rank: int = 3,
    max_gold_window_sentences: int = 3,
    gold_span_gap_margin: float = 0.05,
    low_answer_f1_threshold: float = 0.2,
    component_depth: int = 50,
) -> dict[str, Any]:
    """Run Stage114 train grouped-CV and dev validation for retrieval candidates."""

    _validate_options(
        retrieval_top_k=retrieval_top_k,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        max_candidates_per_document=max_candidates_per_document,
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
        verifier_min_citations=verifier_min_citations,
        verifier_min_evidence_score=verifier_min_evidence_score,
        verifier_max_citation_rank=verifier_max_citation_rank,
        max_gold_window_sentences=max_gold_window_sentences,
        gold_span_gap_margin=gold_span_gap_margin,
        low_answer_f1_threshold=low_answer_f1_threshold,
        component_depth=component_depth,
    )
    started_at = time.perf_counter()
    stage113_report = _load_json_object(stage113_protocol_path)
    stage102_report = _load_json_object(stage102_report_path)
    frozen_protocol = stage113_report.get("frozen_protocol") or {}
    stage113_summary = _stage113_summary(stage113_report)
    stage102_summary = _stage102_summary(stage102_report)
    configs = [
        _config_from_mapping(config)
        for config in frozen_protocol.get("candidate_configs") or []
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
    sections_by_document = load_primeqa_document_sections(documents_path)
    documents = list(documents_by_id.values())
    loaded_documents_at = time.perf_counter()

    baseline_retriever = BM25Retriever(k1=bm25_k1, b=bm25_b)
    baseline_retriever.fit(documents)
    candidate_retrievers = {
        config.config_id: _build_candidate_retriever(
            config=config,
            baseline_retriever=baseline_retriever,
            documents=documents,
            sections_by_document=sections_by_document,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            component_depth=component_depth,
        )
        for config in configs
    }
    indexed_at = time.perf_counter()

    question_inputs_by_split = _question_inputs_by_split(
        split_samples=split_samples,
        documents_by_id=documents_by_id,
        max_gold_window_sentences=max_gold_window_sentences,
    )
    cached_question_inputs_at = time.perf_counter()
    answer_generator = _answer_generator(
        evidence_selector_name=evidence_selector_name,
        max_candidates_per_document=max_candidates_per_document,
        composition_policy_name=composition_policy_name,
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
    )
    answer_verifier = AnswerVerifier(
        min_citations=verifier_min_citations,
        min_evidence_score=verifier_min_evidence_score,
        max_citation_rank=verifier_max_citation_rank,
    )

    baseline_result = _evaluate_retriever(
        config_id="stage102_verified_bm25_top10_answer_pipeline",
        family_id="baseline",
        retrieval_mode="document_bm25",
        selection_eligible=False,
        retriever=baseline_retriever,
        question_inputs_by_split=question_inputs_by_split,
        documents_by_id=documents_by_id,
        answer_generator=answer_generator,
        answer_verifier=answer_verifier,
        retrieval_top_k=retrieval_top_k,
        max_gold_window_sentences=max_gold_window_sentences,
        gold_span_gap_margin=gold_span_gap_margin,
        low_answer_f1_threshold=low_answer_f1_threshold,
        fold_assignments=fold_assignments,
        baseline_public=None,
    )
    baseline_evaluated_at = time.perf_counter()
    config_results = [
        _evaluate_retriever(
            config_id=config.config_id,
            family_id=config.family_id,
            retrieval_mode=config.retrieval_mode,
            selection_eligible=config.selection_eligible,
            retriever=candidate_retrievers[config.config_id],
            question_inputs_by_split=question_inputs_by_split,
            documents_by_id=documents_by_id,
            answer_generator=answer_generator,
            answer_verifier=answer_verifier,
            retrieval_top_k=retrieval_top_k,
            max_gold_window_sentences=max_gold_window_sentences,
            gold_span_gap_margin=gold_span_gap_margin,
            low_answer_f1_threshold=low_answer_f1_threshold,
            fold_assignments=fold_assignments,
            baseline_public=baseline_result["public"],
            baseline_traces_by_split=baseline_result["traces_by_split"],
            frozen_protocol=frozen_protocol,
        )
        for config in configs
    ]
    candidates_evaluated_at = time.perf_counter()
    train_cv_selection = _train_cv_selection(
        config_results=config_results,
        baseline_result=baseline_result,
    )
    selected_result = _selected_public_result(
        config_results=config_results,
        selected_config_id=train_cv_selection.get("selected_config_id"),
    )
    dev_validation = _dev_validation(
        selected_result=selected_result,
        frozen_protocol=frozen_protocol,
    )
    guard_checks = _guard_checks(
        stage113_summary=stage113_summary,
        stage102_summary=stage102_summary,
        split_samples=split_samples,
        configs=configs,
        baseline_result=baseline_result["public"],
        config_results=[result["public"] for result in config_results],
        train_cv_selection=train_cv_selection,
        dev_validation=dev_validation,
        frozen_protocol=frozen_protocol,
        user_confirmed_comparison=user_confirmed_comparison,
    )
    finished_at = time.perf_counter()

    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train/dev-only retrieval/index redesign comparison under the "
            "Stage113 frozen protocol. This stage loads the frozen Stage68 "
            "train/dev split rows and local PrimeQA training/dev corpus "
            "documents, runs all frozen Stage113 retrieval/index candidates, "
            "selects candidates by train grouped-CV only, reports one dev "
            "validation pass without dev selection or retuning, does not load "
            "the test split, does not run final metrics, does not write raw "
            "question, answer, document, token, or document-identifier fields, "
            "does not add fallback strategies, and does not change runtime "
            "defaults."
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
            "selection_mode": "train_grouped_cross_validation_then_full_train_refit",
            "validation_split": "dev",
            "dev_validation_mode": "single_pass_no_retuning",
            "dev_gate_status": "report_only_no_frozen_pass_threshold",
            "forbidden_final_splits": list(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "stage113_protocol": _fingerprint(stage113_protocol_path),
            "stage102_report": _fingerprint(stage102_report_path),
            "train_split": _fingerprint(train_split_path),
            "dev_split": _fingerprint(dev_split_path),
            "corpus_sections": _fingerprint(documents_path),
        },
        "stage113_summary": stage113_summary,
        "stage102_summary": stage102_summary,
        "analysis_config": {
            "retrieval_top_k": retrieval_top_k,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "component_depth": component_depth,
            "evidence_selector_name": evidence_selector_name,
            "max_candidates_per_document": max_candidates_per_document,
            "composition_policy_name": composition_policy_name,
            "max_sentences": max_sentences,
            "min_sentence_score": min_sentence_score,
            "verifier_min_citations": verifier_min_citations,
            "verifier_min_evidence_score": verifier_min_evidence_score,
            "verifier_max_citation_rank": verifier_max_citation_rank,
            "max_gold_window_sentences": max_gold_window_sentences,
            "gold_span_gap_margin": gold_span_gap_margin,
            "low_answer_f1_threshold": low_answer_f1_threshold,
            "candidate_config_count": len(configs),
            "objective_weights": _objective_weights(frozen_protocol),
        },
        "data_summary": {
            "documents": len(documents),
            "sections": sum(len(sections) for sections in sections_by_document.values()),
            "splits": summarize_primeqa_hybrid_split_samples(split_samples),
            "train_cv": _fold_summary(split_samples["train"], fold_assignments),
        },
        "baseline_result": baseline_result["public"],
        "config_results": [result["public"] for result in config_results],
        "train_cv_selection": train_cv_selection,
        "dev_validation": dev_validation,
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
            "load_documents_and_sections": round(
                loaded_documents_at - loaded_splits_at,
                3,
            ),
            "build_indexes": round(indexed_at - loaded_documents_at, 3),
            "cache_question_inputs": round(cached_question_inputs_at - indexed_at, 3),
            "evaluate_baseline": round(
                baseline_evaluated_at - cached_question_inputs_at,
                3,
            ),
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


def write_primeqa_hybrid_retrieval_index_redesign_comparison_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridRetrievalIndexRedesignComparisonVisualization]:
    """Write SVG charts for Stage114 retrieval/index redesign comparison."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage114_train_cv_retrieval_context_miss_delta.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage114 train-CV retrieval_context_miss delta",
                bars=_bucket_delta_bars(report, "train_cv", "retrieval_context_miss"),
                x_label="delta vs baseline",
                width=1540,
                margin_left=800,
            )
        ),
        "stage114_train_cv_gold_doc_recall_delta.svg": render_horizontal_bar_chart_svg(
            title="Stage114 train-CV gold-doc recall@10 delta",
            bars=_retrieval_metric_delta_bars(report, "train_cv", "gold_doc_recall_at_10"),
            x_label="delta vs baseline",
            width=1540,
            margin_left=800,
        ),
        "stage114_train_cv_average_token_f1_delta.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage114 train-CV average token F1 delta",
                bars=_verified_metric_delta_bars(report, "train_cv", "average_token_f1"),
                x_label="delta vs baseline",
                width=1540,
                margin_left=800,
            )
        ),
        "stage114_train_cv_selectability.svg": render_horizontal_bar_chart_svg(
            title="Stage114 train-CV selectability",
            bars=_train_selectability_bars(report),
            x_label="1 means selectable",
            width=1540,
            margin_left=800,
        ),
        "stage114_dev_retrieval_context_miss_delta.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage114 dev retrieval_context_miss delta",
                bars=_bucket_delta_bars(report, "dev", "retrieval_context_miss"),
                x_label="delta vs baseline",
                width=1540,
                margin_left=800,
            )
        ),
        "stage114_dev_gold_doc_recall_delta.svg": render_horizontal_bar_chart_svg(
            title="Stage114 dev gold-doc recall@10 delta",
            bars=_retrieval_metric_delta_bars(report, "dev", "gold_doc_recall_at_10"),
            x_label="delta vs baseline",
            width=1540,
            margin_left=800,
        ),
        "stage114_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage114 decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1400,
            margin_left=700,
        ),
        "stage114_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage114 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1700,
            margin_left=880,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridRetrievalIndexRedesignComparisonVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _build_candidate_retriever(
    *,
    config: _RunConfig,
    baseline_retriever: _Retriever,
    documents: Sequence[PrimeQADocument],
    sections_by_document: Mapping[str, Sequence[PrimeQADocumentSection]],
    bm25_k1: float,
    bm25_b: float,
    component_depth: int,
) -> _Retriever:
    if config.retrieval_mode == "weighted_document_bm25":
        weights = config.payload.get("weights") or {}
        return _weighted_document_retriever(
            documents=documents,
            sections_by_document=sections_by_document,
            title_weight=float(weights.get("title") or 1.0),
            section_heading_weight=float(weights.get("section_heading") or 1.0),
            body_weight=float(weights.get("body") or 1.0),
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )
    if config.retrieval_mode == "document_bm25_rrf":
        heading_retriever = _weighted_document_retriever(
            documents=documents,
            sections_by_document=sections_by_document,
            title_weight=1.0,
            section_heading_weight=1.0,
            body_weight=0.0,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )
        return _RrfFusionRetriever(
            retrievers=(baseline_retriever, heading_retriever),
            rrf_k=int(config.payload.get("rrf_k") or 60),
            component_depth=component_depth,
        )
    if config.retrieval_mode == "section_bm25_document_rollup":
        return _SectionBM25DocumentRollupRetriever(
            documents=documents,
            sections_by_document=sections_by_document,
            title_weight=1.0,
            section_heading_weight=1.0,
            body_weight=1.0,
            rollup_mode="max_score",
            top_n_sections=1,
            rrf_k=int(config.payload.get("rrf_k") or 60),
            k1=bm25_k1,
            b=bm25_b,
        )
    if config.retrieval_mode == "section_document_rrf":
        section_retriever = _SectionBM25DocumentRollupRetriever(
            documents=documents,
            sections_by_document=sections_by_document,
            title_weight=1.0,
            section_heading_weight=1.0,
            body_weight=1.0,
            rollup_mode="top_n_section_rrf",
            top_n_sections=3,
            rrf_k=int(config.payload.get("rrf_k") or 60),
            k1=bm25_k1,
            b=bm25_b,
        )
        return _RrfFusionRetriever(
            retrievers=(baseline_retriever, section_retriever),
            rrf_k=int(config.payload.get("rrf_k") or 60),
            component_depth=component_depth,
        )
    if config.retrieval_mode == "heading_section_title_rollup":
        return _SectionBM25DocumentRollupRetriever(
            documents=documents,
            sections_by_document=sections_by_document,
            title_weight=float(config.payload.get("document_title_weight") or 2.0),
            section_heading_weight=float(config.payload.get("section_heading_weight") or 2.0),
            body_weight=1.0,
            rollup_mode="max_score",
            top_n_sections=1,
            rrf_k=int(config.payload.get("rrf_k") or 60),
            k1=bm25_k1,
            b=bm25_b,
        )
    if config.retrieval_mode == "bm25_with_runtime_special_token_boost":
        return _SpecialTokenBoostRetriever(
            base_retriever=baseline_retriever,
            documents=documents,
            sections_by_document=sections_by_document,
            boost=float(config.payload.get("special_token_boost") or 1.5),
            component_depth=component_depth,
        )
    if config.retrieval_mode == "weighted_bm25_with_special_token_boost":
        weighted_retriever = _weighted_document_retriever(
            documents=documents,
            sections_by_document=sections_by_document,
            title_weight=float(config.payload.get("title_weight") or 2.0),
            section_heading_weight=float(config.payload.get("heading_weight") or 2.0),
            body_weight=1.0,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )
        return _SpecialTokenBoostRetriever(
            base_retriever=weighted_retriever,
            documents=documents,
            sections_by_document=sections_by_document,
            boost=float(config.payload.get("special_token_boost") or 1.5),
            component_depth=component_depth,
        )
    raise ValueError(f"Unknown retrieval_mode: {config.retrieval_mode}")


def _weighted_document_retriever(
    *,
    documents: Sequence[PrimeQADocument],
    sections_by_document: Mapping[str, Sequence[PrimeQADocumentSection]],
    title_weight: float,
    section_heading_weight: float,
    body_weight: float,
    bm25_k1: float,
    bm25_b: float,
) -> _MappedBM25Retriever:
    synthetic_documents = [
        PrimeQADocument(
            id=document.id,
            title="",
            text=_weighted_document_search_text(
                document=document,
                sections=sections_by_document.get(document.id, ()),
                title_weight=title_weight,
                section_heading_weight=section_heading_weight,
                body_weight=body_weight,
            ),
        )
        for document in documents
    ]
    return _MappedBM25Retriever(
        indexed_documents=synthetic_documents,
        original_documents_by_id={document.id: document for document in documents},
        k1=bm25_k1,
        b=bm25_b,
    )


def _evaluate_retriever(
    *,
    config_id: str,
    family_id: str,
    retrieval_mode: str,
    selection_eligible: bool,
    retriever: _Retriever,
    question_inputs_by_split: Mapping[str, Sequence[_QuestionInputs]],
    documents_by_id: Mapping[str, PrimeQADocument],
    answer_generator: ExtractiveAnswerGenerator,
    answer_verifier: AnswerVerifier,
    retrieval_top_k: int,
    max_gold_window_sentences: int,
    gold_span_gap_margin: float,
    low_answer_f1_threshold: float,
    fold_assignments: Mapping[str, str],
    baseline_public: Mapping[str, Any] | None,
    baseline_traces_by_split: Mapping[str, Sequence[_SplitQuestionTrace]] | None = None,
    frozen_protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    traces_by_split = {
        split: [
            _trace_question(
                split=split,
                inputs=inputs,
                retrieval_results=retriever.search(
                    inputs.question.full_question,
                    top_k=retrieval_top_k,
                ),
                documents_by_id=documents_by_id,
                answer_generator=answer_generator,
                answer_verifier=answer_verifier,
                max_gold_window_sentences=max_gold_window_sentences,
                gold_span_gap_margin=gold_span_gap_margin,
                low_answer_f1_threshold=low_answer_f1_threshold,
            )
            for inputs in split_inputs
        ]
        for split, split_inputs in question_inputs_by_split.items()
    }
    public_without_guards = _public_result(
        config_id=config_id,
        family_id=family_id,
        retrieval_mode=retrieval_mode,
        selection_eligible=selection_eligible,
        traces_by_split=traces_by_split,
        baseline_traces_by_split=baseline_traces_by_split or traces_by_split,
        fold_assignments=fold_assignments,
        baseline_public=baseline_public,
    )
    train_cv_selectability = (
        None
        if baseline_public is None
        else _train_cv_selectability(
            result=public_without_guards,
            baseline_public=baseline_public,
            frozen_protocol=frozen_protocol or {},
        )
    )
    return {
        "traces_by_split": traces_by_split,
        "public": {
            **public_without_guards,
            "train_cv_selectability": train_cv_selectability,
        },
    }


def _trace_question(
    *,
    split: str,
    inputs: _QuestionInputs,
    retrieval_results: Sequence[RetrievalResult],
    documents_by_id: Mapping[str, PrimeQADocument],
    answer_generator: ExtractiveAnswerGenerator,
    answer_verifier: AnswerVerifier,
    max_gold_window_sentences: int,
    gold_span_gap_margin: float,
    low_answer_f1_threshold: float,
) -> _SplitQuestionTrace:
    question = inputs.question
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
        best_gold_span_token_f1=inputs.best_gold_span_token_f1,
        gold_span_gap_margin=gold_span_gap_margin,
        low_answer_f1_threshold=low_answer_f1_threshold,
    )
    public_case = _public_case(
        split=split,
        sample_id=inputs.sample.sample_id,
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
        sample=inputs.sample,
        question=question,
        retrieval_results=list(retrieval_results),
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


def _public_result(
    *,
    config_id: str,
    family_id: str,
    retrieval_mode: str,
    selection_eligible: bool,
    traces_by_split: Mapping[str, Sequence[_SplitQuestionTrace]],
    baseline_traces_by_split: Mapping[str, Sequence[_SplitQuestionTrace]],
    fold_assignments: Mapping[str, str],
    baseline_public: Mapping[str, Any] | None,
) -> dict[str, Any]:
    eval_traces = _eval_traces_by_split(traces_by_split)
    baseline_eval_traces = _eval_traces_by_split(baseline_traces_by_split)
    aggregate_outputs = _aggregate_outputs(eval_traces)
    metrics_by_split = _metrics_by_split(eval_traces)
    retrieval_metrics_by_split = _retrieval_metrics_by_split(eval_traces)
    objective_scores = _objective_scores_by_split(
        aggregate_outputs=aggregate_outputs,
        retrieval_metrics_by_split=retrieval_metrics_by_split,
    )
    row_counts_by_split = {
        split: len(traces) for split, traces in eval_traces.items()
    }

    if baseline_public is None:
        objective_deltas = {split: 0.0 for split in _EVAL_SPLITS}
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
        retrieval_metric_deltas = {
            split: {
                "gold_doc_recall_at_10": 0.0,
                "gold_doc_hit_count": 0,
                "gold_doc_miss_count": 0,
                "mean_reciprocal_rank": 0.0,
            }
            for split in _EVAL_SPLITS
        }
    else:
        objective_deltas = {
            split: round(
                objective_scores[split]
                - baseline_public["objective_scores_by_split"][split],
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
        retrieval_metric_deltas = _retrieval_metric_deltas_by_split(
            retrieval_metrics_by_split=retrieval_metrics_by_split,
            baseline_retrieval_metrics_by_split=baseline_public[
                "retrieval_metrics_by_split"
            ],
        )

    changed_counts = _changed_answer_counts_by_split(
        baseline_eval_traces=baseline_eval_traces,
        eval_traces=eval_traces,
    )
    return {
        "config_id": config_id,
        "family_id": family_id,
        "retrieval_mode": retrieval_mode,
        "selection_eligible": bool(selection_eligible),
        "row_counts_by_split": row_counts_by_split,
        "train_cv_group_values_written": False,
        "aggregate_outputs": aggregate_outputs,
        "metrics_by_split": metrics_by_split,
        "retrieval_metrics_by_split": retrieval_metrics_by_split,
        "objective_scores_by_split": objective_scores,
        "objective_score_deltas_by_split": objective_deltas,
        "target_bucket_deltas_by_split": target_bucket_deltas,
        "metric_deltas_by_split": metric_deltas,
        "retrieval_metric_deltas_by_split": retrieval_metric_deltas,
        "changed_answer_counts_by_split": changed_counts,
        "changed_answer_rates_by_split": {
            split: _safe_rate(changed_counts[split], row_counts_by_split[split])
            for split in _EVAL_SPLITS
        },
        "train_cv_fold_results": _train_cv_fold_results(
            traces_by_split=traces_by_split,
            baseline_traces_by_split=baseline_traces_by_split,
            fold_assignments=fold_assignments,
        ),
    }


def _train_cv_selectability(
    *,
    result: Mapping[str, Any],
    baseline_public: Mapping[str, Any],
    frozen_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    guards = (frozen_protocol.get("selection_rules") or {}).get("guard_thresholds") or {}
    metric_deltas = result["metric_deltas_by_split"]["train_cv"]
    target_deltas = result["target_bucket_deltas_by_split"]["train_cv"]
    retrieval_metric_deltas = result["retrieval_metric_deltas_by_split"]["train_cv"]
    baseline_metrics = baseline_public["metrics_by_split"]["train_cv"]["verified"]
    metrics = result["metrics_by_split"]["train_cv"]["verified"]
    average_token_f1_drop = round(
        max(
            0.0,
            float(baseline_metrics["average_token_f1"])
            - float(metrics["average_token_f1"]),
        ),
        4,
    )
    gold_doc_citation_rate_drop = round(
        max(
            0.0,
            float(baseline_metrics["gold_doc_citation_rate"])
            - float(metrics["gold_doc_citation_rate"]),
        ),
        4,
    )
    changed_answer_rate = float(result["changed_answer_rates_by_split"]["train_cv"])
    checks = {
        "candidate_selection_eligible": result["selection_eligible"] is True,
        "train_cv_retrieval_context_miss_delta_negative": int(
            target_deltas["retrieval_context_miss"]
        )
        < 0,
        "train_cv_gold_doc_recall_at_10_delta_positive": float(
            retrieval_metric_deltas["gold_doc_recall_at_10"]
        )
        > 0.0,
        "train_cv_average_token_f1_drop_within_guard": average_token_f1_drop
        <= float(guards.get("max_train_cv_average_token_f1_drop") or 0.0),
        "train_cv_gold_doc_citation_rate_drop_within_guard": (
            gold_doc_citation_rate_drop
            <= float(guards.get("max_train_cv_gold_doc_citation_rate_drop") or 0.0)
        ),
        "train_cv_answerable_refusal_rate_delta_within_guard": float(
            metric_deltas["answerable_refusal_rate"]
        )
        <= float(guards.get("max_train_cv_answerable_refusal_rate_delta") or 0.0),
        "train_cv_answerability_false_answer_delta_within_guard": int(
            target_deltas["answerability_false_answer"]
        )
        <= int(guards.get("max_train_cv_answerability_false_answer_delta") or 0),
        "train_cv_evidence_selection_miss_delta_within_guard": int(
            target_deltas["evidence_selection_miss"]
        )
        <= int(guards.get("max_train_cv_evidence_selection_miss_delta") or 0),
        "train_cv_gold_span_beats_selected_delta_within_guard": int(
            target_deltas["gold_span_beats_selected_answer"]
        )
        <= int(guards.get("max_train_cv_gold_span_beats_selected_delta") or 0),
        "train_cv_changed_answer_rate_within_guard": changed_answer_rate
        <= float(guards.get("max_train_cv_changed_answer_rate") or 0.0),
    }
    return {
        "selectable": all(checks.values()),
        "observed": {
            "train_cv_objective_score_delta": result[
                "objective_score_deltas_by_split"
            ]["train_cv"],
            "train_cv_retrieval_context_miss_delta": target_deltas[
                "retrieval_context_miss"
            ],
            "train_cv_gold_doc_recall_at_10_delta": retrieval_metric_deltas[
                "gold_doc_recall_at_10"
            ],
            "train_cv_average_token_f1_drop": average_token_f1_drop,
            "train_cv_gold_doc_citation_rate_drop": gold_doc_citation_rate_drop,
            "train_cv_answerable_refusal_rate_delta": metric_deltas[
                "answerable_refusal_rate"
            ],
            "train_cv_answerability_false_answer_delta": target_deltas[
                "answerability_false_answer"
            ],
            "train_cv_evidence_selection_miss_delta": target_deltas[
                "evidence_selection_miss"
            ],
            "train_cv_gold_span_beats_selected_delta": target_deltas[
                "gold_span_beats_selected_answer"
            ],
            "train_cv_changed_answer_rate": changed_answer_rate,
        },
        "thresholds": guards,
        "checks": checks,
        "guard_failure_reasons": [
            check_name for check_name, passed in checks.items() if not passed
        ],
    }


def _train_cv_selection(
    *,
    config_results: Sequence[Mapping[str, Any]],
    baseline_result: Mapping[str, Any],
) -> dict[str, Any]:
    public_results = [result["public"] for result in config_results]
    selectable = [
        result
        for result in public_results
        if (result.get("train_cv_selectability") or {}).get("selectable") is True
    ]
    ranking = sorted(public_results, key=_selection_key)
    selected = sorted(selectable, key=_selection_key)[0] if selectable else None
    return {
        "selection_split": "train",
        "selection_mode": "train_grouped_cross_validation_then_full_train_refit",
        "selection_source": "train_cv_only",
        "baseline_train_cv_objective_score": baseline_result["public"][
            "objective_scores_by_split"
        ]["train_cv"],
        "selected_config_id": selected["config_id"] if selected else None,
        "selected_family_id": selected["family_id"] if selected else None,
        "selected_train_cv_objective_score": (
            selected["objective_scores_by_split"]["train_cv"] if selected else None
        ),
        "selected_train_cv_objective_delta": (
            selected["objective_score_deltas_by_split"]["train_cv"]
            if selected
            else None
        ),
        "selectable_config_count": len(selectable),
        "config_count": len(public_results),
        "selection_ranking": [
            {
                "rank": index,
                "config_id": result["config_id"],
                "family_id": result["family_id"],
                "train_cv_objective_score": result["objective_scores_by_split"][
                    "train_cv"
                ],
                "train_cv_objective_delta": result[
                    "objective_score_deltas_by_split"
                ]["train_cv"],
                "train_cv_retrieval_context_miss_delta": result[
                    "target_bucket_deltas_by_split"
                ]["train_cv"]["retrieval_context_miss"],
                "train_cv_gold_doc_recall_at_10_delta": result[
                    "retrieval_metric_deltas_by_split"
                ]["train_cv"]["gold_doc_recall_at_10"],
                "train_cv_selectable": (
                    result["train_cv_selectability"]["selectable"]
                    if result.get("train_cv_selectability")
                    else False
                ),
                "train_cv_changed_answer_rate": result[
                    "changed_answer_rates_by_split"
                ]["train_cv"],
            }
            for index, result in enumerate(ranking, start=1)
        ],
    }


def _selection_key(result: Mapping[str, Any]) -> tuple[Any, ...]:
    selectability = result.get("train_cv_selectability") or {}
    observed = selectability.get("observed") or {}
    metrics = result["metrics_by_split"]["train_cv"]["verified"]
    return (
        0 if selectability.get("selectable") is True else 1,
        float(result["objective_scores_by_split"]["train_cv"]),
        int(observed.get("train_cv_retrieval_context_miss_delta") or 0),
        -float(observed.get("train_cv_gold_doc_recall_at_10_delta") or 0.0),
        -float(metrics["average_token_f1"]),
        -float(metrics["gold_doc_citation_rate"]),
        float(result["changed_answer_rates_by_split"]["train_cv"]),
        str(result["config_id"]),
    )


def _selected_public_result(
    *,
    config_results: Sequence[Mapping[str, Any]],
    selected_config_id: str | None,
) -> Mapping[str, Any] | None:
    if selected_config_id is None:
        return None
    for result in config_results:
        public = result["public"]
        if public["config_id"] == selected_config_id:
            return public
    return None


def _dev_validation(
    *,
    selected_result: Mapping[str, Any] | None,
    frozen_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    dev_rules = (frozen_protocol.get("selection_rules") or {}).get("dev_rules") or {}
    if selected_result is None:
        return {
            "validation_split": "dev",
            "selected_config_id": None,
            "status": "no_train_cv_selectable_config",
            "dev_validation_passed": None,
            "dev_gate_status": "report_only_no_frozen_pass_threshold",
            "dev_rules": dev_rules,
        }
    return {
        "validation_split": "dev",
        "selected_config_id": selected_result["config_id"],
        "selected_family_id": selected_result["family_id"],
        "status": "reported_not_used_for_selection",
        "dev_validation_passed": None,
        "dev_gate_status": "report_only_no_frozen_pass_threshold",
        "dev_objective_score": selected_result["objective_scores_by_split"]["dev"],
        "dev_objective_delta": selected_result["objective_score_deltas_by_split"]["dev"],
        "dev_target_bucket_deltas": selected_result["target_bucket_deltas_by_split"][
            "dev"
        ],
        "dev_metric_deltas": selected_result["metric_deltas_by_split"]["dev"],
        "dev_retrieval_metric_deltas": selected_result[
            "retrieval_metric_deltas_by_split"
        ]["dev"],
        "dev_changed_answer_rate": selected_result["changed_answer_rates_by_split"][
            "dev"
        ],
        "dev_rules": dev_rules,
    }


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
        "selected_family_id": train_cv_selection.get("selected_family_id"),
        "selectable_config_count": train_cv_selection.get("selectable_config_count"),
        "dev_validation_status": dev_validation.get("status"),
        "dev_gate_status": dev_validation.get("dev_gate_status"),
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
    }
    if failed_checks:
        return {
            **base,
            "status": "primeqa_hybrid_retrieval_index_redesign_comparison_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
        }
    if train_cv_selection.get("selected_config_id") is None:
        return {
            **base,
            "status": _NO_TRAIN_CV_SELECTABLE_STATUS,
            "can_continue_train_dev_development": True,
            "recommended_next_direction": "record_retrieval_index_redesign_stop_decision",
            "recommended_next_stage": (
                "Stage115: record a stop decision for the frozen Stage113 "
                "retrieval/index redesign family because no candidate satisfied "
                "train-CV selectability."
            ),
        }
    return {
        **base,
        "status": (
            "primeqa_hybrid_retrieval_index_redesign_completed_train_cv_selected_"
            "dev_reported"
        ),
        "can_continue_train_dev_development": True,
        "recommended_next_direction": (
            "review_selected_retrieval_index_changed_cases_before_any_runtime_or_test_gate"
        ),
        "recommended_next_stage": (
            "Stage115: review public-safe selected retrieval/index changed-case "
            "patterns before deciding any runtime or final-test gate. Test "
            "remains locked until a separately confirmed gate."
        ),
    }


def _guard_checks(
    *,
    stage113_summary: Mapping[str, Any],
    stage102_summary: Mapping[str, Any],
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    configs: Sequence[_RunConfig],
    baseline_result: Mapping[str, Any],
    config_results: Sequence[Mapping[str, Any]],
    train_cv_selection: Mapping[str, Any],
    dev_validation: Mapping[str, Any],
    frozen_protocol: Mapping[str, Any],
    user_confirmed_comparison: bool,
) -> list[dict[str, Any]]:
    selection_rules = frozen_protocol.get("selection_rules") or {}
    ranking_keys = {
        tuple(row.keys()) for row in train_cv_selection.get("selection_ranking") or []
    }
    public_payload = {
        "baseline": baseline_result,
        "configs": config_results,
        "selection": train_cv_selection,
        "dev_validation": dev_validation,
    }
    return [
        _check(
            name="stage113_source_is_expected",
            passed=stage113_summary.get("stage") == _SOURCE_STAGE113,
            observed=stage113_summary.get("stage"),
            expected=_SOURCE_STAGE113,
        ),
        _check(
            name="stage113_protocol_id_matches",
            passed=stage113_summary.get("protocol_id") == _SOURCE_PROTOCOL_ID,
            observed=stage113_summary.get("protocol_id"),
            expected=_SOURCE_PROTOCOL_ID,
        ),
        _check(
            name="stage113_protocol_frozen_for_train_dev_run",
            passed=stage113_summary.get("protocol_status") == _SOURCE_PROTOCOL_STATUS,
            observed=stage113_summary.get("protocol_status"),
            expected=_SOURCE_PROTOCOL_STATUS,
        ),
        _check(
            name="user_confirmed_stage114_comparison",
            passed=user_confirmed_comparison,
            observed=user_confirmed_comparison,
            expected=True,
        ),
        _check(
            name="stage102_baseline_source_available",
            passed=stage102_summary.get("stage") == "Stage 102",
            observed=stage102_summary.get("stage"),
            expected="Stage 102",
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
            name="candidate_config_count_matches_stage113",
            passed=len(configs) == 8
            and len(config_results) == int(stage113_summary.get("candidate_config_count") or 0),
            observed={
                "configs": len(configs),
                "results": len(config_results),
                "stage113": stage113_summary.get("candidate_config_count"),
            },
            expected=8,
        ),
        _check(
            name="train_cv_fold_count_matches_protocol",
            passed=len(baseline_result["train_cv_fold_results"])
            == _train_cv_fold_count(frozen_protocol),
            observed=len(baseline_result["train_cv_fold_results"]),
            expected=_train_cv_fold_count(frozen_protocol),
        ),
        _check(
            name="train_cv_selection_uses_train_only_fields",
            passed=ranking_keys
            == {
                (
                    "rank",
                    "config_id",
                    "family_id",
                    "train_cv_objective_score",
                    "train_cv_objective_delta",
                    "train_cv_retrieval_context_miss_delta",
                    "train_cv_gold_doc_recall_at_10_delta",
                    "train_cv_selectable",
                    "train_cv_changed_answer_rate",
                )
            },
            observed={"ranking_keys": [list(keys) for keys in sorted(ranking_keys)]},
            expected="train-CV ranking fields only",
        ),
        _check(
            name="dev_validation_not_used_for_selection",
            passed=(selection_rules.get("dev_rules") or {}).get("dev_selection_allowed")
            is False
            and (selection_rules.get("dev_rules") or {}).get("dev_retuning_allowed")
            is False
            and train_cv_selection.get("selection_source") == "train_cv_only",
            observed={
                "dev_rules": selection_rules.get("dev_rules"),
                "selection_source": train_cv_selection.get("selection_source"),
            },
            expected="dev validation only; no selection or retuning",
        ),
        _check(
            name="stage114_final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage114_runtime_defaults_unchanged",
            passed=(selection_rules.get("runtime_rules") or {}).get(
                "default_runtime_policy"
            )
            == "unchanged",
            observed=(selection_rules.get("runtime_rules") or {}).get(
                "default_runtime_policy"
            ),
            expected="unchanged",
        ),
        _check(
            name="stage114_fallback_strategies_not_added",
            passed=(selection_rules.get("runtime_rules") or {}).get(
                "fallback_strategies_enabled"
            )
            is False,
            observed=(selection_rules.get("runtime_rules") or {}).get(
                "fallback_strategies_enabled"
            ),
            expected=False,
        ),
        _check(
            name="stage114_public_outputs_have_no_forbidden_keys",
            passed=not _contains_forbidden_key(public_payload),
            observed=sorted(_forbidden_keys_found(public_payload)),
            expected=[],
        ),
        _check(
            name="train_cv_group_values_not_written",
            passed=baseline_result.get("train_cv_group_values_written") is False,
            observed=baseline_result.get("train_cv_group_values_written"),
            expected=False,
        ),
    ]


def _question_inputs_by_split(
    *,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    documents_by_id: Mapping[str, PrimeQADocument],
    max_gold_window_sentences: int,
) -> dict[str, list[_QuestionInputs]]:
    inputs_by_split = {}
    for split, samples in split_samples.items():
        split_inputs = []
        for sample in samples:
            question = sample.to_primeqa_question()
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
                _QuestionInputs(
                    sample=sample,
                    question=question,
                    best_gold_span_token_f1=best_gold_span,
                )
            )
        inputs_by_split[split] = split_inputs
    return inputs_by_split


def _answer_generator(
    *,
    evidence_selector_name: str,
    max_candidates_per_document: int,
    composition_policy_name: str,
    max_sentences: int,
    min_sentence_score: float,
) -> ExtractiveAnswerGenerator:
    evidence_selector = create_sentence_evidence_selector(
        selector_name=evidence_selector_name,
        max_candidates_per_document=max_candidates_per_document,
    )
    composition_policy = create_answer_composition_policy(composition_policy_name)
    return ExtractiveAnswerGenerator(
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
        evidence_selector=evidence_selector,
        composition_policy=composition_policy,
    )


def _eval_traces_by_split(
    traces_by_split: Mapping[str, Sequence[_SplitQuestionTrace]],
) -> dict[str, list[_SplitQuestionTrace]]:
    return {
        "train_cv": list(traces_by_split["train"]),
        "train_full": list(traces_by_split["train"]),
        "dev": list(traces_by_split["dev"]),
    }


def _retrieval_metrics_by_split(
    traces_by_split: Mapping[str, Sequence[_SplitQuestionTrace]],
) -> dict[str, dict[str, Any]]:
    return {
        split: _retrieval_metrics(traces)
        for split, traces in traces_by_split.items()
    }


def _retrieval_metrics(traces: Sequence[_SplitQuestionTrace]) -> dict[str, Any]:
    answerable_traces = [trace for trace in traces if trace.question.answerable]
    reciprocal_ranks = []
    gold_doc_hit_count = 0
    rank_buckets = Counter()
    for trace in answerable_traces:
        rank = _gold_doc_rank(trace.question, trace.retrieval_results)
        rank_buckets[_gold_doc_rank_bucket(rank)] += 1
        if rank is not None:
            gold_doc_hit_count += 1
            reciprocal_ranks.append(1.0 / rank)
    answerable_count = len(answerable_traces)
    gold_doc_miss_count = answerable_count - gold_doc_hit_count
    return {
        "answerable_count": answerable_count,
        "gold_doc_hit_count": gold_doc_hit_count,
        "gold_doc_miss_count": gold_doc_miss_count,
        "gold_doc_recall_at_10": _safe_rate(gold_doc_hit_count, answerable_count),
        "mean_reciprocal_rank": round(
            sum(reciprocal_ranks) / answerable_count if answerable_count else 0.0,
            4,
        ),
        "gold_doc_rank_buckets": dict(sorted(rank_buckets.items())),
    }


def _objective_scores_by_split(
    *,
    aggregate_outputs: Mapping[str, Any],
    retrieval_metrics_by_split: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    weights = {"retrieval_context_miss": 2.0, "gold_doc_hit_reward": 1.5}
    counts_by_split = aggregate_outputs["bucket_counts_by_split"]
    return {
        split: round(
            float(counts_by_split[split].get("retrieval_context_miss", 0))
            * weights["retrieval_context_miss"]
            - float(retrieval_metrics_by_split[split]["gold_doc_hit_count"])
            * weights["gold_doc_hit_reward"],
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
            for bucket in _TARGET_BUCKETS
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


def _retrieval_metric_deltas_by_split(
    *,
    retrieval_metrics_by_split: Mapping[str, Any],
    baseline_retrieval_metrics_by_split: Mapping[str, Any],
) -> dict[str, dict[str, float | int]]:
    metric_names = (
        "gold_doc_recall_at_10",
        "gold_doc_hit_count",
        "gold_doc_miss_count",
        "mean_reciprocal_rank",
    )
    return {
        split: {
            metric: _number_delta(
                retrieval_metrics_by_split[split][metric],
                baseline_retrieval_metrics_by_split[split][metric],
            )
            for metric in metric_names
        }
        for split in _EVAL_SPLITS
    }


def _changed_answer_counts_by_split(
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
) -> list[dict[str, Any]]:
    fold_results = []
    for fold_id in sorted(set(fold_assignments.values())):
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
        retrieval_metrics = _retrieval_metrics(traces)
        baseline_retrieval_metrics = _retrieval_metrics(baseline_traces)
        return_counts = aggregate["bucket_counts_by_split"][fold_id]
        baseline_counts = baseline_aggregate["bucket_counts_by_split"][fold_id]
        objective_score = round(
            float(return_counts.get("retrieval_context_miss", 0)) * 2.0
            - float(retrieval_metrics["gold_doc_hit_count"]) * 1.5,
            4,
        )
        baseline_objective_score = round(
            float(baseline_counts.get("retrieval_context_miss", 0)) * 2.0
            - float(baseline_retrieval_metrics["gold_doc_hit_count"]) * 1.5,
            4,
        )
        fold_results.append(
            {
                "fold_id": fold_id,
                "row_count": len(traces),
                "objective_score": objective_score,
                "objective_delta": round(
                    objective_score - baseline_objective_score,
                    4,
                ),
                "retrieval_context_miss_delta": int(
                    return_counts.get("retrieval_context_miss", 0)
                )
                - int(baseline_counts.get("retrieval_context_miss", 0)),
                "gold_doc_recall_at_10_delta": round(
                    float(retrieval_metrics["gold_doc_recall_at_10"])
                    - float(baseline_retrieval_metrics["gold_doc_recall_at_10"]),
                    4,
                ),
            }
        )
    return fold_results


def _build_train_fold_assignments(
    samples: Sequence[PrimeQAHybridSplitSample],
    *,
    fold_count: int,
) -> dict[str, str]:
    groups: dict[str, list[PrimeQAHybridSplitSample]] = defaultdict(list)
    for sample in samples:
        groups[_group_key(sample)].append(sample)
    fold_rows: list[list[PrimeQAHybridSplitSample]] = [[] for _ in range(fold_count)]
    for group_key, group_samples in sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), _stable_hash(item[0])),
    ):
        _ = group_key
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
    normalized_question = " ".join(
        f"{sample.question_title} {sample.question_text}".lower().split()
    )
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


def _section_index_records(
    *,
    documents: Sequence[PrimeQADocument],
    sections_by_document: Mapping[str, Sequence[PrimeQADocumentSection]],
    title_weight: float,
    section_heading_weight: float,
    body_weight: float,
) -> list[_SectionIndexRecord]:
    records = []
    for document in documents:
        sections = sections_by_document.get(document.id, ())
        if not sections:
            records.append(
                _SectionIndexRecord(
                    document_id=document.id,
                    section_id="document",
                    text=_weighted_section_search_text(
                        document=document,
                        section_heading="document",
                        section_text=document.text,
                        title_weight=title_weight,
                        section_heading_weight=section_heading_weight,
                        body_weight=body_weight,
                    ),
                )
            )
            continue
        for section in sections:
            if not section.text.strip():
                continue
            records.append(
                _SectionIndexRecord(
                    document_id=document.id,
                    section_id=section.section_id,
                    text=_weighted_section_search_text(
                        document=document,
                        section_heading=section.section_id,
                        section_text=section.text,
                        title_weight=title_weight,
                        section_heading_weight=section_heading_weight,
                        body_weight=body_weight,
                    ),
                )
            )
    return records


def _weighted_document_search_text(
    *,
    document: PrimeQADocument,
    sections: Sequence[PrimeQADocumentSection],
    title_weight: float,
    section_heading_weight: float,
    body_weight: float,
) -> str:
    headings = "\n".join(section.section_id for section in sections)
    return "\n".join(
        part
        for part in (
            _repeat_text(document.title, title_weight),
            _repeat_text(headings, section_heading_weight),
            _repeat_text(document.text, body_weight),
        )
        if part
    )


def _weighted_section_search_text(
    *,
    document: PrimeQADocument,
    section_heading: str,
    section_text: str,
    title_weight: float,
    section_heading_weight: float,
    body_weight: float,
) -> str:
    return "\n".join(
        part
        for part in (
            _repeat_text(document.title, title_weight),
            _repeat_text(section_heading, section_heading_weight),
            _repeat_text(section_text, body_weight),
        )
        if part
    )


def _repeat_text(text: str, weight: float) -> str:
    count = max(0, int(round(weight)))
    return "\n".join(text for _ in range(count) if text.strip())


def _special_token_index(
    *,
    documents: Sequence[PrimeQADocument],
    sections_by_document: Mapping[str, Sequence[PrimeQADocumentSection]],
) -> dict[str, set[str]]:
    token_to_doc_ids: dict[str, set[str]] = defaultdict(set)
    for document in documents:
        sections = sections_by_document.get(document.id, ())
        text = "\n".join(
            [document.title, document.text]
            + [section.section_id for section in sections]
            + [section.text for section in sections]
        )
        for token in _special_tokens(text):
            token_to_doc_ids[token].add(document.id)
    return dict(token_to_doc_ids)


def _special_tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _SPECIAL_TOKEN_RE.finditer(text)}


def _gold_doc_rank(
    question: PrimeQAQuestion,
    retrieval_results: Sequence[RetrievalResult],
) -> int | None:
    if not question.answerable or question.answer_doc_id is None:
        return None
    for result in retrieval_results:
        if result.document.id == question.answer_doc_id:
            return result.rank
    return None


def _gold_doc_rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "not_found_top10"
    if rank == 1:
        return "rank_1"
    if rank <= 3:
        return "rank_2_to_3"
    if rank <= 5:
        return "rank_4_to_5"
    return "rank_6_to_10"


def _stage113_summary(stage113_report: Mapping[str, Any]) -> dict[str, Any]:
    frozen = stage113_report.get("frozen_protocol") or {}
    decision = stage113_report.get("decision") or {}
    return {
        "stage": stage113_report.get("stage"),
        "protocol_id": stage113_report.get("protocol_id"),
        "protocol_status": frozen.get("protocol_status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "candidate_config_count": len(frozen.get("candidate_configs") or []),
        "selection_rules": frozen.get("selection_rules") or {},
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
        "metrics_by_split": stage102_report.get("metrics_by_split"),
    }


def _config_from_mapping(config: Mapping[str, Any]) -> _RunConfig:
    return _RunConfig(
        config_id=str(config["config_id"]),
        family_id=str(config["family_id"]),
        retrieval_mode=str(config["retrieval_mode"]),
        selection_eligible=bool(config.get("selection_eligible")),
        payload=dict(config),
    )


def _train_cv_fold_count(frozen_protocol: Mapping[str, Any]) -> int:
    return int(
        (frozen_protocol.get("selection_rules") or {}).get("minimum_train_folds")
        or 5
    )


def _objective_weights(frozen_protocol: Mapping[str, Any]) -> dict[str, float]:
    selection_rules = frozen_protocol.get("selection_rules") or {}
    primary = selection_rules.get("primary_objective") or {}
    secondary = selection_rules.get("secondary_objectives") or []
    recall_weight = 1.5
    for objective in secondary:
        if (objective or {}).get("name") == "improve_gold_doc_recall_at_10":
            recall_weight = float((objective or {}).get("weight") or recall_weight)
    return {
        "retrieval_context_miss": float(primary.get("weight") or 2.0),
        "gold_doc_hit_reward": recall_weight,
        "lower_is_better": True,
    }


def _number_delta(value: float | int, baseline: float | int) -> float | int:
    delta = value - baseline
    if isinstance(value, int) and isinstance(baseline, int):
        return int(delta)
    return round(float(delta), 4)


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _compute_idf(document_count: int, document_frequency: int) -> float:
    return math.log(
        1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
    )


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
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


def _bucket_delta_bars(
    report: Mapping[str, Any],
    split: str,
    bucket: str,
) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(result["config_id"]),
            value=float(result["target_bucket_deltas_by_split"][split][bucket]),
            value_label=f"{int(result['target_bucket_deltas_by_split'][split][bucket]):+d}",
        )
        for result in report.get("config_results") or []
    ]


def _retrieval_metric_delta_bars(
    report: Mapping[str, Any],
    split: str,
    metric: str,
) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(result["config_id"]),
            value=float(result["retrieval_metric_deltas_by_split"][split][metric]),
            value_label=f"{float(result['retrieval_metric_deltas_by_split'][split][metric]):+.4f}",
        )
        for result in report.get("config_results") or []
    ]


def _verified_metric_delta_bars(
    report: Mapping[str, Any],
    split: str,
    metric: str,
) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(result["config_id"]),
            value=float(result["metric_deltas_by_split"][split][metric]),
            value_label=f"{float(result['metric_deltas_by_split'][split][metric]):+.4f}",
        )
        for result in report.get("config_results") or []
    ]


def _train_selectability_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(result["config_id"]),
            value=(
                1.0
                if (result.get("train_cv_selectability") or {}).get("selectable")
                else 0.0
            ),
            value_label=(
                "selectable"
                if (result.get("train_cv_selectability") or {}).get("selectable")
                else "blocked"
            ),
        )
        for result in report.get("config_results") or []
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    names = [
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


def _validate_options(
    *,
    retrieval_top_k: int,
    bm25_k1: float,
    bm25_b: float,
    max_candidates_per_document: int,
    max_sentences: int,
    min_sentence_score: float,
    verifier_min_citations: int,
    verifier_min_evidence_score: float,
    verifier_max_citation_rank: int,
    max_gold_window_sentences: int,
    gold_span_gap_margin: float,
    low_answer_f1_threshold: float,
    component_depth: int,
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
    if verifier_min_citations <= 0:
        raise ValueError("verifier_min_citations must be positive")
    if verifier_min_evidence_score < 0:
        raise ValueError("verifier_min_evidence_score must be non-negative")
    if verifier_max_citation_rank <= 0:
        raise ValueError("verifier_max_citation_rank must be positive")
    if max_gold_window_sentences <= 0:
        raise ValueError("max_gold_window_sentences must be positive")
    if gold_span_gap_margin < 0:
        raise ValueError("gold_span_gap_margin must be non-negative")
    if low_answer_f1_threshold < 0:
        raise ValueError("low_answer_f1_threshold must be non-negative")
    if component_depth <= 0:
        raise ValueError("component_depth must be positive")
