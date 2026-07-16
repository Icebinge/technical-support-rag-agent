from __future__ import annotations

import hashlib
import math
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.answer_composition import create_answer_composition_policy
from ts_rag_agent.application.answer_verification import AnswerVerifier
from ts_rag_agent.application.evidence_selection import create_sentence_evidence_selector
from ts_rag_agent.application.primeqa_hybrid_first_stage_recall_expansion_validation import (
    _baseline_outcomes,
    _channels_for_route_set,
    _evaluation_channels,
    _EvaluationChannel,
    _rank_pool,
    _result_cache,
    _results_for_channels,
)
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    EncoderFactory,
    _build_dense_channels,
    _build_lexical_channels,
    _build_train_fold_assignments,
    _fingerprint,
    _load_json_object,
    _rounded_mean,
    _rounded_percentile,
    _rounded_ratio,
    _section_summary,
)
from ts_rag_agent.application.primeqa_hybrid_prefix_preserving_recall_expansion_validation import (
    _candidate_configs_from_protocol,
    _prefix_identity_violation_count,
    _prefix_preserving_pool,
)
from ts_rag_agent.application.rag_answering import ExtractiveAnswerGenerator, evaluate_answers
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.answer import GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
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

_STAGE = "Stage 129"
_CREATED_AT = "2026-07-16"
_ANALYSIS_ID = "primeqa_hybrid_agent_retrieval_integration_validation_v1"
_SOURCE_STAGE128_STATUS = "primeqa_hybrid_agent_retrieval_integration_protocol_frozen"
_SOURCE_STAGE128_PROTOCOL_ID = "primeqa_hybrid_agent_retrieval_integration_protocol_v1"
_SOURCE_STAGE128_NEXT = "run_agent_retrieval_integration_train_cv_dev_validation"
_SOURCE_STAGE125_STATUS = (
    "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_protocol_frozen"
)
_SELECTED_CONFIG_ID = "prefix_existing_dense_broad_append200_v1"
_SELECTED_FAMILY_ID = "stage116_prefix_existing_dense_append_family_v1"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_TRAIN_SPLIT = "train"
_DEV_SPLIT = "dev"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_FORBIDDEN_FINAL_SPLITS = ("test",)
_BASELINE_PROFILE_ID = "stage102_bm25_top10_verified_baseline"
_STAGE116_PROFILE_ID = "stage116_top200_agent_pool_control"
_STAGE128_PROFILE_ID = "stage128_prefix_append_top400_agent_pool"
_BASELINE_PREFIX_DEPTH = 200
_TARGET_POOL_DEPTH = 400
_APPEND_BUDGET = 200
_DEFAULT_TRAIN_FOLD_COUNT = 5
_DEFAULT_BM25_K1 = 1.5
_DEFAULT_BM25_B = 0.75
_DEFAULT_ENCODER_BATCH_SIZE = 64
_DEFAULT_EVIDENCE_SELECTOR = "bm25-sentence"
_DEFAULT_MAX_CANDIDATES_PER_DOCUMENT = 3
_DEFAULT_COMPOSITION_POLICY = "top-k"
_DEFAULT_MAX_SENTENCES = 3
_DEFAULT_MIN_SENTENCE_SCORE = 2.0
_DEFAULT_MIN_EVIDENCE_SCORE = 7.0
_BASELINE_MAX_CITATION_RANK = 3
_AGENT_EVIDENCE_CONTEXT_DEPTH = 10
_SHORTLIST_DOCUMENT_TEXT_MAX_CHARS = 5000
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer",
        "answer_doc_id",
        "answer_text",
        "candidate_doc_ids",
        "cited_doc_ids",
        "document_body",
        "document_id",
        "document_text",
        "document_title",
        "gold_answer",
        "matched_token_strings",
        "question_id",
        "question_text",
        "question_title",
        "raw_answer_text",
        "raw_document_text",
        "raw_question_text",
        "retrieved_doc_ids",
        "source_doc_ids",
    }
)


@dataclass(frozen=True)
class _ProfileConfig:
    profile_id: str
    family_id: str
    retrieval_mode: str
    retrieval_depth: int
    answer_context_depth: int
    verifier_max_citation_rank: int
    is_stage128_candidate: bool = False
    is_stage116_control: bool = False


@dataclass(frozen=True)
class _QuestionTrace:
    sample: PrimeQAHybridSplitSample
    question: PrimeQAQuestion
    retrieval_results: list[RetrievalResult]
    answer_context_results: list[RetrievalResult]
    original_answer: GeneratedAnswer
    verified_answer: GeneratedAnswer
    verification_reasons: tuple[str, ...]


class _DocumentEvidenceShortlister:
    """Cheap runtime-visible document preselector for large candidate pools."""

    def __init__(self, max_text_chars: int = _SHORTLIST_DOCUMENT_TEXT_MAX_CHARS) -> None:
        if max_text_chars <= 0:
            raise ValueError("max_text_chars must be positive")
        self._max_text_chars = max_text_chars
        self._term_cache: dict[str, set[str]] = {}

    def shortlist(
        self,
        *,
        question: PrimeQAQuestion,
        candidates: Sequence[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_terms = set(tokenize_text(question.full_question))
        if not query_terms:
            return list(candidates[:top_k])
        ranked = sorted(
            candidates,
            key=lambda result: (
                -self._score(query_terms=query_terms, result=result),
                result.rank,
                result.document.id,
            ),
        )
        return ranked[:top_k]

    def _score(self, *, query_terms: set[str], result: RetrievalResult) -> float:
        document_terms = self._document_terms(result.document)
        overlap_count = len(query_terms & document_terms)
        overlap_ratio = overlap_count / max(1, len(query_terms))
        retrieval_prior = 1.0 / math.log2(result.rank + 1)
        return overlap_count + overlap_ratio + 0.35 * retrieval_prior

    def _document_terms(self, document: PrimeQADocument) -> set[str]:
        if document.id not in self._term_cache:
            text = f"{document.title}\n{document.text[: self._max_text_chars]}"
            self._term_cache[document.id] = set(tokenize_text(text))
        return self._term_cache[document.id]


@dataclass(frozen=True)
class PrimeQAHybridAgentRetrievalIntegrationValidationVisualization:
    """One generated Stage129 agent retrieval integration validation chart."""

    name: str
    path: str


def run_primeqa_hybrid_agent_retrieval_integration_validation(
    *,
    stage128_protocol_path: Path,
    stage125_protocol_path: Path,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    user_confirmed_validation: bool,
    confirmation_note: str,
    stage80_report_path: Path | None = None,
    include_dense_channels: bool = True,
    bm25_k1: float = _DEFAULT_BM25_K1,
    bm25_b: float = _DEFAULT_BM25_B,
    train_fold_count: int = _DEFAULT_TRAIN_FOLD_COUNT,
    encoder_batch_size: int = _DEFAULT_ENCODER_BATCH_SIZE,
    encoder_device: str | None = None,
    encoder_factory: EncoderFactory | None = None,
    evidence_selector_name: str = _DEFAULT_EVIDENCE_SELECTOR,
    max_candidates_per_document: int = _DEFAULT_MAX_CANDIDATES_PER_DOCUMENT,
    composition_policy_name: str = _DEFAULT_COMPOSITION_POLICY,
    max_sentences: int = _DEFAULT_MAX_SENTENCES,
    min_sentence_score: float = _DEFAULT_MIN_SENTENCE_SCORE,
    min_evidence_score: float = _DEFAULT_MIN_EVIDENCE_SCORE,
) -> dict[str, Any]:
    """Run the Stage129 train-CV/dev agent retrieval integration validation."""

    _validate_options(
        train_fold_count=train_fold_count,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        encoder_batch_size=encoder_batch_size,
        max_candidates_per_document=max_candidates_per_document,
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
        min_evidence_score=min_evidence_score,
    )
    started_at = time.perf_counter()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    stage128_protocol = _load_json_object(stage128_protocol_path)
    stage125_protocol = _load_json_object(stage125_protocol_path)
    stage128_summary = _stage128_summary(stage128_protocol)
    selected_config = _selected_append_config(
        stage125_protocol=stage125_protocol,
        stage128_summary=stage128_summary,
    )
    loaded_protocols_at = time.perf_counter()

    split_samples = {
        _TRAIN_SPLIT: load_primeqa_hybrid_split_samples(train_split_path),
        _DEV_SPLIT: load_primeqa_hybrid_split_samples(dev_split_path),
    }
    train_fold_assignments = _build_train_fold_assignments(
        split_samples[_TRAIN_SPLIT],
        fold_count=train_fold_count,
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
    dense_preflight_at = time.perf_counter()

    pre_checks = _pre_evaluation_guard_checks(
        stage128_summary=stage128_summary,
        selected_config=selected_config,
        user_confirmed_validation=user_confirmed_validation,
        confirmation_note=confirmation_note,
        split_samples=split_samples,
        include_dense_channels=include_dense_channels,
        dense_summary=dense_summary,
        train_fold_count=train_fold_count,
    )
    if not all(check["passed"] for check in pre_checks):
        checked_at = time.perf_counter()
        report = _blocked_report(
            stage128_protocol_path=stage128_protocol_path,
            stage125_protocol_path=stage125_protocol_path,
            stage80_report_path=stage80_report_path,
            train_split_path=train_split_path,
            dev_split_path=dev_split_path,
            documents_path=documents_path,
            user_confirmed_validation=user_confirmed_validation,
            confirmation_note=confirmation_note,
            stage128_summary=stage128_summary,
            selected_config=selected_config,
            split_samples=split_samples,
            documents=documents,
            sections_by_document=sections_by_document,
            dense_summary=dense_summary,
            guard_checks=pre_checks,
            timing_seconds={
                "load_protocols": round(loaded_protocols_at - started_at, 3),
                "load_splits_and_build_train_folds": round(
                    loaded_splits_at - loaded_protocols_at,
                    3,
                ),
                "load_documents_sections": round(
                    loaded_documents_at - loaded_splits_at,
                    3,
                ),
                "dense_preflight": round(dense_preflight_at - loaded_documents_at, 3),
                "guard_checks": round(checked_at - dense_preflight_at, 3),
                "total": round(checked_at - started_at, 3),
            },
        )
        return {**report, "public_safe_contract": _public_safe_contract(report)}

    lexical_channels = _build_lexical_channels(
        documents=documents,
        sections_by_document=sections_by_document,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        component_depth=_selected_channel_top_k(selected_config),
    )
    evaluation_channels = _evaluation_channels(
        lexical_channels=lexical_channels,
        dense_channels=dense_channels,
    )
    channel_catalog = _public_channel_catalog(evaluation_channels)
    baseline_retriever = BM25Retriever(k1=bm25_k1, b=bm25_b)
    baseline_retriever.fit(documents)
    indexed_at = time.perf_counter()

    candidate_pools_by_split = _candidate_pools_by_split(
        split_samples=split_samples,
        selected_config=selected_config,
        channels=evaluation_channels,
    )
    pools_built_at = time.perf_counter()

    profiles = _profile_configs(selected_config)
    traces_by_profile = {
        profile.profile_id: _evaluate_profile(
            profile=profile,
            split_samples=split_samples,
            documents_by_id=documents_by_id,
            baseline_retriever=baseline_retriever,
            candidate_pools_by_split=candidate_pools_by_split,
            evidence_selector_name=evidence_selector_name,
            max_candidates_per_document=max_candidates_per_document,
            composition_policy_name=composition_policy_name,
            max_sentences=max_sentences,
            min_sentence_score=min_sentence_score,
            min_evidence_score=min_evidence_score,
        )
        for profile in profiles
    }
    evaluated_at = time.perf_counter()

    profile_reports = {
        profile.profile_id: _profile_report(
            profile=profile,
            traces_by_split=traces_by_profile[profile.profile_id],
            fold_assignments=train_fold_assignments,
            baseline_traces_by_split=traces_by_profile[_BASELINE_PROFILE_ID],
            stage116_traces_by_split=traces_by_profile[_STAGE116_PROFILE_ID],
        )
        for profile in profiles
    }
    stage128_report = profile_reports[_STAGE128_PROFILE_ID]
    stage116_report = profile_reports[_STAGE116_PROFILE_ID]
    baseline_report = profile_reports[_BASELINE_PROFILE_ID]
    train_cv_validation = _train_cv_validation(
        stage128_report=stage128_report,
        stage116_report=stage116_report,
        baseline_report=baseline_report,
    )
    dev_report = _dev_report(stage128_report=stage128_report)
    guard_checks = pre_checks + _post_evaluation_guard_checks(
        stage128_summary=stage128_summary,
        selected_config=selected_config,
        candidate_pools_by_split=candidate_pools_by_split,
        profile_reports=profile_reports,
        train_cv_validation=train_cv_validation,
    )
    checked_at = time.perf_counter()
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train/dev-only validation for the frozen Stage128 agent retrieval "
            "integration protocol. This stage loads train/dev frozen split rows, "
            "local corpus documents, the public-safe Stage128 protocol, and the "
            "public-safe Stage125 executable append config. It evaluates whether "
            "the Stage128 400-depth candidate pool can be consumed by evidence "
            "selection and citation validation without reducing train-CV answer "
            "quality. It does not load the test split, does not run final test "
            "metrics, does not write raw question, answer, document, token, or "
            "candidate-row fields, does not add fallback strategies, and does "
            "not change runtime defaults."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_validation),
            "confirmation_note": confirmation_note,
        },
        "split_contract": _split_contract(),
        "source_files": _source_files(
            stage128_protocol_path=stage128_protocol_path,
            stage125_protocol_path=stage125_protocol_path,
            stage80_report_path=stage80_report_path,
            train_split_path=train_split_path,
            dev_split_path=dev_split_path,
            documents_path=documents_path,
        ),
        "stage128_summary": stage128_summary,
        "selected_append_config": _public_selected_config(selected_config),
        "analysis_config": {
            "include_dense_channels": include_dense_channels,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "train_fold_count": train_fold_count,
            "encoder_batch_size": encoder_batch_size,
            "encoder_device": encoder_device,
            "evidence_selector": evidence_selector_name,
            "max_candidates_per_document": max_candidates_per_document,
            "composition_policy": composition_policy_name,
            "max_sentences": max_sentences,
            "min_sentence_score": min_sentence_score,
            "min_evidence_score": min_evidence_score,
            "agent_evidence_shortlister": "query_document_overlap_with_rank_prior",
            "agent_evidence_context_depth": _AGENT_EVIDENCE_CONTEXT_DEPTH,
            "shortlist_document_text_max_chars": _SHORTLIST_DOCUMENT_TEXT_MAX_CHARS,
            "baseline_profile_id": _BASELINE_PROFILE_ID,
            "stage116_control_profile_id": _STAGE116_PROFILE_ID,
            "stage128_candidate_profile_id": _STAGE128_PROFILE_ID,
        },
        "loaded_data_summary": {
            "split_samples": summarize_primeqa_hybrid_split_samples(split_samples),
            "document_count": len(documents),
            **_section_summary(sections_by_document),
            "test_split_loaded": False,
        },
        "dense_channel_preflight": dense_summary,
        "channel_catalog": channel_catalog,
        "candidate_pool_summary": _candidate_pool_summary(candidate_pools_by_split),
        "profile_reports": profile_reports,
        "train_cv_validation": train_cv_validation,
        "dev_report_observations": dev_report,
        "guard_checks": guard_checks,
        "decision": _decision(
            guard_checks=guard_checks,
            train_cv_validation=train_cv_validation,
            dev_report=dev_report,
        ),
        "timing_seconds": {
            "load_protocols": round(loaded_protocols_at - started_at, 3),
            "load_splits_and_build_train_folds": round(
                loaded_splits_at - loaded_protocols_at,
                3,
            ),
            "load_documents_sections": round(
                loaded_documents_at - loaded_splits_at,
                3,
            ),
            "dense_preflight": round(dense_preflight_at - loaded_documents_at, 3),
            "build_indexes": round(indexed_at - dense_preflight_at, 3),
            "build_candidate_pools": round(pools_built_at - indexed_at, 3),
            "evaluate_agent_profiles": round(evaluated_at - pools_built_at, 3),
            "summarize_and_guard": round(checked_at - evaluated_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_agent_retrieval_integration_validation_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridAgentRetrievalIntegrationValidationVisualization]:
    """Write SVG charts for Stage129 validation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage129_train_cv_verified_f1.svg": render_horizontal_bar_chart_svg(
            title="Stage129 train-CV verified F1",
            bars=_verified_metric_bars(report, split="train_cv", metric="average_token_f1"),
            x_label="verified average token F1",
            width=1320,
            margin_left=620,
        ),
        "stage129_train_cv_gold_citation_rate.svg": render_horizontal_bar_chart_svg(
            title="Stage129 train-CV gold citation rate",
            bars=_verified_metric_bars(
                report,
                split="train_cv",
                metric="gold_doc_citation_rate",
            ),
            x_label="verified gold citation rate",
            width=1320,
            margin_left=620,
        ),
        "stage129_train_cv_answer_quality_deltas.svg": render_horizontal_bar_chart_svg(
            title="Stage129 train-CV Stage128 vs Stage116 deltas",
            bars=_train_cv_delta_bars(report),
            x_label="delta",
            width=1480,
            margin_left=760,
        ),
        "stage129_target_depth_recall.svg": render_horizontal_bar_chart_svg(
            title="Stage129 target-depth gold recall",
            bars=_target_depth_recall_bars(report),
            x_label="gold-document hit rate",
            width=1320,
            margin_left=620,
        ),
        "stage129_selected_evidence_region_mix.svg": render_horizontal_bar_chart_svg(
            title="Stage129 selected evidence region mix",
            bars=_selected_evidence_region_bars(report),
            x_label="citation count",
            width=1420,
            margin_left=720,
        ),
        "stage129_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage129 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=1900,
            margin_left=1060,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridAgentRetrievalIntegrationValidationVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _candidate_pools_by_split(
    *,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    selected_config: Mapping[str, Any],
    channels: Sequence[_EvaluationChannel],
) -> dict[str, dict[str, dict[str, Any]]]:
    generation = selected_config["append_generation"]
    channel_top_k = int(generation["channel_top_k"])
    target_pool_depth = int(generation["target_pool_depth"])
    configured_channels = _channels_for_route_set(
        channels=channels,
        route_set=str(generation["route_set"]),
    )
    pools_by_split = {}
    for split, samples in split_samples.items():
        result_cache = _result_cache(
            samples=samples,
            channels=channels,
            top_k=channel_top_k,
        )
        baseline_outcomes = _baseline_outcomes(
            samples=samples,
            channels=channels,
            result_cache=result_cache,
        )
        split_pools = {}
        for sample in samples:
            baseline = baseline_outcomes[sample.sample_id]
            prefix = list(baseline["ranked_pool"][:_BASELINE_PREFIX_DEPTH])
            results_by_channel = _results_for_channels(
                sample=sample,
                channels=configured_channels,
                result_cache=result_cache,
                channel_top_k=channel_top_k,
            )
            append_source_ranked = _rank_pool(
                channels=configured_channels,
                results_by_channel=results_by_channel,
                algorithm=str(generation["append_source_algorithm"]),
                rrf_k=int(generation["rrf_k"]),
                target_pool_depth=target_pool_depth,
            )
            stage128_pool = _prefix_preserving_pool(
                prefix=prefix,
                append_source_ranked=append_source_ranked,
                target_pool_depth=target_pool_depth,
            )
            split_pools[sample.sample_id] = {
                "prefix_pool": prefix,
                "stage128_pool": stage128_pool,
                "prefix_identity_violation_count": _prefix_identity_violation_count(
                    prefix=prefix,
                    ranked_pool=stage128_pool,
                ),
                "append_count": max(0, len(stage128_pool) - len(prefix)),
            }
        pools_by_split[split] = split_pools
    return pools_by_split


def _evaluate_profile(
    *,
    profile: _ProfileConfig,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    documents_by_id: Mapping[str, PrimeQADocument],
    baseline_retriever: BM25Retriever,
    candidate_pools_by_split: Mapping[str, Mapping[str, Mapping[str, Any]]],
    evidence_selector_name: str,
    max_candidates_per_document: int,
    composition_policy_name: str,
    max_sentences: int,
    min_sentence_score: float,
    min_evidence_score: float,
) -> dict[str, list[_QuestionTrace]]:
    answer_generator = _answer_generator(
        evidence_selector_name=evidence_selector_name,
        max_candidates_per_document=max_candidates_per_document,
        composition_policy_name=composition_policy_name,
        max_sentences=max_sentences,
        min_sentence_score=min_sentence_score,
    )
    answer_verifier = AnswerVerifier(
        min_citations=1,
        min_evidence_score=min_evidence_score,
        max_citation_rank=profile.verifier_max_citation_rank,
    )
    shortlister = _DocumentEvidenceShortlister()
    return {
        split: [
            _trace_sample(
                sample=sample,
                candidate_pool_results=_retrieval_results_for_profile(
                    profile=profile,
                    sample=sample,
                    split=split,
                    documents_by_id=documents_by_id,
                    baseline_retriever=baseline_retriever,
                    candidate_pools_by_split=candidate_pools_by_split,
                ),
                shortlister=shortlister,
                answer_context_depth=profile.answer_context_depth,
                answer_generator=answer_generator,
                answer_verifier=answer_verifier,
            )
            for sample in samples
        ]
        for split, samples in split_samples.items()
    }


def _retrieval_results_for_profile(
    *,
    profile: _ProfileConfig,
    sample: PrimeQAHybridSplitSample,
    split: str,
    documents_by_id: Mapping[str, PrimeQADocument],
    baseline_retriever: BM25Retriever,
    candidate_pools_by_split: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[RetrievalResult]:
    question = sample.to_primeqa_question()
    if profile.profile_id == _BASELINE_PROFILE_ID:
        return baseline_retriever.search(
            question.full_question,
            top_k=profile.retrieval_depth,
        )

    pool_key = "prefix_pool" if profile.is_stage116_control else "stage128_pool"
    doc_ids = list(candidate_pools_by_split[split][sample.sample_id][pool_key])
    return [
        RetrievalResult(
            document=documents_by_id[doc_id],
            score=_rank_score(rank),
            rank=rank,
        )
        for rank, doc_id in enumerate(doc_ids[: profile.retrieval_depth], start=1)
        if doc_id in documents_by_id
    ]


def _trace_sample(
    *,
    sample: PrimeQAHybridSplitSample,
    candidate_pool_results: Sequence[RetrievalResult],
    shortlister: _DocumentEvidenceShortlister,
    answer_context_depth: int,
    answer_generator: ExtractiveAnswerGenerator,
    answer_verifier: AnswerVerifier,
) -> _QuestionTrace:
    question = sample.to_primeqa_question()
    answer_context_results = shortlister.shortlist(
        question=question,
        candidates=candidate_pool_results,
        top_k=answer_context_depth,
    )
    original_answer = answer_generator.generate(question, answer_context_results)
    verification = answer_verifier.verify(original_answer, candidate_pool_results)
    return _QuestionTrace(
        sample=sample,
        question=question,
        retrieval_results=list(candidate_pool_results),
        answer_context_results=answer_context_results,
        original_answer=original_answer,
        verified_answer=verification.verified_answer,
        verification_reasons=tuple(verification.reasons),
    )


def _profile_report(
    *,
    profile: _ProfileConfig,
    traces_by_split: Mapping[str, Sequence[_QuestionTrace]],
    fold_assignments: Mapping[str, str],
    baseline_traces_by_split: Mapping[str, Sequence[_QuestionTrace]],
    stage116_traces_by_split: Mapping[str, Sequence[_QuestionTrace]],
) -> dict[str, Any]:
    split_reports = {
        split: _split_profile_report(
            split=split,
            traces=traces,
            baseline_traces=baseline_traces_by_split[split],
            stage116_traces=stage116_traces_by_split[split],
        )
        for split, traces in traces_by_split.items()
    }
    train_cv = _split_profile_report(
        split="train_cv",
        traces=traces_by_split[_TRAIN_SPLIT],
        baseline_traces=baseline_traces_by_split[_TRAIN_SPLIT],
        stage116_traces=stage116_traces_by_split[_TRAIN_SPLIT],
    )
    train_fold_reports = _train_fold_reports(
        traces=traces_by_split[_TRAIN_SPLIT],
        baseline_traces=baseline_traces_by_split[_TRAIN_SPLIT],
        stage116_traces=stage116_traces_by_split[_TRAIN_SPLIT],
        fold_assignments=fold_assignments,
    )
    return {
        "profile_id": profile.profile_id,
        "family_id": profile.family_id,
        "retrieval_mode": profile.retrieval_mode,
        "retrieval_depth": profile.retrieval_depth,
        "answer_context_depth": profile.answer_context_depth,
        "verifier_max_citation_rank": profile.verifier_max_citation_rank,
        "split_reports": {
            "train_cv": train_cv,
            "train_full": split_reports[_TRAIN_SPLIT],
            "dev": split_reports[_DEV_SPLIT],
        },
        "train_fold_reports": train_fold_reports,
        "train_cv_group_values_written": False,
    }


def _split_profile_report(
    *,
    split: str,
    traces: Sequence[_QuestionTrace],
    baseline_traces: Sequence[_QuestionTrace],
    stage116_traces: Sequence[_QuestionTrace],
) -> dict[str, Any]:
    questions = [trace.question for trace in traces]
    original_metrics = evaluate_answers(
        questions,
        [trace.original_answer for trace in traces],
    )
    verified_metrics = evaluate_answers(
        questions,
        [trace.verified_answer for trace in traces],
    )
    retrieval = _retrieval_summary(traces)
    selected_evidence = _selected_evidence_summary(traces)
    return {
        "split": split,
        "row_count": len(traces),
        "original_metrics": asdict(original_metrics),
        "verified_metrics": asdict(verified_metrics),
        "retrieval_summary": retrieval,
        "selected_evidence_summary": selected_evidence,
        "verification_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for trace in traces
                    for reason in trace.verification_reasons
                ).items()
            )
        ),
        "changed_verified_answers_vs_baseline": _changed_verified_answers(
            traces,
            baseline_traces,
        ),
        "changed_verified_answers_vs_stage116_control": _changed_verified_answers(
            traces,
            stage116_traces,
        ),
    }


def _retrieval_summary(traces: Sequence[_QuestionTrace]) -> dict[str, Any]:
    answerable = [trace for trace in traces if trace.question.answerable]
    total = len(answerable)
    hit_counts = {10: 0, 50: 0, 100: 0, 200: 0, 400: 0}
    rank_buckets = Counter()
    reciprocal_ranks = []
    pool_sizes = []
    for trace in answerable:
        pool_sizes.append(len(trace.retrieval_results))
        rank = _gold_rank(trace)
        rank_buckets[_rank_bucket(rank)] += 1
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
        for top_k in hit_counts:
            if rank is not None and rank <= top_k:
                hit_counts[top_k] += 1
    return {
        "answerable_count": total,
        "pool_size": _distribution_summary(pool_sizes),
        "gold_hit_counts": hit_counts,
        "gold_hit_rates": {
            str(top_k): _rounded_ratio(count, total) for top_k, count in hit_counts.items()
        },
        "gold_miss_count_at_profile_depth": total - _hit_count_at_profile_depth(answerable),
        "gold_hit_count_at_profile_depth": _hit_count_at_profile_depth(answerable),
        "gold_hit_rate_at_profile_depth": _rounded_ratio(
            _hit_count_at_profile_depth(answerable),
            total,
        ),
        "mean_reciprocal_rank": round(
            sum(reciprocal_ranks) / total if total else 0.0,
            4,
        ),
        "gold_rank_buckets": dict(sorted(rank_buckets.items())),
    }


def _selected_evidence_summary(traces: Sequence[_QuestionTrace]) -> dict[str, Any]:
    citation_count = 0
    answered_count = 0
    gold_citation_count = 0
    region_counts = Counter()
    route_family_counts = Counter()
    citation_counts_per_answer = []
    for trace in traces:
        if trace.verified_answer.refused:
            continue
        answered_count += 1
        citation_counts_per_answer.append(len(trace.verified_answer.citations))
        if trace.question.answerable:
            cited_doc_ids = {
                citation.document_id for citation in trace.verified_answer.citations
            }
            if trace.question.answer_doc_id in cited_doc_ids:
                gold_citation_count += 1
        for citation in trace.verified_answer.citations:
            citation_count += 1
            region = _rank_region(citation.retrieval_rank)
            region_counts[region] += 1
            route_family_counts[_route_family_for_region(region)] += 1
    return {
        "answered_count": answered_count,
        "citation_count": citation_count,
        "gold_citation_count": gold_citation_count,
        "citation_counts_per_answer": _distribution_summary(citation_counts_per_answer),
        "rank_region_counts": dict(sorted(region_counts.items())),
        "route_family_counts": dict(sorted(route_family_counts.items())),
    }


def _train_fold_reports(
    *,
    traces: Sequence[_QuestionTrace],
    baseline_traces: Sequence[_QuestionTrace],
    stage116_traces: Sequence[_QuestionTrace],
    fold_assignments: Mapping[str, str],
) -> list[dict[str, Any]]:
    reports = []
    for fold_id in sorted(set(fold_assignments.values())):
        fold_traces = [
            trace
            for trace in traces
            if fold_assignments[trace.sample.sample_id] == fold_id
        ]
        fold_baseline = [
            trace
            for trace in baseline_traces
            if fold_assignments[trace.sample.sample_id] == fold_id
        ]
        fold_stage116 = [
            trace
            for trace in stage116_traces
            if fold_assignments[trace.sample.sample_id] == fold_id
        ]
        report = _split_profile_report(
            split=fold_id,
            traces=fold_traces,
            baseline_traces=fold_baseline,
            stage116_traces=fold_stage116,
        )
        reports.append(
            {
                "fold_id": fold_id,
                "row_count": report["row_count"],
                "verified_average_token_f1": report["verified_metrics"][
                    "average_token_f1"
                ],
                "verified_gold_doc_citation_rate": report["verified_metrics"][
                    "gold_doc_citation_rate"
                ],
                "answerable_refusal_rate": report["verified_metrics"][
                    "answerable_refusal_rate"
                ],
                "gold_hit_rate_at_profile_depth": report["retrieval_summary"][
                    "gold_hit_rate_at_profile_depth"
                ],
                "changed_verified_answers_vs_stage116_control": report[
                    "changed_verified_answers_vs_stage116_control"
                ],
            }
        )
    return reports


def _train_cv_validation(
    *,
    stage128_report: Mapping[str, Any],
    stage116_report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
) -> dict[str, Any]:
    stage128_train = stage128_report["split_reports"]["train_cv"]
    stage116_train = stage116_report["split_reports"]["train_cv"]
    baseline_train = baseline_report["split_reports"]["train_cv"]
    deltas_vs_stage116 = _split_deltas(
        candidate=stage128_train,
        baseline=stage116_train,
        changed_answer_count=stage128_train[
            "changed_verified_answers_vs_stage116_control"
        ],
    )
    deltas_vs_baseline = _split_deltas(
        candidate=stage128_train,
        baseline=baseline_train,
        changed_answer_count=stage128_train["changed_verified_answers_vs_baseline"],
    )
    checks = {
        "verified_f1_delta_vs_stage116_non_negative": (
            deltas_vs_stage116["verified_average_token_f1_delta"] >= 0
        ),
        "gold_citation_count_delta_vs_stage116_non_negative": (
            deltas_vs_stage116["verified_gold_citation_count_delta"] >= 0
        ),
        "answerable_refusal_rate_delta_vs_stage116_non_positive": (
            deltas_vs_stage116["answerable_refusal_rate_delta"] <= 0
        ),
        "target_depth_recall_delta_vs_stage116_positive": (
            deltas_vs_stage116["gold_hit_count_at_profile_depth_delta"] > 0
        ),
    }
    return {
        "selection_split": "train",
        "selection_mode": "train_grouped_cross_validation_agent_integration_validation",
        "selection_source": "fixed_stage128_selected_config_no_candidate_search",
        "selected_profile_id": _STAGE128_PROFILE_ID,
        "baseline_profile_id": _BASELINE_PROFILE_ID,
        "stage116_control_profile_id": _STAGE116_PROFILE_ID,
        "deltas_vs_stage116_control": deltas_vs_stage116,
        "deltas_vs_baseline": deltas_vs_baseline,
        "checks": checks,
        "passed": all(checks.values()),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "train_cv_group_values_written": False,
    }


def _dev_report(stage128_report: Mapping[str, Any]) -> dict[str, Any]:
    dev = stage128_report["split_reports"]["dev"]
    return {
        "validation_split": "dev",
        "profile_id": _STAGE128_PROFILE_ID,
        "status": "reported_not_used_for_selection",
        "dev_selection_allowed": False,
        "dev_retuning_allowed": False,
        "dev_verified_average_token_f1": dev["verified_metrics"]["average_token_f1"],
        "dev_gold_doc_citation_rate": dev["verified_metrics"][
            "gold_doc_citation_rate"
        ],
        "dev_answerable_refusal_rate": dev["verified_metrics"][
            "answerable_refusal_rate"
        ],
        "dev_gold_hit_rate_at_profile_depth": dev["retrieval_summary"][
            "gold_hit_rate_at_profile_depth"
        ],
        "dev_changed_verified_answers_vs_stage116_control": dev[
            "changed_verified_answers_vs_stage116_control"
        ],
        "dev_gate_status": "report_only_no_runtime_or_test_gate",
    }


def _split_deltas(
    *,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    changed_answer_count: int,
) -> dict[str, Any]:
    candidate_verified = candidate["verified_metrics"]
    baseline_verified = baseline["verified_metrics"]
    candidate_retrieval = candidate["retrieval_summary"]
    baseline_retrieval = baseline["retrieval_summary"]
    candidate_evidence = candidate["selected_evidence_summary"]
    baseline_evidence = baseline["selected_evidence_summary"]
    return {
        "verified_average_token_f1_delta": _float_delta(
            candidate_verified["average_token_f1"],
            baseline_verified["average_token_f1"],
        ),
        "verified_gold_doc_citation_rate_delta": _float_delta(
            candidate_verified["gold_doc_citation_rate"],
            baseline_verified["gold_doc_citation_rate"],
        ),
        "verified_gold_citation_count_delta": int(
            candidate_evidence["gold_citation_count"]
        )
        - int(baseline_evidence["gold_citation_count"]),
        "answerable_refusal_rate_delta": _float_delta(
            candidate_verified["answerable_refusal_rate"],
            baseline_verified["answerable_refusal_rate"],
        ),
        "unanswerable_refusal_rate_delta": _float_delta(
            candidate_verified["unanswerable_refusal_rate"],
            baseline_verified["unanswerable_refusal_rate"],
        ),
        "gold_hit_count_at_profile_depth_delta": int(
            candidate_retrieval["gold_hit_count_at_profile_depth"]
        )
        - int(baseline_retrieval["gold_hit_count_at_profile_depth"]),
        "gold_hit_rate_at_profile_depth_delta": _float_delta(
            candidate_retrieval["gold_hit_rate_at_profile_depth"],
            baseline_retrieval["gold_hit_rate_at_profile_depth"],
        ),
        "changed_verified_answers": int(changed_answer_count),
    }


def _pre_evaluation_guard_checks(
    *,
    stage128_summary: Mapping[str, Any],
    selected_config: Mapping[str, Any] | None,
    user_confirmed_validation: bool,
    confirmation_note: str,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    include_dense_channels: bool,
    dense_summary: Mapping[str, Any],
    train_fold_count: int,
) -> list[dict[str, Any]]:
    return [
        _check(
            name="user_confirmed_stage129_validation",
            passed=user_confirmed_validation and "Stage129" in confirmation_note,
            observed=confirmation_note,
            expected="user confirmed Stage129 validation",
        ),
        _check(
            name="stage128_protocol_frozen",
            passed=stage128_summary.get("status") == _SOURCE_STAGE128_STATUS,
            observed=stage128_summary.get("status"),
            expected=_SOURCE_STAGE128_STATUS,
        ),
        _check(
            name="stage128_protocol_id_matches",
            passed=stage128_summary.get("protocol_id") == _SOURCE_STAGE128_PROTOCOL_ID,
            observed=stage128_summary.get("protocol_id"),
            expected=_SOURCE_STAGE128_PROTOCOL_ID,
        ),
        _check(
            name="stage128_recommends_stage129_validation",
            passed=stage128_summary.get("recommended_next_direction")
            == _SOURCE_STAGE128_NEXT,
            observed=stage128_summary.get("recommended_next_direction"),
            expected=_SOURCE_STAGE128_NEXT,
        ),
        _check(
            name="selected_append_config_available",
            passed=selected_config is not None,
            observed=None if selected_config is None else selected_config.get("config_id"),
            expected=_SELECTED_CONFIG_ID,
        ),
        _check(
            name="selected_append_config_matches_stage128",
            passed=selected_config is not None
            and selected_config.get("config_id") == stage128_summary.get("selected_config_id")
            and selected_config.get("family_id") == stage128_summary.get("selected_family_id"),
            observed={
                "stage125_config_id": None
                if selected_config is None
                else selected_config.get("config_id"),
                "stage128_config_id": stage128_summary.get("selected_config_id"),
            },
            expected=_SELECTED_CONFIG_ID,
        ),
        _check(
            name="only_train_dev_splits_loaded",
            passed=tuple(split_samples) == _ALLOWED_DEVELOPMENT_SPLITS,
            observed=list(split_samples),
            expected=list(_ALLOWED_DEVELOPMENT_SPLITS),
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
            expected={split: split for split in _ALLOWED_DEVELOPMENT_SPLITS},
        ),
        _check(
            name="train_fold_count_matches_stage128_minimum",
            passed=train_fold_count >= 5,
            observed=train_fold_count,
            expected=">= 5",
        ),
        _check(
            name="dense_channels_use_existing_cache_only",
            passed=(not include_dense_channels)
            or (
                bool(dense_summary.get("can_run_without_download"))
                and bool(dense_summary.get("no_model_download_attempted"))
            ),
            observed=dense_summary,
            expected="can run without download and no model download attempted",
        ),
    ]


def _post_evaluation_guard_checks(
    *,
    stage128_summary: Mapping[str, Any],
    selected_config: Mapping[str, Any],
    candidate_pools_by_split: Mapping[str, Mapping[str, Mapping[str, Any]]],
    profile_reports: Mapping[str, Mapping[str, Any]],
    train_cv_validation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    pool_summary = _candidate_pool_summary(candidate_pools_by_split)
    stage128_report = profile_reports[_STAGE128_PROFILE_ID]
    public_payload = {
        "profiles": profile_reports,
        "train_cv_validation": train_cv_validation,
        "pool_summary": pool_summary,
    }
    return [
        _check(
            name="stage129_prefix_identity_preserved",
            passed=pool_summary["all_splits_prefix_identity_violation_count"] == 0,
            observed=pool_summary["all_splits_prefix_identity_violation_count"],
            expected=0,
        ),
        _check(
            name="stage129_append_budget_not_exceeded",
            passed=pool_summary["all_splits_append_budget_exceeded_count"] == 0,
            observed=pool_summary["all_splits_append_budget_exceeded_count"],
            expected=0,
        ),
        _check(
            name="stage129_target_pool_depth_matches_protocol",
            passed=int(selected_config["append_generation"]["target_pool_depth"])
            == _TARGET_POOL_DEPTH
            and all(
                int(report["retrieval_depth"]) in {10, _BASELINE_PREFIX_DEPTH, _TARGET_POOL_DEPTH}
                for report in profile_reports.values()
            ),
            observed={
                "selected_target_depth": selected_config["append_generation"][
                    "target_pool_depth"
                ],
                "profile_depths": {
                    profile_id: report["retrieval_depth"]
                    for profile_id, report in profile_reports.items()
                },
            },
            expected=_TARGET_POOL_DEPTH,
        ),
        _check(
            name="stage129_stage128_candidate_pool_does_not_regress_target_recall",
            passed=train_cv_validation["deltas_vs_stage116_control"][
                "gold_hit_count_at_profile_depth_delta"
            ]
            >= 0,
            observed=train_cv_validation["deltas_vs_stage116_control"][
                "gold_hit_count_at_profile_depth_delta"
            ],
            expected=">= 0",
        ),
        _check(
            name="stage129_agent_answer_quality_train_cv_guard",
            passed=bool(train_cv_validation["passed"]),
            observed=train_cv_validation["checks"],
            expected="all train-CV answer-quality checks pass",
        ),
        _check(
            name="stage129_dev_report_only",
            passed=True,
            observed="dev reported only; no selection or retuning",
            expected="dev report only",
        ),
        _check(
            name="stage129_test_locked",
            passed=stage128_summary.get("can_run_final_test_metrics_now") is False,
            observed={
                "can_open_final_test_gate_now": stage128_summary.get(
                    "can_open_final_test_gate_now"
                ),
                "can_run_final_test_metrics_now": stage128_summary.get(
                    "can_run_final_test_metrics_now"
                ),
                "can_use_test_for_tuning": stage128_summary.get("can_use_test_for_tuning"),
            },
            expected="test locked",
        ),
        _check(
            name="stage129_runtime_defaults_unchanged",
            passed=stage128_summary.get("default_runtime_policy") == "unchanged",
            observed=stage128_summary.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage129_no_fallback_strategies",
            passed=stage128_summary.get("fallback_strategies_enabled") is False,
            observed=stage128_summary.get("fallback_strategies_enabled"),
            expected=False,
        ),
        _check(
            name="stage129_public_outputs_have_no_forbidden_keys",
            passed=not _contains_forbidden_key(public_payload),
            observed=sorted(_forbidden_keys_found(public_payload)),
            expected=[],
        ),
        _check(
            name="stage129_train_cv_group_values_not_written",
            passed=stage128_report.get("train_cv_group_values_written") is False,
            observed=stage128_report.get("train_cv_group_values_written"),
            expected=False,
        ),
    ]


def _decision(
    *,
    guard_checks: Sequence[Mapping[str, Any]],
    train_cv_validation: Mapping[str, Any],
    dev_report: Mapping[str, Any],
) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    base = {
        "analysis_id": _ANALYSIS_ID,
        "selected_profile_id": _STAGE128_PROFILE_ID,
        "train_cv_validation_passed": bool(train_cv_validation.get("passed")),
        "train_cv_failed_checks": list(train_cv_validation.get("failed_checks") or []),
        "dev_validation_status": dev_report.get("status"),
        "dev_gate_status": dev_report.get("dev_gate_status"),
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "runtime_defaultization_allowed_now": False,
        "fallback_strategies_enabled": False,
        "default_runtime_policy": "unchanged",
    }
    if failed_checks:
        return {
            **base,
            "status": "primeqa_hybrid_agent_retrieval_integration_validation_blocked_or_failed",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": True,
            "recommended_next_direction": (
                "review_stage129_agent_integration_failure_patterns"
            ),
        }
    return {
        **base,
        "status": "primeqa_hybrid_agent_retrieval_integration_validation_completed",
        "failed_checks": [],
        "can_continue_train_dev_development": True,
        "recommended_next_direction": (
            "review_stage129_agent_integration_changed_cases_before_runtime_or_test_gate"
        ),
    }


def _blocked_report(
    *,
    stage128_protocol_path: Path,
    stage125_protocol_path: Path,
    stage80_report_path: Path | None,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    user_confirmed_validation: bool,
    confirmation_note: str,
    stage128_summary: Mapping[str, Any],
    selected_config: Mapping[str, Any] | None,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    documents: Sequence[Any],
    sections_by_document: Mapping[str, Sequence[Any]],
    dense_summary: Mapping[str, Any],
    guard_checks: Sequence[Mapping[str, Any]],
    timing_seconds: Mapping[str, float],
) -> dict[str, Any]:
    dev_report = {
        "validation_split": "dev",
        "status": "not_run_pre_evaluation_guard_failed",
        "dev_gate_status": "not_run",
    }
    train_cv_validation = {
        "selection_split": "train",
        "passed": False,
        "failed_checks": ["pre_evaluation_guard_failed"],
    }
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": "Stage129 blocked before evaluation by guard checks.",
        "user_confirmation": {
            "confirmed": bool(user_confirmed_validation),
            "confirmation_note": confirmation_note,
        },
        "split_contract": _split_contract(),
        "source_files": _source_files(
            stage128_protocol_path=stage128_protocol_path,
            stage125_protocol_path=stage125_protocol_path,
            stage80_report_path=stage80_report_path,
            train_split_path=train_split_path,
            dev_split_path=dev_split_path,
            documents_path=documents_path,
        ),
        "stage128_summary": dict(stage128_summary),
        "selected_append_config": _public_selected_config(selected_config),
        "loaded_data_summary": {
            "split_samples": summarize_primeqa_hybrid_split_samples(split_samples),
            "document_count": len(documents),
            **_section_summary(sections_by_document),
            "test_split_loaded": False,
        },
        "dense_channel_preflight": dense_summary,
        "channel_catalog": [],
        "candidate_pool_summary": {},
        "profile_reports": {},
        "train_cv_validation": train_cv_validation,
        "dev_report_observations": dev_report,
        "guard_checks": list(guard_checks),
        "decision": _decision(
            guard_checks=guard_checks,
            train_cv_validation=train_cv_validation,
            dev_report=dev_report,
        ),
        "timing_seconds": dict(timing_seconds),
    }


def _profile_configs(selected_config: Mapping[str, Any]) -> list[_ProfileConfig]:
    _ = selected_config
    return [
        _ProfileConfig(
            profile_id=_BASELINE_PROFILE_ID,
            family_id="stage102_verified_bm25_baseline",
            retrieval_mode="document_bm25_top10_current_verified_baseline",
            retrieval_depth=10,
            answer_context_depth=10,
            verifier_max_citation_rank=_BASELINE_MAX_CITATION_RANK,
        ),
        _ProfileConfig(
            profile_id=_STAGE116_PROFILE_ID,
            family_id="stage116_agent_pool_control",
            retrieval_mode="stage116_immutable_prefix_top200_candidate_pool",
            retrieval_depth=_BASELINE_PREFIX_DEPTH,
            answer_context_depth=_AGENT_EVIDENCE_CONTEXT_DEPTH,
            verifier_max_citation_rank=_BASELINE_PREFIX_DEPTH,
            is_stage116_control=True,
        ),
        _ProfileConfig(
            profile_id=_STAGE128_PROFILE_ID,
            family_id=_SELECTED_FAMILY_ID,
            retrieval_mode="stage116_prefix_plus_stage128_append_top400_candidate_pool",
            retrieval_depth=_TARGET_POOL_DEPTH,
            answer_context_depth=_AGENT_EVIDENCE_CONTEXT_DEPTH,
            verifier_max_citation_rank=_TARGET_POOL_DEPTH,
            is_stage128_candidate=True,
        ),
    ]


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


def _selected_append_config(
    *,
    stage125_protocol: Mapping[str, Any],
    stage128_summary: Mapping[str, Any],
) -> dict[str, Any] | None:
    selected_id = stage128_summary.get("selected_config_id")
    for config in _candidate_configs_from_protocol(stage125_protocol):
        if config.get("config_id") == selected_id:
            return config
    return None


def _stage128_summary(stage128_protocol: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage128_protocol.get("decision") or {}
    frozen = stage128_protocol.get("frozen_protocol") or {}
    selected = frozen.get("selected_retrieval_config") or {}
    contract = frozen.get("agent_retrieval_contract") or {}
    validation_plan = frozen.get("validation_plan") or {}
    public_safe = stage128_protocol.get("public_safe_contract") or {}
    return {
        "stage": stage128_protocol.get("stage"),
        "protocol_id": stage128_protocol.get("protocol_id"),
        "status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "selected_config_id": decision.get("selected_config_id")
        or selected.get("config_id"),
        "selected_family_id": decision.get("selected_family_id")
        or selected.get("family_id"),
        "candidate_pool_output_depth": contract.get("candidate_pool_output_depth"),
        "candidate_pool_is_not_automatic_answer_context": contract.get(
            "candidate_pool_is_not_automatic_answer_context"
        ),
        "answer_context_policy": contract.get("answer_context_policy"),
        "validation_plan_next_stage": validation_plan.get("next_stage"),
        "validation_plan_action": validation_plan.get("action"),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "runtime_defaultization_allowed_now": decision.get(
            "runtime_defaultization_allowed_now"
        ),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "public_safe_forbidden_keys_found": public_safe.get("forbidden_keys_found") or [],
    }


def _public_selected_config(selected_config: Mapping[str, Any] | None) -> dict[str, Any]:
    if selected_config is None:
        return {"config_found": False}
    generation = selected_config.get("append_generation") or {}
    return {
        "config_found": True,
        "config_id": selected_config.get("config_id"),
        "family_id": selected_config.get("family_id"),
        "source_stage124_config_id": selected_config.get("source_stage124_config_id"),
        "append_source_algorithm": generation.get("append_source_algorithm"),
        "route_set": generation.get("route_set"),
        "channel_top_k": generation.get("channel_top_k"),
        "rrf_k": generation.get("rrf_k"),
        "append_start_rank": generation.get("append_start_rank"),
        "append_budget": generation.get("append_budget"),
        "target_pool_depth": generation.get("target_pool_depth"),
    }


def _public_channel_catalog(
    channels: Sequence[_EvaluationChannel],
) -> list[dict[str, Any]]:
    rows = []
    for channel in channels:
        rows.append(
            {
                "channel_id": channel.channel_id,
                "family": channel.family,
                "weight": channel.weight,
                "description": channel.description,
            }
        )
    return rows


def _candidate_pool_summary(
    candidate_pools_by_split: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    split_rows = {}
    all_prefix_violations = 0
    all_append_budget_exceeded = 0
    for split, pools in candidate_pools_by_split.items():
        prefix_violations = sum(
            int(row["prefix_identity_violation_count"]) for row in pools.values()
        )
        append_counts = [int(row["append_count"]) for row in pools.values()]
        append_budget_exceeded = sum(count > _APPEND_BUDGET for count in append_counts)
        all_prefix_violations += prefix_violations
        all_append_budget_exceeded += append_budget_exceeded
        split_rows[split] = {
            "row_count": len(pools),
            "prefix_identity_violation_count": prefix_violations,
            "append_budget_exceeded_count": append_budget_exceeded,
            "append_count": _distribution_summary(append_counts),
        }
    return {
        "splits": split_rows,
        "all_splits_prefix_identity_violation_count": all_prefix_violations,
        "all_splits_append_budget_exceeded_count": all_append_budget_exceeded,
    }


def _split_contract() -> dict[str, Any]:
    return {
        "split_name": _SPLIT_NAME,
        "protocol_version": _PROTOCOL_VERSION,
        "development_splits": list(_ALLOWED_DEVELOPMENT_SPLITS),
        "selection_split": _TRAIN_SPLIT,
        "validation_split": _DEV_SPLIT,
        "dev_selection_used": False,
        "dev_retuning_used": False,
        "forbidden_final_splits": sorted(_FORBIDDEN_FINAL_SPLITS),
    }


def _source_files(
    *,
    stage128_protocol_path: Path,
    stage125_protocol_path: Path,
    stage80_report_path: Path | None,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
) -> dict[str, Any]:
    files = {
        "stage128_protocol": _fingerprint(stage128_protocol_path),
        "stage125_protocol": _fingerprint(stage125_protocol_path),
        "train_split": _fingerprint(train_split_path),
        "dev_split": _fingerprint(dev_split_path),
        "corpus_documents": _fingerprint(documents_path),
    }
    if stage80_report_path is not None:
        files["stage80_dense_cache_report"] = _fingerprint(stage80_report_path)
    return files


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


def _public_safe_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    forbidden_keys = sorted(_forbidden_keys_found(report))
    return {
        "public_safe_summary_only": True,
        "raw_question_text_written": False,
        "raw_answer_text_written": False,
        "raw_document_text_written": False,
        "raw_document_ids_written": False,
        "raw_candidate_rows_written": False,
        "raw_sample_ids_written": False,
        "test_split_loaded": False,
        "final_test_metrics_run": False,
        "forbidden_keys_found": forbidden_keys,
    }


def _contains_forbidden_key(value: Any) -> bool:
    return bool(_forbidden_keys_found(value))


def _forbidden_keys_found(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in _FORBIDDEN_PUBLIC_KEYS:
                found.add(key_text)
            found.update(_forbidden_keys_found(child))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for child in value:
            found.update(_forbidden_keys_found(child))
    return found


def _validate_options(
    *,
    train_fold_count: int,
    bm25_k1: float,
    bm25_b: float,
    encoder_batch_size: int,
    max_candidates_per_document: int,
    max_sentences: int,
    min_sentence_score: float,
    min_evidence_score: float,
) -> None:
    if train_fold_count <= 0:
        raise ValueError("train_fold_count must be positive")
    if bm25_k1 <= 0:
        raise ValueError("bm25_k1 must be positive")
    if not 0 <= bm25_b <= 1:
        raise ValueError("bm25_b must be between 0 and 1")
    if encoder_batch_size <= 0:
        raise ValueError("encoder_batch_size must be positive")
    if max_candidates_per_document <= 0:
        raise ValueError("max_candidates_per_document must be positive")
    if max_sentences <= 0:
        raise ValueError("max_sentences must be positive")
    if min_sentence_score < 0:
        raise ValueError("min_sentence_score must be non-negative")
    if min_evidence_score < 0:
        raise ValueError("min_evidence_score must be non-negative")


def _rank_score(rank: int) -> float:
    return round(1.0 / math.log2(rank + 1), 8)


def _gold_rank(trace: _QuestionTrace) -> int | None:
    if not trace.question.answerable or trace.question.answer_doc_id is None:
        return None
    for result in trace.retrieval_results:
        if result.document.id == trace.question.answer_doc_id:
            return result.rank
    return None


def _hit_count_at_profile_depth(traces: Sequence[_QuestionTrace]) -> int:
    return sum(1 for trace in traces if _gold_rank(trace) is not None)


def _rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "missing"
    if rank <= 10:
        return "001-010"
    if rank <= 50:
        return "011-050"
    if rank <= 100:
        return "051-100"
    if rank <= 200:
        return "101-200"
    if rank <= 400:
        return "201-400"
    return "401+"


def _rank_region(rank: int) -> str:
    if rank <= 10:
        return "rank_001_010"
    if rank <= _BASELINE_PREFIX_DEPTH:
        return "stage116_immutable_prefix_011_200"
    if rank <= _TARGET_POOL_DEPTH:
        return "stage128_append_expansion_201_400"
    return "outside_stage128_pool"


def _route_family_for_region(region: str) -> str:
    if region.startswith("stage128_append"):
        return "stage128_append_fusion_candidate"
    if region.startswith("stage116"):
        return "stage116_prefix_candidate"
    if region.startswith("rank_001"):
        return "top_rank_candidate"
    return "outside_stage128_protocol"


def _changed_verified_answers(
    traces: Sequence[_QuestionTrace],
    baseline_traces: Sequence[_QuestionTrace],
) -> int:
    return sum(
        _answer_signature(trace.verified_answer)
        != _answer_signature(baseline_trace.verified_answer)
        for trace, baseline_trace in zip(traces, baseline_traces, strict=True)
    )


def _answer_signature(answer: GeneratedAnswer) -> tuple[Any, ...]:
    return (
        bool(answer.refused),
        " ".join(answer.answer.lower().split()),
        tuple(
            (citation.retrieval_rank, round(citation.evidence_score, 4))
            for citation in answer.citations
        ),
    )


def _distribution_summary(values: Sequence[int]) -> dict[str, Any]:
    return {
        "average": _rounded_mean(values),
        "median": _rounded_percentile(values, 50),
        "p95": _rounded_percentile(values, 95),
        "max": max(values, default=0),
    }


def _float_delta(value: Any, baseline: Any) -> float:
    return round(float(value) - float(baseline), 4)


def _selected_channel_top_k(selected_config: Mapping[str, Any] | None) -> int:
    if selected_config is None:
        return _TARGET_POOL_DEPTH
    return int(selected_config["append_generation"]["channel_top_k"])


def _verified_metric_bars(
    report: Mapping[str, Any],
    *,
    split: str,
    metric: str,
) -> list[BarDatum]:
    return [
        BarDatum(
            label=profile_id,
            value=float(profile["split_reports"][split]["verified_metrics"][metric]),
            value_label=(
                f"{float(profile['split_reports'][split]['verified_metrics'][metric]):.4f}"
            ),
        )
        for profile_id, profile in report.get("profile_reports", {}).items()
    ]


def _train_cv_delta_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    deltas = (report.get("train_cv_validation") or {}).get(
        "deltas_vs_stage116_control"
    ) or {}
    return [
        BarDatum(
            label="verified_average_token_f1_delta",
            value=float(deltas.get("verified_average_token_f1_delta") or 0.0),
            value_label=f"{float(deltas.get('verified_average_token_f1_delta') or 0.0):+.4f}",
        ),
        BarDatum(
            label="verified_gold_doc_citation_rate_delta",
            value=float(deltas.get("verified_gold_doc_citation_rate_delta") or 0.0),
            value_label=(
                f"{float(deltas.get('verified_gold_doc_citation_rate_delta') or 0.0):+.4f}"
            ),
        ),
        BarDatum(
            label="answerable_refusal_rate_delta",
            value=float(deltas.get("answerable_refusal_rate_delta") or 0.0),
            value_label=f"{float(deltas.get('answerable_refusal_rate_delta') or 0.0):+.4f}",
        ),
        BarDatum(
            label="gold_hit_rate_at_profile_depth_delta",
            value=float(deltas.get("gold_hit_rate_at_profile_depth_delta") or 0.0),
            value_label=(
                f"{float(deltas.get('gold_hit_rate_at_profile_depth_delta') or 0.0):+.4f}"
            ),
        ),
    ]


def _target_depth_recall_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for profile_id, profile in report.get("profile_reports", {}).items():
        for split in ("train_cv", "dev"):
            value = float(
                profile["split_reports"][split]["retrieval_summary"][
                    "gold_hit_rate_at_profile_depth"
                ]
            )
            bars.append(
                BarDatum(
                    label=f"{profile_id}:{split}",
                    value=value,
                    value_label=f"{value:.4f}",
                )
            )
    return bars


def _selected_evidence_region_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    profile = (report.get("profile_reports") or {}).get(_STAGE128_PROFILE_ID) or {}
    bars = []
    for split in ("train_cv", "dev"):
        region_counts = (
            profile.get("split_reports", {})
            .get(split, {})
            .get("selected_evidence_summary", {})
            .get("rank_region_counts", {})
        )
        for region, count in sorted(region_counts.items()):
            bars.append(
                BarDatum(
                    label=f"{split}:{region}",
                    value=float(count),
                    value_label=str(count),
                )
            )
    return bars


def _guard_check_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(check["name"]),
            value=1.0 if check.get("passed") else 0.0,
            value_label="passed" if check.get("passed") else "failed",
        )
        for check in report.get("guard_checks", [])
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
