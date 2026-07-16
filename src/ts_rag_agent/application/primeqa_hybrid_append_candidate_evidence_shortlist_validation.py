from __future__ import annotations

import hashlib
import math
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.answer_verification import AnswerVerifier
from ts_rag_agent.application.primeqa_hybrid_agent_retrieval_integration_validation import (
    _AGENT_EVIDENCE_CONTEXT_DEPTH,
    _BASELINE_PREFIX_DEPTH,
    _BASELINE_PROFILE_ID,
    _DEFAULT_BM25_B,
    _DEFAULT_BM25_K1,
    _DEFAULT_COMPOSITION_POLICY,
    _DEFAULT_ENCODER_BATCH_SIZE,
    _DEFAULT_EVIDENCE_SELECTOR,
    _DEFAULT_MAX_CANDIDATES_PER_DOCUMENT,
    _DEFAULT_MAX_SENTENCES,
    _DEFAULT_MIN_EVIDENCE_SCORE,
    _DEFAULT_MIN_SENTENCE_SCORE,
    _DEV_SPLIT,
    _SELECTED_CONFIG_ID,
    _STAGE116_PROFILE_ID,
    _TARGET_POOL_DEPTH,
    _TRAIN_SPLIT,
    _answer_generator,
    _build_dense_channels,
    _build_lexical_channels,
    _candidate_pool_summary,
    _candidate_pools_by_split,
    _contains_forbidden_key,
    _evaluate_profile,
    _evaluation_channels,
    _fingerprint,
    _load_json_object,
    _profile_configs,
    _profile_report,
    _ProfileConfig,
    _public_channel_catalog,
    _public_selected_config,
    _QuestionTrace,
    _retrieval_results_for_profile,
    _selected_append_config,
    _split_deltas,
    _split_profile_report,
    _stage128_summary,
    _trace_sample,
    _train_fold_reports,
    _validate_options,
)
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    EncoderFactory,
    _build_train_fold_assignments,
    _rounded_ratio,
    _section_summary,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
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

_STAGE = "Stage 132"
_CREATED_AT = "2026-07-16"
_ANALYSIS_ID = "primeqa_hybrid_append_candidate_evidence_shortlist_validation_v1"
_SOURCE_STAGE131_STATUS = (
    "primeqa_hybrid_append_candidate_evidence_shortlist_redesign_protocol_frozen"
)
_SOURCE_STAGE131_PROTOCOL_ID = (
    "primeqa_hybrid_append_candidate_evidence_shortlist_redesign_protocol_v1"
)
_SOURCE_STAGE131_NEXT = "run_append_candidate_evidence_shortlist_train_cv_dev_validation"
_ALLOWED_DEVELOPMENT_SPLITS = (_TRAIN_SPLIT, _DEV_SPLIT)
_FORBIDDEN_FINAL_SPLITS = ("test",)
_STAGE131_FAMILY_ID = "stage131_append_candidate_evidence_shortlist_redesign"
_STAGE129_FAILED_PROFILE_ID = "stage128_prefix_append_top400_agent_pool"
_MAX_REPLACEMENT_APPEND_SLOTS = 2
_MINIMUM_TRAIN_FOLDS = 5
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
class _AppendShortlistProfile:
    profile_id: str
    config_id: str
    selection_role: str
    protected_prefix_slots: int
    replacement_append_slots: int
    append_sidecar_slots: int
    append_sidecar_can_generate_answer_text: bool
    append_sidecar_can_support_citation_verification: bool
    answer_context_depth: int = _AGENT_EVIDENCE_CONTEXT_DEPTH
    retrieval_depth: int = _TARGET_POOL_DEPTH
    verifier_max_citation_rank: int = _TARGET_POOL_DEPTH


@dataclass(frozen=True)
class PrimeQAHybridAppendCandidateEvidenceShortlistValidationVisualization:
    """One generated Stage132 append-shortlist validation chart."""

    name: str
    path: str


class _AppendCandidateEvidenceShortlister:
    """Prefix-protected shortlister for Stage128 append candidates."""

    def __init__(
        self,
        *,
        protected_prefix_slots: int,
        replacement_append_slots: int,
        max_text_chars: int = 5000,
    ) -> None:
        if protected_prefix_slots < 0:
            raise ValueError("protected_prefix_slots must be non-negative")
        if replacement_append_slots < 0:
            raise ValueError("replacement_append_slots must be non-negative")
        if max_text_chars <= 0:
            raise ValueError("max_text_chars must be positive")
        self._protected_prefix_slots = protected_prefix_slots
        self._replacement_append_slots = replacement_append_slots
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

        prefix = [result for result in candidates if result.rank <= _BASELINE_PREFIX_DEPTH]
        append = [
            result
            for result in candidates
            if _BASELINE_PREFIX_DEPTH < result.rank <= _TARGET_POOL_DEPTH
        ]
        ranked_prefix = self._rank(query_terms=query_terms, candidates=prefix)
        ranked_append = self._rank(query_terms=query_terms, candidates=append)

        protected_count = min(self._protected_prefix_slots, top_k, len(ranked_prefix))
        protected = ranked_prefix[:protected_count]
        remaining_slots = max(0, top_k - len(protected))
        if remaining_slots == 0:
            return protected[:top_k]

        prefix_fill = ranked_prefix[protected_count : protected_count + remaining_slots]
        append_budget = min(self._replacement_append_slots, remaining_slots)
        append_fill = ranked_append[:append_budget]
        fill_pool = self._rank(
            query_terms=query_terms,
            candidates=[*prefix_fill, *append_fill],
        )
        selected = [*protected, *fill_pool[:remaining_slots]]
        return self._rank(query_terms=query_terms, candidates=selected)[:top_k]

    def _rank(
        self,
        *,
        query_terms: set[str],
        candidates: Sequence[RetrievalResult],
    ) -> list[RetrievalResult]:
        return sorted(
            candidates,
            key=lambda result: (
                -self._score(query_terms=query_terms, result=result),
                result.rank,
                result.document.id,
            ),
        )

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


def run_primeqa_hybrid_append_candidate_evidence_shortlist_validation(
    *,
    stage131_protocol_path: Path,
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
    train_fold_count: int = _MINIMUM_TRAIN_FOLDS,
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
    """Run the Stage132 train-CV/dev append-candidate shortlist validation."""

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

    stage131_protocol = _load_json_object(stage131_protocol_path)
    stage128_protocol = _load_json_object(stage128_protocol_path)
    stage125_protocol = _load_json_object(stage125_protocol_path)
    stage131_summary = _stage131_summary(stage131_protocol)
    stage128_summary = _stage128_summary(stage128_protocol)
    shortlist_profiles = _shortlist_profiles_from_protocol(stage131_protocol)
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
        stage131_summary=stage131_summary,
        stage128_summary=stage128_summary,
        selected_config=selected_config,
        shortlist_profiles=shortlist_profiles,
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
            stage131_protocol_path=stage131_protocol_path,
            stage128_protocol_path=stage128_protocol_path,
            stage125_protocol_path=stage125_protocol_path,
            stage80_report_path=stage80_report_path,
            train_split_path=train_split_path,
            dev_split_path=dev_split_path,
            documents_path=documents_path,
            user_confirmed_validation=user_confirmed_validation,
            confirmation_note=confirmation_note,
            stage131_summary=stage131_summary,
            stage128_summary=stage128_summary,
            selected_config=selected_config,
            shortlist_profiles=shortlist_profiles,
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

    control_profiles = _stage129_control_profiles(selected_config)
    control_traces = {
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
        for profile in control_profiles
    }
    append_traces = {
        profile.profile_id: _evaluate_append_profile(
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
        for profile in shortlist_profiles
    }
    evaluated_at = time.perf_counter()

    baseline_traces = control_traces[_BASELINE_PROFILE_ID]
    stage116_traces = control_traces[_STAGE116_PROFILE_ID]
    control_profile_reports = {
        profile.profile_id: _profile_report(
            profile=profile,
            traces_by_split=control_traces[profile.profile_id],
            fold_assignments=train_fold_assignments,
            baseline_traces_by_split=baseline_traces,
            stage116_traces_by_split=stage116_traces,
        )
        for profile in control_profiles
    }
    append_profile_reports = {
        profile.profile_id: _append_profile_report(
            profile=profile,
            traces_by_split=append_traces[profile.profile_id],
            fold_assignments=train_fold_assignments,
            baseline_traces_by_split=baseline_traces,
            stage116_traces_by_split=stage116_traces,
        )
        for profile in shortlist_profiles
    }
    profile_reports = {**control_profile_reports, **append_profile_reports}
    train_candidate_reviews = _train_candidate_reviews(
        shortlist_profiles=shortlist_profiles,
        profile_reports=append_profile_reports,
        stage116_report=control_profile_reports[_STAGE116_PROFILE_ID],
        baseline_report=control_profile_reports[_BASELINE_PROFILE_ID],
        stage131_summary=stage131_summary,
    )
    train_selection = _select_config_on_train(train_candidate_reviews)
    dev_report = _dev_report_observations(
        train_selection=train_selection,
        train_candidate_reviews=train_candidate_reviews,
        profile_reports=append_profile_reports,
        stage116_report=control_profile_reports[_STAGE116_PROFILE_ID],
    )
    report_payload = {
        "profile_reports": profile_reports,
        "train_candidate_reviews": train_candidate_reviews,
        "train_selection": train_selection,
        "candidate_pool_summary": _candidate_pool_summary(candidate_pools_by_split),
    }
    post_checks = _post_evaluation_guard_checks(
        report_payload=report_payload,
        stage131_summary=stage131_summary,
        selected_config=selected_config,
        shortlist_profiles=shortlist_profiles,
    )
    guard_checks = pre_checks + post_checks
    checked_at = time.perf_counter()

    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train/dev-only validation for the frozen Stage131 append-candidate "
            "evidence shortlist redesign protocol. This stage loads train/dev "
            "frozen split rows, local corpus documents, the public-safe Stage131 "
            "protocol, the public-safe Stage128 candidate-pool protocol, and "
            "the public-safe Stage125 executable append config. It validates "
            "the three Stage131 shortlist configs against Stage116 top200 "
            "control using train grouped cross-validation and reports dev once "
            "without retuning. It does not load the test split, does not run "
            "final test metrics, does not write raw question, answer, document, "
            "token, or candidate-row fields, does not add fallback strategies, "
            "and does not change runtime defaults."
        ),
        "user_confirmation": {
            "confirmed": bool(user_confirmed_validation),
            "confirmation_note": confirmation_note,
        },
        "split_contract": _split_contract(),
        "source_files": _source_files(
            stage131_protocol_path=stage131_protocol_path,
            stage128_protocol_path=stage128_protocol_path,
            stage125_protocol_path=stage125_protocol_path,
            stage80_report_path=stage80_report_path,
            train_split_path=train_split_path,
            dev_split_path=dev_split_path,
            documents_path=documents_path,
        ),
        "stage131_summary": stage131_summary,
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
            "answer_context_depth": _AGENT_EVIDENCE_CONTEXT_DEPTH,
            "max_replacement_append_slots": _MAX_REPLACEMENT_APPEND_SLOTS,
            "baseline_profile_id": _BASELINE_PROFILE_ID,
            "stage116_control_profile_id": _STAGE116_PROFILE_ID,
        },
        "loaded_data_summary": {
            "split_samples": summarize_primeqa_hybrid_split_samples(split_samples),
            "document_count": len(documents),
            **_section_summary(sections_by_document),
            "test_split_loaded": False,
        },
        "dense_channel_preflight": dense_summary,
        "channel_catalog": channel_catalog,
        "candidate_pool_summary": report_payload["candidate_pool_summary"],
        "profile_reports": profile_reports,
        "train_candidate_reviews": train_candidate_reviews,
        "train_selection": train_selection,
        "dev_report_observations": dev_report,
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks=guard_checks, train_selection=train_selection),
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
            "evaluate_shortlist_profiles": round(evaluated_at - pools_built_at, 3),
            "summarize_and_guard": round(checked_at - evaluated_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }
    return {**report, "public_safe_contract": _public_safe_contract(report)}


def write_primeqa_hybrid_append_candidate_evidence_shortlist_validation_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridAppendCandidateEvidenceShortlistValidationVisualization]:
    """Write SVG charts for Stage132 validation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage132_train_cv_verified_f1_delta.svg": render_horizontal_bar_chart_svg(
            title="Stage132 train-CV verified F1 delta vs Stage116",
            bars=_train_candidate_delta_bars(
                report,
                metric="verified_average_token_f1_delta",
                value_format="{:+.4f}",
            ),
            x_label="delta",
            width=1580,
            margin_left=820,
        ),
        "stage132_train_cv_gold_citation_delta.svg": render_horizontal_bar_chart_svg(
            title="Stage132 train-CV gold citation count delta vs Stage116",
            bars=_train_candidate_delta_bars(
                report,
                metric="verified_gold_citation_count_delta",
                value_format="{:+.0f}",
            ),
            x_label="delta",
            width=1580,
            margin_left=820,
        ),
        "stage132_train_cv_changed_answer_rate.svg": render_horizontal_bar_chart_svg(
            title="Stage132 train-CV changed answer rate",
            bars=_changed_answer_rate_bars(report),
            x_label="rate",
            width=1480,
            margin_left=760,
        ),
        "stage132_selected_evidence_region_mix.svg": render_horizontal_bar_chart_svg(
            title="Stage132 selected evidence region mix",
            bars=_selected_evidence_region_bars(report),
            x_label="citation count",
            width=1680,
            margin_left=900,
        ),
        "stage132_train_config_guard_status.svg": render_horizontal_bar_chart_svg(
            title="Stage132 train config guard status",
            bars=_train_config_guard_bars(report),
            x_label="1 means pass",
            width=1900,
            margin_left=1060,
        ),
        "stage132_decision_flags.svg": render_horizontal_bar_chart_svg(
            title="Stage132 decision flags",
            bars=_decision_flag_bars(report),
            x_label="1 means true",
            width=1500,
            margin_left=820,
        ),
        "stage132_guard_check_status.svg": render_horizontal_bar_chart_svg(
            title="Stage132 guard checks",
            bars=_guard_check_bars(report),
            x_label="1 means passed",
            width=2000,
            margin_left=1160,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridAppendCandidateEvidenceShortlistValidationVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _stage131_summary(stage131_protocol: Mapping[str, Any]) -> dict[str, Any]:
    decision = stage131_protocol.get("decision") or {}
    frozen = stage131_protocol.get("frozen_protocol") or {}
    plan = frozen.get("selection_and_validation_plan") or {}
    source = frozen.get("source_failure_review") or {}
    public_safe = stage131_protocol.get("public_safe_contract") or {}
    stage130_summary = stage131_protocol.get("stage130_summary") or {}
    return {
        "stage": stage131_protocol.get("stage"),
        "protocol_id": stage131_protocol.get("protocol_id"),
        "decision_status": decision.get("status"),
        "recommended_next_direction": decision.get("recommended_next_direction"),
        "can_run_append_shortlist_validation_now": decision.get(
            "can_run_append_shortlist_validation_now"
        ),
        "can_open_final_test_gate_now": decision.get("can_open_final_test_gate_now"),
        "can_run_final_test_metrics_now": decision.get("can_run_final_test_metrics_now"),
        "can_use_test_for_tuning": decision.get("can_use_test_for_tuning"),
        "runtime_defaultization_allowed_now": decision.get(
            "runtime_defaultization_allowed_now"
        ),
        "fallback_strategies_enabled": decision.get("fallback_strategies_enabled"),
        "default_runtime_policy": decision.get("default_runtime_policy"),
        "stage128_direct_agent_integration_path_remains_blocked": decision.get(
            "stage128_direct_agent_integration_path_remains_blocked"
        ),
        "source_stage130_status": source.get("status"),
        "candidate_config_count": len(frozen.get("candidate_shortlist_configs") or []),
        "selection_split": plan.get("selection_split"),
        "selection_mode": plan.get("selection_mode"),
        "minimum_train_folds": plan.get("minimum_train_folds"),
        "validation_split": plan.get("validation_split"),
        "dev_mode": plan.get("dev_mode"),
        "primary_train_cv_guard": plan.get("primary_train_cv_guard"),
        "test_rules": plan.get("test_rules") or {},
        "runtime_rules": plan.get("runtime_rules") or {},
        "stage130_train_changed_answer_rate": stage130_summary.get(
            "train_changed_answer_rate"
        ),
        "stage130_train_gold_citation_delta": stage130_summary.get(
            "train_gold_citation_delta"
        ),
        "stage130_train_gold_hit_delta": stage130_summary.get("train_gold_hit_delta"),
        "public_safe_forbidden_keys_found": public_safe.get("forbidden_keys_found") or [],
    }


def _shortlist_profiles_from_protocol(
    stage131_protocol: Mapping[str, Any],
) -> list[_AppendShortlistProfile]:
    configs = (
        (stage131_protocol.get("frozen_protocol") or {}).get(
            "candidate_shortlist_configs"
        )
        or []
    )
    profiles = []
    for config in configs:
        config_id = str(config["config_id"])
        profiles.append(
            _AppendShortlistProfile(
                profile_id=f"stage132_{config_id}",
                config_id=config_id,
                selection_role=str(config["selection_role"]),
                protected_prefix_slots=int(config["protected_prefix_slots"]),
                replacement_append_slots=int(config["replacement_append_slots"]),
                append_sidecar_slots=int(config["append_sidecar_slots"]),
                append_sidecar_can_generate_answer_text=bool(
                    config["append_sidecar_can_generate_answer_text"]
                ),
                append_sidecar_can_support_citation_verification=bool(
                    config["append_sidecar_can_support_citation_verification"]
                ),
                answer_context_depth=int(config["answer_context_depth"]),
            )
        )
    return profiles


def _stage129_control_profiles(
    selected_config: Mapping[str, Any],
) -> list[_ProfileConfig]:
    profiles = _profile_configs(selected_config)
    return [
        profile
        for profile in profiles
        if profile.profile_id in {_BASELINE_PROFILE_ID, _STAGE116_PROFILE_ID}
    ]


def _evaluate_append_profile(
    *,
    profile: _AppendShortlistProfile,
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
    retrieval_profile = _ProfileConfig(
        profile_id=profile.profile_id,
        family_id=_STAGE131_FAMILY_ID,
        retrieval_mode="stage128_pool_with_stage131_prefix_protected_shortlist",
        retrieval_depth=profile.retrieval_depth,
        answer_context_depth=profile.answer_context_depth,
        verifier_max_citation_rank=profile.verifier_max_citation_rank,
        is_stage128_candidate=True,
    )
    shortlister = _AppendCandidateEvidenceShortlister(
        protected_prefix_slots=profile.protected_prefix_slots,
        replacement_append_slots=profile.replacement_append_slots,
    )
    return {
        split: [
            _trace_sample(
                sample=sample,
                candidate_pool_results=_retrieval_results_for_profile(
                    profile=retrieval_profile,
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


def _append_profile_report(
    *,
    profile: _AppendShortlistProfile,
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
    return {
        "profile_id": profile.profile_id,
        "family_id": _STAGE131_FAMILY_ID,
        "config_id": profile.config_id,
        "selection_role": profile.selection_role,
        "retrieval_mode": "stage128_pool_with_stage131_prefix_protected_shortlist",
        "retrieval_depth": profile.retrieval_depth,
        "answer_context_depth": profile.answer_context_depth,
        "verifier_max_citation_rank": profile.verifier_max_citation_rank,
        "shortlist_config": {
            "protected_prefix_slots": profile.protected_prefix_slots,
            "replacement_append_slots": profile.replacement_append_slots,
            "append_sidecar_slots": profile.append_sidecar_slots,
            "append_sidecar_can_generate_answer_text": (
                profile.append_sidecar_can_generate_answer_text
            ),
            "append_sidecar_can_support_citation_verification": (
                profile.append_sidecar_can_support_citation_verification
            ),
        },
        "split_reports": {
            "train_cv": train_cv,
            "train_full": split_reports[_TRAIN_SPLIT],
            "dev": split_reports[_DEV_SPLIT],
        },
        "train_fold_reports": _train_fold_reports(
            traces=traces_by_split[_TRAIN_SPLIT],
            baseline_traces=baseline_traces_by_split[_TRAIN_SPLIT],
            stage116_traces=stage116_traces_by_split[_TRAIN_SPLIT],
            fold_assignments=fold_assignments,
        ),
        "train_cv_group_values_written": False,
    }


def _train_candidate_reviews(
    *,
    shortlist_profiles: Sequence[_AppendShortlistProfile],
    profile_reports: Mapping[str, Mapping[str, Any]],
    stage116_report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
    stage131_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reviews = []
    for profile in shortlist_profiles:
        report = profile_reports[profile.profile_id]
        train = report["split_reports"]["train_cv"]
        stage116_train = stage116_report["split_reports"]["train_cv"]
        baseline_train = baseline_report["split_reports"]["train_cv"]
        deltas_vs_stage116 = _split_deltas(
            candidate=train,
            baseline=stage116_train,
            changed_answer_count=train["changed_verified_answers_vs_stage116_control"],
        )
        deltas_vs_baseline = _split_deltas(
            candidate=train,
            baseline=baseline_train,
            changed_answer_count=train["changed_verified_answers_vs_baseline"],
        )
        region_shift = _selected_citation_region_shift(
            candidate=train,
            control=stage116_train,
        )
        train_guard = _config_train_guard(
            deltas_vs_stage116=deltas_vs_stage116,
            candidate_train=train,
            region_shift=region_shift,
            stage131_summary=stage131_summary,
        )
        reviews.append(
            {
                "profile_id": profile.profile_id,
                "config_id": profile.config_id,
                "family_id": _STAGE131_FAMILY_ID,
                "selection_role": profile.selection_role,
                "protected_prefix_slots": profile.protected_prefix_slots,
                "replacement_append_slots": profile.replacement_append_slots,
                "append_sidecar_slots": profile.append_sidecar_slots,
                "deltas_vs_stage116_control": deltas_vs_stage116,
                "deltas_vs_baseline": deltas_vs_baseline,
                "selected_citation_region_shift": region_shift,
                "train_changed_verified_answer_rate_vs_stage116": _rounded_ratio(
                    int(train["changed_verified_answers_vs_stage116_control"]),
                    int(train["row_count"]),
                ),
                "train_cv_guard": train_guard,
            }
        )
    return reviews


def _config_train_guard(
    *,
    deltas_vs_stage116: Mapping[str, Any],
    candidate_train: Mapping[str, Any],
    region_shift: Mapping[str, Any],
    stage131_summary: Mapping[str, Any],
) -> dict[str, Any]:
    changed_rate = _rounded_ratio(
        int(candidate_train["changed_verified_answers_vs_stage116_control"]),
        int(candidate_train["row_count"]),
    )
    stage129_changed_rate = float(
        stage131_summary.get("stage130_train_changed_answer_rate") or 1.0
    )
    gold_delta = int(deltas_vs_stage116["verified_gold_citation_count_delta"])
    append_selected = int(region_shift["append_region_selected_citation_count"])
    prefix_delta = int(region_shift["prefix_like_selected_citation_delta"])
    checks = [
        _check(
            name="verified_f1_delta_vs_stage116_non_negative",
            passed=float(deltas_vs_stage116["verified_average_token_f1_delta"]) >= 0.0,
            observed=deltas_vs_stage116["verified_average_token_f1_delta"],
            expected=">= 0",
        ),
        _check(
            name="gold_citation_count_delta_vs_stage116_non_negative",
            passed=gold_delta >= 0,
            observed=gold_delta,
            expected=">= 0",
        ),
        _check(
            name="answerable_refusal_rate_delta_vs_stage116_non_positive",
            passed=float(deltas_vs_stage116["answerable_refusal_rate_delta"]) <= 0.0,
            observed=deltas_vs_stage116["answerable_refusal_rate_delta"],
            expected="<= 0",
        ),
        _check(
            name="unanswerable_refusal_rate_delta_vs_stage116_non_positive",
            passed=float(deltas_vs_stage116["unanswerable_refusal_rate_delta"]) <= 0.0,
            observed=deltas_vs_stage116["unanswerable_refusal_rate_delta"],
            expected="<= 0",
        ),
        _check(
            name="target_depth_recall_delta_vs_stage116_positive",
            passed=int(deltas_vs_stage116["gold_hit_count_at_profile_depth_delta"]) > 0,
            observed=deltas_vs_stage116["gold_hit_count_at_profile_depth_delta"],
            expected="> 0",
        ),
        _check(
            name="changed_verified_answer_rate_not_above_stage129_candidate",
            passed=changed_rate <= stage129_changed_rate,
            observed=changed_rate,
            expected=f"<= {stage129_changed_rate:.4f}",
        ),
        _check(
            name=(
                "append_selected_citations_do_not_displace_prefix_like_citations_"
                "without_gold_gain"
            ),
            passed=not (append_selected > 0 and prefix_delta < 0 and gold_delta <= 0),
            observed={
                "append_selected_citations": append_selected,
                "prefix_like_selected_citation_delta": prefix_delta,
                "gold_citation_count_delta": gold_delta,
            },
            expected="no prefix-like displacement without positive gold citation gain",
        ),
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "failed_checks": [check["name"] for check in checks if not check["passed"]],
    }


def _select_config_on_train(
    train_candidate_reviews: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    eligible = [
        review
        for review in train_candidate_reviews
        if review["train_cv_guard"]["passed"]
    ]
    ranked = sorted(
        eligible,
        key=lambda review: (
            -int(review["deltas_vs_stage116_control"]["verified_gold_citation_count_delta"]),
            -float(review["deltas_vs_stage116_control"]["verified_average_token_f1_delta"]),
            float(review["train_changed_verified_answer_rate_vs_stage116"]),
            int(review["replacement_append_slots"]),
            str(review["config_id"]),
        ),
    )
    selected = ranked[0] if ranked else None
    return {
        "selection_split": _TRAIN_SPLIT,
        "selection_mode": (
            "train_grouped_cross_validation_append_shortlist_config_selection"
        ),
        "dev_used_for_selection": False,
        "dev_used_for_retuning": False,
        "candidate_count": len(train_candidate_reviews),
        "eligible_config_count": len(eligible),
        "selected_config_id": selected["config_id"] if selected else None,
        "selected_profile_id": selected["profile_id"] if selected else None,
        "selected_family_id": selected["family_id"] if selected else None,
        "selected_train_summary": (
            _selection_metric_summary(selected) if selected else None
        ),
        "selection_ranking": [
            {
                "config_id": review["config_id"],
                "profile_id": review["profile_id"],
                "protected_prefix_slots": review["protected_prefix_slots"],
                "replacement_append_slots": review["replacement_append_slots"],
                "train_verified_f1_delta_vs_stage116": review[
                    "deltas_vs_stage116_control"
                ]["verified_average_token_f1_delta"],
                "train_gold_citation_count_delta_vs_stage116": review[
                    "deltas_vs_stage116_control"
                ]["verified_gold_citation_count_delta"],
                "train_target_depth_gold_hit_delta_vs_stage116": review[
                    "deltas_vs_stage116_control"
                ]["gold_hit_count_at_profile_depth_delta"],
                "train_changed_answer_rate_vs_stage116": review[
                    "train_changed_verified_answer_rate_vs_stage116"
                ],
                "guard_passed": review["train_cv_guard"]["passed"],
                "failed_checks": review["train_cv_guard"]["failed_checks"],
            }
            for review in sorted(
                train_candidate_reviews,
                key=lambda review: (
                    not bool(review["train_cv_guard"]["passed"]),
                    -int(
                        review["deltas_vs_stage116_control"][
                            "verified_gold_citation_count_delta"
                        ]
                    ),
                    -float(
                        review["deltas_vs_stage116_control"][
                            "verified_average_token_f1_delta"
                        ]
                    ),
                    float(review["train_changed_verified_answer_rate_vs_stage116"]),
                    int(review["replacement_append_slots"]),
                    str(review["config_id"]),
                ),
            )
        ],
    }


def _selection_metric_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    deltas = review["deltas_vs_stage116_control"]
    return {
        "config_id": review["config_id"],
        "profile_id": review["profile_id"],
        "verified_average_token_f1_delta": deltas["verified_average_token_f1_delta"],
        "verified_gold_citation_count_delta": deltas[
            "verified_gold_citation_count_delta"
        ],
        "gold_hit_count_at_profile_depth_delta": deltas[
            "gold_hit_count_at_profile_depth_delta"
        ],
        "changed_verified_answer_rate": review[
            "train_changed_verified_answer_rate_vs_stage116"
        ],
    }


def _dev_report_observations(
    *,
    train_selection: Mapping[str, Any],
    train_candidate_reviews: Sequence[Mapping[str, Any]],
    profile_reports: Mapping[str, Mapping[str, Any]],
    stage116_report: Mapping[str, Any],
) -> dict[str, Any]:
    dev_reviews = []
    for review in train_candidate_reviews:
        profile = profile_reports[review["profile_id"]]
        candidate_dev = profile["split_reports"]["dev"]
        stage116_dev = stage116_report["split_reports"]["dev"]
        deltas = _split_deltas(
            candidate=candidate_dev,
            baseline=stage116_dev,
            changed_answer_count=candidate_dev[
                "changed_verified_answers_vs_stage116_control"
            ],
        )
        dev_reviews.append(
            {
                "config_id": review["config_id"],
                "profile_id": review["profile_id"],
                "status": "reported_not_used_for_selection",
                "deltas_vs_stage116_control": deltas,
                "changed_verified_answer_rate_vs_stage116": _rounded_ratio(
                    int(candidate_dev["changed_verified_answers_vs_stage116_control"]),
                    int(candidate_dev["row_count"]),
                ),
                "selected_citation_region_shift": _selected_citation_region_shift(
                    candidate=candidate_dev,
                    control=stage116_dev,
                ),
            }
        )
    selected_profile_id = train_selection.get("selected_profile_id")
    return {
        "validation_split": _DEV_SPLIT,
        "status": "reported_not_used_for_selection",
        "dev_used_for_selection": False,
        "dev_used_for_retuning": False,
        "selected_config_id": train_selection.get("selected_config_id"),
        "selected_profile_id": selected_profile_id,
        "selected_dev_summary": (
            next(
                (
                    review
                    for review in dev_reviews
                    if review["profile_id"] == selected_profile_id
                ),
                None,
            )
            if selected_profile_id
            else None
        ),
        "config_dev_reviews": dev_reviews,
        "dev_gate_status": "report_only_no_runtime_or_test_gate",
    }


def _selected_citation_region_shift(
    *,
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, Any]:
    control_regions = control["selected_evidence_summary"]["rank_region_counts"]
    candidate_regions = candidate["selected_evidence_summary"]["rank_region_counts"]
    region_keys = sorted(set(control_regions) | set(candidate_regions))
    deltas = {
        region: int(candidate_regions.get(region, 0)) - int(control_regions.get(region, 0))
        for region in region_keys
    }
    append_count = int(candidate_regions.get("stage128_append_expansion_201_400", 0))
    control_prefix_like = int(control_regions.get("rank_001_010", 0)) + int(
        control_regions.get("stage116_immutable_prefix_011_200", 0)
    )
    candidate_prefix_like = int(candidate_regions.get("rank_001_010", 0)) + int(
        candidate_regions.get("stage116_immutable_prefix_011_200", 0)
    )
    return {
        "control_rank_region_counts": dict(sorted(control_regions.items())),
        "candidate_rank_region_counts": dict(sorted(candidate_regions.items())),
        "rank_region_count_deltas": deltas,
        "append_region_selected_citation_count": append_count,
        "prefix_like_selected_citation_delta": candidate_prefix_like
        - control_prefix_like,
    }


def _pre_evaluation_guard_checks(
    *,
    stage131_summary: Mapping[str, Any],
    stage128_summary: Mapping[str, Any],
    selected_config: Mapping[str, Any] | None,
    shortlist_profiles: Sequence[_AppendShortlistProfile],
    user_confirmed_validation: bool,
    confirmation_note: str,
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    include_dense_channels: bool,
    dense_summary: Mapping[str, Any],
    train_fold_count: int,
) -> list[dict[str, Any]]:
    return [
        _check(
            name="user_confirmed_stage132_validation",
            passed=user_confirmed_validation and "Stage132" in confirmation_note,
            observed=confirmation_note,
            expected="user confirmed Stage132 validation",
        ),
        _check(
            name="stage131_protocol_frozen",
            passed=stage131_summary.get("decision_status") == _SOURCE_STAGE131_STATUS,
            observed=stage131_summary.get("decision_status"),
            expected=_SOURCE_STAGE131_STATUS,
        ),
        _check(
            name="stage131_protocol_id_matches",
            passed=stage131_summary.get("protocol_id") == _SOURCE_STAGE131_PROTOCOL_ID,
            observed=stage131_summary.get("protocol_id"),
            expected=_SOURCE_STAGE131_PROTOCOL_ID,
        ),
        _check(
            name="stage131_recommends_stage132_validation",
            passed=stage131_summary.get("recommended_next_direction")
            == _SOURCE_STAGE131_NEXT,
            observed=stage131_summary.get("recommended_next_direction"),
            expected=_SOURCE_STAGE131_NEXT,
        ),
        _check(
            name="stage131_runtime_and_test_boundaries_locked",
            passed=stage131_summary.get("can_open_final_test_gate_now") is False
            and stage131_summary.get("can_run_final_test_metrics_now") is False
            and stage131_summary.get("can_use_test_for_tuning") is False
            and stage131_summary.get("runtime_defaultization_allowed_now") is False
            and stage131_summary.get("fallback_strategies_enabled") is False
            and stage131_summary.get("default_runtime_policy") == "unchanged",
            observed=stage131_summary,
            expected="test locked, runtime unchanged, fallback disabled",
        ),
        _check(
            name="stage128_selected_candidate_pool_available",
            passed=selected_config is not None
            and selected_config.get("config_id") == _SELECTED_CONFIG_ID
            and stage128_summary.get("selected_config_id") == _SELECTED_CONFIG_ID,
            observed={
                "selected_config": None
                if selected_config is None
                else selected_config.get("config_id"),
                "stage128_selected_config": stage128_summary.get("selected_config_id"),
            },
            expected=_SELECTED_CONFIG_ID,
        ),
        _check(
            name="stage132_expected_shortlist_profiles_loaded",
            passed=len(shortlist_profiles) == 3
            and all(
                profile.replacement_append_slots <= _MAX_REPLACEMENT_APPEND_SLOTS
                for profile in shortlist_profiles
            ),
            observed=[
                {
                    "config_id": profile.config_id,
                    "replacement_append_slots": profile.replacement_append_slots,
                }
                for profile in shortlist_profiles
            ],
            expected="3 configs, each with <= 2 replacement append slots",
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
            name="train_fold_count_matches_stage131_minimum",
            passed=train_fold_count >= _MINIMUM_TRAIN_FOLDS,
            observed=train_fold_count,
            expected=f">= {_MINIMUM_TRAIN_FOLDS}",
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
    report_payload: Mapping[str, Any],
    stage131_summary: Mapping[str, Any],
    selected_config: Mapping[str, Any],
    shortlist_profiles: Sequence[_AppendShortlistProfile],
) -> list[dict[str, Any]]:
    pool_summary = report_payload["candidate_pool_summary"]
    train_selection = report_payload["train_selection"]
    public_payload = {
        "profile_reports": report_payload["profile_reports"],
        "train_candidate_reviews": report_payload["train_candidate_reviews"],
        "train_selection": train_selection,
        "candidate_pool_summary": pool_summary,
    }
    return [
        _check(
            name="stage132_prefix_identity_preserved",
            passed=pool_summary["all_splits_prefix_identity_violation_count"] == 0,
            observed=pool_summary["all_splits_prefix_identity_violation_count"],
            expected=0,
        ),
        _check(
            name="stage132_append_budget_not_exceeded",
            passed=pool_summary["all_splits_append_budget_exceeded_count"] == 0,
            observed=pool_summary["all_splits_append_budget_exceeded_count"],
            expected=0,
        ),
        _check(
            name="stage132_target_pool_depth_matches_stage128_protocol",
            passed=int(selected_config["append_generation"]["target_pool_depth"])
            == _TARGET_POOL_DEPTH,
            observed=selected_config["append_generation"]["target_pool_depth"],
            expected=_TARGET_POOL_DEPTH,
        ),
        _check(
            name="stage132_shortlist_profiles_match_protocol",
            passed=len(report_payload["train_candidate_reviews"])
            == len(shortlist_profiles)
            == int(stage131_summary.get("candidate_config_count") or 0),
            observed={
                "reviews": len(report_payload["train_candidate_reviews"]),
                "profiles": len(shortlist_profiles),
                "protocol_configs": stage131_summary.get("candidate_config_count"),
            },
            expected=3,
        ),
        _check(
            name="stage132_train_selection_uses_train_only",
            passed=train_selection.get("selection_split") == _TRAIN_SPLIT
            and train_selection.get("dev_used_for_selection") is False
            and train_selection.get("dev_used_for_retuning") is False,
            observed={
                "selection_split": train_selection.get("selection_split"),
                "dev_used_for_selection": train_selection.get("dev_used_for_selection"),
                "dev_used_for_retuning": train_selection.get("dev_used_for_retuning"),
            },
            expected="train only, dev not used",
        ),
        _check(
            name="stage132_dev_report_only",
            passed=True,
            observed="dev reported only; no selection or retuning",
            expected="dev report only",
        ),
        _check(
            name="stage132_test_locked",
            passed=stage131_summary.get("can_run_final_test_metrics_now") is False
            and stage131_summary.get("can_use_test_for_tuning") is False,
            observed={
                "can_run_final_test_metrics_now": stage131_summary.get(
                    "can_run_final_test_metrics_now"
                ),
                "can_use_test_for_tuning": stage131_summary.get("can_use_test_for_tuning"),
            },
            expected="test locked",
        ),
        _check(
            name="stage132_runtime_defaults_unchanged",
            passed=stage131_summary.get("default_runtime_policy") == "unchanged",
            observed=stage131_summary.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage132_no_fallback_strategies",
            passed=stage131_summary.get("fallback_strategies_enabled") is False,
            observed=stage131_summary.get("fallback_strategies_enabled"),
            expected=False,
        ),
        _check(
            name="stage132_public_outputs_have_no_forbidden_keys",
            passed=not _contains_forbidden_key(public_payload)
            and _public_safe_contract(public_payload)["forbidden_keys_found"] == [],
            observed=_public_safe_contract(public_payload)["forbidden_keys_found"],
            expected=[],
        ),
        _check(
            name="stage132_train_cv_group_values_not_written",
            passed=all(
                report.get("train_cv_group_values_written") is False
                for report in report_payload["profile_reports"].values()
            ),
            observed={
                profile_id: report.get("train_cv_group_values_written")
                for profile_id, report in report_payload["profile_reports"].items()
            },
            expected=False,
        ),
    ]


def _decision(
    *,
    guard_checks: Sequence[Mapping[str, Any]],
    train_selection: Mapping[str, Any],
) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    base = {
        "analysis_id": _ANALYSIS_ID,
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
            "status": (
                "primeqa_hybrid_append_candidate_evidence_shortlist_validation_blocked"
            ),
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "selected_config_id": None,
            "selected_profile_id": None,
            "recommended_next_direction": "fix_stage132_validation_guards",
        }
    if train_selection.get("selected_config_id"):
        return {
            **base,
            "status": (
                "primeqa_hybrid_append_candidate_evidence_shortlist_validation_completed"
            ),
            "failed_checks": [],
            "selected_config_id": train_selection["selected_config_id"],
            "selected_profile_id": train_selection["selected_profile_id"],
            "selected_family_id": train_selection["selected_family_id"],
            "eligible_config_count": train_selection["eligible_config_count"],
            "can_continue_train_dev_development": True,
            "recommended_next_direction": (
                "review_append_candidate_evidence_shortlist_selected_config"
            ),
        }
    return {
        **base,
        "status": (
            "primeqa_hybrid_append_candidate_evidence_shortlist_validation_"
            "completed_no_selection"
        ),
        "failed_checks": [],
        "selected_config_id": None,
        "selected_profile_id": None,
        "eligible_config_count": 0,
        "positive_signal_found": any(
            int(row.get("train_gold_citation_count_delta_vs_stage116") or 0) > 0
            or float(row.get("train_verified_f1_delta_vs_stage116") or 0.0) > 0.0
            for row in train_selection.get("selection_ranking") or []
        ),
        "can_continue_train_dev_development": True,
        "recommended_next_direction": (
            "review_append_candidate_evidence_shortlist_validation_failures"
        ),
    }


def _blocked_report(
    *,
    stage131_protocol_path: Path,
    stage128_protocol_path: Path,
    stage125_protocol_path: Path,
    stage80_report_path: Path | None,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
    user_confirmed_validation: bool,
    confirmation_note: str,
    stage131_summary: Mapping[str, Any],
    stage128_summary: Mapping[str, Any],
    selected_config: Mapping[str, Any] | None,
    shortlist_profiles: Sequence[_AppendShortlistProfile],
    split_samples: Mapping[str, Sequence[PrimeQAHybridSplitSample]],
    documents: Sequence[PrimeQADocument],
    sections_by_document: Mapping[str, Sequence[Any]],
    dense_summary: Mapping[str, Any],
    guard_checks: Sequence[Mapping[str, Any]],
    timing_seconds: Mapping[str, float],
) -> dict[str, Any]:
    train_selection = _empty_train_selection(shortlist_profiles)
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": "Stage132 blocked before evaluation by guard checks.",
        "user_confirmation": {
            "confirmed": bool(user_confirmed_validation),
            "confirmation_note": confirmation_note,
        },
        "split_contract": _split_contract(),
        "source_files": _source_files(
            stage131_protocol_path=stage131_protocol_path,
            stage128_protocol_path=stage128_protocol_path,
            stage125_protocol_path=stage125_protocol_path,
            stage80_report_path=stage80_report_path,
            train_split_path=train_split_path,
            dev_split_path=dev_split_path,
            documents_path=documents_path,
        ),
        "stage131_summary": dict(stage131_summary),
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
        "train_candidate_reviews": [],
        "train_selection": train_selection,
        "dev_report_observations": {
            "validation_split": _DEV_SPLIT,
            "status": "not_run_pre_evaluation_guard_failed",
            "dev_gate_status": "not_run",
        },
        "guard_checks": list(guard_checks),
        "decision": _decision(guard_checks=guard_checks, train_selection=train_selection),
        "timing_seconds": dict(timing_seconds),
    }


def _empty_train_selection(
    shortlist_profiles: Sequence[_AppendShortlistProfile],
) -> dict[str, Any]:
    return {
        "selection_split": _TRAIN_SPLIT,
        "selection_mode": (
            "train_grouped_cross_validation_append_shortlist_config_selection"
        ),
        "dev_used_for_selection": False,
        "dev_used_for_retuning": False,
        "candidate_count": len(shortlist_profiles),
        "eligible_config_count": 0,
        "selected_config_id": None,
        "selected_profile_id": None,
        "selected_family_id": None,
        "selected_train_summary": None,
        "selection_ranking": [],
    }


def _split_contract() -> dict[str, Any]:
    return {
        "split_name": "primeqa_hybrid_stage68_v1",
        "protocol_version": "primeqa_hybrid_split_v1",
        "development_splits": list(_ALLOWED_DEVELOPMENT_SPLITS),
        "selection_split": _TRAIN_SPLIT,
        "validation_split": _DEV_SPLIT,
        "dev_selection_used": False,
        "dev_retuning_used": False,
        "forbidden_final_splits": sorted(_FORBIDDEN_FINAL_SPLITS),
    }


def _source_files(
    *,
    stage131_protocol_path: Path,
    stage128_protocol_path: Path,
    stage125_protocol_path: Path,
    stage80_report_path: Path | None,
    train_split_path: Path,
    dev_split_path: Path,
    documents_path: Path,
) -> dict[str, Any]:
    files = {
        "stage131_protocol": _fingerprint(stage131_protocol_path),
        "stage128_protocol": _fingerprint(stage128_protocol_path),
        "stage125_protocol": _fingerprint(stage125_protocol_path),
        "train_split": _fingerprint(train_split_path),
        "dev_split": _fingerprint(dev_split_path),
        "corpus_documents": _fingerprint(documents_path),
    }
    if stage80_report_path is not None:
        files["stage80_dense_cache_report"] = _fingerprint(stage80_report_path)
    return files


def _selected_channel_top_k(selected_config: Mapping[str, Any] | None) -> int:
    if selected_config is None:
        return _TARGET_POOL_DEPTH
    return int(selected_config["append_generation"]["channel_top_k"])


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _train_candidate_delta_bars(
    report: Mapping[str, Any],
    *,
    metric: str,
    value_format: str,
) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(review["config_id"]),
            value=float(review["deltas_vs_stage116_control"][metric]),
            value_label=value_format.format(
                float(review["deltas_vs_stage116_control"][metric])
            ),
        )
        for review in report.get("train_candidate_reviews") or []
    ]


def _changed_answer_rate_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(review["config_id"]),
            value=float(review["train_changed_verified_answer_rate_vs_stage116"]),
            value_label=(
                f"{float(review['train_changed_verified_answer_rate_vs_stage116']):.2%}"
            ),
        )
        for review in report.get("train_candidate_reviews") or []
    ]


def _selected_evidence_region_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    bars = []
    for review in report.get("train_candidate_reviews") or []:
        shift = review["selected_citation_region_shift"]
        bars.append(
            BarDatum(
                label=f"{review['config_id']} append citations",
                value=float(shift["append_region_selected_citation_count"]),
                value_label=str(shift["append_region_selected_citation_count"]),
            )
        )
        bars.append(
            BarDatum(
                label=f"{review['config_id']} prefix-like delta",
                value=float(shift["prefix_like_selected_citation_delta"]),
                value_label=f"{int(shift['prefix_like_selected_citation_delta']):+d}",
            )
        )
    return bars


def _train_config_guard_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=f"{review['config_id']} guard",
            value=1.0 if review["train_cv_guard"]["passed"] else 0.0,
            value_label="pass" if review["train_cv_guard"]["passed"] else "fail",
        )
        for review in report.get("train_candidate_reviews") or []
    ]


def _decision_flag_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    decision = report.get("decision") or {}
    flags = (
        "can_continue_train_dev_development",
        "can_open_final_test_gate_now",
        "can_run_final_test_metrics_now",
        "can_use_test_for_tuning",
        "runtime_defaultization_allowed_now",
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
        for check in report.get("guard_checks") or []
    ]
